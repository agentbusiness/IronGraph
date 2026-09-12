// Test-only module. Clippy's `allow-expect-in-tests` covers `#[test]` bodies but not the helper
// functions those tests call, and a failed expectation in a fixture is the intended way for a test
// to fail. The production denial of `expect` is unaffected.
#![allow(clippy::expect_used)]

use std::sync::{Mutex, MutexGuard, OnceLock};

use candle_core::{DType, Device, Tensor};

use super::{
    METAL_ROW_INSTRUCTION_WORDS, METAL_ROW_OPCODE_INTEGER_CONSTANT, METAL_ROW_PROOF_HEADER_WORDS,
    METAL_ROW_SCOPE_EXPRESSION, MetalResidentRowFinalize, MetalResidentRowProof,
    MetalResidentRowProofArgs, MetalRowSortStrategy, candle_error, decode_metal_row_packet,
    metal_resident_row_program_scratch_breakdown, metal_resident_row_program_scratch_bytes,
    metal_row_finalize_args, metal_row_metadata_words, metal_row_proof_args,
    metal_row_sort_strategy,
};
use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    execution::{
        BackendKind, DeviceMemoryGovernor, ResidentDeviceCompletion, ResidentEntityBinding,
        ResidentExecutionId, ResidentNodeBinding, ResidentNodePipelineRequest,
        ResidentObligationScope, ResidentRowColumn, ResidentRowInstruction, ResidentRowOperation,
        ResidentRowProgram, ResidentRowProgramManifest, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentRowProgramResultParts, ResidentRowSortKey,
        ResidentRowValueType,
    },
    graph::LayerMask,
    types::{LabelId, PropertyId},
};

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const PROOF_ORDER: u32 = 4;
const PROOF_OPTIMALITY: u32 = 8;
const PROOF_GRAPH_ALIGNMENT: u32 = 32;
const PROOF_DESCRIPTOR: u32 = 64;
const ROW_EVALUATION_OK: i64 = u32::MAX as i64;
const STRING_REFERENCE_MARKER: u32 = 0x8000_0000;

fn metal_proof_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn key(register: u16, descending: bool, nulls_first: bool) -> ResidentRowSortKey {
    ResidentRowSortKey {
        register,
        descending,
        nulls_first,
    }
}

fn request(
    seed: u64,
    rows: usize,
    register_count: usize,
    sort_keys: Vec<ResidentRowSortKey>,
    offset: usize,
    limit: usize,
) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: (0..register_count)
            .map(|_| ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::IntegerConstant(0),
            })
            .collect(),
    };
    request_with_program(seed, rows, program, sort_keys, offset, limit, vec![0])
}

fn request_with_program(
    seed: u64,
    rows: usize,
    program: ResidentRowProgram,
    sort_keys: Vec<ResidentRowSortKey>,
    offset: usize,
    limit: usize,
    final_registers: Vec<u16>,
) -> Result<ResidentRowProgramRequest> {
    let manifest = ResidentRowProgramManifest::build(
        &program,
        &sort_keys,
        offset,
        limit,
        rows,
        &final_registers,
        seed.checked_mul(100)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| crate::Error::internal("proof test obligation ID overflow"))?,
    )?;
    Ok(ResidentRowProgramRequest {
        project: PROJECT,
        expected_bookmark: Bookmark { term: 7, index: 9 },
        expected_graph_revision: 11,
        expected_layout_version: 13,
        execution: ResidentExecutionId {
            high: 0x5052_4f4f_4654_4553,
            low: seed,
        },
        input: ResidentNodePipelineRequest {
            project: PROJECT,
            labels: vec![LabelId(1)],
            layers: LayerMask::ALL,
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
        },
        program,
        manifest,
        sort_keys,
        offset,
        limit,
        max_output_rows: rows,
        final_registers,
    })
}

fn generated_string_request(
    seed: u64,
    rows: usize,
    capacity: usize,
    descending: bool,
    nulls_first: bool,
) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            ResidentRowInstruction {
                output_type: ResidentRowValueType::String,
                operation: ResidentRowOperation::StringConstant("x".repeat(capacity)),
            },
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::IntegerConstant(0),
            },
        ],
    };
    request_with_program(
        seed,
        rows,
        program,
        vec![key(0, descending, nulls_first)],
        0,
        rows,
        vec![1],
    )
}

fn temporal_request(
    seed: u64,
    rows: usize,
    value_type: ResidentRowValueType,
    maximum_timezone_bytes: u32,
    final_projection: bool,
) -> Result<ResidentRowProgramRequest> {
    let binding = ResidentEntityBinding::Node(ResidentNodeBinding::Start);
    let property = PropertyId(1);
    let operation = match value_type {
        ResidentRowValueType::Date => ResidentRowOperation::LoadDateProperty { binding, property },
        ResidentRowValueType::LocalTime => {
            ResidentRowOperation::LoadLocalTimeProperty { binding, property }
        }
        ResidentRowValueType::ZonedTime => {
            ResidentRowOperation::LoadZonedTimeProperty { binding, property }
        }
        ResidentRowValueType::LocalDateTime => {
            ResidentRowOperation::LoadLocalDateTimeProperty { binding, property }
        }
        ResidentRowValueType::ZonedDateTime => ResidentRowOperation::LoadZonedDateTimeProperty {
            binding,
            property,
            maximum_timezone_bytes,
        },
        _ => {
            return Err(crate::Error::internal(
                "temporal proof test requested a non-temporal register",
            ));
        }
    };
    request_with_program(
        seed,
        rows,
        ResidentRowProgram {
            instructions: vec![ResidentRowInstruction {
                output_type: value_type,
                operation,
            }],
        },
        vec![key(0, false, false)],
        0,
        rows,
        final_projection.then_some(0).into_iter().collect(),
    )
}

fn evaluated_words(registers: &[Vec<i64>], validity: &[Vec<u8>]) -> Vec<i64> {
    assert_eq!(registers.len(), validity.len());
    let rows = registers.first().map_or(0, Vec::len);
    assert!(registers.iter().all(|column| column.len() == rows));
    assert!(validity.iter().all(|column| column.len() == rows));
    let status_base = 1 + registers.len() * rows * 2;
    let mut words = vec![0_i64; status_base + rows];
    // The production evaluator initializes this word before executing any row. These proof tests
    // construct the post-evaluation tensor directly, so they must carry the same no-error marker.
    words[0] = ROW_EVALUATION_OK;
    words[status_base..].fill(ROW_EVALUATION_OK);
    for (register, values) in registers.iter().enumerate() {
        let start = 1 + register * rows;
        words[start..start + rows].copy_from_slice(values);
        let validity_start = 1 + registers.len() * rows + register * rows;
        for (target, source) in words[validity_start..validity_start + rows]
            .iter_mut()
            .zip(&validity[register])
        {
            *target = i64::from(*source);
        }
    }
    words
}

fn register_value_offset(rows: usize, register: usize, row: usize) -> usize {
    1 + register * rows + row
}

fn register_validity_offset(
    rows: usize,
    register_count: usize,
    register: usize,
    row: usize,
) -> usize {
    1 + register_count * rows + register * rows + row
}

fn packed_string_reference(offset: usize, length: usize) -> i64 {
    let offset = u32::try_from(offset).expect("proof-test string offset must fit u32");
    let length = u32::try_from(length).expect("proof-test string length must fit u32");
    assert_eq!(offset & STRING_REFERENCE_MARKER, 0);
    ((((STRING_REFERENCE_MARKER | offset) as u64) << 32) | u64::from(length)) as i64
}

fn write_evaluated_bytes(words: &mut [i64], offset: usize, bytes: &[u8]) {
    let end = offset
        .checked_add(bytes.len())
        .expect("proof-test byte range must not overflow");
    assert!(end <= std::mem::size_of_val(words));
    for (index, byte) in bytes.iter().copied().enumerate() {
        let absolute = offset + index;
        let word = absolute / std::mem::size_of::<i64>();
        let lane = absolute % std::mem::size_of::<i64>();
        let mut encoded = words[word].to_ne_bytes();
        encoded[lane] = byte;
        words[word] = i64::from_ne_bytes(encoded);
    }
}

fn generated_string_evaluated_words(
    request: &ResidentRowProgramRequest,
    strings: &[&[u8]],
    validity: &[u8],
) -> Result<Vec<i64>> {
    assert_eq!(strings.len(), validity.len());
    assert!(validity.iter().all(|valid| *valid <= 1));
    let rows = strings.len();
    let register_count = request.program.instructions.len();
    let status_base = 1 + register_count * rows * 2;
    let fixed_words = status_base + rows;
    let expected_words = super::metal_row_evaluate_words(request, rows)?;
    assert!(expected_words >= fixed_words);
    let capacity = request
        .program
        .string_register_capacity(0)?
        .expect("generated STRING register must own an arena slot");
    assert!(strings.iter().all(|value| value.len() <= capacity));

    let mut words = vec![0_i64; expected_words];
    words[0] = ROW_EVALUATION_OK;
    words[status_base..fixed_words].fill(ROW_EVALUATION_OK);
    for register in 0..register_count {
        for row in 0..rows {
            words[register_validity_offset(rows, register_count, register, row)] = 1;
        }
    }
    let arena_base = fixed_words * std::mem::size_of::<i64>();
    for (row, (value, valid)) in strings.iter().zip(validity).enumerate() {
        let offset = arena_base + row * capacity;
        write_evaluated_bytes(&mut words, offset, value);
        words[register_value_offset(rows, 0, row)] = packed_string_reference(offset, value.len());
        words[register_validity_offset(rows, register_count, 0, row)] = i64::from(*valid);
    }
    Ok(words)
}

fn temporal_evaluated_words(
    request: &ResidentRowProgramRequest,
    payload: &[i64],
    validity: &[u8],
    sidecars: &[&[i64]],
    timezones: Option<&[&[u8]]>,
) -> Result<Vec<i64>> {
    assert_eq!(request.program.instructions.len(), 1);
    assert_eq!(payload.len(), validity.len());
    assert!(validity.iter().all(|valid| *valid <= 1));
    let rows = payload.len();
    let expected_words = super::metal_row_evaluate_words(request, rows)?;
    let mut words = evaluated_words(&[payload.to_vec()], &[validity.to_vec()]);
    words.resize(expected_words, 0);

    let (sidecar_base, sidecar_lanes) = super::metal_row_temporal_slots(request, rows)?[0];
    assert_eq!(
        sidecar_lanes,
        sidecars.len() + usize::from(timezones.is_some())
    );
    for (lane, values) in sidecars.iter().enumerate() {
        assert_eq!(values.len(), rows);
        let start = sidecar_base + lane * rows;
        words[start..start + rows].copy_from_slice(values);
    }

    if let Some(timezones) = timezones {
        assert_eq!(timezones.len(), rows);
        let (arena_base, capacity) = super::metal_row_string_slots(request, rows)?[0];
        let arena_base = arena_base as usize;
        let capacity = capacity as usize;
        let reference_start = sidecar_base + sidecars.len() * rows;
        for (row, timezone) in timezones.iter().enumerate() {
            assert!(timezone.len() <= capacity);
            let offset = arena_base + row * capacity;
            write_evaluated_bytes(&mut words, offset, timezone);
            words[reference_start + row] = packed_string_reference(offset, timezone.len());
        }
    }
    Ok(words)
}

fn zoned_time_key(local_nanos: i64, offset_seconds: i64) -> i64 {
    local_nanos.saturating_sub(offset_seconds * 1_000_000_000)
}

fn graph_image(rows: usize, positions: &[u64], offset: usize, output_rows: usize) -> Vec<u32> {
    let input = (0..rows)
        .map(|row| 10_000_u32 + row as u32)
        .collect::<Vec<_>>();
    let mut image = input.clone();
    image.extend(
        positions[offset..offset + output_rows]
            .iter()
            .map(|position| {
                input[usize::try_from(*position).expect("proof position exceeds usize")]
            }),
    );
    image
}

fn proof_position_tensor(positions: &[u64], device: &Device) -> Result<Tensor> {
    if positions.is_empty() {
        return Tensor::zeros(1, DType::I64, device).map_err(candle_error);
    }
    let words = positions
        .iter()
        .map(|position| *position as i64)
        .collect::<Vec<_>>();
    Tensor::from_slice(&words, words.len(), device).map_err(candle_error)
}

fn execute_proof(
    device: &Device,
    request: &ResidentRowProgramRequest,
    registers: &[Vec<i64>],
    validity: &[Vec<u8>],
    positions: &[u64],
    graph: &[u32],
) -> Result<(Tensor, Tensor, MetalResidentRowProofArgs)> {
    execute_proof_with_graph_columns(device, request, registers, validity, positions, graph, 1)
}

fn execute_proof_with_graph_columns(
    device: &Device,
    request: &ResidentRowProgramRequest,
    registers: &[Vec<i64>],
    validity: &[Vec<u8>],
    positions: &[u64],
    graph: &[u32],
    graph_column_count: usize,
) -> Result<(Tensor, Tensor, MetalResidentRowProofArgs)> {
    let rows = registers.first().map_or(0, Vec::len);
    let evaluated_words = evaluated_words(registers, validity);
    execute_proof_with_evaluated_words(
        device,
        request,
        rows,
        &evaluated_words,
        positions,
        graph,
        graph_column_count,
    )
}

fn execute_proof_with_evaluated_words(
    device: &Device,
    request: &ResidentRowProgramRequest,
    rows: usize,
    evaluated_words: &[i64],
    positions: &[u64],
    graph: &[u32],
    graph_column_count: usize,
) -> Result<(Tensor, Tensor, MetalResidentRowProofArgs)> {
    let output_rows = request.output_cardinality(rows)?;
    let strategy = metal_row_sort_strategy(request, rows, output_rows)?;
    let args = metal_row_proof_args(
        request,
        rows,
        output_rows,
        positions.len(),
        graph_column_count,
        strategy,
    )?;
    assert_eq!(evaluated_words.len(), args.evaluated_word_count as usize);
    let evaluated = Tensor::from_slice(&evaluated_words, evaluated_words.len(), device)
        .map_err(candle_error)?;
    let positions = proof_position_tensor(positions, device)?;
    let graph = Tensor::from_slice(graph, graph.len(), device).map_err(candle_error)?;
    let proof = evaluated
        .apply_op3_no_bwd(
            &positions,
            &graph,
            &MetalResidentRowProof { arguments: args },
        )
        .map_err(candle_error)?;
    Ok((evaluated, proof, args))
}

fn proof_status(proof: &Tensor) -> Result<u32> {
    let words = proof.to_vec1::<u32>().map_err(candle_error)?;
    assert_eq!(words[0], super::METAL_ROW_PROOF_MAGIC);
    let output_rows = u64::from(words[8]) | (u64::from(words[9]) << 32);
    let proof_position_words = usize::try_from(output_rows)
        .expect("proof output rows must fit usize")
        .max(1)
        .checked_mul(2)
        .expect("proof position shape must not overflow");
    assert_eq!(
        words.len(),
        METAL_ROW_PROOF_HEADER_WORDS + proof_position_words
    );
    Ok(words[1])
}

#[test]
fn row_packet_status_distinguishes_arithmetic_temporal_and_proof_failures() -> Result<()> {
    let request = request(90, 1, 1, Vec::new(), 0, 1)?;
    let mut words = vec![0_i64; super::METAL_ROW_PACKET_HEADER_WORDS];
    words[0] = super::METAL_ROW_PACKET_MAGIC as i64;

    words[1] = 4;
    let error = decode_metal_row_packet(&words, &request)
        .err()
        .expect("status 4 must fail");
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(error.message, "integer negation overflow");

    words[1] = 5;
    let error = decode_metal_row_packet(&words, &request)
        .err()
        .expect("status 5 must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        error.message,
        "Metal resident row device proof rejected result publication"
    );

    words[1] = 6;
    let error = decode_metal_row_packet(&words, &request)
        .err()
        .expect("status 6 must fail");
    assert_eq!(error.code, ErrorCode::TemporalRange);
    assert_eq!(
        error.message,
        "temporal duration arithmetic is outside the supported range"
    );

    words[1] = 7;
    let error = decode_metal_row_packet(&words, &request)
        .err()
        .expect("status 7 must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(
        error.message,
        "native Metal calendar-duration arithmetic requires a fixed-offset timezone"
    );

    words[1] = 8;
    let error = decode_metal_row_packet(&words, &request)
        .err()
        .expect("unknown status must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        error.message,
        "Metal resident row packet has an unknown completion status"
    );
    Ok(())
}

fn zero_input_string_request(seed: u64) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![ResidentRowInstruction {
            output_type: ResidentRowValueType::String,
            operation: ResidentRowOperation::InputColumn(ResidentRowColumn::String {
                offsets: vec![0],
                bytes: Vec::new(),
                validity: Vec::new(),
            }),
        }],
    };
    let mut request =
        request_with_program(seed, 0, program, vec![key(0, false, false)], 0, 0, vec![0])?;
    request.input.labels.clear();
    request.validate()?;
    Ok(request)
}

fn synthetic_zero_input_packet(request: &ResidentRowProgramRequest) -> Result<Vec<i64>> {
    let strategy = metal_row_sort_strategy(request, 0, 0)?;
    assert_eq!(strategy, MetalRowSortStrategy::EmptyWindow);
    let proof = metal_row_proof_args(request, 0, 0, 0, 0, strategy)?;
    assert_eq!(proof.strategy, super::METAL_ROW_PROOF_STRATEGY_EMPTY);
    assert_eq!(proof.position_count, 0);
    let scratch_bytes = request.scratch_bytes(0)?;
    let finalize = metal_row_finalize_args(request, 0, 0, scratch_bytes, &proof)?;
    let mut packet = vec![0_i64; finalize.output_word_count as usize];
    packet[..super::METAL_ROW_PACKET_HEADER_WORDS].copy_from_slice(&[
        super::METAL_ROW_PACKET_MAGIC as i64,
        0,
        finalize.execution_high as i64,
        finalize.execution_low as i64,
        finalize.project_high as i64,
        finalize.project_low as i64,
        finalize.bookmark_term as i64,
        finalize.bookmark_index as i64,
        finalize.graph_revision as i64,
        finalize.layout_version as i64,
        finalize.fingerprint_0 as i64,
        finalize.fingerprint_1 as i64,
        finalize.fingerprint_2 as i64,
        finalize.fingerprint_3 as i64,
        0,
        0,
        0,
        finalize.projection_count as i64,
        finalize.receipt_count as i64,
        finalize.scratch_bytes as i64,
    ]);

    let mut cursor = super::METAL_ROW_PACKET_HEADER_WORDS;
    for register in &request.final_registers {
        packet[cursor] = i64::from(*register);
        packet[cursor + 1] =
            request.program.register_type(*register).ok_or_else(|| {
                crate::Error::internal("zero-input projection register disappeared")
            })? as u8 as i64;
        cursor += super::metal_row_projection_packet_words(request, *register, 0)?;
    }
    for obligation in request.obligations() {
        let ResidentObligationScope::Expression(index) = obligation.scope else {
            return Err(crate::Error::internal(
                "zero-input receipt uses a non-expression scope",
            ));
        };
        packet[cursor..cursor + super::METAL_ROW_PACKET_RECEIPT_WORDS].copy_from_slice(&[
            request.execution.high as i64,
            request.execution.low as i64,
            obligation.id as i64,
            obligation.kind as u8 as i64,
            METAL_ROW_SCOPE_EXPRESSION as i64,
            i64::from(index),
            0,
            0,
            ResidentDeviceCompletion::Metal as u8 as i64,
        ]);
        cursor += super::METAL_ROW_PACKET_RECEIPT_WORDS;
    }
    assert_eq!(cursor, packet.len());
    Ok(packet)
}

#[test]
fn zero_input_packet_binds_empty_positions_receipts_and_fingerprint() -> Result<()> {
    let request = zero_input_string_request(91)?;
    assert_eq!(request.scalar_input_rows()?, Some(0));
    assert_eq!(request.output_cardinality(0)?, 0);
    let packet = synthetic_zero_input_packet(&request)?;
    let decoded = decode_metal_row_packet(&packet, &request)?;
    assert_eq!(decoded.input_cardinality, 0);
    assert!(decoded.source_positions.is_empty());
    assert_eq!(decoded.manifest_fingerprint, request.manifest.fingerprint);
    assert_eq!(decoded.receipts.len(), request.obligations().count());
    assert!(decoded.receipts.iter().all(|receipt| {
        receipt.input_cardinality == 0
            && receipt.output_cardinality == 0
            && receipt.completion == ResidentDeviceCompletion::Metal
    }));
    assert_eq!(
        decoded.projected_columns,
        [crate::ResidentRowProjectedColumn {
            register: 0,
            column: ResidentRowColumn::String {
                offsets: vec![0],
                bytes: Vec::new(),
                validity: Vec::new(),
            },
        }]
    );
    validate_temporal_packet(&packet, &request)?;

    let mut forged_position_count = packet.clone();
    forged_position_count[16] = 1;
    let error = decode_metal_row_packet(&forged_position_count, &request)
        .err()
        .expect("a zero-input packet cannot publish a source position");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut forged_fingerprint = packet.clone();
    forged_fingerprint[10] ^= 1;
    let error = validate_temporal_packet(&forged_fingerprint, &request)
        .expect_err("a zero-input packet cannot change its manifest fingerprint");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let receipt_start = super::METAL_ROW_PACKET_HEADER_WORDS
        + super::metal_row_projection_packet_words(&request, 0, 0)?;
    let mut forged_receipt = packet;
    forged_receipt[receipt_start + 6] = 1;
    let error = validate_temporal_packet(&forged_receipt, &request)
        .expect_err("a zero-input receipt cannot claim a nonzero input cardinality");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

fn metadata(request: &ResidentRowProgramRequest, device: &Device) -> Result<Tensor> {
    let mut words = Vec::with_capacity(metal_row_metadata_words(request)?);
    let rows = request.input.max_output_rows;
    let temporal_slots = super::metal_row_temporal_slots(request, rows)?;
    let string_slots = super::metal_row_string_slots(request, rows)?;
    for (register, instruction) in request.program.instructions.iter().enumerate() {
        let value_type = instruction.output_type as u8 as i64;
        let sidecar_base = temporal_slots[register].0 as i64;
        let (arena_base, arena_capacity) = string_slots[register];
        let descriptor = match &instruction.operation {
            ResidentRowOperation::IntegerConstant(value) => [
                METAL_ROW_OPCODE_INTEGER_CONSTANT as i64,
                value_type,
                *value,
                0,
                0,
                0,
                0,
                0,
            ],
            ResidentRowOperation::StringConstant(_) => [
                super::METAL_ROW_OPCODE_STRING_CONSTANT as i64,
                value_type,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            ResidentRowOperation::LoadDateProperty { .. } => [
                super::METAL_ROW_OPCODE_LOAD_DATE as i64,
                value_type,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            ResidentRowOperation::LoadLocalTimeProperty { .. } => [
                super::METAL_ROW_OPCODE_LOAD_LOCAL_TIME as i64,
                value_type,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            ResidentRowOperation::LoadZonedTimeProperty { .. } => [
                super::METAL_ROW_OPCODE_LOAD_ZONED_TIME as i64,
                value_type,
                0,
                0,
                0,
                sidecar_base,
                0,
                0,
            ],
            ResidentRowOperation::LoadLocalDateTimeProperty { .. } => [
                super::METAL_ROW_OPCODE_LOAD_LOCAL_DATETIME as i64,
                value_type,
                0,
                0,
                0,
                sidecar_base,
                0,
                0,
            ],
            ResidentRowOperation::LoadZonedDateTimeProperty { binding, .. } => [
                match binding {
                    ResidentEntityBinding::Node(_) => {
                        super::METAL_ROW_OPCODE_LOAD_NODE_ZONED_DATETIME as i64
                    }
                    ResidentEntityBinding::Relationship(_) => {
                        super::METAL_ROW_OPCODE_LOAD_EDGE_ZONED_DATETIME as i64
                    }
                },
                value_type,
                0,
                0,
                0,
                sidecar_base,
                i64::from(arena_base),
                i64::from(arena_capacity),
            ],
            _ => {
                return Err(crate::Error::internal(
                    "proof test metadata helper received an unsupported instruction",
                ));
            }
        };
        words.extend(descriptor);
    }
    words.extend(
        request
            .final_registers
            .iter()
            .map(|register| i64::from(*register)),
    );
    for obligation in request.obligations() {
        let ResidentObligationScope::Expression(index) = obligation.scope else {
            return Err(crate::Error::internal(
                "proof test obligation scope changed",
            ));
        };
        words.extend([
            obligation.id as i64,
            obligation.kind as u8 as i64,
            METAL_ROW_SCOPE_EXPRESSION as i64,
            i64::from(index),
        ]);
    }
    assert_eq!(
        request.program.instructions.len() * METAL_ROW_INSTRUCTION_WORDS
            + request.final_registers.len()
            + request.obligations().count() * 4,
        words.len()
    );
    Tensor::from_slice(&words, words.len(), device).map_err(candle_error)
}

fn finalize_temporal_packet(
    device: &Device,
    request: &ResidentRowProgramRequest,
    evaluated_words: &[i64],
    positions: &[u64],
) -> Result<Vec<i64>> {
    let rows = request.input.max_output_rows;
    let graph = graph_image(rows, positions, 0, positions.len());
    let (evaluated, proof, proof_args) = execute_proof_with_evaluated_words(
        device,
        request,
        rows,
        evaluated_words,
        positions,
        &graph,
        1,
    )?;
    assert_eq!(proof_status(&proof)?, 0);
    let metadata = metadata(request, device)?;
    let finalize_args = metal_row_finalize_args(
        request,
        rows,
        positions.len(),
        request.scratch_bytes(rows)?,
        &proof_args,
    )?;
    evaluated
        .apply_op3_no_bwd(
            &metadata,
            &proof,
            &MetalResidentRowFinalize {
                arguments: finalize_args,
            },
        )
        .map_err(candle_error)?
        .to_vec1::<i64>()
        .map_err(candle_error)
}

fn validate_temporal_packet(packet: &[i64], request: &ResidentRowProgramRequest) -> Result<()> {
    let decoded = decode_metal_row_packet(packet, request)?;
    let mut rows = super::empty_row_pipeline_result();
    if request.has_graph_input() {
        rows.start_rows = decoded
            .source_positions
            .iter()
            .map(|position| {
                u32::try_from(*position).map_err(|_| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "proof fixture source position exceeds graph-row addressability",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
    }
    ResidentRowProgramResult::from_untrusted_parts(ResidentRowProgramResultParts {
        project: decoded.project,
        execution: decoded.execution,
        bookmark: decoded.bookmark,
        graph_revision: decoded.graph_revision,
        layout_version: decoded.layout_version,
        manifest_fingerprint: decoded.manifest_fingerprint,
        input_cardinality: decoded.input_cardinality,
        source_positions: decoded.source_positions,
        rows,
        projected_columns: decoded.projected_columns,
        receipts: decoded.receipts,
        scratch_bytes: decoded.scratch_bytes,
    })
    .validate(request, BackendKind::Metal)?;
    Ok(())
}

fn temporal_projection_start(packet: &[i64]) -> usize {
    let source_count = usize::try_from(packet[16]).expect("packet source count must fit usize");
    super::METAL_ROW_PACKET_HEADER_WORDS + source_count
}

fn temporal_proof_status(
    device: &Device,
    request: &ResidentRowProgramRequest,
    evaluated_words: &[i64],
    positions: &[u64],
) -> Result<u32> {
    let rows = request.input.max_output_rows;
    let graph = graph_image(rows, positions, 0, positions.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        device,
        request,
        rows,
        evaluated_words,
        positions,
        &graph,
        1,
    )?;
    proof_status(&proof)
}

fn temporal_proof_contract_error(
    device: &Device,
    request: &ResidentRowProgramRequest,
    evaluated_words: &[i64],
    positions: &[u64],
    args: MetalResidentRowProofArgs,
) -> Result<String> {
    let rows = request.input.max_output_rows;
    let graph = graph_image(rows, positions, 0, positions.len());
    let evaluated =
        Tensor::from_slice(evaluated_words, evaluated_words.len(), device).map_err(candle_error)?;
    let positions = proof_position_tensor(positions, device)?;
    let graph = Tensor::from_slice(&graph, graph.len(), device).map_err(candle_error)?;
    let error = evaluated
        .apply_op3_no_bwd(
            &positions,
            &graph,
            &MetalResidentRowProof { arguments: args },
        )
        .err()
        .expect("corrupt temporal proof arguments must fail before publication");
    Ok(error.to_string())
}

#[test]
fn proof_tensors_and_metadata_are_inside_the_exact_scratch_boundary() -> Result<()> {
    let request = request(
        8,
        4,
        2,
        vec![key(0, false, false), key(1, true, true)],
        0,
        4,
    )?;
    let strategy = metal_row_sort_strategy(&request, 4, 4)?;
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, 4, strategy)?;
    assert!(breakdown.proof_graph_device > 0);
    assert!(breakdown.proof_membership_device > 0);
    assert!(breakdown.proof_argument_bytes > 0);
    assert!(breakdown.finalize_position_device > 0);
    assert_eq!(std::mem::size_of::<super::MetalResidentRowProofKey>(), 32);
    assert_eq!(std::mem::size_of::<MetalResidentRowProofArgs>(), 2_240);
    assert_eq!(
        std::mem::size_of::<super::MetalResidentRowFinalizeArgs>(),
        232
    );
    let required = metal_resident_row_program_scratch_bytes(&request, 4)?;
    assert_eq!(required, breakdown.required_bytes()?);
    let short = DeviceMemoryGovernor::new(required - 1, 0);
    let error = short
        .reserve_scratch(required)
        .expect_err("one byte below the proof-inclusive requirement must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    let exact = DeviceMemoryGovernor::new(required, 0);
    assert_eq!(exact.reserve_scratch(required)?.bytes(), required);
    Ok(())
}

#[test]
fn device_proof_rejects_reversed_and_unstable_full_radix_positions() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = request(
        1,
        4,
        2,
        vec![key(0, false, false), key(1, false, false)],
        0,
        4,
    )?;
    assert_eq!(
        metal_row_sort_strategy(&request, 4, 4)?,
        MetalRowSortStrategy::Radix
    );
    let registers = vec![vec![0, 1, 2, 3], vec![0, 0, 0, 0]];
    let validity = vec![vec![1; 4], vec![1; 4]];

    let reversed = vec![3, 2, 1, 0];
    let graph = graph_image(4, &reversed, 0, 4);
    let (_, proof, _) = execute_proof(&device, &request, &registers, &validity, &reversed, &graph)?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);

    let equal_registers = vec![vec![7; 4], vec![9; 4]];
    let unstable = vec![1, 0, 2, 3];
    let graph = graph_image(4, &unstable, 0, 4);
    let (_, proof, _) = execute_proof(
        &device,
        &request,
        &equal_registers,
        &validity,
        &unstable,
        &graph,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);
    Ok(())
}

#[test]
fn device_proof_requires_unsorted_identity_and_accepts_an_exact_empty_window() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let identity_request = request(6, 3, 1, Vec::new(), 0, 3)?;
    let registers = vec![vec![3, 2, 1]];
    let validity = vec![vec![1; 3]];
    let wrong_identity = vec![0, 2, 1];
    let graph = graph_image(3, &wrong_identity, 0, 3);
    let (_, proof, _) = execute_proof(
        &device,
        &identity_request,
        &registers,
        &validity,
        &wrong_identity,
        &graph,
    )?;
    assert_ne!(proof_status(&proof)? & 16, 0);

    let empty_request = request(7, 3, 1, vec![key(0, false, false)], 0, 0)?;
    let graph = graph_image(3, &[], 0, 0);
    let (_, proof, _) = execute_proof(&device, &empty_request, &registers, &validity, &[], &graph)?;
    assert_eq!(proof_status(&proof)?, 0);
    Ok(())
}

#[test]
fn device_proof_rejects_wrong_top_k_boundary_and_null_placement() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let top_k_request = request(2, 4, 1, vec![key(0, false, false)], 0, 2)?;
    assert_eq!(
        metal_row_sort_strategy(&top_k_request, 4, 2)?,
        MetalRowSortStrategy::SingleKeyTopK { window: 2 }
    );
    let registers = vec![vec![0, 1, 2, 3]];
    let validity = vec![vec![1; 4]];
    let wrong_boundary = vec![0, 2];
    let graph = graph_image(4, &wrong_boundary, 0, 2);
    let (_, proof, _) = execute_proof(
        &device,
        &top_k_request,
        &registers,
        &validity,
        &wrong_boundary,
        &graph,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_OPTIMALITY, 0);

    let null_request = request(3, 3, 1, vec![key(0, false, false)], 0, 2)?;
    let null_registers = vec![vec![99, 1, 2]];
    let null_validity = vec![vec![0, 1, 1]];
    let null_first = vec![0, 1];
    let graph = graph_image(3, &null_first, 0, 2);
    let (_, proof, _) = execute_proof(
        &device,
        &null_request,
        &null_registers,
        &null_validity,
        &null_first,
        &graph,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);
    Ok(())
}

#[test]
fn device_proof_rejects_mixed_direction_order_and_one_corrupt_graph_column() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = request(
        4,
        4,
        2,
        vec![key(0, false, false), key(1, true, true)],
        0,
        4,
    )?;
    let registers = vec![vec![0, 0, 1, 1], vec![5, 3, 9, 7]];
    let validity = vec![vec![1; 4], vec![1; 4]];
    let wrong_secondary_direction = vec![1, 0, 2, 3];
    let graph = graph_image(4, &wrong_secondary_direction, 0, 4);
    let (_, proof, _) = execute_proof(
        &device,
        &request,
        &registers,
        &validity,
        &wrong_secondary_direction,
        &graph,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);

    let correct = vec![0, 1, 2, 3];
    let mut graph = graph_image(4, &correct, 0, 4);
    graph[4 + 2] ^= 1;
    let (_, proof, _) = execute_proof(&device, &request, &registers, &validity, &correct, &graph)?;
    assert_ne!(proof_status(&proof)? & PROOF_GRAPH_ALIGNMENT, 0);
    Ok(())
}

#[test]
fn device_proof_checks_every_graph_column_without_hashing() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = request(9, 4, 1, vec![key(0, false, false)], 0, 4)?;
    let registers = vec![vec![0, 1, 2, 3]];
    let validity = vec![vec![1; 4]];
    let positions = vec![0, 1, 2, 3];
    let graph_column_count = 3_usize;
    let input_columns = (0..graph_column_count)
        .map(|column| {
            (0..4)
                .map(|row| 10_000_u32 + (column as u32) * 100 + row)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut graph = input_columns.iter().flatten().copied().collect::<Vec<_>>();
    for column in &input_columns {
        graph.extend(positions.iter().map(|position| column[*position as usize]));
    }

    // Corrupt a non-first output column. Exact per-cell comparison must detect this even though
    // every other graph cell and every source position remains correct.
    let output_base = graph_column_count * 4;
    graph[output_base + 2 * 4 + 1] ^= 1;
    let (_, proof, _) = execute_proof_with_graph_columns(
        &device,
        &request,
        &registers,
        &validity,
        &positions,
        &graph,
        graph_column_count,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_GRAPH_ALIGNMENT, 0);
    Ok(())
}

#[test]
fn only_a_complete_device_proof_can_publish_metal_receipts() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = request(5, 4, 1, vec![key(0, false, false)], 0, 2)?;
    let registers = vec![vec![0, 1, 2, 3]];
    let validity = vec![vec![1; 4]];
    let correct = vec![0, 1];
    let graph = graph_image(4, &correct, 0, 2);
    let (evaluated, proof, proof_args) =
        execute_proof(&device, &request, &registers, &validity, &correct, &graph)?;
    assert_eq!(proof_status(&proof)?, 0);
    let metadata = metadata(&request, &device)?;
    let finalize_args =
        metal_row_finalize_args(&request, 4, 2, request.scratch_bytes(4)?, &proof_args)?;
    let packet = evaluated
        .apply_op3_no_bwd(
            &metadata,
            &proof,
            &MetalResidentRowFinalize {
                arguments: finalize_args,
            },
        )
        .map_err(candle_error)?
        .to_vec1::<i64>()
        .map_err(candle_error)?;
    let decoded = decode_metal_row_packet(&packet, &request)?;
    assert_eq!(decoded.source_positions, correct);
    assert!(
        decoded
            .receipts
            .iter()
            .all(|receipt| receipt.completion == ResidentDeviceCompletion::Metal)
    );

    let wrong = vec![0, 2];
    let graph = graph_image(4, &wrong, 0, 2);
    let (evaluated, rejected_proof, proof_args) =
        execute_proof(&device, &request, &registers, &validity, &wrong, &graph)?;
    assert_ne!(proof_status(&rejected_proof)?, 0);
    let finalize_args =
        metal_row_finalize_args(&request, 4, 2, request.scratch_bytes(4)?, &proof_args)?;
    let rejected_packet = evaluated
        .apply_op3_no_bwd(
            &metadata,
            &rejected_proof,
            &MetalResidentRowFinalize {
                arguments: finalize_args,
            },
        )
        .map_err(candle_error)?
        .to_vec1::<i64>()
        .map_err(candle_error)?;
    let error = match decode_metal_row_packet(&rejected_packet, &request) {
        Ok(_) => {
            return Err(crate::Error::internal(
                "a failed device proof published completion receipts",
            ));
        }
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

#[test]
fn device_proof_orders_generated_strings_by_exact_bytes_in_both_directions() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let strings: Vec<&[u8]> = vec![
        b"a",
        b"a\0",
        b"aa",
        "é".as_bytes(),
        "β".as_bytes(),
        "😀".as_bytes(),
        b"",
        b"a\0b",
    ];
    let validity = vec![1; strings.len()];
    let ascending = vec![6, 0, 1, 7, 2, 3, 4, 5];

    let request = generated_string_request(100, strings.len(), 8, false, false)?;
    assert_eq!(
        metal_row_sort_strategy(&request, strings.len(), strings.len())?,
        MetalRowSortStrategy::Radix
    );
    let evaluated = generated_string_evaluated_words(&request, &strings, &validity)?;
    let graph = graph_image(strings.len(), &ascending, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &ascending,
        &graph,
        1,
    )?;
    assert_eq!(proof_status(&proof)?, 0);

    // `aa` cannot precede `a\0`: the proof must compare the embedded NUL rather than treating
    // either value as a C string or comparing packed offsets numerically.
    let wrong_embedded_nul = vec![6, 0, 2, 1, 7, 3, 4, 5];
    let graph = graph_image(strings.len(), &wrong_embedded_nul, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &wrong_embedded_nul,
        &graph,
        1,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);

    let descending = ascending.iter().rev().copied().collect::<Vec<_>>();
    let request = generated_string_request(101, strings.len(), 8, true, false)?;
    let evaluated = generated_string_evaluated_words(&request, &strings, &validity)?;
    let graph = graph_image(strings.len(), &descending, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &descending,
        &graph,
        1,
    )?;
    assert_eq!(proof_status(&proof)?, 0);

    let graph = graph_image(strings.len(), &ascending, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &ascending,
        &graph,
        1,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);
    Ok(())
}

#[test]
fn device_proof_uses_string_validity_before_reading_packed_payloads() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let strings: Vec<&[u8]> = vec![b"b", b"hidden", b"a"];
    let validity = vec![1, 0, 1];
    let request = generated_string_request(102, strings.len(), 8, false, false)?;
    let mut evaluated = generated_string_evaluated_words(&request, &strings, &validity)?;
    let evaluated_bytes = std::mem::size_of_val(evaluated.as_slice());
    evaluated[register_value_offset(strings.len(), 0, 1)] =
        packed_string_reference(evaluated_bytes - 1, 2);

    // NULLS LAST must accept this even though the hidden payload is deliberately out of bounds.
    // The device proof is required to consult validity before touching a NULL payload.
    let correct = vec![2, 0, 1];
    let graph = graph_image(strings.len(), &correct, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &correct,
        &graph,
        1,
    )?;
    assert_eq!(proof_status(&proof)?, 0);

    let null_first = vec![1, 2, 0];
    let graph = graph_image(strings.len(), &null_first, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &null_first,
        &graph,
        1,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_ORDER, 0);

    let register_count = request.program.instructions.len();
    evaluated[register_validity_offset(strings.len(), register_count, 0, 1)] = 2;
    let graph = graph_image(strings.len(), &correct, 0, strings.len());
    let (_, proof, _) = execute_proof_with_evaluated_words(
        &device,
        &request,
        strings.len(),
        &evaluated,
        &correct,
        &graph,
        1,
    )?;
    assert_ne!(proof_status(&proof)? & PROOF_DESCRIPTOR, 0);
    Ok(())
}

#[test]
fn device_proof_rejects_corrupt_generated_string_references_and_blocks_publication() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let strings: Vec<&[u8]> = vec![b"a", b"b"];
    let validity = vec![1, 1];
    let request = generated_string_request(103, strings.len(), 8, false, false)?;
    let canonical = generated_string_evaluated_words(&request, &strings, &validity)?;
    let evaluated_bytes = std::mem::size_of_val(canonical.as_slice());
    let positions = vec![0, 1];
    let graph = graph_image(strings.len(), &positions, 0, strings.len());
    let first_value = register_value_offset(strings.len(), 0, 0);
    let second_value = register_value_offset(strings.len(), 0, 1);

    let mut missing_marker = canonical.clone();
    let markerless_offset = u32::try_from(evaluated_bytes - 8).map_err(|_| {
        crate::Error::internal("proof-test markerless offset unexpectedly exceeds u32")
    })?;
    missing_marker[first_value] = ((u64::from(markerless_offset) << 32) | 1) as i64;
    missing_marker[second_value] = ((u64::from(markerless_offset + 1) << 32) | 1) as i64;

    let mut offset_out_of_bounds = canonical.clone();
    offset_out_of_bounds[first_value] = packed_string_reference(0x7fff_ffff, 1);

    let mut length_out_of_bounds = canonical.clone();
    length_out_of_bounds[first_value] = packed_string_reference(evaluated_bytes - 1, 2);

    let mut rank_reference_mismatch = canonical.clone();
    rank_reference_mismatch[first_value] = 0;
    assert_ne!(
        ((rank_reference_mismatch[second_value] as u64 >> 32) as u32) & STRING_REFERENCE_MARKER,
        0
    );

    let corruptions = [
        ("missing offset marker", missing_marker),
        ("offset outside evaluated arena", offset_out_of_bounds),
        ("length outside evaluated arena", length_out_of_bounds),
        ("rank/reference mode mismatch", rank_reference_mismatch),
    ];
    let mut rejected = None;
    for (name, evaluated) in corruptions {
        let (evaluated, proof, proof_args) = execute_proof_with_evaluated_words(
            &device,
            &request,
            strings.len(),
            &evaluated,
            &positions,
            &graph,
            1,
        )?;
        assert_ne!(
            proof_status(&proof)? & PROOF_DESCRIPTOR,
            0,
            "Metal proof accepted {name}"
        );
        if rejected.is_none() {
            rejected = Some((evaluated, proof, proof_args));
        }
    }

    let (evaluated, proof, proof_args) = rejected.expect("at least one corruption must be tested");
    let metadata = metadata(&request, &device)?;
    let finalize_args = metal_row_finalize_args(
        &request,
        strings.len(),
        strings.len(),
        request.scratch_bytes(strings.len())?,
        &proof_args,
    )?;
    let packet = evaluated
        .apply_op3_no_bwd(
            &metadata,
            &proof,
            &MetalResidentRowFinalize {
                arguments: finalize_args,
            },
        )
        .map_err(candle_error)?
        .to_vec1::<i64>()
        .map_err(candle_error)?;
    let error = match decode_metal_row_packet(&packet, &request) {
        Ok(_) => {
            return Err(crate::Error::internal(
                "a corrupt generated STRING proof published Metal receipts",
            ));
        }
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        error.message,
        "Metal resident row device proof rejected result publication"
    );
    Ok(())
}

#[test]
fn temporal_metal_proof_rejects_wrong_date_and_local_time_order() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let fixtures = [
        (
            ResidentRowValueType::Date,
            vec![-2, 0, 9],
            vec![0, 1, 2],
            vec![0, 2, 1],
        ),
        (
            ResidentRowValueType::LocalTime,
            vec![10, 0, 5],
            vec![1, 2, 0],
            vec![1, 0, 2],
        ),
    ];
    for (index, (value_type, payload, correct, corrupt)) in fixtures.into_iter().enumerate() {
        let request = temporal_request(200 + index as u64, 3, value_type, 0, false)?;
        let evaluated = temporal_evaluated_words(&request, &payload, &[1; 3], &[], None)?;
        assert_eq!(
            temporal_proof_status(&device, &request, &evaluated, &correct)?,
            0
        );
        assert_ne!(
            temporal_proof_status(&device, &request, &evaluated, &corrupt)? & PROOF_ORDER,
            0,
            "Metal proof accepted a misordered {value_type:?} column"
        );
    }
    Ok(())
}

#[test]
fn temporal_metal_proof_rejects_zoned_time_normalized_and_offset_tuple_corruption() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = temporal_request(202, 3, ResidentRowValueType::ZonedTime, 0, false)?;
    let local_nanos = vec![3_600_000_000_000, 0, 1];
    let offsets = vec![3_600, 0, 0];
    let normalized = local_nanos
        .iter()
        .zip(&offsets)
        .map(|(local, offset)| zoned_time_key(*local, *offset))
        .collect::<Vec<_>>();
    let correct = vec![1, 0, 2];
    let evaluated = temporal_evaluated_words(
        &request,
        &normalized,
        &[1; 3],
        &[&local_nanos, &offsets],
        None,
    )?;
    assert_eq!(
        temporal_proof_status(&device, &request, &evaluated, &correct)?,
        0
    );

    let wrong_offset_tie = vec![0, 1, 2];
    assert_ne!(
        temporal_proof_status(&device, &request, &evaluated, &wrong_offset_tie)? & PROOF_ORDER,
        0
    );

    let mut wrong_normalized = normalized.clone();
    wrong_normalized[0] += 1;
    let corrupt = temporal_evaluated_words(
        &request,
        &wrong_normalized,
        &[1; 3],
        &[&local_nanos, &offsets],
        None,
    )?;
    assert_ne!(
        temporal_proof_status(&device, &request, &corrupt, &correct)? & PROOF_DESCRIPTOR,
        0
    );

    let mut out_of_range_offsets = offsets.clone();
    out_of_range_offsets[1] = i64::from(i32::MAX) + 1;
    let corrupt = temporal_evaluated_words(
        &request,
        &normalized,
        &[1; 3],
        &[&local_nanos, &out_of_range_offsets],
        None,
    )?;
    assert_ne!(
        temporal_proof_status(&device, &request, &corrupt, &correct)? & PROOF_DESCRIPTOR,
        0
    );
    Ok(())
}

#[test]
fn temporal_metal_proof_rejects_local_datetime_nanos_corruption() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = temporal_request(203, 3, ResidentRowValueType::LocalDateTime, 0, false)?;
    let seconds = vec![10, 10, 11];
    let nanos = vec![2, 1, 0];
    let correct = vec![1, 0, 2];
    let evaluated = temporal_evaluated_words(&request, &seconds, &[1; 3], &[&nanos], None)?;
    assert_eq!(
        temporal_proof_status(&device, &request, &evaluated, &correct)?,
        0
    );

    assert_ne!(
        temporal_proof_status(&device, &request, &evaluated, &[0, 1, 2])? & PROOF_ORDER,
        0
    );

    let corrupt_nanos = vec![1_000_000_000, 1, 0];
    let corrupt = temporal_evaluated_words(&request, &seconds, &[1; 3], &[&corrupt_nanos], None)?;
    assert_ne!(
        temporal_proof_status(&device, &request, &corrupt, &correct)? & PROOF_DESCRIPTOR,
        0
    );
    Ok(())
}

#[test]
fn temporal_metal_proof_compares_zoned_datetime_timezone_utf8_bytes_exactly() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = temporal_request(204, 3, ResidentRowValueType::ZonedDateTime, 4, false)?;
    let seconds = vec![10; 3];
    let nanos = vec![7; 3];
    let timezones: Vec<&[u8]> = vec!["Aé".as_bytes(), b"A\0", b"Aa"];
    let correct = vec![1, 2, 0];
    let evaluated =
        temporal_evaluated_words(&request, &seconds, &[1; 3], &[&nanos], Some(&timezones))?;
    assert_eq!(
        temporal_proof_status(&device, &request, &evaluated, &correct)?,
        0
    );
    assert_ne!(
        temporal_proof_status(&device, &request, &evaluated, &[1, 0, 2])? & PROOF_ORDER,
        0,
        "Metal proof compared timezone references instead of exact UTF-8 bytes"
    );
    Ok(())
}

#[test]
fn temporal_metal_proof_rejects_zoned_datetime_timezone_reference_bounds() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let request = temporal_request(205, 2, ResidentRowValueType::ZonedDateTime, 4, false)?;
    let seconds = vec![10; 2];
    let nanos = vec![0; 2];
    let timezones: Vec<&[u8]> = vec![b"A", b"B"];
    let canonical =
        temporal_evaluated_words(&request, &seconds, &[1; 2], &[&nanos], Some(&timezones))?;
    let sidecar_base = super::metal_row_temporal_slots(&request, 2)?[0].0;
    let second_reference = sidecar_base + 2 + 1;
    let evaluated_bytes = std::mem::size_of_val(canonical.as_slice());

    let mut outside_arena = canonical.clone();
    outside_arena[second_reference] = packed_string_reference(evaluated_bytes - 1, 2);
    assert_ne!(
        temporal_proof_status(&device, &request, &outside_arena, &[0, 1])? & PROOF_DESCRIPTOR,
        0
    );

    let mut missing_marker = canonical;
    missing_marker[second_reference] = 1;
    assert_ne!(
        temporal_proof_status(&device, &request, &missing_marker, &[0, 1])? & PROOF_DESCRIPTOR,
        0
    );
    Ok(())
}

#[test]
fn temporal_metal_proof_rejects_sidecar_overlap_width_and_end_overflow() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let rows = 2;
    let request = temporal_request(206, rows, ResidentRowValueType::ZonedTime, 0, false)?;
    let local = vec![0, 1];
    let offsets = vec![0, 0];
    let evaluated = temporal_evaluated_words(&request, &local, &[1; 2], &[&local, &offsets], None)?;
    let strategy = metal_row_sort_strategy(&request, rows, rows)?;
    let canonical_args = metal_row_proof_args(&request, rows, rows, rows, 1, strategy)?;
    let fixed_words = 1 + rows * request.program.instructions.len() * 2 + rows;

    let mut overlap = canonical_args;
    overlap.sort_keys[0].sidecar_base = (fixed_words - 1) as u64;
    let mut wrong_width = canonical_args;
    wrong_width.sort_keys[0].sidecar_lanes = 1;
    let mut end_overflow = canonical_args;
    end_overflow.sort_keys[0].sidecar_base = evaluated.len() as u64;
    for (name, corrupt) in [
        ("fixed-register overlap", overlap),
        ("lane-width mismatch", wrong_width),
        ("evaluated-image end overflow", end_overflow),
    ] {
        let error = temporal_proof_contract_error(&device, &request, &evaluated, &[0, 1], corrupt)?;
        assert!(
            error.contains("Metal resident row proof tensor contract is invalid"),
            "unexpected error for {name}: {error}"
        );
    }

    let date_request = temporal_request(207, rows, ResidentRowValueType::Date, 0, false)?;
    let date_evaluated = temporal_evaluated_words(&date_request, &[0, 1], &[1; 2], &[], None)?;
    let strategy = metal_row_sort_strategy(&date_request, rows, rows)?;
    let mut forbidden_scalar_sidecar =
        metal_row_proof_args(&date_request, rows, rows, rows, 1, strategy)?;
    forbidden_scalar_sidecar.sort_keys[0].sidecar_base = date_evaluated.len() as u64;
    forbidden_scalar_sidecar.sort_keys[0].sidecar_lanes = 1;
    let error = temporal_proof_contract_error(
        &device,
        &date_request,
        &date_evaluated,
        &[0, 1],
        forbidden_scalar_sidecar,
    )?;
    assert!(error.contains("Metal resident row proof tensor contract is invalid"));
    Ok(())
}

#[test]
fn temporal_projection_packet_rejects_corrupt_fixed_components() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };

    let date_request = temporal_request(208, 2, ResidentRowValueType::Date, 0, true)?;
    let date_evaluated = temporal_evaluated_words(&date_request, &[1, 2], &[1; 2], &[], None)?;
    let mut date_packet =
        finalize_temporal_packet(&device, &date_request, &date_evaluated, &[0, 1])?;
    validate_temporal_packet(&date_packet, &date_request)?;
    let projection = temporal_projection_start(&date_packet);
    date_packet[projection + 1] = ResidentRowValueType::LocalTime as u8 as i64;
    let error = validate_temporal_packet(&date_packet, &date_request)
        .expect_err("a Date packet relabeled as LocalTime must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let local_time_request = temporal_request(209, 2, ResidentRowValueType::LocalTime, 0, true)?;
    let local_time_evaluated =
        temporal_evaluated_words(&local_time_request, &[1, 2], &[1; 2], &[], None)?;
    let mut local_time_packet =
        finalize_temporal_packet(&device, &local_time_request, &local_time_evaluated, &[0, 1])?;
    validate_temporal_packet(&local_time_packet, &local_time_request)?;
    let projection = temporal_projection_start(&local_time_packet);
    local_time_packet[projection + 2 + 2] = 2;
    let error = validate_temporal_packet(&local_time_packet, &local_time_request)
        .expect_err("invalid LocalTime validity must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let zoned_time_request = temporal_request(210, 2, ResidentRowValueType::ZonedTime, 0, true)?;
    let local = vec![0, 1];
    let offsets = vec![0, 0];
    let zoned_time_evaluated = temporal_evaluated_words(
        &zoned_time_request,
        &local,
        &[1; 2],
        &[&local, &offsets],
        None,
    )?;
    let mut zoned_time_packet =
        finalize_temporal_packet(&device, &zoned_time_request, &zoned_time_evaluated, &[0, 1])?;
    validate_temporal_packet(&zoned_time_packet, &zoned_time_request)?;
    let projection = temporal_projection_start(&zoned_time_packet);
    zoned_time_packet[projection + 2 + 2] = i64::MAX;
    let error = validate_temporal_packet(&zoned_time_packet, &zoned_time_request)
        .expect_err("out-of-i32 ZonedTime offset must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let local_datetime_request =
        temporal_request(211, 2, ResidentRowValueType::LocalDateTime, 0, true)?;
    let nanos = vec![0, 1];
    let local_datetime_evaluated =
        temporal_evaluated_words(&local_datetime_request, &[1, 2], &[1; 2], &[&nanos], None)?;
    let mut local_datetime_packet = finalize_temporal_packet(
        &device,
        &local_datetime_request,
        &local_datetime_evaluated,
        &[0, 1],
    )?;
    validate_temporal_packet(&local_datetime_packet, &local_datetime_request)?;
    let projection = temporal_projection_start(&local_datetime_packet);
    local_datetime_packet[projection + 2 + 2] = -1;
    let error = validate_temporal_packet(&local_datetime_packet, &local_datetime_request)
        .expect_err("negative LocalDateTime nanos must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

#[test]
fn temporal_zoned_datetime_packet_rejects_invalid_utf8_lengths_nanos_and_bounds() -> Result<()> {
    let _guard = crate::metal_test_guard();
    let _guard = metal_proof_guard();
    let Some(device) = crate::metal_test_device() else {
        return Ok(());
    };
    let rows = 2;
    let capacity = 16;
    let request = temporal_request(
        212,
        rows,
        ResidentRowValueType::ZonedDateTime,
        capacity,
        true,
    )?;
    let nanos = vec![0, 1];
    let timezones: Vec<&[u8]> = vec![b"UTC", b"Europe/Paris"];
    let evaluated =
        temporal_evaluated_words(&request, &[1, 2], &[1; 2], &[&nanos], Some(&timezones))?;
    let packet = finalize_temporal_packet(&device, &request, &evaluated, &[0, 1])?;
    validate_temporal_packet(&packet, &request)?;
    let projection = temporal_projection_start(&packet);
    let payload = projection + super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS;
    let nanos_start = payload + rows;
    let lengths_start = nanos_start + rows;
    let validity_start = lengths_start + rows;
    let packed_start = validity_start + rows;

    let mut invalid_utf8 = packet.clone();
    let mut first_packed = invalid_utf8[packed_start].to_le_bytes();
    first_packed[0] = 0xff;
    invalid_utf8[packed_start] = i64::from_le_bytes(first_packed);
    let error = validate_temporal_packet(&invalid_utf8, &request)
        .expect_err("invalid timezone UTF-8 must fail result validation");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        error.message,
        "resident row zoned-datetime column has an invalid canonical timezone shape"
    );

    let mut over_capacity = packet.clone();
    over_capacity[lengths_start] = i64::from(capacity) + 1;
    let error = validate_temporal_packet(&over_capacity, &request)
        .expect_err("timezone length beyond its admitted slot must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut null_with_bytes = packet.clone();
    null_with_bytes[validity_start] = 0;
    let error = validate_temporal_packet(&null_with_bytes, &request)
        .expect_err("NULL timezone with hidden bytes must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut negative_nanos = packet.clone();
    negative_nanos[nanos_start] = -1;
    let error = validate_temporal_packet(&negative_nanos, &request)
        .expect_err("negative ZonedDateTime nanos must fail");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut truncated = packet;
    truncated.pop();
    let error = validate_temporal_packet(&truncated, &request)
        .expect_err("truncated timezone byte arena must fail packet bounds");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}
