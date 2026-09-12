// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentScalarCell,
        ResidentScalarCellTag, ResidentScalarListEntry, ResidentScalarMapEntry,
        ResidentScalarProgramInstruction, ResidentScalarProgramOpcode,
        ResidentScalarProgramOperand, ResidentScalarProgramRequest, ResidentScalarProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const G: u32 = u32::MAX;
const M: u32 = u32::MAX - 1;
const S: u32 = u32::MAX - 2;
const L: u32 = u32::MAX - 3;
const UNKNOWN_SENTINEL: u32 = u32::MAX - 4;

const NULL: u16 = 0;
const FALSE: u16 = 1;
const TRUE: u16 = 2;
const ZERO: u16 = 3;
const ONE: u16 = 4;
const TEXT: u16 = 5;
const LIST: u16 = 6;
const MAP: u16 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticStatus {
    Generic,
    MapKey,
    IndexSource,
    ListIndex,
}

impl SemanticStatus {
    const ALL: [Self; 4] = [
        Self::Generic,
        Self::MapKey,
        Self::IndexSource,
        Self::ListIndex,
    ];

    const fn root(self) -> u32 {
        match self {
            Self::Generic => G,
            Self::MapKey => M,
            Self::IndexSource => S,
            Self::ListIndex => L,
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::Generic => {
                "InvalidArgumentType: resident scalar operation received incompatible operands"
            }
            Self::MapKey => "MapElementAccessByNonString: map index requires a STRING",
            Self::IndexSource => {
                "InvalidArgumentType: indexing requires a LIST, MAP, NODE, or RELATIONSHIP"
            }
            Self::ListIndex => "InvalidArgumentType: list index requires an INTEGER",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarCase {
    ListIndex,
    ToInteger,
    ListMembership,
    ListSliceMembership,
    NumericSubtract,
    BuildList,
    ToFloat,
    ToString,
    NumericAdd,
    NumericMultiply,
    NumericDivide,
    NumericModulo,
    NumericPower,
    NumericPositive,
    NumericNegative,
    CypherAdd,
    ListSlice,
}

impl ScalarCase {
    const ALL: [Self; 17] = [
        Self::ListIndex,
        Self::ToInteger,
        Self::ListMembership,
        Self::ListSliceMembership,
        Self::NumericSubtract,
        Self::BuildList,
        Self::ToFloat,
        Self::ToString,
        Self::NumericAdd,
        Self::NumericMultiply,
        Self::NumericDivide,
        Self::NumericModulo,
        Self::NumericPower,
        Self::NumericPositive,
        Self::NumericNegative,
        Self::CypherAdd,
        Self::ListSlice,
    ];

    const fn operand_count(self) -> usize {
        match self {
            Self::ToInteger
            | Self::ToFloat
            | Self::ToString
            | Self::NumericPositive
            | Self::NumericNegative => 1,
            Self::ListSliceMembership => 4,
            Self::ListSlice => 3,
            Self::BuildList => 2,
            Self::ListIndex
            | Self::ListMembership
            | Self::NumericSubtract
            | Self::NumericAdd
            | Self::NumericMultiply
            | Self::NumericDivide
            | Self::NumericModulo
            | Self::NumericPower
            | Self::CypherAdd => 2,
        }
    }
}

struct ProgramBuilder {
    cells: Vec<ResidentScalarCell>,
    map_entries: Vec<ResidentScalarMapEntry>,
    list_entries: Vec<ResidentScalarListEntry>,
    instructions: Vec<ResidentScalarProgramInstruction>,
}

impl ProgramBuilder {
    fn new() -> Self {
        Self {
            cells: vec![
                cell(ResidentScalarCellTag::Null, 0, 0),
                cell(ResidentScalarCellTag::Boolean, 0, 0),
                cell(ResidentScalarCellTag::Boolean, 1, 0),
                cell(ResidentScalarCellTag::Integer, 0, 0),
                cell(ResidentScalarCellTag::Integer, 1, 0),
                cell(ResidentScalarCellTag::String, 1, 1),
                cell(ResidentScalarCellTag::List, 0, 1),
                cell(ResidentScalarCellTag::Map, 0, 1),
            ],
            map_entries: vec![ResidentScalarMapEntry { key: 1, value: ONE }],
            list_entries: vec![ResidentScalarListEntry { value: ONE }],
            instructions: Vec::new(),
        }
    }

    fn integer_control(&mut self, value: u64) -> ResidentScalarProgramOperand {
        let index = u16::try_from(self.cells.len()).expect("bounded scalar test cells");
        self.cells
            .push(cell(ResidentScalarCellTag::Integer, value, 0));
        ResidentScalarProgramOperand::Cell(index)
    }

    fn reserve_list_arena(
        &mut self,
        capacity: usize,
    ) -> (ResidentScalarProgramOperand, ResidentScalarProgramOperand) {
        let offset = self.list_entries.len();
        self.list_entries
            .extend((0..capacity).map(|_| ResidentScalarListEntry { value: NULL }));
        (
            self.integer_control(offset as u64),
            self.integer_control(capacity as u64),
        )
    }

    fn push(&mut self, instruction: ResidentScalarProgramInstruction) -> u16 {
        let register = u16::try_from(self.instructions.len()).expect("bounded scalar registers");
        self.instructions.push(instruction);
        register
    }

    fn push_status(&mut self, status: SemanticStatus) -> u16 {
        match status {
            SemanticStatus::Generic => self.push(instruction(
                ResidentScalarProgramOpcode::NumericPositive,
                ResidentScalarProgramOperand::Cell(TEXT),
                ResidentScalarProgramOperand::Cell(NULL),
            )),
            SemanticStatus::MapKey => self.push(instruction(
                ResidentScalarProgramOpcode::ListIndex,
                ResidentScalarProgramOperand::Cell(MAP),
                ResidentScalarProgramOperand::Cell(ZERO),
            )),
            SemanticStatus::IndexSource => self.push(instruction(
                ResidentScalarProgramOpcode::ListIndex,
                ResidentScalarProgramOperand::Cell(ONE),
                ResidentScalarProgramOperand::Cell(ZERO),
            )),
            SemanticStatus::ListIndex => self.push(instruction(
                ResidentScalarProgramOpcode::ListIndex,
                ResidentScalarProgramOperand::Cell(LIST),
                ResidentScalarProgramOperand::Cell(TEXT),
            )),
        }
    }

    fn push_valid_register(&mut self) -> u16 {
        self.push(instruction(
            ResidentScalarProgramOpcode::NumericPositive,
            ResidentScalarProgramOperand::Cell(ONE),
            ResidentScalarProgramOperand::Cell(NULL),
        ))
    }

    fn push_list_shaped_status(&mut self, status: SemanticStatus) -> u16 {
        let value = self.push_status(status);
        let (offset, _) = self.reserve_list_arena(1);
        let count = self.integer_control(1);
        self.push(ResidentScalarProgramInstruction {
            opcode: ResidentScalarProgramOpcode::BuildList,
            left: ResidentScalarProgramOperand::Register(value),
            right: count,
            third: Some(offset),
            fourth: None,
        })
    }

    fn finish(mut self, output: u16) -> ResidentScalarProgramRequest {
        let instruction_count = self.instructions.len();
        self.cells
            .extend((0..instruction_count).map(|_| cell(ResidentScalarCellTag::Null, 0, 0)));
        let mut string_offsets = Vec::with_capacity(instruction_count + 2);
        let mut string_bytes = b"x".to_vec();
        string_offsets.push(0);
        string_offsets.push(1);
        for _ in 0..instruction_count {
            string_bytes.resize(string_bytes.len() + 32, 0);
            string_offsets.push(string_bytes.len() as u32);
        }
        ResidentScalarProgramRequest {
            scalar_cells: self.cells,
            map_entries: self.map_entries,
            scalar_list_entries: self.list_entries,
            string_offsets,
            string_bytes,
            null_cell: NULL,
            false_cell: Some(FALSE),
            true_cell: Some(TRUE),
            instructions: self.instructions,
            output_values: vec![ResidentScalarProgramOperand::Register(output)],
        }
    }
}

const fn cell(tag: ResidentScalarCellTag, payload: u64, auxiliary: u32) -> ResidentScalarCell {
    ResidentScalarCell {
        tag,
        payload,
        auxiliary,
    }
}

const fn instruction(
    opcode: ResidentScalarProgramOpcode,
    left: ResidentScalarProgramOperand,
    right: ResidentScalarProgramOperand,
) -> ResidentScalarProgramInstruction {
    ResidentScalarProgramInstruction {
        opcode,
        left,
        right,
        third: None,
        fourth: None,
    }
}

fn push_target(
    builder: &mut ProgramBuilder,
    case: ScalarCase,
    operands: &[ResidentScalarProgramOperand],
) -> u16 {
    match case {
        ScalarCase::ListIndex => builder.push(instruction(
            ResidentScalarProgramOpcode::ListIndex,
            operands[0],
            operands[1],
        )),
        ScalarCase::ToInteger => builder.push(instruction(
            ResidentScalarProgramOpcode::ToInteger,
            operands[0],
            ResidentScalarProgramOperand::Cell(NULL),
        )),
        ScalarCase::ListMembership => builder.push(instruction(
            ResidentScalarProgramOpcode::ListMembership,
            operands[0],
            operands[1],
        )),
        ScalarCase::ListSliceMembership => builder.push(ResidentScalarProgramInstruction {
            opcode: ResidentScalarProgramOpcode::ListSliceMembership,
            left: operands[0],
            right: operands[1],
            third: Some(operands[2]),
            fourth: Some(operands[3]),
        }),
        ScalarCase::NumericSubtract => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericSubtract,
            operands[0],
            operands[1],
        )),
        ScalarCase::BuildList => {
            let ResidentScalarProgramOperand::Register(first) = operands[0] else {
                panic!("BuildList test elements must be registers");
            };
            for (index, operand) in operands.iter().enumerate() {
                assert_eq!(
                    *operand,
                    ResidentScalarProgramOperand::Register(first + index as u16),
                    "BuildList test registers must be contiguous",
                );
            }
            let (offset, _) = builder.reserve_list_arena(operands.len());
            let count = builder.integer_control(operands.len() as u64);
            builder.push(ResidentScalarProgramInstruction {
                opcode: ResidentScalarProgramOpcode::BuildList,
                left: operands[0],
                right: count,
                third: Some(offset),
                fourth: None,
            })
        }
        ScalarCase::ToFloat => builder.push(instruction(
            ResidentScalarProgramOpcode::ToFloat,
            operands[0],
            ResidentScalarProgramOperand::Cell(NULL),
        )),
        ScalarCase::ToString => builder.push(instruction(
            ResidentScalarProgramOpcode::ToString,
            operands[0],
            ResidentScalarProgramOperand::Cell(NULL),
        )),
        ScalarCase::NumericAdd => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericAdd,
            operands[0],
            operands[1],
        )),
        ScalarCase::NumericMultiply => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericMultiply,
            operands[0],
            operands[1],
        )),
        ScalarCase::NumericDivide => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericDivide,
            operands[0],
            operands[1],
        )),
        ScalarCase::NumericModulo => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericModulo,
            operands[0],
            operands[1],
        )),
        ScalarCase::NumericPower => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericPower,
            operands[0],
            operands[1],
        )),
        ScalarCase::NumericPositive => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericPositive,
            operands[0],
            ResidentScalarProgramOperand::Cell(NULL),
        )),
        ScalarCase::NumericNegative => builder.push(instruction(
            ResidentScalarProgramOpcode::NumericNegative,
            operands[0],
            ResidentScalarProgramOperand::Cell(NULL),
        )),
        ScalarCase::CypherAdd => {
            let (offset, capacity) = builder.reserve_list_arena(2);
            builder.push(ResidentScalarProgramInstruction {
                opcode: ResidentScalarProgramOpcode::CypherAdd,
                left: operands[0],
                right: operands[1],
                third: Some(offset),
                fourth: Some(capacity),
            })
        }
        ScalarCase::ListSlice => {
            let (offset, capacity) = builder.reserve_list_arena(1);
            let ResidentScalarProgramOperand::Cell(offset_cell) = offset else {
                unreachable!()
            };
            let ResidentScalarProgramOperand::Cell(capacity_cell) = capacity else {
                unreachable!()
            };
            let offset = builder.cells[offset_cell as usize].payload;
            let capacity = builder.cells[capacity_cell as usize].payload;
            let descriptor = builder.integer_control((offset << 32) | capacity);
            builder.push(ResidentScalarProgramInstruction {
                opcode: ResidentScalarProgramOpcode::ListSlice,
                left: operands[0],
                right: operands[1],
                third: Some(operands[2]),
                fourth: Some(descriptor),
            })
        }
    }
}

fn default_operands(case: ScalarCase) -> Vec<ResidentScalarProgramOperand> {
    let cell = ResidentScalarProgramOperand::Cell;
    match case {
        ScalarCase::ListIndex => vec![cell(LIST), cell(ZERO)],
        ScalarCase::ToInteger
        | ScalarCase::ToFloat
        | ScalarCase::ToString
        | ScalarCase::NumericPositive
        | ScalarCase::NumericNegative => vec![cell(ONE)],
        ScalarCase::ListMembership => vec![cell(ONE), cell(LIST)],
        ScalarCase::ListSliceMembership => vec![cell(ONE), cell(LIST), cell(ZERO), cell(ONE)],
        ScalarCase::BuildList => unreachable!("BuildList elements are registers"),
        ScalarCase::NumericSubtract
        | ScalarCase::NumericAdd
        | ScalarCase::NumericMultiply
        | ScalarCase::NumericDivide
        | ScalarCase::NumericModulo
        | ScalarCase::NumericPower => vec![cell(ONE), cell(ONE)],
        ScalarCase::CypherAdd => vec![cell(LIST), cell(LIST)],
        ScalarCase::ListSlice => vec![cell(LIST), cell(ZERO), cell(ONE)],
    }
}

fn single_status_request(
    case: ScalarCase,
    position: usize,
    status: SemanticStatus,
) -> ResidentScalarProgramRequest {
    let mut builder = ProgramBuilder::new();
    if case == ScalarCase::BuildList {
        let mut elements = Vec::with_capacity(2);
        for index in 0..2 {
            let register = if index == position {
                builder.push_status(status)
            } else {
                builder.push_valid_register()
            };
            elements.push(ResidentScalarProgramOperand::Register(register));
        }
        let output = push_target(&mut builder, case, &elements);
        return builder.finish(output);
    }
    let status_register = if case == ScalarCase::ListSlice && position == 0 {
        builder.push_list_shaped_status(status)
    } else {
        builder.push_status(status)
    };
    let mut operands = default_operands(case);
    operands[position] = ResidentScalarProgramOperand::Register(status_register);
    let output = push_target(&mut builder, case, &operands);
    builder.finish(output)
}

fn ordered_pair_request(
    case: ScalarCase,
    first_position: usize,
    first_status: SemanticStatus,
    second_position: usize,
    second_status: SemanticStatus,
) -> ResidentScalarProgramRequest {
    assert!(first_position < second_position);
    let mut builder = ProgramBuilder::new();
    if case == ScalarCase::BuildList {
        let first = builder.push_status(first_status);
        let second = builder.push_status(second_status);
        let output = push_target(
            &mut builder,
            case,
            &[
                ResidentScalarProgramOperand::Register(first),
                ResidentScalarProgramOperand::Register(second),
            ],
        );
        return builder.finish(output);
    }
    let first =
        if matches!(case, ScalarCase::CypherAdd | ScalarCase::ListSlice) && first_position == 0 {
            builder.push_list_shaped_status(first_status)
        } else {
            builder.push_status(first_status)
        };
    let second = builder.push_status(second_status);
    let mut operands = default_operands(case);
    operands[first_position] = ResidentScalarProgramOperand::Register(first);
    operands[second_position] = ResidentScalarProgramOperand::Register(second);
    let output = push_target(&mut builder, case, &operands);
    builder.finish(output)
}

fn execute_error(backend: &dyn ExecutionBackend, request: &ResidentScalarProgramRequest) -> Error {
    backend
        .execute_scalar_program(request, &CancellationToken::new())
        .expect_err("resident scalar status program must fail")
}

fn assert_status(error: &Error, status: SemanticStatus) {
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(error.message, status.message());
}

fn assert_complete_status_matrix(backend: &dyn ExecutionBackend) {
    for case in ScalarCase::ALL {
        for position in 0..case.operand_count() {
            for status in SemanticStatus::ALL {
                let request = single_status_request(case, position, status);
                let error = execute_error(backend, &request);
                assert_eq!(
                    (error.code, error.message.as_ref()),
                    (ErrorCode::QueryType, status.message()),
                    "{case:?} semantic operand {position} did not preserve {status:?}",
                );
            }
        }
    }
}

fn assert_complete_precedence_matrix(backend: &dyn ExecutionBackend) {
    for case in ScalarCase::ALL {
        let operand_count = case.operand_count();
        if operand_count < 2 {
            continue;
        }
        for first_position in 0..operand_count - 1 {
            for second_position in first_position + 1..operand_count {
                for first_status in SemanticStatus::ALL {
                    for second_status in SemanticStatus::ALL {
                        let request = ordered_pair_request(
                            case,
                            first_position,
                            first_status,
                            second_position,
                            second_status,
                        );
                        let error = execute_error(backend, &request);
                        assert_eq!(
                            (error.code, error.message.as_ref()),
                            (ErrorCode::QueryType, first_status.message()),
                            "{case:?} operands {first_position}/{second_position} did not preserve {first_status:?} before {second_status:?}",
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_propagates_all_four_statuses_through_every_semantic_operand() {
    let cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
    assert_complete_status_matrix(&cpu);
}

#[test]
fn cpu_preserves_first_status_for_every_ordered_operand_pair() {
    let cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
    assert_complete_precedence_matrix(&cpu);
}

fn table_word_layout(request: &ResidentScalarProgramRequest) -> (usize, usize, usize) {
    let roots = request.output_values.len();
    let list_base = roots + request.scalar_cells.len() * 4 + request.map_entries.len() * 2;
    let offset_base = list_base + request.scalar_list_entries.len();
    let byte_base = offset_base + request.string_offsets.len();
    (list_base, offset_base, byte_base)
}

#[test]
fn build_list_preflights_before_mutating_and_status_cannot_hide_partial_output() -> Result<()> {
    let request = single_status_request(ScalarCase::BuildList, 1, SemanticStatus::ListIndex);
    let words = request.cpu_materialized_words()?;
    assert_eq!(words[0], L);
    let (list_base, _, _) = table_word_layout(&request);
    let actual = &words[list_base..list_base + request.scalar_list_entries.len()];
    let expected = request
        .scalar_list_entries
        .iter()
        .map(|entry| u32::from(entry.value))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);

    let mut corrupted = words;
    *corrupted.last_mut().expect("non-empty canonical frame") = 1;
    let error = request
        .decode_materialized_words(&corrupted)
        .expect_err("typed status must not hide a modified output arena");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

#[test]
fn decoder_validates_frame_before_exact_status_messages() -> Result<()> {
    for status in SemanticStatus::ALL {
        let request = single_status_request(ScalarCase::ToInteger, 0, status);
        let words = request.cpu_materialized_words()?;
        assert_eq!(words[0], status.root());
        assert_status(
            &request
                .decode_materialized_words(&words)
                .expect_err("status root must decode as QueryType"),
            status,
        );
    }

    let request = single_status_request(ScalarCase::NumericPositive, 0, SemanticStatus::ListIndex);
    let mut unknown = request.cpu_materialized_words()?;
    unknown[0] = UNKNOWN_SENTINEL;
    let error = request
        .decode_materialized_words(&unknown)
        .expect_err("unknown status sentinel must be corruption");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut corrupt_cell = request.cpu_materialized_words()?;
    corrupt_cell[0] = L;
    corrupt_cell[request.output_values.len()] = 99;
    let error = request
        .decode_materialized_words(&corrupt_cell)
        .expect_err("typed root must not hide a corrupt scalar tag");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut corrupt_string = request.cpu_materialized_words()?;
    corrupt_string[0] = L;
    let (_, _, byte_base) = table_word_layout(&request);
    corrupt_string[byte_base] = u32::from(b'y');
    let error = request
        .decode_materialized_words(&corrupt_string)
        .expect_err("typed root must not hide changed immutable string bytes");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

fn successful_to_string_request() -> ResidentScalarProgramRequest {
    let mut builder = ProgramBuilder::new();
    let string = builder.push(instruction(
        ResidentScalarProgramOpcode::ToString,
        ResidentScalarProgramOperand::Cell(ONE),
        ResidentScalarProgramOperand::Cell(NULL),
    ));
    let _other = builder.push_valid_register();
    builder.finish(string)
}

#[test]
fn only_the_owning_to_string_instruction_may_change_its_slot() -> Result<()> {
    let request = successful_to_string_request();
    let words = request.cpu_materialized_words()?;
    request.decode_materialized_words(&words)?;

    let (_, _, byte_base) = table_word_layout(&request);
    let input_string_count = request.string_offsets.len() - request.instructions.len() - 1;
    let second_slot_id = input_string_count + 2;
    let second_slot_start = request.string_offsets[second_slot_id - 1] as usize;
    let mut corrupted = words;
    corrupted[byte_base + second_slot_start] = u32::from(b'!');
    let error = request
        .decode_materialized_words(&corrupted)
        .expect_err("a non-ToString instruction cannot mutate its string slot");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}

fn exact_scratch_bytes(request: &ResidentScalarProgramRequest) -> usize {
    let table_words = request.scalar_cells.len() * 4
        + request.map_entries.len() * 2
        + request.scalar_list_entries.len()
        + request.string_offsets.len()
        + request.string_bytes.len();
    let program_words = request.instructions.len() * 9 + request.output_values.len() * 2;
    let input_words = program_words + table_words;
    let output_words = request.output_values.len() + table_words;
    (input_words + output_words) * size_of::<u32>() * 2
}

#[test]
fn scalar_program_scratch_admits_exactly_and_rejects_one_byte_short() -> Result<()> {
    let request = successful_to_string_request();
    let scratch = exact_scratch_bytes(&request);
    let exact = CpuBackend::new(scratch, 0);
    exact.execute_scalar_program(&request, &CancellationToken::new())?;

    let short = CpuBackend::new(scratch - 1, 0);
    let error = short
        .execute_scalar_program(&request, &CancellationToken::new())
        .expect_err("one-byte-short scalar scratch must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
static METAL_TEST_GUARD: Mutex<()> = Mutex::new(());

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_matches_cpu_for_complete_status_and_precedence_matrices() -> Result<()> {
    let _guard = METAL_TEST_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cpu = CpuBackend::new(128 * 1024 * 1024, 16 * 1024 * 1024);
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?;

    for case in ScalarCase::ALL {
        for position in 0..case.operand_count() {
            for status in SemanticStatus::ALL {
                let request = single_status_request(case, position, status);
                let cpu_error = execute_error(&cpu, &request);
                let metal_error = execute_error(&metal, &request);
                assert_eq!(
                    (metal_error.code, metal_error.message.as_ref()),
                    (cpu_error.code, cpu_error.message.as_ref()),
                    "real Metal diverged for {case:?} operand {position} carrying {status:?}",
                );
            }
        }
    }
    for case in ScalarCase::ALL {
        let operand_count = case.operand_count();
        if operand_count < 2 {
            continue;
        }
        for first_position in 0..operand_count - 1 {
            for second_position in first_position + 1..operand_count {
                for first_status in SemanticStatus::ALL {
                    for second_status in SemanticStatus::ALL {
                        let request = ordered_pair_request(
                            case,
                            first_position,
                            first_status,
                            second_position,
                            second_status,
                        );
                        let cpu_error = execute_error(&cpu, &request);
                        let metal_error = execute_error(&metal, &request);
                        assert_eq!(
                            (metal_error.code, metal_error.message.as_ref()),
                            (cpu_error.code, cpu_error.message.as_ref()),
                            "real Metal precedence diverged for {case:?} operands {first_position}/{second_position}: {first_status:?} before {second_status:?}",
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_scalar_scratch_admits_exactly_and_rejects_one_byte_short() -> Result<()> {
    let _guard = METAL_TEST_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = successful_to_string_request();
    let scratch = exact_scratch_bytes(&request);
    let exact = MetalBackend::new(0, scratch, 0)?;
    exact.execute_scalar_program(&request, &CancellationToken::new())?;

    let short = MetalBackend::new(0, scratch - 1, 0)?;
    let error = short
        .execute_scalar_program(&request, &CancellationToken::new())
        .expect_err("one-byte-short Metal scalar scratch must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    Ok(())
}

struct ObservedScalarBackend {
    inner: Box<dyn ExecutionBackend>,
    calls: Arc<AtomicUsize>,
}

impl ObservedScalarBackend {
    fn strict_cpu() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(128 * 1024 * 1024, 16 * 1024 * 1024)),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn strict_metal() -> Result<Self> {
        Ok(Self {
            inner: Box::new(MetalBackend::new(0, 128 * 1024 * 1024, 16 * 1024 * 1024)?),
            calls: Arc::new(AtomicUsize::new(0)),
        })
    }
}

impl ExecutionBackend for ObservedScalarBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            calls: Arc::clone(&self.calls),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.inner.resident_graph_revision(project)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.inner.resident_bookmark(project)
    }

    fn scan_nodes(
        &self,
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner.scan_nodes(project, label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_node_i64(project, property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_project_in(project, targets, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.inner.search_vectors(request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.inner.sort_rows(request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.inner.join_node_i64(request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.inner.group_node_i64(request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_scalar_program(&self) -> bool {
        true
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_scalar_program(request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_i64(values, validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_out(sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.inner
            .exact_l2(matrix, rows, dimension, query, cancellation)
    }
}

fn query_context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: ProjectId::random(),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

const QUERY_STATUS_CASES: [(&str, SemanticStatus); 6] = [
    (
        "WITH [1] AS l, 'x' AS k RETURN +l[k]",
        SemanticStatus::ListIndex,
    ),
    (
        "WITH {x: 1} AS m, 0 AS k RETURN toInteger(m[k])",
        SemanticStatus::MapKey,
    ),
    (
        "WITH [1] AS l, 'x' AS k, {x: 1} AS m, 0 AS z RETURN l[k] + m[z]",
        SemanticStatus::ListIndex,
    ),
    (
        "WITH [1] AS l, 'x' AS k RETURN l[k] IN null",
        SemanticStatus::ListIndex,
    ),
    (
        "WITH [1] AS l, 'x' AS k RETURN l[k] IN []",
        SemanticStatus::ListIndex,
    ),
    (
        "WITH [1] AS l, 'x' AS k RETURN l[k] IN [1][0..0]",
        SemanticStatus::ListIndex,
    ),
];

fn assert_queries_use_one_native_scalar_call(backend: &ObservedScalarBackend) {
    for (query, status) in QUERY_STATUS_CASES {
        let before = backend.calls.load(Ordering::SeqCst);
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut query_context(&graph, backend))
            .expect_err("query must return its native scalar status");
        assert_status(&error, status);
        assert_eq!(
            backend.calls.load(Ordering::SeqCst),
            before + 1,
            "query did not execute exactly one native scalar program: {query}",
        );
    }
}

#[test]
fn strict_cpu_query_routes_dynamic_index_statuses_through_the_native_scalar_vm() {
    let backend = ObservedScalarBackend::strict_cpu();
    assert_queries_use_one_native_scalar_call(&backend);
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn strict_real_metal_query_routes_match_cpu_native_statuses() -> Result<()> {
    let _guard = METAL_TEST_GUARD
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let backend = ObservedScalarBackend::strict_metal()?;
    assert_queries_use_one_native_scalar_call(&backend);
    Ok(())
}
