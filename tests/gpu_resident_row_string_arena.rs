// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::mem::size_of;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentEntityBinding,
        ResidentExecutionId, ResidentNodeBinding, ResidentNodePipelineRequest,
        ResidentProjectImage, ResidentRowColumn, ResidentRowInstruction, ResidentRowOperation,
        ResidentRowProgram, ResidentRowProgramManifest, ResidentRowProgramRequest,
        ResidentRowSortKey, ResidentRowValueType, ValidatedResidentRowProgramResult,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;
const SCALAR_ROWS: usize = 8;
const GRAPH_ROWS: usize = 9;

struct Harness {
    graph: GraphStore,
    bookmark: Bookmark,
    label: LabelId,
    title: PropertyId,
    name: PropertyId,
}

impl Harness {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("StringArena")?;
        let title = graph.catalog_mut().intern_property("title")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let titles = [
            Some(""),
            Some("a"),
            Some("a"),
            Some("aa"),
            Some("é"),
            Some("🙂"),
            None,
            Some("a"),
            Some("a"),
        ];
        let names = [
            Some(""),
            Some(""),
            Some("a"),
            Some(""),
            Some("β"),
            Some("x"),
            Some("z"),
            None,
            Some(""),
        ];
        for row in 0..GRAPH_ROWS {
            let mut properties = Vec::new();
            if let Some(value) = titles[row] {
                properties.push((title, ScalarValue::String(value.into())));
            }
            if let Some(value) = names[row] {
                properties.push((name, ScalarValue::String(value.into())));
            }
            graph.insert_node(NodeInput {
                id: NodeId(1_000 + row as u64),
                layer: Layer::Observed,
                revision: 1 + row as u64,
                labels: vec![label],
                properties,
            })?;
        }
        let bookmark = Bookmark {
            term: 31,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            label,
            title,
            name,
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

    fn cpu(&self) -> Result<CpuBackend> {
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(self.image()?)?;
        Ok(backend)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn metal(&self) -> Result<MetalBackend> {
        let mut backend = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        backend.admit_project(self.image()?)?;
        Ok(backend)
    }

    fn request(
        &self,
        seed: u64,
        input: ResidentNodePipelineRequest,
        program: ResidentRowProgram,
        sort_keys: Vec<ResidentRowSortKey>,
        final_registers: Vec<u16>,
    ) -> Result<ResidentRowProgramRequest> {
        let max_output_rows = input.max_output_rows;
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &sort_keys,
            0,
            usize::MAX,
            max_output_rows,
            &final_registers,
            seed.checked_mul(1_000)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::internal("string-arena obligation ID overflow"))?,
        )?;
        Ok(ResidentRowProgramRequest {
            project: PROJECT,
            expected_bookmark: self.bookmark,
            expected_graph_revision: self.graph.revision(),
            expected_layout_version: self.graph.layout_version(),
            execution: ResidentExecutionId {
                high: 0x5354_5249_4e47_4152,
                low: seed,
            },
            input,
            program,
            manifest,
            sort_keys,
            offset: 0,
            limit: usize::MAX,
            max_output_rows,
            final_registers,
        })
    }

    fn scalar_request(
        &self,
        seed: u64,
        descending: bool,
        nulls_first: bool,
    ) -> Result<ResidentRowProgramRequest> {
        self.request(
            seed,
            scalar_input(SCALAR_ROWS),
            scalar_concat_program(),
            vec![ResidentRowSortKey {
                register: 4,
                descending,
                nulls_first,
            }],
            vec![5],
        )
    }

    fn graph_request(
        &self,
        seed: u64,
        descending: bool,
        nulls_first: bool,
    ) -> Result<ResidentRowProgramRequest> {
        self.request(
            seed,
            graph_input(self.label, GRAPH_ROWS),
            graph_concat_program(self.title, self.name, 4, 2),
            vec![ResidentRowSortKey {
                register: 4,
                descending,
                nulls_first,
            }],
            vec![0],
        )
    }

    fn coalesce_request(&self, seed: u64) -> Result<ResidentRowProgramRequest> {
        self.request(
            seed,
            scalar_input(SCALAR_ROWS),
            scalar_coalesce_program(),
            Vec::new(),
            vec![2],
        )
    }
}

fn scalar_input(rows: usize) -> ResidentNodePipelineRequest {
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

fn graph_input(label: LabelId, rows: usize) -> ResidentNodePipelineRequest {
    ResidentNodePipelineRequest {
        project: PROJECT,
        labels: vec![label],
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

fn instruction(
    output_type: ResidentRowValueType,
    operation: ResidentRowOperation,
) -> ResidentRowInstruction {
    ResidentRowInstruction {
        output_type,
        operation,
    }
}

fn strings(values: &[Option<&str>]) -> ResidentRowColumn {
    let mut offsets = Vec::with_capacity(values.len().saturating_add(1));
    let mut bytes = Vec::new();
    let mut validity = Vec::with_capacity(values.len());
    offsets.push(0);
    for value in values {
        if let Some(value) = value {
            bytes.extend_from_slice(value.as_bytes());
            validity.push(1);
        } else {
            validity.push(0);
        }
        offsets.push(u32::try_from(bytes.len()).expect("test string column exceeds u32"));
    }
    ResidentRowColumn::String {
        offsets,
        bytes,
        validity,
    }
}

fn scalar_concat_program() -> ResidentRowProgram {
    ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(strings(&[
                    Some(""),
                    Some("é"),
                    Some("a"),
                    Some("a"),
                    Some("a"),
                    None,
                    Some("a"),
                    Some("aa"),
                ])),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant("\0é".to_owned()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(strings(&[
                    Some("🙂"),
                    Some(""),
                    Some("🙂"),
                    Some("β"),
                    Some("β"),
                    Some("z"),
                    None,
                    Some(""),
                ])),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 2, right: 3 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                    values: (100..108).collect(),
                    validity: vec![1; SCALAR_ROWS],
                }),
            ),
        ],
    }
}

fn scalar_coalesce_program() -> ResidentRowProgram {
    ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(strings(&[
                    Some(""),
                    None,
                    Some("left"),
                    None,
                    Some("é"),
                    None,
                    Some("\0"),
                    None,
                ])),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(strings(&[
                    Some("right0"),
                    Some("right1"),
                    None,
                    None,
                    Some("fallback"),
                    Some("🙂"),
                    Some("right6"),
                    Some(""),
                ])),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringCoalesce { left: 0, right: 1 },
            ),
        ],
    }
}

fn graph_concat_program(
    title: PropertyId,
    name: PropertyId,
    title_maximum_bytes: u32,
    name_maximum_bytes: u32,
) -> ResidentRowProgram {
    let start = ResidentEntityBinding::Node(ResidentNodeBinding::Start);
    ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: start,
                    property: title,
                    maximum_bytes: title_maximum_bytes,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(" ".to_owned()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: start,
                    property: name,
                    maximum_bytes: name_maximum_bytes,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 2, right: 3 },
            ),
        ],
    }
}

fn execute_validated<B: ExecutionBackend>(
    backend: &B,
    request: &ResidentRowProgramRequest,
    kind: BackendKind,
) -> Result<ValidatedResidentRowProgramResult> {
    backend
        .execute_row_program(request, &CancellationToken::new())?
        .validate(request, kind)
}

fn integer_projection(result: &ValidatedResidentRowProgramResult) -> (&[i64], &[u8]) {
    let ResidentRowColumn::Integer { values, validity } = &result.projected_columns()[0].column
    else {
        panic!("expected the integer identity projection");
    };
    (values, validity)
}

fn assert_completion(
    result: &ValidatedResidentRowProgramResult,
    expected: ResidentDeviceCompletion,
) {
    assert!(!result.receipts().is_empty());
    assert!(
        result
            .receipts()
            .iter()
            .all(|receipt| receipt.completion == expected)
    );
}

#[test]
fn string_columns_constants_concat_capacity_and_exact_scratch_are_admitted() -> Result<()> {
    let harness = Harness::new()?;
    let request = harness.scalar_request(1, false, false)?;
    request.validate()?;

    let ResidentRowOperation::InputColumn(ResidentRowColumn::String {
        offsets,
        bytes,
        validity,
    }) = &request.program.instructions[0].operation
    else {
        panic!("register zero must be the canonical STRING input");
    };
    assert_eq!(offsets, &[0, 0, 2, 3, 4, 5, 5, 6, 8]);
    assert_eq!(bytes, &[0xc3, 0xa9, b'a', b'a', b'a', b'a', b'a', b'a']);
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0, 1, 1]);

    let ResidentRowOperation::StringConstant(constant) = &request.program.instructions[1].operation
    else {
        panic!("register one must be the exact STRING constant");
    };
    assert_eq!(constant.as_bytes(), &[0x00, 0xc3, 0xa9]);

    let capacities = (0..request.program.instructions.len())
        .map(|register| request.program.string_register_capacity(register as u16))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        capacities,
        vec![Some(2), Some(3), Some(5), Some(4), Some(9), None]
    );

    // Five STRING registers own 23 arena bytes per input row. Every STRING/INTEGER register also
    // owns an eight-byte payload and one validity byte; the selected INTEGER owns the same output
    // width. Scalar input has no aligned graph columns.
    let expected_scratch = SCALAR_ROWS
        .checked_mul(54 + 23 + size_of::<u64>() + 9)
        .and_then(|bytes| {
            bytes.checked_add(7 * size_of::<irongraph::gpu::ResidentExecutionReceipt>())
        })
        .expect("test scratch calculation overflowed");
    assert_eq!(request.obligations().count(), 7);
    assert_eq!(request.scratch_bytes(SCALAR_ROWS)?, expected_scratch);
    Ok(())
}

#[test]
fn cpu_string_constant_concat_utf8_nulls_and_stable_order_are_exact() -> Result<()> {
    let harness = Harness::new()?;
    let cpu = harness.cpu()?;

    let ascending = harness.scalar_request(2, false, false)?;
    let ascending_result = execute_validated(&cpu, &ascending, BackendKind::Cpu)?;
    assert_eq!(
        ascending_result.source_positions(),
        &[0, 3, 4, 2, 7, 1, 5, 6]
    );
    assert_eq!(
        integer_projection(&ascending_result),
        (
            &[100, 103, 104, 102, 107, 101, 105, 106][..],
            &[1; SCALAR_ROWS][..]
        )
    );
    assert_completion(&ascending_result, ResidentDeviceCompletion::CpuReference);
    assert_eq!(
        ascending_result.scratch_bytes(),
        ascending.scratch_bytes(SCALAR_ROWS)?
    );

    let descending = harness.scalar_request(3, true, true)?;
    let descending_result = execute_validated(&cpu, &descending, BackendKind::Cpu)?;
    assert_eq!(
        descending_result.source_positions(),
        &[5, 6, 1, 7, 2, 3, 4, 0]
    );
    assert_eq!(
        integer_projection(&descending_result).0,
        &[105, 106, 101, 107, 102, 103, 104, 100]
    );
    assert_completion(&descending_result, ResidentDeviceCompletion::CpuReference);
    Ok(())
}

#[test]
fn cpu_load_string_property_concat_nulls_and_byte_order_are_exact() -> Result<()> {
    let harness = Harness::new()?;
    let cpu = harness.cpu()?;

    let ascending = harness.graph_request(4, false, false)?;
    assert_eq!(
        (0..ascending.program.instructions.len())
            .map(|register| ascending.program.string_register_capacity(register as u16))
            .collect::<Result<Vec<_>>>()?,
        vec![Some(4), Some(1), Some(5), Some(2), Some(7)]
    );
    // Two aligned graph copies use U32 rows; stable source positions use U64. The projected title
    // owns an eight-byte reference, validity byte, and four-byte-per-row UTF-8 slot.
    let expected_scratch = GRAPH_ROWS
        .checked_mul(45 + 19 + (2 * size_of::<u32>()) + size_of::<u64>() + 9 + 4)
        .and_then(|bytes| {
            bytes.checked_add(6 * size_of::<irongraph::gpu::ResidentExecutionReceipt>())
        })
        .expect("test graph scratch calculation overflowed");
    assert_eq!(ascending.scratch_bytes(GRAPH_ROWS)?, expected_scratch);

    let ascending_result = execute_validated(&cpu, &ascending, BackendKind::Cpu)?;
    assert_eq!(
        ascending_result.source_positions(),
        &[0, 1, 8, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        ascending_result.rows().start_rows,
        [0, 1, 8, 2, 3, 4, 5, 6, 7]
    );
    assert!(matches!(
        &ascending_result.projected_columns()[0].column,
        ResidentRowColumn::String {
            offsets,
            bytes,
            validity,
        } if offsets == &[0, 0, 1, 2, 3, 5, 7, 11, 11, 12]
            && bytes == "aaaaaé🙂a".as_bytes()
            && validity == &[1, 1, 1, 1, 1, 1, 1, 0, 1]
    ));
    assert_completion(&ascending_result, ResidentDeviceCompletion::CpuReference);

    let descending = harness.graph_request(5, true, true)?;
    let descending_result = execute_validated(&cpu, &descending, BackendKind::Cpu)?;
    assert_eq!(
        descending_result.source_positions(),
        &[6, 7, 5, 4, 3, 2, 1, 8, 0]
    );
    assert_completion(&descending_result, ResidentDeviceCompletion::CpuReference);
    Ok(())
}

#[test]
fn cpu_string_coalesce_keeps_valid_empty_left_and_uses_right_only_for_null() -> Result<()> {
    let harness = Harness::new()?;
    let request = harness.coalesce_request(51)?;
    let result = execute_validated(&harness.cpu()?, &request, BackendKind::Cpu)?;
    assert!(matches!(
        &result.projected_columns()[0].column,
        ResidentRowColumn::String {
            offsets,
            bytes,
            validity,
        } if offsets == &[0, 0, 6, 10, 10, 12, 16, 17, 17]
            && bytes.as_slice() == b"right1left\xc3\xa9\xf0\x9f\x99\x82\0"
            && validity == &[1, 1, 1, 0, 1, 1, 1, 1]
    ));
    assert_completion(&result, ResidentDeviceCompletion::CpuReference);
    Ok(())
}

#[test]
fn cpu_string_scratch_admits_exactly_and_rejects_one_byte_short() -> Result<()> {
    let harness = Harness::new()?;
    let cpu = harness.cpu()?;
    let request = harness.scalar_request(6, false, false)?;
    let logical = request.scratch_bytes(SCALAR_ROWS)?;
    let available = cpu.available_query_scratch_bytes();
    assert!(available > logical);

    let exact_blocker = cpu.reserve_query_scratch(available - logical)?;
    execute_validated(&cpu, &request, BackendKind::Cpu)?;
    drop(exact_blocker);

    let short_blocker = cpu.reserve_query_scratch(available - logical + 1)?;
    let error = cpu
        .execute_row_program(&request, &CancellationToken::new())
        .expect_err("one-byte-short STRING arena scratch must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    drop(short_blocker);
    execute_validated(&cpu, &request, BackendKind::Cpu)?;
    Ok(())
}

#[test]
fn string_abi_rejects_malformed_columns_and_capacity_overflow_and_projects_exact_utf8() -> Result<()>
{
    for malformed in [
        ResidentRowColumn::String {
            offsets: vec![0, 1],
            bytes: vec![0xff],
            validity: vec![1],
        },
        ResidentRowColumn::String {
            offsets: vec![0, 1],
            bytes: vec![b'x'],
            validity: vec![0],
        },
    ] {
        let error = ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(malformed),
            )],
        }
        .validate()
        .expect_err("malformed STRING input must fail closed");
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        assert_eq!(
            error.message.as_ref(),
            "resident row string column has an invalid canonical UTF-8 shape"
        );
    }

    let start = ResidentEntityBinding::Node(ResidentNodeBinding::Start);
    let overflow = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: start,
                    property: PropertyId(1),
                    maximum_bytes: u32::MAX,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: start,
                    property: PropertyId(2),
                    maximum_bytes: u32::MAX,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    }
    .validate()
    .expect_err("STRING concatenation capacity beyond u32 must fail closed");
    assert_eq!(overflow.code, ErrorCode::ResultBudgetExceeded);
    assert_eq!(
        overflow.message.as_ref(),
        "resident string register capacity exceeds u32"
    );

    let harness = Harness::new()?;
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::String,
            ResidentRowOperation::InputColumn(strings(&[Some("é")])),
        )],
    };
    let projection = harness.request(7, scalar_input(1), program, Vec::new(), vec![0])?;
    projection.validate()?;
    let projected = execute_validated(&harness.cpu()?, &projection, BackendKind::Cpu)?;
    assert!(matches!(
        &projected.projected_columns()[0].column,
        ResidentRowColumn::String {
            offsets,
            bytes,
            validity,
        } if offsets == &[0, 2] && bytes == "é".as_bytes() && validity == &[1]
    ));
    Ok(())
}

#[test]
fn cpu_load_string_property_requires_the_exact_admitted_maximum_width() -> Result<()> {
    let harness = Harness::new()?;
    let cpu = harness.cpu()?;
    for (seed, maximum_bytes) in [(8, 3), (9, 5)] {
        let request = harness.request(
            seed,
            graph_input(harness.label, GRAPH_ROWS),
            graph_concat_program(harness.title, harness.name, maximum_bytes, 2),
            vec![ResidentRowSortKey {
                register: 4,
                descending: false,
                nulls_first: false,
            }],
            Vec::new(),
        )?;
        request.validate()?;
        let error = cpu
            .execute_row_program(&request, &CancellationToken::new())
            .expect_err("dishonest maximum_bytes must fail before STRING arena execution");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert_eq!(
            error.message.as_ref(),
            "resident row string property maximum UTF-8 width does not match the resident dictionary"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_matches_cpu_for_scalar_and_property_string_arenas_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let harness = Harness::new()?;
    let cpu = harness.cpu()?;
    let metal = harness.metal()?;
    let available_before = metal.available_query_scratch_bytes();

    for request in [
        harness.scalar_request(101, false, false)?,
        harness.scalar_request(102, true, true)?,
        harness.graph_request(103, false, false)?,
        harness.graph_request(104, true, true)?,
        harness.coalesce_request(107)?,
    ] {
        let cpu_result = execute_validated(&cpu, &request, BackendKind::Cpu)?;
        let metal_result = execute_validated(&metal, &request, BackendKind::Metal)?;
        assert_eq!(
            metal_result.source_positions(),
            cpu_result.source_positions()
        );
        assert_eq!(metal_result.rows(), cpu_result.rows());
        assert_eq!(
            metal_result.projected_columns(),
            cpu_result.projected_columns()
        );
        assert_eq!(metal_result.scratch_bytes(), cpu_result.scratch_bytes());
        assert_completion(&cpu_result, ResidentDeviceCompletion::CpuReference);
        assert_completion(&metal_result, ResidentDeviceCompletion::Metal);
        assert_eq!(metal_result.receipts().len(), request.obligations().count());
    }
    assert_eq!(metal.available_query_scratch_bytes(), available_before);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_string_arena_scratch_and_width_mismatch_fail_closed() -> Result<()> {
    let _guard = metal_test_guard();
    let harness = Harness::new()?;
    let metal = harness.metal()?;
    let request = harness.graph_request(105, false, false)?;
    let logical = request.scratch_bytes(GRAPH_ROWS)?;
    let available = metal.available_query_scratch_bytes();
    assert!(available > logical);
    let blocker = metal.reserve_query_scratch(available - logical + 1)?;
    let error = metal
        .execute_row_program(&request, &CancellationToken::new())
        .expect_err("one-byte-short Metal STRING arena scratch must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    drop(blocker);
    execute_validated(&metal, &request, BackendKind::Metal)?;

    let dishonest = harness.request(
        106,
        graph_input(harness.label, GRAPH_ROWS),
        graph_concat_program(harness.title, harness.name, 3, 2),
        vec![ResidentRowSortKey {
            register: 4,
            descending: false,
            nulls_first: false,
        }],
        Vec::new(),
    )?;
    let error = metal
        .execute_row_program(&dishonest, &CancellationToken::new())
        .expect_err("Metal must reject an underestimated property STRING slot");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(
        error.message.as_ref(),
        "resident row string property maximum UTF-8 width does not match the resident dictionary"
    );
    Ok(())
}
