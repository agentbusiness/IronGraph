// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::hint::black_box;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Mutex, MutexGuard, OnceLock};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentEntityBinding,
        ResidentExecutionId, ResidentNodeBinding, ResidentNodePipelineRequest,
        ResidentProjectImage, ResidentRowColumn, ResidentRowInstruction,
        ResidentRowManifestFingerprint, ResidentRowOperation, ResidentRowProgram,
        ResidentRowProgramManifest, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentRowProgramResultParts, ResidentRowSortKey, ResidentRowValueType,
        ValidatedResidentRowProgramResult,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;
const QUIET_NAN_BITS: u64 = 0x7ff8_0000_0000_0042;

struct Harness {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Harness {
    fn new() -> Self {
        let graph = GraphStore::default();
        let bookmark = Bookmark {
            term: 17,
            index: graph.revision(),
        };
        Self { graph, bookmark }
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
        let input_rows = program
            .instructions
            .iter()
            .find_map(|instruction| match &instruction.operation {
                ResidentRowOperation::InputColumn(column) => Some(column.len()),
                _ => None,
            })
            .ok_or_else(|| Error::internal("adversarial scalar program has no input column"))?;
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &sort_keys,
            offset,
            limit,
            max_output_rows,
            &final_registers,
            seed.checked_mul(1_000)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| Error::internal("adversarial obligation ID overflow"))?,
        )?;
        Ok(ResidentRowProgramRequest {
            project: PROJECT,
            expected_bookmark: self.bookmark,
            expected_graph_revision: self.graph.revision(),
            expected_layout_version: self.graph.layout_version(),
            execution: ResidentExecutionId {
                high: 0x4e55_4d45_5249_4352,
                low: seed,
            },
            input: scalar_input(input_rows),
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

fn instruction(
    output_type: ResidentRowValueType,
    operation: ResidentRowOperation,
) -> ResidentRowInstruction {
    ResidentRowInstruction {
        output_type,
        operation,
    }
}

fn integers(values: &[i64], validity: &[u8]) -> ResidentRowColumn {
    ResidentRowColumn::Integer {
        values: values.to_vec(),
        validity: validity.to_vec(),
    }
}

fn valid_integers(values: &[i64]) -> ResidentRowColumn {
    integers(values, &vec![1; values.len()])
}

fn booleans(values: &[u8], validity: &[u8]) -> ResidentRowColumn {
    ResidentRowColumn::Boolean {
        values: values.to_vec(),
        validity: validity.to_vec(),
    }
}

fn floats(bits: &[u64], validity: &[u8]) -> ResidentRowColumn {
    ResidentRowColumn::Float {
        bits: bits.to_vec(),
        validity: validity.to_vec(),
    }
}

fn valid_floats(bits: &[u64]) -> ResidentRowColumn {
    floats(bits, &vec![1; bits.len()])
}

fn execute_raw<B: ExecutionBackend>(
    backend: &B,
    request: &ResidentRowProgramRequest,
) -> Result<ResidentRowProgramResult> {
    backend.execute_row_program(request, &CancellationToken::new())
}

fn execute_validated<B: ExecutionBackend>(
    backend: &B,
    request: &ResidentRowProgramRequest,
    kind: BackendKind,
) -> Result<ValidatedResidentRowProgramResult> {
    execute_raw(backend, request)?.validate(request, kind)
}

fn integer_projection(result: &ValidatedResidentRowProgramResult, index: usize) -> (&[i64], &[u8]) {
    let ResidentRowColumn::Integer { values, validity } = &result.projected_columns()[index].column
    else {
        panic!("expected INTEGER projection {index}");
    };
    (values, validity)
}

fn float_projection(result: &ValidatedResidentRowProgramResult, index: usize) -> (&[u64], &[u8]) {
    let ResidentRowColumn::Float { bits, validity } = &result.projected_columns()[index].column
    else {
        panic!("expected FLOAT projection {index}");
    };
    (bits, validity)
}

fn integer_request(harness: &Harness, seed: u64) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[7, -7, 7, -7, 0])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[3, 3, -3, -3, -5])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericSubtract { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerModulo { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 0 },
            ),
        ],
    };
    harness.request(
        seed,
        program,
        Vec::new(),
        0,
        usize::MAX,
        5,
        vec![2, 3, 4, 5, 6, 2],
    )
}

fn modulo_boundary_request(harness: &Harness, seed: u64) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[i64::MIN, 7, -7, 7, -7])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[-1, -3, -3, 3, 3])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerModulo { left: 0, right: 1 },
            ),
        ],
    };
    harness.request(seed, program, Vec::new(), 0, usize::MAX, 5, vec![2])
}

fn null_arithmetic_request(harness: &Harness, seed: u64) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(integers(
                    &[8, i64::MAX, 9, i64::MIN, i64::MIN],
                    &[1, 0, 1, 0, 0],
                )),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(integers(&[2, 1, 0, 0, 0], &[1, 1, 0, 0, 1])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericSubtract { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerModulo { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 0 },
            ),
        ],
    };
    harness.request(
        seed,
        program,
        Vec::new(),
        0,
        usize::MAX,
        5,
        vec![2, 3, 4, 5, 6],
    )
}

fn mixed_float_inputs() -> (Vec<i64>, Vec<u8>, Vec<u64>, Vec<u64>) {
    (
        vec![2, -3, 0, 2, 1, 9_007_199_254_740_993, 4],
        vec![1, 1, 1, 1, 1, 1, 0],
        vec![
            0.5_f64.to_bits(),
            (-0.0_f64).to_bits(),
            (-0.0_f64).to_bits(),
            f64::INFINITY.to_bits(),
            QUIET_NAN_BITS,
            0.0_f64.to_bits(),
            3.0_f64.to_bits(),
        ],
        vec![
            (-0.5_f64).to_bits(),
            (-0.0_f64).to_bits(),
            2.0_f64.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            1.0_f64.to_bits(),
            (-0.0_f64).to_bits(),
            1.0_f64.to_bits(),
        ],
    )
}

fn mixed_float_request(harness: &Harness, seed: u64) -> Result<ResidentRowProgramRequest> {
    let (integer_values, integer_validity, left_float, right_float) = mixed_float_inputs();
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(integers(&integer_values, &integer_validity)),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::InputColumn(valid_floats(&left_float)),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::InputColumn(valid_floats(&right_float)),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericSubtract { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericAdd { left: 1, right: 2 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericSubtract { left: 1, right: 2 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericMultiply { left: 1, right: 2 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericNegate { operand: 1 },
            ),
        ],
    };
    harness.request(
        seed,
        program,
        Vec::new(),
        0,
        usize::MAX,
        integer_values.len(),
        vec![3, 4, 5, 6, 7, 8, 9, 3],
    )
}

#[allow(clippy::too_many_arguments)]
fn sort_request(
    harness: &Harness,
    seed: u64,
    descending: bool,
    nulls_first: bool,
    offset: usize,
    limit: usize,
    max_output_rows: usize,
) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[
                    100, 101, 102, 103, 104, 105, 106, 107,
                ])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(integers(
                    &[2, 1, 1, 0, 0, 3, 99, 88],
                    &[1, 1, 1, 1, 1, 1, 0, 0],
                )),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(0),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericSubtract { left: 1, right: 2 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 0 },
            ),
        ],
    };
    harness.request(
        seed,
        program,
        vec![ResidentRowSortKey {
            register: 3,
            descending,
            nulls_first,
        }],
        offset,
        limit,
        max_output_rows,
        vec![0, 4, 0],
    )
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn successful_requests(
    harness: &Harness,
) -> Result<Vec<(&'static str, ResidentRowProgramRequest)>> {
    Ok(vec![
        ("integer arithmetic", integer_request(harness, 101)?),
        ("modulo boundaries", modulo_boundary_request(harness, 102)?),
        ("null arithmetic", null_arithmetic_request(harness, 103)?),
        (
            "ascending nulls last",
            sort_request(harness, 105, false, false, 0, usize::MAX, 8)?,
        ),
        (
            "ascending nulls first",
            sort_request(harness, 106, false, true, 0, usize::MAX, 8)?,
        ),
        (
            "descending nulls first",
            sort_request(harness, 107, true, true, 0, usize::MAX, 8)?,
        ),
        (
            "descending nulls last",
            sort_request(harness, 108, true, false, 0, usize::MAX, 8)?,
        ),
        (
            "offset and limit",
            sort_request(harness, 109, false, false, 2, 3, 3)?,
        ),
        (
            "empty window",
            sort_request(harness, 110, false, false, 3, 0, 0)?,
        ),
        ("mixed float", mixed_float_request(harness, 104)?),
    ])
}

struct ExpectedFailure {
    name: &'static str,
    request: ResidentRowProgramRequest,
    code: ErrorCode,
    message: &'static str,
}

fn binary_failure_request(
    harness: &Harness,
    seed: u64,
    left: i64,
    right: i64,
    operation: ResidentRowOperation,
) -> Result<ResidentRowProgramRequest> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[left])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[right])),
            ),
            instruction(ResidentRowValueType::Integer, operation),
        ],
    };
    harness.request(seed, program, Vec::new(), 0, 1, 1, vec![2])
}

fn runtime_failures(harness: &Harness) -> Result<Vec<ExpectedFailure>> {
    let add = binary_failure_request(
        harness,
        201,
        i64::MAX,
        1,
        ResidentRowOperation::NumericAdd { left: 0, right: 1 },
    )?;
    let subtract = binary_failure_request(
        harness,
        202,
        i64::MIN,
        1,
        ResidentRowOperation::NumericSubtract { left: 0, right: 1 },
    )?;
    let multiply = binary_failure_request(
        harness,
        203,
        i64::MAX,
        2,
        ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
    )?;
    let modulo = binary_failure_request(
        harness,
        204,
        7,
        0,
        ResidentRowOperation::IntegerModulo { left: 0, right: 1 },
    )?;
    let negate_program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[i64::MIN])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 0 },
            ),
        ],
    };
    let negate = harness.request(205, negate_program, Vec::new(), 0, 1, 1, vec![1])?;
    Ok(vec![
        ExpectedFailure {
            name: "INTEGER add overflow",
            request: add,
            code: ErrorCode::QueryType,
            message: "integer arithmetic overflow",
        },
        ExpectedFailure {
            name: "INTEGER subtract overflow",
            request: subtract,
            code: ErrorCode::QueryType,
            message: "integer arithmetic overflow",
        },
        ExpectedFailure {
            name: "INTEGER multiply overflow",
            request: multiply,
            code: ErrorCode::QueryType,
            message: "integer arithmetic overflow",
        },
        ExpectedFailure {
            name: "INTEGER modulo by zero",
            request: modulo,
            code: ErrorCode::QueryType,
            message: "modulo by zero",
        },
        ExpectedFailure {
            name: "INTEGER negate MIN",
            request: negate,
            code: ErrorCode::QueryType,
            message: "integer negation overflow",
        },
    ])
}

fn invalid_requests(harness: &Harness) -> Result<Vec<ExpectedFailure>> {
    let base = integer_request(harness, 301)?;

    let mut forward = base.clone();
    forward.program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[1, 2])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 2 },
            ),
        ],
    };

    let mut non_numeric = base.clone();
    non_numeric.program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::InputColumn(booleans(&[0, 1], &[1, 1])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 0 },
            ),
        ],
    };

    let mut wrong_promotion = base.clone();
    wrong_promotion.program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[1, 2])),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::InputColumn(valid_floats(&[
                    1.0_f64.to_bits(),
                    2.0_f64.to_bits(),
                ])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
        ],
    };

    let mut mixed_cardinality = base.clone();
    mixed_cardinality.program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[1, 2])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[1, 2, 3])),
            ),
        ],
    };

    let mut graph_descriptor = base.clone();
    graph_descriptor.input.labels = vec![LabelId(999)];

    let mut property_load = base;
    property_load.program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(valid_integers(&[1])),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(999),
                },
            ),
        ],
    };
    property_load.input.max_output_rows = 1;

    Ok(vec![
        ExpectedFailure {
            name: "forward register reference",
            request: forward,
            code: ErrorCode::QueryType,
            message: "resident row instruction reads a register before it is defined",
        },
        ExpectedFailure {
            name: "non-numeric arithmetic input",
            request: non_numeric,
            code: ErrorCode::QueryType,
            message: "resident numeric instruction requires numeric input registers",
        },
        ExpectedFailure {
            name: "wrong INTEGER/FLOAT promotion",
            request: wrong_promotion,
            code: ErrorCode::QueryType,
            message: "resident numeric instruction output does not match INTEGER/FLOAT promotion",
        },
        ExpectedFailure {
            name: "mixed scalar cardinalities",
            request: mixed_cardinality,
            code: ErrorCode::GpuAdmissionFailure,
            message: "resident typed input columns have different cardinalities",
        },
        ExpectedFailure {
            name: "graph descriptor mixed with scalar input",
            request: graph_descriptor,
            code: ErrorCode::GpuAdmissionFailure,
            message: "resident typed literal input has a non-empty graph source descriptor",
        },
        ExpectedFailure {
            name: "graph property load mixed with scalar input",
            request: property_load,
            code: ErrorCode::GpuAdmissionFailure,
            message: "resident typed literal input cannot be mixed with graph-property loads",
        },
    ])
}

fn assert_exact_error(error: &Error, code: ErrorCode, message: &str, context: &str) {
    assert_eq!(error.code, code, "{context}: wrong error code: {error:?}");
    assert_eq!(
        error.message, message,
        "{context}: wrong error message: {error:?}"
    );
}

fn assert_failure_on<B: ExecutionBackend>(backend: &B, failure: &ExpectedFailure) -> Error {
    let error = match execute_raw(backend, &failure.request) {
        Ok(_) => panic!("{} unexpectedly succeeded", failure.name),
        Err(error) => error,
    };
    assert_exact_error(&error, failure.code, failure.message, failure.name);
    error
}

fn canonical_float_binary(left: f64, right: f64, operation: ResidentRowOperation) -> u64 {
    let left = black_box(left);
    let right = black_box(right);
    match operation {
        ResidentRowOperation::NumericAdd { .. } => (left + right).to_bits(),
        ResidentRowOperation::NumericSubtract { .. } => (left - right).to_bits(),
        ResidentRowOperation::NumericMultiply { .. } => (left * right).to_bits(),
        _ => panic!("test requested a non-floating binary operation"),
    }
}

fn assert_integer_success(result: &ValidatedResidentRowProgramResult) {
    assert_eq!(result.source_positions(), &[0, 1, 2, 3, 4]);
    let expected = [
        vec![10, -4, 4, -10, -5],
        vec![4, -10, 10, -4, 5],
        vec![21, -21, -21, 21, 0],
        vec![1, -1, 1, -1, 0],
        vec![-7, 7, -7, 7, 0],
        vec![10, -4, 4, -10, -5],
    ];
    for (index, expected) in expected.iter().enumerate() {
        let (values, validity) = integer_projection(result, index);
        assert_eq!(values, expected);
        assert_eq!(validity, &[1, 1, 1, 1, 1]);
    }
    assert_eq!(result.projected_columns()[0].register, 2);
    assert_eq!(result.projected_columns()[5].register, 2);
    assert_eq!(
        result.projected_columns()[0].column,
        result.projected_columns()[5].column
    );
}

fn assert_modulo_boundaries(result: &ValidatedResidentRowProgramResult) {
    let (values, validity) = integer_projection(result, 0);
    assert_eq!(values, &[0, 1, -1, 1, -1]);
    assert_eq!(validity, &[1, 1, 1, 1, 1]);
}

fn assert_null_arithmetic(result: &ValidatedResidentRowProgramResult) {
    for index in 0..4 {
        let (values, validity) = integer_projection(result, index);
        let expected = match index {
            0 => [10, 0, 0, 0, 0],
            1 => [6, 0, 0, 0, 0],
            2 => [16, 0, 0, 0, 0],
            3 => [0, 0, 0, 0, 0],
            _ => unreachable!(),
        };
        assert_eq!(values, &expected);
        assert_eq!(validity, &[1, 0, 0, 0, 0]);
    }
    let (values, validity) = integer_projection(result, 4);
    assert_eq!(values, &[-8, 0, -9, 0, 0]);
    assert_eq!(validity, &[1, 0, 1, 0, 0]);
}

fn assert_mixed_float_success(result: &ValidatedResidentRowProgramResult) {
    let (integer_values, integer_validity, left_bits, right_bits) = mixed_float_inputs();
    let mut expected = vec![Vec::new(); 7];
    for row in 0..integer_values.len() {
        let left = f64::from_bits(left_bits[row]);
        let right = f64::from_bits(right_bits[row]);
        if integer_validity[row] == 0 {
            expected[0].push(0);
            expected[1].push(0);
            expected[2].push(0);
        } else {
            let integer = integer_values[row] as f64;
            expected[0].push(canonical_float_binary(
                integer,
                left,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ));
            expected[1].push(canonical_float_binary(
                integer,
                left,
                ResidentRowOperation::NumericSubtract { left: 0, right: 1 },
            ));
            expected[2].push(canonical_float_binary(
                integer,
                left,
                ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
            ));
        }
        expected[3].push(canonical_float_binary(
            left,
            right,
            ResidentRowOperation::NumericAdd { left: 1, right: 2 },
        ));
        expected[4].push(canonical_float_binary(
            left,
            right,
            ResidentRowOperation::NumericSubtract { left: 1, right: 2 },
        ));
        expected[5].push(canonical_float_binary(
            left,
            right,
            ResidentRowOperation::NumericMultiply { left: 1, right: 2 },
        ));
        expected[6].push(left_bits[row] ^ (1_u64 << 63));
    }

    for (index, expected) in expected.iter().enumerate() {
        let (bits, validity) = float_projection(result, index);
        assert_eq!(bits, expected, "FLOAT projection {index} changed IEEE bits");
        if index < 3 {
            assert_eq!(validity, integer_validity);
        } else {
            assert_eq!(validity, &[1, 1, 1, 1, 1, 1, 1]);
        }
    }
    assert_eq!(
        result.projected_columns()[0].column,
        result.projected_columns()[7].column
    );

    let (mixed_add, _) = float_projection(result, 0);
    let (mixed_subtract, _) = float_projection(result, 1);
    let (mixed_multiply, _) = float_projection(result, 2);
    let (float_add, _) = float_projection(result, 3);
    let (float_subtract, _) = float_projection(result, 4);
    assert_eq!(mixed_add[2], 0.0_f64.to_bits());
    assert_eq!(mixed_multiply[2], (-0.0_f64).to_bits());
    assert_eq!(mixed_add[3], f64::INFINITY.to_bits());
    assert_eq!(mixed_subtract[3], f64::NEG_INFINITY.to_bits());
    assert_eq!(mixed_multiply[3], f64::INFINITY.to_bits());
    assert_eq!(float_add[1], (-0.0_f64).to_bits());
    assert_eq!(float_subtract[1], 0.0_f64.to_bits());
    assert!(f64::from_bits(mixed_add[4]).is_nan());
    assert!(f64::from_bits(float_add[3]).is_nan());
}

fn assert_sort_result(result: &ValidatedResidentRowProgramResult, expected: &[u64]) {
    assert_eq!(result.source_positions(), expected);
    assert_eq!(
        result
            .projected_columns()
            .iter()
            .map(|column| column.register)
            .collect::<Vec<_>>(),
        vec![0, 4, 0]
    );
    let expected_values = expected
        .iter()
        .map(|position| 100_i64 + i64::try_from(*position).expect("tiny test position fits i64"))
        .collect::<Vec<_>>();
    let expected_negated = expected_values
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    let (values, validity) = integer_projection(result, 0);
    assert_eq!(values, expected_values);
    assert!(validity.iter().all(|valid| *valid == 1));
    let (values, validity) = integer_projection(result, 1);
    assert_eq!(values, expected_negated);
    assert!(validity.iter().all(|valid| *valid == 1));
    assert_eq!(
        result.projected_columns()[0].column,
        result.projected_columns()[2].column
    );
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn assert_differential_equal(
    context: &str,
    cpu: &ValidatedResidentRowProgramResult,
    metal: &ValidatedResidentRowProgramResult,
) {
    assert_eq!(metal.project(), cpu.project(), "{context}: project");
    assert_eq!(metal.execution(), cpu.execution(), "{context}: execution");
    assert_eq!(metal.bookmark(), cpu.bookmark(), "{context}: bookmark");
    assert_eq!(
        metal.graph_revision(),
        cpu.graph_revision(),
        "{context}: graph revision"
    );
    assert_eq!(
        metal.layout_version(),
        cpu.layout_version(),
        "{context}: layout version"
    );
    assert_eq!(
        metal.manifest_fingerprint(),
        cpu.manifest_fingerprint(),
        "{context}: manifest fingerprint"
    );
    assert_eq!(
        metal.input_cardinality(),
        cpu.input_cardinality(),
        "{context}: input cardinality"
    );
    assert_eq!(
        metal.source_positions(),
        cpu.source_positions(),
        "{context}: source positions"
    );
    assert_eq!(metal.rows(), cpu.rows(), "{context}: graph columns");
    assert_eq!(
        metal.projected_columns().len(),
        cpu.projected_columns().len(),
        "{context}: projected-column count"
    );
    for (projection, (cpu, metal)) in cpu
        .projected_columns()
        .iter()
        .zip(metal.projected_columns())
        .enumerate()
    {
        assert_eq!(
            metal.register, cpu.register,
            "{context}: projection {projection} register"
        );
        match (&cpu.column, &metal.column) {
            (
                ResidentRowColumn::Boolean {
                    values: cpu_values,
                    validity: cpu_validity,
                },
                ResidentRowColumn::Boolean {
                    values: metal_values,
                    validity: metal_validity,
                },
            ) => {
                assert_eq!(
                    metal_values, cpu_values,
                    "{context}: projection {projection} Boolean payload"
                );
                assert_eq!(
                    metal_validity, cpu_validity,
                    "{context}: projection {projection} Boolean validity"
                );
            }
            (
                ResidentRowColumn::Integer {
                    values: cpu_values,
                    validity: cpu_validity,
                },
                ResidentRowColumn::Integer {
                    values: metal_values,
                    validity: metal_validity,
                },
            ) => {
                assert_eq!(
                    metal_validity, cpu_validity,
                    "{context}: projection {projection} INTEGER validity"
                );
                assert_eq!(metal_values.len(), cpu_values.len());
                for (row, (cpu_value, metal_value)) in
                    cpu_values.iter().zip(metal_values).enumerate()
                {
                    assert_eq!(
                        metal_value, cpu_value,
                        "{context}: projection {projection}, register {}, row {row} INTEGER payload",
                        cpu.register
                    );
                }
            }
            (
                ResidentRowColumn::Float {
                    bits: cpu_bits,
                    validity: cpu_validity,
                },
                ResidentRowColumn::Float {
                    bits: metal_bits,
                    validity: metal_validity,
                },
            ) => {
                assert_eq!(
                    metal_validity, cpu_validity,
                    "{context}: projection {projection} FLOAT validity"
                );
                assert_eq!(metal_bits.len(), cpu_bits.len());
                for (row, (cpu_bits, metal_bits)) in cpu_bits.iter().zip(metal_bits).enumerate() {
                    assert_eq!(
                        metal_bits, cpu_bits,
                        "{context}: projection {projection}, register {}, row {row} IEEE-754 bits",
                        cpu.register
                    );
                }
            }
            _ => panic!(
                "{context}: projection {projection}, register {} changed physical type",
                cpu.register
            ),
        }
    }
    assert_eq!(
        metal.scratch_bytes(),
        cpu.scratch_bytes(),
        "{context}: logical scratch"
    );
    assert_eq!(
        metal.receipts().len(),
        cpu.receipts().len(),
        "{context}: receipt count"
    );
    for (index, (cpu, metal)) in cpu.receipts().iter().zip(metal.receipts()).enumerate() {
        assert_eq!(metal.execution, cpu.execution, "{context}: receipt {index}");
        assert_eq!(
            metal.obligation, cpu.obligation,
            "{context}: receipt {index}"
        );
        assert_eq!(
            metal.input_cardinality, cpu.input_cardinality,
            "{context}: receipt {index} input"
        );
        assert_eq!(
            metal.output_cardinality, cpu.output_cardinality,
            "{context}: receipt {index} output"
        );
        assert_eq!(
            cpu.completion,
            ResidentDeviceCompletion::CpuReference,
            "{context}: CPU receipt {index}"
        );
        assert_eq!(
            metal.completion,
            ResidentDeviceCompletion::Metal,
            "{context}: Metal receipt {index}"
        );
    }
}

fn assert_corrupt_result(
    parts: ResidentRowProgramResultParts,
    request: &ResidentRowProgramRequest,
    backend: BackendKind,
    message: &str,
) {
    let error =
        match ResidentRowProgramResult::from_untrusted_parts(parts).validate(request, backend) {
            Ok(_) => panic!("corrupt result unexpectedly validated: {message}"),
            Err(error) => error,
        };
    assert_exact_error(&error, ErrorCode::CorruptStorage, message, message);
}

fn assert_result_corruption_fences(
    raw: &ResidentRowProgramResult,
    request: &ResidentRowProgramRequest,
    backend: BackendKind,
) {
    let mut parts = raw.clone().into_untrusted_parts();
    parts.projected_columns.swap(0, 1);
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row projections are not in the selected register order",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    let rows = parts.source_positions.len();
    parts.projected_columns[0].column = ResidentRowColumn::Float {
        bits: vec![0; rows],
        validity: vec![1; rows],
    };
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row typed column has an invalid type, payload, or validity shape",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    let ResidentRowColumn::Integer { validity, .. } = &mut parts.projected_columns[0].column else {
        panic!("corruption fixture changed projection type");
    };
    validity[0] = 2;
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row typed column has an invalid type, payload, or validity shape",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts[0].completion = match backend {
        BackendKind::Cpu => ResidentDeviceCompletion::Metal,
        BackendKind::Metal => ResidentDeviceCompletion::CpuReference,
        BackendKind::Cuda => unreachable!(),
    };
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row receipt has invalid execution or completion provenance",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts[0].execution.low ^= 1;
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row receipt has invalid execution or completion provenance",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.source_positions.swap(0, 1);
    assert_corrupt_result(
        parts,
        request,
        backend,
        "unsorted resident row result does not preserve stable input order",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.manifest_fingerprint = ResidentRowManifestFingerprint([0xa5; 32]);
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row result belongs to a different execution, manifest, or graph image",
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.scratch_bytes = parts.scratch_bytes.saturating_add(1);
    assert_corrupt_result(
        parts,
        request,
        backend,
        "resident row result reports an invalid scratch shape",
    );
}

#[test]
fn cpu_integer_arithmetic_modulo_overflow_and_null_contract() -> Result<()> {
    let harness = Harness::new();
    let cpu = harness.cpu()?;

    let integer = integer_request(&harness, 1)?;
    assert_integer_success(&execute_validated(&cpu, &integer, BackendKind::Cpu)?);

    let modulo = modulo_boundary_request(&harness, 2)?;
    assert_modulo_boundaries(&execute_validated(&cpu, &modulo, BackendKind::Cpu)?);

    let nulls = null_arithmetic_request(&harness, 3)?;
    assert_null_arithmetic(&execute_validated(&cpu, &nulls, BackendKind::Cpu)?);

    for failure in runtime_failures(&harness)? {
        assert_failure_on(&cpu, &failure);
    }
    Ok(())
}

#[test]
fn cpu_mixed_integer_float_ieee_bits_are_exact() -> Result<()> {
    let harness = Harness::new();
    let cpu = harness.cpu()?;
    let request = mixed_float_request(&harness, 4)?;
    let result = execute_validated(&cpu, &request, BackendKind::Cpu)?;
    assert_mixed_float_success(&result);
    Ok(())
}

#[test]
fn cpu_hidden_sort_key_stability_null_placement_and_windows_are_exact() -> Result<()> {
    let harness = Harness::new();
    let cpu = harness.cpu()?;
    let cases = [
        (
            sort_request(&harness, 5, false, false, 0, usize::MAX, 8)?,
            vec![3, 4, 1, 2, 0, 5, 6, 7],
        ),
        (
            sort_request(&harness, 6, false, true, 0, usize::MAX, 8)?,
            vec![6, 7, 3, 4, 1, 2, 0, 5],
        ),
        (
            sort_request(&harness, 7, true, true, 0, usize::MAX, 8)?,
            vec![6, 7, 5, 0, 1, 2, 3, 4],
        ),
        (
            sort_request(&harness, 8, true, false, 0, usize::MAX, 8)?,
            vec![5, 0, 1, 2, 3, 4, 6, 7],
        ),
        (
            sort_request(&harness, 9, false, false, 2, 3, 3)?,
            vec![1, 2, 0],
        ),
        (
            sort_request(&harness, 10, false, false, 3, 0, 0)?,
            Vec::new(),
        ),
    ];
    for (request, expected) in cases {
        assert!(!request.final_registers.contains(&3));
        let result = execute_validated(&cpu, &request, BackendKind::Cpu)?;
        assert_sort_result(&result, &expected);
    }
    Ok(())
}

#[test]
fn request_validation_rejects_adversarial_register_and_source_shapes() -> Result<()> {
    let harness = Harness::new();
    for failure in invalid_requests(&harness)? {
        let error = match failure.request.validate() {
            Ok(()) => panic!("{} unexpectedly validated", failure.name),
            Err(error) => error,
        };
        assert_exact_error(&error, failure.code, failure.message, failure.name);
    }
    Ok(())
}

#[test]
fn cpu_result_validation_rejects_projection_receipt_position_and_scratch_corruption() -> Result<()>
{
    let harness = Harness::new();
    let cpu = harness.cpu()?;
    let request = integer_request(&harness, 11)?;
    let raw = execute_raw(&cpu, &request)?;
    raw.clone().validate(&request, BackendKind::Cpu)?;
    assert_result_corruption_fences(&raw, &request, BackendKind::Cpu);
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
fn real_metal_matches_cpu_for_every_successful_adversarial_program() -> Result<()> {
    let _guard = metal_test_guard();
    let harness = Harness::new();
    let cpu = harness.cpu()?;
    let metal = harness.metal()?;

    for (name, request) in successful_requests(&harness)? {
        let cpu_raw = execute_raw(&cpu, &request)?;
        let metal_raw = execute_raw(&metal, &request)?;
        let cpu_result = cpu_raw.clone().validate(&request, BackendKind::Cpu)?;
        let metal_result = metal_raw.clone().validate(&request, BackendKind::Metal)?;
        assert_differential_equal(name, &cpu_result, &metal_result);
        if name == "integer arithmetic" {
            assert_result_corruption_fences(&cpu_raw, &request, BackendKind::Cpu);
            assert_result_corruption_fences(&metal_raw, &request, BackendKind::Metal);
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_matches_cpu_error_codes_and_messages_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let harness = Harness::new();
    let cpu = harness.cpu()?;
    let metal = harness.metal()?;

    for failure in invalid_requests(&harness)?
        .into_iter()
        .chain(runtime_failures(&harness)?)
    {
        let cpu_error = assert_failure_on(&cpu, &failure);
        let metal_error = assert_failure_on(&metal, &failure);
        assert_eq!(
            metal_error.code, cpu_error.code,
            "{}: CPU/Metal error code mismatch",
            failure.name
        );
        assert_eq!(
            metal_error.message, cpu_error.message,
            "{}: CPU/Metal error message mismatch",
            failure.name
        );
    }
    Ok(())
}
