// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Mutex, MutexGuard, OnceLock};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentDeviceCompletion,
        ResidentExecutionId, ResidentExecutionObligation, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierExpression as Expr, ResidentQuantifierFunction,
        ResidentQuantifierGeneration, ResidentQuantifierKind, ResidentQuantifierOutput,
        ResidentQuantifierProgram, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentQuantifierProjection, ResidentQuantifierSlot,
        ResidentQuantifierSource, ResidentQuantifierStage, ResidentQuantifierUnary,
        ResidentQuantifierValue as Value, ResidentRangeProgramRequest,
    },
    graph::{GraphStore, IndexCatalog, TemporalStore},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(0x5155_414e_5449_4649_4552));
const MEMORY_LIMIT: usize = 256 * 1024 * 1024;
const RESERVED: usize = 8 * 1024 * 1024;

fn metal_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Fixture {
    cpu: CpuBackend,
    backend: MetalBackend,
    generation: ResidentQuantifierGeneration,
}

impl Fixture {
    fn new() -> Result<Self> {
        let graph = GraphStore::default();
        let bookmark = Bookmark {
            term: 17,
            index: graph.revision(),
        };
        let generation = ResidentQuantifierGeneration {
            project: PROJECT,
            bookmark,
            graph_revision: graph.revision(),
            layout_version: graph.layout_version(),
            catalog_generation: graph.catalog().optimizer_generation(),
        };
        let mut cpu = CpuBackend::new(MEMORY_LIMIT, RESERVED);
        cpu.admit_project(ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?)?;
        let mut backend = MetalBackend::new(0, MEMORY_LIMIT, RESERVED)?;
        backend.admit_project(ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?)?;
        Ok(Self {
            cpu,
            backend,
            generation,
        })
    }

    fn request(
        &self,
        low: u64,
        seed: u64,
        slot_count: u16,
        stages: Vec<ResidentQuantifierStage>,
        outputs: Vec<(&str, u16)>,
    ) -> Result<ResidentQuantifierProgramRequest> {
        ResidentQuantifierProgramRequest::build(
            self.generation,
            ResidentExecutionId {
                high: 0x4d45_5441_4c51_5541,
                low,
            },
            ResidentQuantifierProgram {
                slot_count,
                stages,
                outputs: outputs
                    .into_iter()
                    .map(|(name, source)| ResidentQuantifierOutput {
                        name: name.to_owned(),
                        source: ResidentQuantifierSlot(source),
                    })
                    .collect(),
            },
            64,
            64,
            seed,
        )
    }

    fn execute(
        &self,
        request: &ResidentQuantifierProgramRequest,
    ) -> Result<ResidentQuantifierProgramResult> {
        self.backend
            .execute_quantifier_program(request, &CancellationToken::new())
    }
}

fn integer(value: i64) -> Expr {
    Expr::Literal(Value::Integer(value))
}

fn boolean(value: bool) -> Expr {
    Expr::Literal(Value::Boolean(value))
}

fn float(bits: u64) -> Expr {
    Expr::Literal(Value::Float(bits))
}

fn slot(index: u16) -> Expr {
    Expr::Slot(ResidentQuantifierSlot(index))
}

fn list(values: impl IntoIterator<Item = Expr>) -> Expr {
    Expr::List(values.into_iter().collect())
}

fn binary(left: Expr, operation: ResidentQuantifierBinary, right: Expr) -> Expr {
    Expr::Binary {
        left: Box::new(left),
        operation,
        right: Box::new(right),
    }
}

fn negative(operand: Expr) -> Expr {
    Expr::Unary {
        operation: ResidentQuantifierUnary::Negative,
        operand: Box::new(operand),
    }
}

fn project(output: u16, expression: Expr) -> ResidentQuantifierProjection {
    ResidentQuantifierProjection {
        output: ResidentQuantifierSlot(output),
        expression,
    }
}

fn range_source_request(
    fixture: &Fixture,
    low: u64,
    start: i64,
    end: i64,
    step: i64,
    integer_operands: [bool; 3],
) -> Result<ResidentQuantifierProgramRequest> {
    const ROWS: usize = 8;
    ResidentQuantifierProgramRequest::build_with_source(
        fixture.generation,
        ResidentExecutionId {
            high: 0x5241_4e47_455f_5155,
            low,
        },
        ResidentQuantifierSource::Range {
            request: ResidentRangeProgramRequest {
                start,
                end,
                step,
                integer_operands,
                max_values: ROWS,
            },
            output: ResidentQuantifierSlot(0),
            obligation: ResidentExecutionObligation {
                id: 0x5155_5241_4e47_0001,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::Expression(u16::MAX - 2),
            },
        },
        ResidentQuantifierProgram {
            slot_count: 2,
            stages: vec![ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![project(1, slot(0))],
            }],
            outputs: vec![ResidentQuantifierOutput {
                name: "value".to_owned(),
                source: ResidentQuantifierSlot(1),
            }],
        },
        ROWS,
        ROWS,
        0x5241_4e47_455f_5345 ^ low,
    )
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_boolean_binary_matches_cpu_three_valued_truth_tables() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let values = [Value::Boolean(true), Value::Boolean(false), Value::Null];
    let mut low = 0x200_u64;
    for operation in [
        ResidentQuantifierBinary::And,
        ResidentQuantifierBinary::Or,
        ResidentQuantifierBinary::Xor,
    ] {
        for left in &values {
            for right in &values {
                low += 1;
                let request = fixture.request(
                    low,
                    0x424f_4f4c_4541_4e00 ^ low,
                    1,
                    vec![ResidentQuantifierStage::Project {
                        keep_scope: false,
                        bindings: vec![project(
                            0,
                            binary(
                                Expr::Literal(left.clone()),
                                operation,
                                Expr::Literal(right.clone()),
                            ),
                        )],
                    }],
                    vec![("result", 0)],
                )?;
                let cpu = fixture
                    .cpu
                    .execute_quantifier_program(&request, &CancellationToken::new())?
                    .validate(&request, BackendKind::Cpu)?;
                let metal = fixture
                    .backend
                    .execute_quantifier_program(&request, &CancellationToken::new())?
                    .validate(&request, BackendKind::Metal)?;
                assert_eq!(
                    metal.rows(),
                    cpu.rows(),
                    "{operation:?} changed when evaluating {left:?} and {right:?}"
                );
            }
        }
    }

    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_range_source_matches_cpu_for_descending_empty_and_type_error() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    for (low, start, end, step, expected) in [
        (0x101, 5, 1, -2, vec![5, 3, 1]),
        (0x102, 3, 1, 1, Vec::new()),
    ] {
        let request = range_source_request(&fixture, low, start, end, step, [true; 3])?;
        let cpu = fixture
            .cpu
            .execute_quantifier_program(&request, &CancellationToken::new())?
            .validate(&request, BackendKind::Cpu)?;
        let metal = fixture
            .backend
            .execute_quantifier_program(&request, &CancellationToken::new())?
            .validate(&request, BackendKind::Metal)?;
        let expected = expected
            .into_iter()
            .map(|value| vec![Value::Integer(value)])
            .collect::<Vec<_>>();
        assert_eq!(cpu.rows(), expected);
        assert_eq!(metal.rows(), cpu.rows());
        let cardinalities = metal
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>();
        let rows = expected.len() as u64;
        assert_eq!(cardinalities, vec![(1, rows), (rows, rows), (rows, rows)]);
        assert!(
            metal
                .receipts()
                .iter()
                .all(|receipt| { receipt.completion == ResidentDeviceCompletion::Metal })
        );
    }

    let invalid = range_source_request(&fixture, 0x103, 0, 3, 1, [false, true, true])?;
    let cpu_error = fixture
        .cpu
        .execute_quantifier_program(&invalid, &CancellationToken::new())
        .unwrap_err();
    let metal_error = fixture
        .backend
        .execute_quantifier_program(&invalid, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(cpu_error.code, ErrorCode::QueryType);
    assert_eq!(metal_error.code, cpu_error.code);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_executes_nested_quantifiers_with_device_receipts() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let inner = Expr::Predicate {
        kind: ResidentQuantifierKind::All,
        variable: ResidentQuantifierSlot(2),
        list: Box::new(list([
            slot(1),
            binary(slot(1), ResidentQuantifierBinary::Add, integer(1)),
        ])),
        predicate: Box::new(binary(
            slot(2),
            ResidentQuantifierBinary::GreaterOrEqual,
            slot(1),
        )),
    };
    let expression = Expr::Predicate {
        kind: ResidentQuantifierKind::Any,
        variable: ResidentQuantifierSlot(1),
        list: Box::new(list([integer(1), integer(2), integer(3)])),
        predicate: Box::new(inner),
    };
    let request = fixture.request(
        1,
        7,
        3,
        vec![ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![project(0, expression)],
        }],
        vec![("result", 0)],
    )?;
    let result = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(result.rows(), &[vec![Value::Boolean(true)]]);
    assert_eq!(result.receipts().len(), 2);
    assert!(
        result
            .receipts()
            .iter()
            .all(|receipt| receipt.completion as u8 == 2)
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_executes_unwind_filter_map_and_list_comprehension() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let stages = vec![
        ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![project(0, list([integer(1), integer(2), integer(3)]))],
        },
        ResidentQuantifierStage::Unwind {
            expression: slot(0),
            output: ResidentQuantifierSlot(1),
        },
        ResidentQuantifierStage::Filter {
            predicate: binary(slot(1), ResidentQuantifierBinary::Greater, integer(1)),
        },
        ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![project(
                2,
                Expr::Map(vec![
                    (
                        "doubled".to_owned(),
                        binary(slot(1), ResidentQuantifierBinary::Multiply, integer(2)),
                    ),
                    (
                        "tail".to_owned(),
                        Expr::ListComprehension {
                            variable: ResidentQuantifierSlot(3),
                            list: Box::new(list([integer(1), slot(1), integer(4)])),
                            predicate: Some(Box::new(binary(
                                slot(3),
                                ResidentQuantifierBinary::Greater,
                                integer(1),
                            ))),
                            projection: Some(Box::new(slot(3))),
                        },
                    ),
                ]),
            )],
        },
    ];
    let request = fixture.request(2, 11, 4, stages, vec![("value", 2)])?;
    let result = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(
        result.rows(),
        &[
            vec![Value::Map(vec![
                ("doubled".to_owned(), Value::Integer(4)),
                (
                    "tail".to_owned(),
                    Value::List(vec![Value::Integer(2), Value::Integer(4)]),
                ),
            ])],
            vec![Value::Map(vec![
                ("doubled".to_owned(), Value::Integer(6)),
                (
                    "tail".to_owned(),
                    Value::List(vec![Value::Integer(3), Value::Integer(4)]),
                ),
            ])],
        ]
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_groups_and_counts_inside_the_same_command() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let request = fixture.request(
        3,
        13,
        4,
        vec![
            ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![project(0, list([integer(1), integer(1), integer(2)]))],
            },
            ResidentQuantifierStage::Unwind {
                expression: slot(0),
                output: ResidentQuantifierSlot(1),
            },
            ResidentQuantifierStage::GroupCount {
                groups: vec![project(2, slot(1))],
                count_outputs: vec![ResidentQuantifierSlot(3)],
            },
        ],
        vec![("key", 2), ("count", 3)],
    )?;
    let result = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(
        result.rows(),
        &[
            vec![Value::Integer(1), Value::Integer(2)],
            vec![Value::Integer(2), Value::Integer(1)],
        ]
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_rand_is_repeatable_and_schedule_independent() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let request = fixture.request(
        4,
        0x1234_5678,
        2,
        vec![ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![
                project(
                    0,
                    Expr::Function {
                        function: ResidentQuantifierFunction::Rand,
                        arguments: Vec::new(),
                    },
                ),
                project(
                    1,
                    Expr::Function {
                        function: ResidentQuantifierFunction::Rand,
                        arguments: Vec::new(),
                    },
                ),
            ],
        }],
        vec![("a", 0), ("b", 1)],
    )?;
    let first = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    let second = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(first.rows(), second.rows());
    assert_ne!(first.rows()[0][0], first.rows()[0][1]);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn result_validation_rejects_missing_and_forged_metal_receipts() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let request = fixture.request(
        5,
        19,
        2,
        vec![ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![project(
                0,
                Expr::Predicate {
                    kind: ResidentQuantifierKind::None,
                    variable: ResidentQuantifierSlot(1),
                    list: Box::new(list([integer(1), integer(2)])),
                    predicate: Box::new(boolean(false)),
                },
            )],
        }],
        vec![("result", 0)],
    )?;
    let parts = fixture.execute(&request)?.into_untrusted_parts();

    let mut missing = parts.clone();
    missing.receipts.pop();
    assert!(
        ResidentQuantifierProgramResult::from_untrusted_parts(missing)
            .validate(&request, BackendKind::Metal)
            .is_err()
    );

    let mut forged = parts;
    forged.receipts[0].execution.low ^= 1;
    assert!(
        ResidentQuantifierProgramResult::from_untrusted_parts(forged)
            .validate(&request, BackendKind::Metal)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_unary_minus_preserves_float_sign_and_payload_bits() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let positive_nan = 0x7ff8_0000_0000_0042_u64;
    let negative_nan = positive_nan ^ (1_u64 << 63);
    let inputs = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        positive_nan,
        negative_nan,
        2.1_f64.to_bits(),
        (-2.1_f64).to_bits(),
    ];
    let names = [
        "positive_zero",
        "negative_zero",
        "positive_infinity",
        "negative_infinity",
        "positive_nan",
        "negative_nan",
        "positive_finite",
        "negative_finite",
    ];
    let bindings = inputs
        .iter()
        .enumerate()
        .map(|(index, bits)| project(index as u16, negative(float(*bits))))
        .collect();
    let outputs = names
        .iter()
        .enumerate()
        .map(|(index, name)| (*name, index as u16))
        .collect();
    let request = fixture.request(
        6,
        23,
        inputs.len() as u16,
        vec![ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings,
        }],
        outputs,
    )?;
    let result = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(
        result.rows(),
        &[inputs
            .iter()
            .map(|bits| Value::Float(*bits ^ (1_u64 << 63)))
            .collect::<Vec<_>>()]
    );
    assert_eq!(result.receipts().len(), 2);
    assert!(result.receipts().iter().all(|receipt| {
        receipt.execution == request.execution && receipt.completion as u8 == 2
    }));
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_nested_float_unary_minus_uses_checked_arena_and_metal_provenance() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let input = 0x7ff0_0000_0000_0042_u64;
    let mut expression = float(input);
    for _ in 0..31 {
        expression = negative(expression);
    }
    let request = fixture.request(
        7,
        29,
        1,
        vec![ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings: vec![project(0, expression)],
        }],
        vec![("value", 0)],
    )?;
    let result = fixture
        .execute(&request)?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(result.rows(), &[vec![Value::Float(input ^ (1_u64 << 63))]]);
    assert_eq!(result.receipts().len(), request.obligations().len());
    assert!(result.receipts().iter().all(|receipt| {
        receipt.execution == request.execution && receipt.completion as u8 == 2
    }));
    Ok(())
}
