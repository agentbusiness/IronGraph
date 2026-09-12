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

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentProjectImage, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentTemporalArithmeticOperand,
        ResidentTemporalArithmeticOperation, ResidentTemporalArithmeticRequest,
        ResidentTemporalArithmeticResult, ResidentTemporalArithmeticResultParts,
        ResidentTemporalArithmeticSource, ResidentTemporalArithmeticStatus,
        ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult, ResidentTemporalValue,
        ResidentTemporalValueFunction, ResidentTemporalValueInput,
        ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 16;
const FEATURE: &str = "features/expressions/temporal/Temporal8.feature";
const TEMPORAL_MAP_WIRE_VERSION: u8 = 1;

const DATE_X: &str = "date({year: 1984, month: 10, day: 11})";
const LOCAL_TIME_X: &str = "localtime({hour: 12, minute: 31, second: 14, nanosecond: 1})";
const TIME_X: &str = "time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'})";
const LOCAL_DATETIME_X: &str = "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1})";
const DATETIME_X: &str = "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'})";

const DURATION_A: &str = "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})";
const DURATION_A_N1: &str = "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})";
const DURATION_B: &str = "duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})";
const DURATION_C: &str = "duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceShape {
    Map,
    Null,
}

#[derive(Clone, Copy, Debug)]
struct SourceSpec {
    function: ResidentTemporalValueFunction,
    expression: &'static str,
    shape: SourceShape,
}

const fn source(function: ResidentTemporalValueFunction, expression: &'static str) -> SourceSpec {
    SourceSpec {
        function,
        expression,
        shape: SourceShape::Map,
    }
}

const fn null_source(
    function: ResidentTemporalValueFunction,
    expression: &'static str,
) -> SourceSpec {
    SourceSpec {
        function,
        expression,
        shape: SourceShape::Null,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArithmeticOperation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl ArithmeticOperation {
    const fn token(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
        }
    }

    #[allow(dead_code)]
    const fn debug_name(self) -> &'static str {
        match self {
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Multiply => "Multiply",
            Self::Divide => "Divide",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ArithmeticRight {
    Register(u16),
    Integer(i64),
    Float(&'static str),
    Null,
}

impl ArithmeticRight {
    fn expression(self) -> String {
        match self {
            Self::Register(_) => "d".to_owned(),
            Self::Integer(value) => value.to_string(),
            Self::Float(value) => value.to_owned(),
            Self::Null => "null".to_owned(),
        }
    }

    #[allow(dead_code)]
    fn debug_name(self) -> String {
        match self {
            Self::Register(register) => format!("Register({register})"),
            Self::Integer(value) => format!("Integer({value})"),
            Self::Float(value) => format!("Float({value})"),
            Self::Null => "Null".to_owned(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ProgramKind {
    Pair {
        output_function: ResidentTemporalValueFunction,
    },
    Scale {
        multiply: ArithmeticRight,
        divide: ArithmeticRight,
    },
    Single {
        output_function: ResidentTemporalValueFunction,
        operation: ArithmeticOperation,
        right: ArithmeticRight,
        alias: &'static str,
    },
}

#[derive(Clone, Copy, Debug)]
struct InstructionSpec {
    #[allow(dead_code)]
    function: ResidentTemporalValueFunction,
    operation: ArithmeticOperation,
    left: u16,
    right: ArithmeticRight,
    alias: &'static str,
}

#[derive(Clone, Copy, Debug)]
enum ExpectedOutcome {
    Success(&'static [&'static str]),
    Error(ErrorCode),
}

#[derive(Clone, Copy, Debug)]
struct ArithmeticCase {
    report_id: Option<u16>,
    scenario: Option<u8>,
    name: &'static str,
    source0: SourceSpec,
    source1: Option<SourceSpec>,
    program: ProgramKind,
    expected: ExpectedOutcome,
}

impl ArithmeticCase {
    fn label(self) -> String {
        match (self.report_id, self.scenario) {
            (Some(id), Some(scenario)) => {
                format!("TCK {id} {FEATURE} [{scenario}] {}", self.name)
            }
            _ => format!("adversarial {}", self.name),
        }
    }

    fn instructions(self) -> Vec<InstructionSpec> {
        match self.program {
            ProgramKind::Pair { output_function } => vec![
                InstructionSpec {
                    function: output_function,
                    operation: ArithmeticOperation::Add,
                    left: 0,
                    right: ArithmeticRight::Register(1),
                    alias: "sum",
                },
                InstructionSpec {
                    function: output_function,
                    operation: ArithmeticOperation::Subtract,
                    left: 0,
                    right: ArithmeticRight::Register(1),
                    alias: "diff",
                },
            ],
            ProgramKind::Scale { multiply, divide } => vec![
                InstructionSpec {
                    function: ResidentTemporalValueFunction::Duration,
                    operation: ArithmeticOperation::Multiply,
                    left: 0,
                    right: multiply,
                    alias: "prod",
                },
                InstructionSpec {
                    function: ResidentTemporalValueFunction::Duration,
                    operation: ArithmeticOperation::Divide,
                    left: 0,
                    right: divide,
                    alias: "div",
                },
            ],
            ProgramKind::Single {
                output_function,
                operation,
                right,
                alias,
            } => vec![InstructionSpec {
                function: output_function,
                operation,
                left: 0,
                right,
                alias,
            }],
        }
    }

    fn native_query(self) -> String {
        match self.program {
            ProgramKind::Pair { .. } => {
                let right = self
                    .source1
                    .expect("pair arithmetic always has a second source");
                format!(
                    "WITH {} AS x, {} AS d\nRETURN x + d AS sum, x - d AS diff",
                    self.source0.expression, right.expression
                )
            }
            ProgramKind::Scale { multiply, divide } => format!(
                "WITH {} AS d\nRETURN d * {} AS prod, d / {} AS div",
                self.source0.expression,
                multiply.expression(),
                divide.expression()
            ),
            ProgramKind::Single {
                operation,
                right,
                alias,
                ..
            } => {
                let left_name = if self.source1.is_some() { "x" } else { "d" };
                let source_clause = if let Some(source1) = self.source1 {
                    format!(
                        "WITH {} AS x, {} AS d",
                        self.source0.expression, source1.expression
                    )
                } else {
                    format!("WITH {} AS d", self.source0.expression)
                };
                format!(
                    "{source_clause}\nRETURN {left_name} {} {} AS {alias}",
                    operation.token(),
                    right.expression()
                )
            }
        }
    }

    fn official_setup(self) -> Option<String> {
        let scenario = self.scenario?;
        match scenario {
            1..=5 => {
                let duration = self.source1?;
                Some(format!(
                    "CREATE (:Duration {{dur: {}}})",
                    duration.expression
                ))
            }
            6 => {
                let right = self.source1?;
                Some(format!(
                    "CREATE (:Duration1 {{date: {}}})\nCREATE (:Duration2 {{date: {}}})",
                    self.source0.expression, right.expression
                ))
            }
            7 => Some(format!(
                "CREATE (:Duration {{date: {}}})",
                self.source0.expression
            )),
            _ => None,
        }
    }

    fn official_query(self) -> Option<String> {
        match self.scenario? {
            1 => Some(String::from(
                "WITH date({year: 1984, month: 10, day: 11}) AS x\nMATCH (d:Duration)\nRETURN x + d.dur AS sum, x - d.dur AS diff",
            )),
            2 => Some(String::from(
                "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\nMATCH (d:Duration)\nRETURN x + d.dur AS sum, x - d.dur AS diff",
            )),
            3 => Some(String::from(
                "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\nMATCH (d:Duration)\nRETURN x + d.dur AS sum, x - d.dur AS diff",
            )),
            4 => Some(String::from(
                "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\nMATCH (d:Duration)\nRETURN x + d.dur AS sum, x - d.dur AS diff",
            )),
            5 => Some(String::from(
                "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\nMATCH (d:Duration)\nRETURN x + d.dur AS sum, x - d.dur AS diff",
            )),
            6 => Some(String::from(
                "MATCH (dur:Duration1), (dur2: Duration2)\nRETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff",
            )),
            7 => {
                let ProgramKind::Scale { multiply, divide } = self.program else {
                    return None;
                };
                Some(format!(
                    "MATCH (d:Duration)\nRETURN d.date * {} AS prod, d.date / {} AS div",
                    multiply.expression(),
                    divide.expression()
                ))
            }
            _ => None,
        }
    }

    fn execution_query(self) -> String {
        self.official_query().unwrap_or_else(|| self.native_query())
    }
}

const fn official_pair(
    report_id: u16,
    scenario: u8,
    name: &'static str,
    left_function: ResidentTemporalValueFunction,
    left_expression: &'static str,
    right_expression: &'static str,
    expected: &'static [&'static str],
) -> ArithmeticCase {
    ArithmeticCase {
        report_id: Some(report_id),
        scenario: Some(scenario),
        name,
        source0: source(left_function, left_expression),
        source1: Some(source(
            ResidentTemporalValueFunction::Duration,
            right_expression,
        )),
        program: ProgramKind::Pair {
            output_function: left_function,
        },
        expected: ExpectedOutcome::Success(expected),
    }
}

const fn official_scale(
    report_id: u16,
    name: &'static str,
    multiply: ArithmeticRight,
    divide: ArithmeticRight,
    expected: &'static [&'static str],
) -> ArithmeticCase {
    ArithmeticCase {
        report_id: Some(report_id),
        scenario: Some(7),
        name,
        source0: source(ResidentTemporalValueFunction::Duration, DURATION_A_N1),
        source1: None,
        program: ProgramKind::Scale { multiply, divide },
        expected: ExpectedOutcome::Success(expected),
    }
}

const OFFICIAL_CASES: [ArithmeticCase; 27] = [
    official_pair(
        3472,
        1,
        "date plus/minus first duration",
        ResidentTemporalValueFunction::Date,
        DATE_X,
        DURATION_A,
        &["date('1997-03-25')", "date('1972-04-27')"],
    ),
    official_pair(
        3473,
        1,
        "date plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::Date,
        DATE_X,
        DURATION_B,
        &["date('1984-10-28')", "date('1984-09-25')"],
    ),
    official_pair(
        3474,
        1,
        "date plus/minus fractional duration",
        ResidentTemporalValueFunction::Date,
        DATE_X,
        DURATION_C,
        &["date('1997-10-11')", "date('1971-10-12')"],
    ),
    official_pair(
        3475,
        2,
        "local time plus/minus first duration",
        ResidentTemporalValueFunction::LocalTime,
        LOCAL_TIME_X,
        DURATION_A,
        &[
            "localtime('04:44:24.000000003')",
            "localtime('20:18:03.999999999')",
        ],
    ),
    official_pair(
        3476,
        2,
        "local time plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::LocalTime,
        LOCAL_TIME_X,
        DURATION_B,
        &[
            "localtime('04:20:24.000000001')",
            "localtime('20:42:04.000000001')",
        ],
    ),
    official_pair(
        3477,
        2,
        "local time plus/minus fractional duration",
        ResidentTemporalValueFunction::LocalTime,
        LOCAL_TIME_X,
        DURATION_C,
        &[
            "localtime('22:29:27.500000004')",
            "localtime('02:33:00.499999998')",
        ],
    ),
    official_pair(
        3478,
        3,
        "time plus/minus first duration",
        ResidentTemporalValueFunction::Time,
        TIME_X,
        DURATION_A,
        &[
            "time('04:44:24.000000003+01:00')",
            "time('20:18:03.999999999+01:00')",
        ],
    ),
    official_pair(
        3479,
        3,
        "time plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::Time,
        TIME_X,
        DURATION_B,
        &[
            "time('04:20:24.000000001+01:00')",
            "time('20:42:04.000000001+01:00')",
        ],
    ),
    official_pair(
        3480,
        3,
        "time plus/minus fractional duration",
        ResidentTemporalValueFunction::Time,
        TIME_X,
        DURATION_C,
        &[
            "time('22:29:27.500000004+01:00')",
            "time('02:33:00.499999998+01:00')",
        ],
    ),
    official_pair(
        3481,
        4,
        "local datetime plus/minus first duration",
        ResidentTemporalValueFunction::LocalDateTime,
        LOCAL_DATETIME_X,
        DURATION_A,
        &[
            "localdatetime('1997-03-26T04:44:24.000000003')",
            "localdatetime('1972-04-26T20:18:03.999999999')",
        ],
    ),
    official_pair(
        3482,
        4,
        "local datetime plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::LocalDateTime,
        LOCAL_DATETIME_X,
        DURATION_B,
        &[
            "localdatetime('1984-10-29T04:20:24.000000001')",
            "localdatetime('1984-09-24T20:42:04.000000001')",
        ],
    ),
    official_pair(
        3483,
        4,
        "local datetime plus/minus fractional duration",
        ResidentTemporalValueFunction::LocalDateTime,
        LOCAL_DATETIME_X,
        DURATION_C,
        &[
            "localdatetime('1997-10-11T22:29:27.500000004')",
            "localdatetime('1971-10-12T02:33:00.499999998')",
        ],
    ),
    official_pair(
        3484,
        5,
        "datetime plus/minus first duration",
        ResidentTemporalValueFunction::DateTime,
        DATETIME_X,
        DURATION_A,
        &[
            "datetime('1997-03-26T04:44:24.000000003+01:00')",
            "datetime('1972-04-26T20:18:03.999999999+01:00')",
        ],
    ),
    official_pair(
        3485,
        5,
        "datetime plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::DateTime,
        DATETIME_X,
        DURATION_B,
        &[
            "datetime('1984-10-29T04:20:24.000000001+01:00')",
            "datetime('1984-09-24T20:42:04.000000001+01:00')",
        ],
    ),
    official_pair(
        3486,
        5,
        "datetime plus/minus fractional duration",
        ResidentTemporalValueFunction::DateTime,
        DATETIME_X,
        DURATION_C,
        &[
            "datetime('1997-10-11T22:29:27.500000004+01:00')",
            "datetime('1971-10-12T02:33:00.499999998+01:00')",
        ],
    ),
    official_pair(
        3487,
        6,
        "equal first durations",
        ResidentTemporalValueFunction::Duration,
        DURATION_A_N1,
        DURATION_A_N1,
        &[
            "duration('P24Y10M28DT32H26M20.000000002S')",
            "duration('PT0S')",
        ],
    ),
    official_pair(
        3488,
        6,
        "first plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_A_N1,
        DURATION_B,
        &[
            "duration('P12Y6MT32H2M20.000000001S')",
            "duration('P12Y4M28DT24M0.000000001S')",
        ],
    ),
    official_pair(
        3489,
        6,
        "first plus/minus fractional duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_A_N1,
        DURATION_C,
        &[
            "duration('P25Y4M43DT50H11M23.500000004S')",
            "duration('P-6M-15DT-17H-45M-3.500000002S')",
        ],
    ),
    official_pair(
        3490,
        6,
        "mixed-sign plus/minus first duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_B,
        DURATION_A_N1,
        &[
            "duration('P12Y6MT32H2M20.000000001S')",
            "duration('P-12Y-4M-28DT-24M-0.000000001S')",
        ],
    ),
    official_pair(
        3491,
        6,
        "equal mixed-sign durations",
        ResidentTemporalValueFunction::Duration,
        DURATION_B,
        DURATION_B,
        &["duration('P2M-28DT31H38M20S')", "duration('PT0S')"],
    ),
    official_pair(
        3492,
        6,
        "mixed-sign plus/minus fractional duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_B,
        DURATION_C,
        &[
            "duration('P13Y15DT49H47M23.500000003S')",
            "duration('P-12Y-10M-43DT-18H-9M-3.500000003S')",
        ],
    ),
    official_pair(
        3493,
        6,
        "fractional plus/minus first duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_C,
        DURATION_A_N1,
        &[
            "duration('P25Y4M43DT50H11M23.500000004S')",
            "duration('P6M15DT17H45M3.500000002S')",
        ],
    ),
    official_pair(
        3494,
        6,
        "fractional plus/minus mixed-sign duration",
        ResidentTemporalValueFunction::Duration,
        DURATION_C,
        DURATION_B,
        &[
            "duration('P13Y15DT49H47M23.500000003S')",
            "duration('P12Y10M43DT18H9M3.500000003S')",
        ],
    ),
    official_pair(
        3495,
        6,
        "equal fractional durations",
        ResidentTemporalValueFunction::Duration,
        DURATION_C,
        DURATION_C,
        &[
            "duration('P25Y10M58DT67H56M27.000000006S')",
            "duration('PT0S')",
        ],
    ),
    official_scale(
        3496,
        "identity multiply/divide",
        ArithmeticRight::Integer(1),
        ArithmeticRight::Integer(1),
        &[
            "duration('P12Y5M14DT16H13M10.000000001S')",
            "duration('P12Y5M14DT16H13M10.000000001S')",
        ],
    ),
    official_scale(
        3497,
        "integer multiply/divide",
        ArithmeticRight::Integer(2),
        ArithmeticRight::Integer(2),
        &[
            "duration('P24Y10M28DT32H26M20.000000002S')",
            "duration('P6Y2M22DT13H21M8S')",
        ],
    ),
    official_scale(
        3498,
        "fractional multiply/divide",
        ArithmeticRight::Float("0.5"),
        ArithmeticRight::Float("0.5"),
        &[
            "duration('P6Y2M22DT13H21M8S')",
            "duration('P24Y10M28DT32H26M20.000000002S')",
        ],
    ),
];

const ADVERSARIAL_CASES: [ArithmeticCase; 11] = [
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "NULL temporal propagates through addition and subtraction",
        source0: null_source(ResidentTemporalValueFunction::Date, "date(null)"),
        source1: Some(source(
            ResidentTemporalValueFunction::Duration,
            "duration({days: 1})",
        )),
        program: ProgramKind::Pair {
            output_function: ResidentTemporalValueFunction::Date,
        },
        expected: ExpectedOutcome::Success(&["null", "null"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "NULL duration propagates through temporal addition and subtraction",
        source0: source(
            ResidentTemporalValueFunction::Date,
            "date({year: 2024, month: 1, day: 31})",
        ),
        source1: Some(null_source(
            ResidentTemporalValueFunction::Duration,
            "duration(null)",
        )),
        program: ProgramKind::Pair {
            output_function: ResidentTemporalValueFunction::Date,
        },
        expected: ExpectedOutcome::Success(&["null", "null"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "leap-year month-end clamp",
        source0: source(
            ResidentTemporalValueFunction::Date,
            "date({year: 2024, month: 1, day: 31})",
        ),
        source1: Some(source(
            ResidentTemporalValueFunction::Duration,
            "duration({months: 1})",
        )),
        program: ProgramKind::Pair {
            output_function: ResidentTemporalValueFunction::Date,
        },
        expected: ExpectedOutcome::Success(&["date('2024-02-29')", "date('2023-12-31')"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "common-year month-end clamp",
        source0: source(
            ResidentTemporalValueFunction::Date,
            "date({year: 2023, month: 1, day: 31})",
        ),
        source1: Some(source(
            ResidentTemporalValueFunction::Duration,
            "duration({months: 1})",
        )),
        program: ProgramKind::Pair {
            output_function: ResidentTemporalValueFunction::Date,
        },
        expected: ExpectedOutcome::Success(&["date('2023-02-28')", "date('2022-12-31')"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "negative fractional duration normalization",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({seconds: -0.5})",
        ),
        source1: Some(source(
            ResidentTemporalValueFunction::Duration,
            "duration({nanoseconds: -500000000})",
        )),
        program: ProgramKind::Pair {
            output_function: ResidentTemporalValueFunction::Duration,
        },
        expected: ExpectedOutcome::Success(&["duration({seconds: -1})", "duration({seconds: 0})"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "zero multiplication and negative fractional division",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({seconds: 1})",
        ),
        source1: None,
        program: ProgramKind::Scale {
            multiply: ArithmeticRight::Integer(0),
            divide: ArithmeticRight::Integer(-2),
        },
        expected: ExpectedOutcome::Success(&[
            "duration({seconds: 0})",
            "duration({seconds: -0.5})",
        ]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "NULL scale operands propagate",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({days: 1, nanoseconds: 1})",
        ),
        source1: None,
        program: ProgramKind::Scale {
            multiply: ArithmeticRight::Null,
            divide: ArithmeticRight::Null,
        },
        expected: ExpectedOutcome::Success(&["null", "null"]),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "integer division by zero is a device error",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({days: 1})",
        ),
        source1: None,
        program: ProgramKind::Single {
            output_function: ResidentTemporalValueFunction::Duration,
            operation: ArithmeticOperation::Divide,
            right: ArithmeticRight::Integer(0),
            alias: "value",
        },
        expected: ExpectedOutcome::Error(ErrorCode::QueryType),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "negative floating zero division is a device error",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({seconds: 1})",
        ),
        source1: None,
        program: ProgramKind::Single {
            output_function: ResidentTemporalValueFunction::Duration,
            operation: ArithmeticOperation::Divide,
            right: ArithmeticRight::Float("-0.0"),
            alias: "value",
        },
        expected: ExpectedOutcome::Error(ErrorCode::QueryType),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "duration minus temporal is a device type error",
        source0: source(
            ResidentTemporalValueFunction::Duration,
            "duration({days: 1})",
        ),
        source1: Some(source(
            ResidentTemporalValueFunction::Date,
            "date({year: 2024, month: 1, day: 1})",
        )),
        program: ProgramKind::Single {
            output_function: ResidentTemporalValueFunction::Duration,
            operation: ArithmeticOperation::Subtract,
            right: ArithmeticRight::Register(1),
            alias: "value",
        },
        expected: ExpectedOutcome::Error(ErrorCode::QueryType),
    },
    ArithmeticCase {
        report_id: None,
        scenario: None,
        name: "temporal multiplication is a device type error",
        source0: source(
            ResidentTemporalValueFunction::Date,
            "date({year: 2024, month: 1, day: 1})",
        ),
        source1: None,
        program: ProgramKind::Single {
            output_function: ResidentTemporalValueFunction::Date,
            operation: ArithmeticOperation::Multiply,
            right: ArithmeticRight::Integer(2),
            alias: "value",
        },
        expected: ExpectedOutcome::Error(ErrorCode::QueryType),
    },
];

fn all_cases() -> impl Iterator<Item = &'static ArithmeticCase> {
    OFFICIAL_CASES.iter().chain(ADVERSARIAL_CASES.iter())
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn new() -> Self {
        let graph = GraphStore::default();
        Self {
            bookmark: Bookmark {
                term: 41,
                index: graph.revision(),
            },
            graph,
        }
    }

    fn for_case(case: ArithmeticCase) -> Result<Self> {
        let mut fixture = Self::new();
        for label in [
            "Duration",
            "Duration1",
            "Duration2",
            "TemporalArithmeticDecoy",
        ] {
            fixture.graph.catalog_mut().intern_label(label)?;
        }
        for property in ["dur", "date", "irrelevant"] {
            fixture.graph.catalog_mut().intern_property(property)?;
        }
        if let Some(setup) = case.official_setup() {
            let output = QueryEngine.execute(&setup, &mut context(&fixture, None))?;
            if !output.temporal_mutations.is_empty() || output.graph_mutations.is_empty() {
                return Err(Error::internal(format!(
                    "{} exact TCK setup did not produce canonical graph mutations",
                    case.label()
                )));
            }
            for mutation in output.graph_mutations {
                fixture.graph.apply(mutation)?;
            }
        }

        let decoy_label = fixture
            .graph
            .catalog()
            .label("TemporalArithmeticDecoy")
            .ok_or_else(|| Error::internal("decoy label disappeared"))?;
        let dur = fixture
            .graph
            .catalog()
            .property("dur")
            .ok_or_else(|| Error::internal("dur property disappeared"))?;
        let date = fixture
            .graph
            .catalog()
            .property("date")
            .ok_or_else(|| Error::internal("date property disappeared"))?;
        let irrelevant = fixture
            .graph
            .catalog()
            .property("irrelevant")
            .ok_or_else(|| Error::internal("irrelevant property disappeared"))?;
        fixture.graph.insert_node(NodeInput {
            id: NodeId(10_000),
            layer: Layer::Observed,
            revision: fixture.graph.revision().saturating_add(1),
            labels: vec![decoy_label],
            properties: vec![
                (
                    dur,
                    ScalarValue::Duration {
                        months: 999,
                        days: 999,
                        seconds: 999,
                        nanos: 999,
                    },
                ),
                (
                    date,
                    ScalarValue::Duration {
                        months: -999,
                        days: -999,
                        seconds: -1_000,
                        nanos: 999_999_001,
                    },
                ),
                (irrelevant, ScalarValue::Integer(i64::MIN)),
            ],
        })?;
        fixture.bookmark = Bookmark {
            term: 41,
            index: fixture.graph.revision(),
        };
        Ok(fixture)
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
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(cpu)
    }

    fn strict_cpu_backend(&self) -> Result<ObservedTemporalBackend> {
        ObservedTemporalBackend::strict_cpu_reference(self.cpu()?)
    }

    fn cpu_masquerading_as_metal(&self) -> Result<ObservedTemporalBackend> {
        ObservedTemporalBackend::new(
            Box::new(self.cpu()?),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedTemporalBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedTemporalBackend::real_metal(metal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeTemporalReceipt {
    request: ResidentTemporalArithmeticRequest,
    result: ResidentTemporalArithmeticResultParts,
}

#[derive(Default)]
struct TemporalObservations {
    pins: AtomicUsize,
    temporal_arithmetic_calls: AtomicUsize,
    legacy_route_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    generation_mutation_attempts: AtomicUsize,
    receipts: Mutex<Vec<NativeTemporalReceipt>>,
}

/// Fail-closed observer around the only accepted execution boundary.
///
/// Before planning, the CPU reference advertises Metal so the generic CPU evaluator cannot be
/// selected. Pinning proves the real inner backend and freezes its bookmark/revision. The pinned
/// wrapper continues to advertise the native class while receipts retain honest CPU or Metal
/// completion. Every route except one temporal SSA dispatch is rejected.
struct ObservedTemporalBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<TemporalObservations>,
}

impl ObservedTemporalBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "temporal arithmetic test did not construct a real Metal backend",
            ));
        }
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    fn new(
        inner: Box<dyn ExecutionBackend>,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("temporal arithmetic backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal arithmetic backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(TemporalObservations::default()),
        })
    }

    fn observations(&self) -> Arc<TemporalObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict temporal arithmetic test rejected `{route}` execution"),
        ))
    }

    fn reject_legacy_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .legacy_route_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject_query_route(route)
    }

    fn reject_generation_mutation<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .generation_mutation_attempts
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("pinned temporal arithmetic generation rejected `{route}` mutation"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal arithmetic project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal(
                    "replacement temporal arithmetic project has no resident graph revision",
                )
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedTemporalBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
            self.advertised_kind
        }
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
        if self.pinned {
            return self.reject_query_route("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_query_route("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "pinned temporal arithmetic generation has wrong backend provenance or fence",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("admit_project");
        }
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("replace_all_projects");
        }
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("evict_project");
        }
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        if self.pinned {
            self.observations
                .generation_mutation_attempts
                .fetch_add(1, Ordering::SeqCst);
        } else {
            self.inner.advance_bookmark(bookmark);
        }
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
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_query_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_query_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_query_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_query_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_legacy_route("execute_node_pipeline")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_legacy_route("execute_row_program")
    }

    fn supports_native_temporal_value_program(&self) -> bool {
        false
    }

    fn execute_temporal_value_program(
        &self,
        _request: &ResidentTemporalValueProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        self.reject_legacy_route("execute_temporal_value_program")
    }

    fn supports_native_temporal_arithmetic_program(&self) -> bool {
        true
    }

    fn execute_temporal_arithmetic_program(
        &self,
        request: &ResidentTemporalArithmeticRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalArithmeticResult> {
        if !self.pinned {
            return self.reject_query_route("execute_temporal_arithmetic_program_without_pin");
        }
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal arithmetic execution escaped its immutable generation or provenance",
            ));
        }
        self.observations
            .temporal_arithmetic_calls
            .fetch_add(1, Ordering::SeqCst);
        let result = self
            .inner
            .execute_temporal_arithmetic_program(request, cancellation)?;
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal arithmetic dispatch mutated its pinned generation",
            ));
        }
        self.observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(NativeTemporalReceipt {
                request: request.clone(),
                result: result.clone().into_untrusted_parts(),
            });
        Ok(result)
    }

    fn execute_temporal_pipeline(
        &self,
        _request: &ResidentTemporalPipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
        self.reject_legacy_route("execute_temporal_pipeline")
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_query_route("exact_l2")
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: i64::MIN,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution: backend.is_some(),
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 1,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn assert_source_input(
    input: &ResidentTemporalValueInput,
    shape: SourceShape,
    source_index: usize,
) -> Result<()> {
    match (shape, input) {
        (SourceShape::Null, ResidentTemporalValueInput::Null) => Ok(()),
        (SourceShape::Map, ResidentTemporalValueInput::Map(packet)) => {
            if packet.len() < 3 || packet[0] != TEMPORAL_MAP_WIRE_VERSION || packet[1] == 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "source {source_index} was not dispatched as a non-empty raw temporal map packet: {packet:?}"
                    ),
                ));
            }
            Ok(())
        }
        _ => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "source {source_index} was folded, precomputed, or lowered through the wrong input shape: {input:?}"
            ),
        )),
    }
}

// This semantic wire assertion stays compileable before the production arithmetic enum lands.
// It prescribes one exact typed SSA form:
// Arithmetic { operation, left, right: Register|Integer|Float|Null }.
#[allow(dead_code)]
fn assert_arithmetic_input(
    input: &ResidentTemporalValueInput,
    instruction: InstructionSpec,
) -> Result<()> {
    let expected = format!(
        "Arithmetic {{ operation: {}, left: {}, right: {} }}",
        instruction.operation.debug_name(),
        instruction.left,
        instruction.right.debug_name()
    );
    let actual = format!("{input:?}");
    if actual != expected {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "arithmetic was folded, precomputed, or dispatched through a non-SSA input: expected {expected}, got {actual}"
            ),
        ));
    }
    Ok(())
}

fn scalar_to_resident(value: &ScalarValue) -> Result<ResidentTemporalValue> {
    Ok(match value {
        ScalarValue::Null => ResidentTemporalValue::Null,
        ScalarValue::Date(days) => ResidentTemporalValue::Date(*days),
        ScalarValue::LocalTime(nanos) => ResidentTemporalValue::LocalTime(*nanos),
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => ResidentTemporalValue::ZonedTime {
            nanos: *nanos,
            offset_seconds: *offset_seconds,
        },
        ScalarValue::LocalDateTime { seconds, nanos } => ResidentTemporalValue::LocalDateTime {
            seconds: *seconds,
            nanos: *nanos,
        },
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            let offset_seconds = parse_fixed_offset(timezone).ok_or_else(|| {
                Error::internal(format!(
                    "arithmetic acceptance expected a fixed-offset datetime, got {timezone}"
                ))
            })?;
            ResidentTemporalValue::ZonedDateTimeFixedOffset {
                seconds: *seconds,
                nanos: *nanos,
                offset_seconds,
            }
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => ResidentTemporalValue::Duration {
            months: *months,
            days: *days,
            seconds: *seconds,
            nanos: *nanos,
        },
        _ => {
            return Err(Error::internal(format!(
                "arithmetic acceptance expected a temporal, duration, or NULL value, got {value:?}"
            )));
        }
    })
}

fn parse_fixed_offset(value: &str) -> Option<i32> {
    if value == "UTC" || value == "Z" || value == "+00:00" || value == "-00:00" {
        return Some(0);
    }
    let bytes = value.as_bytes();
    if bytes.len() != 6 || !matches!(bytes[0], b'+' | b'-') || bytes[3] != b':' {
        return None;
    }
    let digit = |byte: u8| byte.is_ascii_digit().then_some(i32::from(byte - b'0'));
    let hours = digit(bytes[1])?
        .checked_mul(10)?
        .checked_add(digit(bytes[2])?)?;
    let minutes = digit(bytes[4])?
        .checked_mul(10)?
        .checked_add(digit(bytes[5])?)?;
    if hours > 23 || minutes > 59 {
        return None;
    }
    let seconds = hours
        .checked_mul(3_600)?
        .checked_add(minutes.checked_mul(60)?)?;
    Some(if bytes[0] == b'-' { -seconds } else { seconds })
}

fn column_type(value: &ScalarValue) -> ColumnType {
    match value {
        ScalarValue::Null => ColumnType::Null,
        ScalarValue::Date(_)
        | ScalarValue::LocalTime(_)
        | ScalarValue::ZonedTime { .. }
        | ScalarValue::LocalDateTime { .. }
        | ScalarValue::ZonedDateTime { .. } => ColumnType::Temporal,
        ScalarValue::Duration { .. } => ColumnType::Duration,
        _ => unreachable!("arithmetic output is always temporal, duration, or NULL"),
    }
}

fn result_scalars(output: &ExecutionOutput) -> Result<Vec<ScalarValue>> {
    if output.result.batches.len() != 1 {
        return Err(Error::internal("expected exactly one result batch"));
    }
    let batch = &output.result.batches[0];
    if batch.row_count != 1 {
        return Err(Error::internal("expected exactly one result row"));
    }
    batch
        .columns
        .iter()
        .map(|column| match column.values.as_slice() {
            [ResultValue::Scalar(value)] => Ok(value.clone()),
            values => Err(Error::internal(format!(
                "expected one scalar cell, got {values:?}"
            ))),
        })
        .collect()
}

fn expected_scalars(fixture: &Fixture, case: ArithmeticCase) -> Result<Vec<ScalarValue>> {
    let ExpectedOutcome::Success(expressions) = case.expected else {
        return Err(Error::internal(
            "error arithmetic case does not have expected scalar values",
        ));
    };
    let projection = expressions
        .iter()
        .enumerate()
        .map(|(index, expression)| format!("{expression} AS expected_{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let output =
        QueryEngine.execute(&format!("RETURN {projection}"), &mut context(fixture, None))?;
    result_scalars(&output)
}

fn assert_native_receipt(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ArithmeticCase,
    receipt: &NativeTemporalReceipt,
) -> Result<()> {
    let request = &receipt.request;
    request.validate()?;
    if request.generation.project != PROJECT
        || request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
        || request.generation.layout_version != fixture.graph.layout_version()
        || request.generation.catalog_generation != fixture.graph.catalog().optimizer_generation()
        || request.fingerprint != request.fingerprint()?
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal arithmetic request has wrong immutable generation or fingerprint: {request:?}"
            ),
        ));
    }
    let instructions = case.instructions();
    let expected_source_count = match case.scenario {
        Some(1..=6) => 2,
        Some(7) => 1,
        Some(_) => return Err(Error::internal("unknown official Temporal8 scenario")),
        None => [Some(case.source0), case.source1]
            .into_iter()
            .flatten()
            .fold(
                Vec::<(ResidentTemporalValueFunction, &'static str)>::new(),
                |mut seen, source| {
                    if !seen.contains(&(source.function, source.expression)) {
                        seen.push((source.function, source.expression));
                    }
                    seen
                },
            )
            .len(),
    };
    if request.sources.len() != expected_source_count
        || request.instructions.len() != instructions.len()
        || request.outputs.len() != instructions.len()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal arithmetic request changed source/instruction/output width: {request:?}"
            ),
        ));
    }

    match case.scenario {
        Some(1..=5) => {
            let ResidentTemporalArithmeticSource::Literal(invocation) = &request.sources[0] else {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "WITH temporal source was not raw",
                ));
            };
            if invocation.function != case.source0.function {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "WITH temporal type changed",
                ));
            }
            assert_source_input(&invocation.input, case.source0.shape, 0)?;
            let property = fixture.graph.catalog().property("dur");
            if request.sources[1]
                != (ResidentTemporalArithmeticSource::DurationProperty { scan: 0, property })
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "d.dur did not lower as a resident property source",
                ));
            }
        }
        Some(6) => {
            let property = fixture.graph.catalog().property("date");
            if request.sources
                != [
                    ResidentTemporalArithmeticSource::DurationProperty { scan: 0, property },
                    ResidentTemporalArithmeticSource::DurationProperty { scan: 1, property },
                ]
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "two-scan duration sources were merged or reordered",
                ));
            }
        }
        Some(7) => {
            let property = fixture.graph.catalog().property("date");
            if request.sources
                != [ResidentTemporalArithmeticSource::DurationProperty { scan: 0, property }]
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "duration scale property source changed",
                ));
            }
        }
        Some(_) => unreachable!(),
        None => {
            let mut expected = [Some(case.source0), case.source1]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            expected.dedup_by_key(|source| (source.function, source.expression));
            for (index, source) in expected.iter().enumerate() {
                let ResidentTemporalArithmeticSource::Literal(invocation) = &request.sources[index]
                else {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "adversarial source was not a raw temporal invocation",
                    ));
                };
                if invocation.function != source.function {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "adversarial source type changed",
                    ));
                }
                assert_source_input(&invocation.input, source.shape, index)?;
            }
        }
    }

    for (offset, instruction) in instructions.iter().copied().enumerate() {
        let actual = &request.instructions[offset];
        let operation = match instruction.operation {
            ArithmeticOperation::Add => ResidentTemporalArithmeticOperation::Add,
            ArithmeticOperation::Subtract => ResidentTemporalArithmeticOperation::Subtract,
            ArithmeticOperation::Multiply => ResidentTemporalArithmeticOperation::Multiply,
            ArithmeticOperation::Divide => ResidentTemporalArithmeticOperation::Divide,
        };
        let right = match instruction.right {
            ArithmeticRight::Register(register) => {
                ResidentTemporalArithmeticOperand::Register(register)
            }
            ArithmeticRight::Integer(value) => ResidentTemporalArithmeticOperand::Integer(value),
            ArithmeticRight::Float(value) => ResidentTemporalArithmeticOperand::FloatBits(
                value
                    .parse::<f64>()
                    .map_err(|_| Error::internal("invalid test float"))?
                    .to_bits(),
            ),
            ArithmeticRight::Null => ResidentTemporalArithmeticOperand::Null,
        };
        if actual.operation != operation
            || actual.left != ResidentTemporalArithmeticOperand::Register(instruction.left)
            || actual.right != right
            || request.outputs[offset].register
                != u16::try_from(expected_source_count + offset)
                    .map_err(|_| Error::internal("test register overflow"))?
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "arithmetic SSA instruction {offset} changed: expected {instruction:?}, got {actual:?}"
                ),
            ));
        }
    }

    let validated = ResidentTemporalArithmeticResult::from_untrusted_parts(receipt.result.clone())
        .validate_for_publication(request, backend.actual_kind)?;
    if validated.receipts().len() != request.obligations().len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "backend completion did not cover every immutable obligation",
        ));
    }
    match case.scenario {
        Some(1..=5) | Some(7) => {
            let label = fixture
                .graph
                .catalog()
                .label("Duration")
                .ok_or_else(|| Error::internal("Duration label disappeared"))?;
            if request.scans.len() != 1
                || request.scans[0].labels != [label]
                || request.manifest.scan_obligations.len() != 1
                || request.manifest.cartesian_obligation.is_some()
                || validated.receipts()[0].input_cardinality
                    != fixture.graph.node_slot_count() as u64
                || validated.receipts()[0].output_cardinality != 1
                || validated.relation_rows() != 1
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "official one-scan query did not prove its exact Duration label/cardinality",
                ));
            }
        }
        Some(6) => {
            let duration1 = fixture
                .graph
                .catalog()
                .label("Duration1")
                .ok_or_else(|| Error::internal("Duration1 label disappeared"))?;
            let duration2 = fixture
                .graph
                .catalog()
                .label("Duration2")
                .ok_or_else(|| Error::internal("Duration2 label disappeared"))?;
            if request.scans.len() != 2
                || request.scans[0].labels != [duration1]
                || request.scans[1].labels != [duration2]
                || request.manifest.scan_obligations.len() != 2
                || request.manifest.cartesian_obligation.is_none()
                || validated.receipts()[0].output_cardinality != 1
                || validated.receipts()[1].output_cardinality != 1
                || validated.receipts()[2].input_cardinality != 2
                || validated.receipts()[2].output_cardinality != 1
                || validated.relation_rows() != 1
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "official two-scan query did not prove both labels and its 1x1 Cartesian relation",
                ));
            }
        }
        Some(_) => unreachable!(),
        None => {
            if !request.scans.is_empty()
                || request.manifest.cartesian_obligation.is_some()
                || validated.relation_rows() != 1
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "graph-free adversarial query acquired a scan or non-unit relation",
                ));
            }
        }
    }
    match case.expected {
        ExpectedOutcome::Success(_) => {
            if validated.status() != ResidentTemporalArithmeticStatus::Success
                || validated.relation_rows() != 1
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "successful temporal arithmetic result has wrong status or cardinality",
                ));
            }
            let expected = expected_scalars(fixture, case)?
                .iter()
                .map(scalar_to_resident)
                .collect::<Result<Vec<_>>>()?;
            for (column, expected) in validated.columns().iter().zip(expected) {
                let expected_validity = u8::from(!matches!(&expected, ResidentTemporalValue::Null));
                if column.values.as_slice() != [expected.clone()]
                    || column.validity.as_slice() != [expected_validity]
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!(
                            "backend-authored temporal arithmetic column changed canonical value or validity: expected {expected:?}/{expected_validity}, got {:?}/{:?}",
                            column.values, column.validity
                        ),
                    ));
                }
            }
        }
        ExpectedOutcome::Error(ErrorCode::QueryType) => {
            if !matches!(
                validated.status(),
                ResidentTemporalArithmeticStatus::QueryType { .. }
            ) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "backend did not author QueryType provenance",
                ));
            }
        }
        ExpectedOutcome::Error(ErrorCode::TemporalRange) => {
            if !matches!(
                validated.status(),
                ResidentTemporalArithmeticStatus::TemporalRange { .. }
            ) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "backend did not author TemporalRange provenance",
                ));
            }
        }
        ExpectedOutcome::Error(code) => {
            return Err(Error::internal(format!(
                "unsupported temporal arithmetic test error {code:?}"
            )));
        }
    }
    Ok(())
}

fn observe_success(
    output: &ExecutionOutput,
    fixture: &Fixture,
    case: ArithmeticCase,
) -> Result<Vec<ScalarValue>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(
            "temporal arithmetic query produced side effects, truncation, or auxiliary work",
        ));
    }
    if output.result.bookmark != fixture.bookmark {
        return Err(Error::internal(
            "temporal arithmetic result changed the immutable bookmark",
        ));
    }

    let expected = expected_scalars(fixture, case)?;
    let instructions = case.instructions();
    let expected_schema = instructions
        .iter()
        .zip(&expected)
        .map(|(instruction, value)| (instruction.alias.to_owned(), column_type(value)))
        .collect::<Vec<_>>();
    if output.result.schema != expected_schema {
        return Err(Error::internal(format!(
            "temporal arithmetic result schema is wrong: expected {expected_schema:?}, got {:?}",
            output.result.schema
        )));
    }
    let actual = result_scalars(output)?;
    if actual != expected {
        return Err(Error::internal(format!(
            "temporal arithmetic output mismatch: expected {expected:?}, got {actual:?}"
        )));
    }
    Ok(actual)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum CaseObservation {
    Values(Vec<ScalarValue>),
    Error(ErrorCode),
}

fn execute_generic_case(fixture: &Fixture, case: ArithmeticCase) -> Result<CaseObservation> {
    match QueryEngine.execute(&case.execution_query(), &mut context(fixture, None)) {
        Ok(output) => {
            let ExpectedOutcome::Success(_) = case.expected else {
                return Err(Error::internal(format!(
                    "{} unexpectedly succeeded in the generic CPU oracle",
                    case.label()
                )));
            };
            observe_success(&output, fixture, case).map(CaseObservation::Values)
        }
        Err(error) => {
            let ExpectedOutcome::Error(expected) = case.expected else {
                return Err(error);
            };
            if error.code != expected {
                return Err(Error::internal(format!(
                    "{} generic CPU error changed: expected {expected:?}, got {:?}: {error}",
                    case.label(),
                    error.code
                )));
            }
            Ok(CaseObservation::Error(error.code))
        }
    }
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ArithmeticCase,
) -> std::result::Result<CaseObservation, String> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let temporal_before = observations
        .temporal_arithmetic_calls
        .load(Ordering::SeqCst);
    let legacy_before = observations.legacy_route_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let mutations_before = observations
        .generation_mutation_attempts
        .load(Ordering::SeqCst);
    let receipts_before = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();

    let execution = QueryEngine.execute(
        &case.execution_query(),
        &mut context(fixture, Some(backend)),
    );

    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations
        .temporal_arithmetic_calls
        .load(Ordering::SeqCst)
        != temporal_before + 1
    {
        return Err(format!(
            "query did not cross exactly one native temporal arithmetic boundary; backend result was {execution:?}"
        ));
    }
    if observations.legacy_route_calls.load(Ordering::SeqCst) != legacy_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered a generic, legacy, graph, row, host-evaluation, or CPU-fallback route"
                .to_owned(),
        );
    }
    if observations
        .generation_mutation_attempts
        .load(Ordering::SeqCst)
        != mutations_before
        || backend.resident_bookmark(PROJECT) != Some(fixture.bookmark)
        || backend.resident_graph_revision(PROJECT) != Some(fixture.graph.revision())
    {
        return Err("query attempted to mutate or replace its pinned generation".to_owned());
    }

    let receipt = {
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if receipts.len() != receipts_before + 1 {
            return Err(format!(
                "native temporal observer retained {} receipts instead of one; backend result was {execution:?}",
                receipts.len().saturating_sub(receipts_before),
            ));
        }
        receipts
            .last()
            .cloned()
            .ok_or_else(|| "native temporal arithmetic receipt disappeared".to_owned())?
    };
    assert_native_receipt(fixture, backend, case, &receipt)
        .map_err(|error| format!("native request/provenance assertion failed: {error}"))?;

    match (execution, case.expected) {
        (Ok(output), ExpectedOutcome::Success(_)) => observe_success(&output, fixture, case)
            .map(CaseObservation::Values)
            .map_err(|error| format!("native output inspection failed: {error}")),
        (Err(error), ExpectedOutcome::Error(expected)) if error.code == expected => {
            Ok(CaseObservation::Error(error.code))
        }
        (Ok(_), ExpectedOutcome::Error(expected)) => Err(format!(
            "expected native {expected:?} error but query succeeded"
        )),
        (Err(error), ExpectedOutcome::Success(_)) => Err(format!(
            "native query failed with {:?}: {error}",
            error.code
        )),
        (Err(error), ExpectedOutcome::Error(expected)) => Err(format!(
            "native error changed: expected {expected:?}, got {:?}: {error}",
            error.code
        )),
    }
}

fn run_native_cpu_suite() -> Vec<String> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = match Fixture::for_case(*case) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: exact fixture failed: {error}", case.label()));
                continue;
            }
        };
        let backend = match fixture.strict_cpu_backend() {
            Ok(backend) => backend,
            Err(error) => {
                failures.push(format!(
                    "{}: resident admission failed: {error}",
                    case.label()
                ));
                continue;
            }
        };
        if let Err(error) = execute_native_case(&fixture, &backend, *case) {
            failures.push(format!("{}: {error}", case.label()));
            continue;
        }
        let observations = backend.observations();
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if observations.pins.load(Ordering::SeqCst) != 1
            || observations
                .temporal_arithmetic_calls
                .load(Ordering::SeqCst)
                != 1
            || observations.legacy_route_calls.load(Ordering::SeqCst) != 0
            || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
            || receipts.len() != 1
            || receipts[0].result.completion != ResidentDeviceCompletion::CpuReference
        {
            failures.push(format!(
                "{}: exact case did not retain one CPU pin/call/backend-authored completion",
                case.label()
            ));
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} native temporal arithmetic suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_temporal8_ids_scenarios_operands_results_and_operations() {
    assert_eq!(OFFICIAL_CASES.len(), 27);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.report_id.expect("official case has report id"))
            .collect::<Vec<_>>(),
        (3472_u16..=3498).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.scenario.expect("official case has scenario"))
            .collect::<Vec<_>>(),
        vec![
            1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7,
        ]
    );

    for case in &OFFICIAL_CASES {
        assert!(case.official_setup().is_some(), "{} setup", case.label());
        let query = case
            .official_query()
            .unwrap_or_else(|| panic!("{} query", case.label()));
        assert!(
            !query.contains('<'),
            "{} retained an outline placeholder",
            case.label()
        );
        assert!(matches!(case.expected, ExpectedOutcome::Success(_)));
        assert_eq!(case.instructions().len(), 2, "{}", case.label());
    }
    assert_eq!(
        OFFICIAL_CASES[24..]
            .iter()
            .map(|case| case.official_query().expect("scale query"))
            .collect::<Vec<_>>(),
        vec![
            "MATCH (d:Duration)\nRETURN d.date * 1 AS prod, d.date / 1 AS div",
            "MATCH (d:Duration)\nRETURN d.date * 2 AS prod, d.date / 2 AS div",
            "MATCH (d:Duration)\nRETURN d.date * 0.5 AS prod, d.date / 0.5 AS div",
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter(|case| matches!(case.program, ProgramKind::Pair { .. }))
            .count(),
        24
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter(|case| matches!(case.program, ProgramKind::Scale { .. }))
            .count(),
        3
    );

    for operation in [
        ArithmeticOperation::Add,
        ArithmeticOperation::Subtract,
        ArithmeticOperation::Multiply,
        ArithmeticOperation::Divide,
    ] {
        assert!(
            all_cases().any(|case| {
                case.instructions()
                    .iter()
                    .any(|instruction| instruction.operation == operation)
            }),
            "manifest omitted {operation:?}"
        );
    }

    assert!(ADVERSARIAL_CASES.iter().any(|case| {
        case.source0.shape == SourceShape::Null
            || case
                .source1
                .is_some_and(|source| source.shape == SourceShape::Null)
    }));
    assert!(
        ADVERSARIAL_CASES
            .iter()
            .any(|case| case.name.contains("month-end"))
    );
    assert!(
        ADVERSARIAL_CASES
            .iter()
            .any(|case| case.name.contains("negative fractional"))
    );
    assert!(
        ADVERSARIAL_CASES
            .iter()
            .any(|case| case.name.contains("division by zero"))
    );
    assert!(
        ADVERSARIAL_CASES
            .iter()
            .any(|case| { matches!(case.expected, ExpectedOutcome::Error(ErrorCode::QueryType)) })
    );
}

#[test]
fn generic_cpu_oracle_confirms_all_official_and_adversarial_semantics() -> Result<()> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::for_case(*case)?;
        if let Err(error) = execute_generic_case(&fixture, *case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert!(
        failures.is_empty(),
        "generic CPU/openCypher semantic oracle had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn strict_cpu_runs_all_27_official_and_11_adversarial_cases_natively() -> Result<()> {
    assert_no_failures("strict CPU reference", run_native_cpu_suite());
    Ok(())
}

#[test]
fn cpu_backend_cannot_masquerade_as_metal_or_publish_arithmetic_answers() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.cpu_masquerading_as_metal()?;
    let observations = backend.observations();
    let error = match backend.pin_project(PROJECT) {
        Ok(_) => panic!("a CPU backend advertised as Metal unexpectedly pinned as real Metal"),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations
            .temporal_arithmetic_calls
            .load(Ordering::SeqCst),
        0
    );
    assert!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
    );
    Ok(())
}

#[test]
fn pinned_temporal_arithmetic_generation_rejects_every_mutation_surface() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    let observations = backend.observations();
    let mut pinned = backend.pin_project(PROJECT)?;
    let bookmark = pinned.resident_bookmark(PROJECT);
    let graph_revision = pinned.resident_graph_revision(PROJECT);

    assert_eq!(
        pinned.admit_project(fixture.image()?).unwrap_err().code,
        ErrorCode::CorruptStorage
    );
    assert_eq!(
        pinned
            .replace_all_projects(vec![fixture.image()?])
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    assert_eq!(
        pinned.evict_project(PROJECT).unwrap_err().code,
        ErrorCode::CorruptStorage
    );
    pinned.advance_bookmark(Bookmark {
        term: fixture.bookmark.term.saturating_add(1),
        index: fixture.bookmark.index.saturating_add(1),
    });

    assert_eq!(pinned.resident_bookmark(PROJECT), bookmark);
    assert_eq!(pinned.resident_graph_revision(PROJECT), graph_revision);
    assert_eq!(
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
        4
    );
    assert_eq!(
        observations
            .temporal_arithmetic_calls
            .load(Ordering::SeqCst),
        0
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "focused cold-compile regression gate: requires real Metal"]
fn real_metal_runs_first_localdatetime_duration_property_case_before_deadline() -> Result<()> {
    let _guard = metal_test_guard();
    let case = OFFICIAL_CASES[9];
    assert_eq!(case.report_id, Some(3481));
    assert_eq!(case.scenario, Some(4));
    let fixture = Fixture::for_case(case)?;
    let cpu = fixture.strict_cpu_backend()?;
    let metal = fixture.real_metal_backend()?;
    let cpu_result = execute_native_case(&fixture, &cpu, case).map_err(Error::internal)?;
    let metal_result = execute_native_case(&fixture, &metal, case).map_err(Error::internal)?;
    assert_eq!(metal_result, cpu_result);
    let observations = metal.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(
        observations
            .temporal_arithmetic_calls
            .load(Ordering::SeqCst),
        1
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: exact Temporal8 plus adversarial arithmetic suite requires real Metal"]
fn real_metal_matches_independent_cpu_with_exact_dispatch_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = match Fixture::for_case(*case) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: exact fixture failed: {error}", case.label()));
                continue;
            }
        };
        let cpu = match fixture.strict_cpu_backend() {
            Ok(backend) => backend,
            Err(error) => {
                failures.push(format!("{}: CPU admission failed: {error}", case.label()));
                continue;
            }
        };
        let metal = match fixture.real_metal_backend() {
            Ok(backend) => backend,
            Err(error) => {
                failures.push(format!("{}: Metal admission failed: {error}", case.label()));
                continue;
            }
        };
        assert_eq!(cpu.actual_kind, BackendKind::Cpu);
        assert_eq!(metal.actual_kind, BackendKind::Metal);

        let cpu_result = execute_native_case(&fixture, &cpu, *case);
        let metal_result = execute_native_case(&fixture, &metal, *case);
        match (cpu_result, metal_result) {
            (Ok(cpu_value), Ok(metal_value)) if cpu_value == metal_value => {}
            (Ok(cpu_value), Ok(metal_value)) => failures.push(format!(
                "{}: CPU/Metal mismatch: CPU={cpu_value:?}, Metal={metal_value:?}",
                case.label()
            )),
            (Err(cpu_error), Ok(_)) => {
                failures.push(format!("{}: strict CPU failed: {cpu_error}", case.label()))
            }
            (Ok(_), Err(metal_error)) => failures.push(format!(
                "{}: real Metal failed: {metal_error}",
                case.label()
            )),
            (Err(cpu_error), Err(metal_error)) => failures.push(format!(
                "{}: strict CPU failed: {cpu_error}; real Metal failed: {metal_error}",
                case.label()
            )),
        }
        for (name, backend, completion) in [
            ("CPU", &cpu, ResidentDeviceCompletion::CpuReference),
            ("Metal", &metal, ResidentDeviceCompletion::Metal),
        ] {
            let observations = backend.observations();
            let receipts = observations
                .receipts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if observations.pins.load(Ordering::SeqCst) != 1
                || observations
                    .temporal_arithmetic_calls
                    .load(Ordering::SeqCst)
                    != 1
                || observations.legacy_route_calls.load(Ordering::SeqCst) != 0
                || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
                || observations
                    .generation_mutation_attempts
                    .load(Ordering::SeqCst)
                    != 0
                || receipts.len() != 1
                || receipts[0].request.generation.bookmark != fixture.bookmark
                || receipts[0].request.generation.graph_revision != fixture.graph.revision()
                || receipts[0].result.completion != completion
            {
                failures.push(format!(
                    "{}: {name} did not retain exactly one pinned, backend-authored completion for the exact fixture",
                    case.label()
                ));
            }
        }
    }
    assert_no_failures("strict CPU versus real Metal parity", failures);
    Ok(())
}

#[test]
#[ignore]
fn debug_exact_temporal8_physical_shapes() -> Result<()> {
    use irongraph::cypher::{bind, parse, plan};

    let mut graph = GraphStore::default();
    for label in ["Duration", "Duration1", "Duration2"] {
        graph.catalog_mut().intern_label(label)?;
    }
    for property in ["dur", "date"] {
        graph.catalog_mut().intern_property(property)?;
    }
    for index in [0_usize, 15, 24] {
        let query = OFFICIAL_CASES[index]
            .official_query()
            .expect("official query");
        let physical = plan(bind(
            parse(&query)?,
            graph.catalog(),
            BindCapabilities::default(),
        )?)?;
        eprintln!(
            "CASE {} QUERY:\n{}\nPLAN: {:#?}",
            OFFICIAL_CASES[index].report_id.unwrap(),
            query,
            physical.operators
        );
    }
    Ok(())
}
