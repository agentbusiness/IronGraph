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
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentSortRequest, ResidentSortResult,
        ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult, ResidentTemporalValue,
        ResidentTemporalValueFunction, ResidentTemporalValueInput, ResidentTemporalValueInvocation,
        ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 16;
const FEATURE: &str = "features/expressions/temporal/Temporal7.feature";
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const TEMPORAL_MAP_WIRE_VERSION: u8 = 1;

fn assert_certified_report_identities(
    identities: impl IntoIterator<Item = (usize, String, String)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = std::collections::BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature.clone(), expanded_name.clone())),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(&feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExpectedComparison {
    operation: CompareOp,
    token: &'static str,
    value: Option<bool>,
}

const LESS_FIVE: [ExpectedComparison; 5] = [
    expected(CompareOp::Greater, ">", Some(false)),
    expected(CompareOp::Less, "<", Some(true)),
    expected(CompareOp::GreaterOrEqual, ">=", Some(false)),
    expected(CompareOp::LessOrEqual, "<=", Some(true)),
    expected(CompareOp::Eq, "=", Some(false)),
];

const EQUAL_FIVE: [ExpectedComparison; 5] = [
    expected(CompareOp::Greater, ">", Some(false)),
    expected(CompareOp::Less, "<", Some(false)),
    expected(CompareOp::GreaterOrEqual, ">=", Some(true)),
    expected(CompareOp::LessOrEqual, "<=", Some(true)),
    expected(CompareOp::Eq, "=", Some(true)),
];

const EQ_FALSE: [ExpectedComparison; 1] = [expected(CompareOp::Eq, "=", Some(false))];
const EQ_TRUE: [ExpectedComparison; 1] = [expected(CompareOp::Eq, "=", Some(true))];

const LESS_ALL_SIX: [ExpectedComparison; 6] = [
    expected(CompareOp::Eq, "=", Some(false)),
    expected(CompareOp::NotEq, "<>", Some(true)),
    expected(CompareOp::Less, "<", Some(true)),
    expected(CompareOp::LessOrEqual, "<=", Some(true)),
    expected(CompareOp::Greater, ">", Some(false)),
    expected(CompareOp::GreaterOrEqual, ">=", Some(false)),
];

const NULL_ALL_SIX: [ExpectedComparison; 6] = [
    expected(CompareOp::Eq, "=", None),
    expected(CompareOp::NotEq, "<>", None),
    expected(CompareOp::Less, "<", None),
    expected(CompareOp::LessOrEqual, "<=", None),
    expected(CompareOp::Greater, ">", None),
    expected(CompareOp::GreaterOrEqual, ">=", None),
];

const CROSS_FAMILY_ALL_SIX: [ExpectedComparison; 6] = [
    expected(CompareOp::Eq, "=", Some(false)),
    expected(CompareOp::NotEq, "<>", Some(true)),
    expected(CompareOp::Less, "<", None),
    expected(CompareOp::LessOrEqual, "<=", None),
    expected(CompareOp::Greater, ">", None),
    expected(CompareOp::GreaterOrEqual, ">=", None),
];

const EQUIVALENT_DURATION_ALL_SIX: [ExpectedComparison; 6] = [
    expected(CompareOp::Eq, "=", Some(true)),
    expected(CompareOp::NotEq, "<>", Some(false)),
    expected(CompareOp::Less, "<", None),
    expected(CompareOp::LessOrEqual, "<=", None),
    expected(CompareOp::Greater, ">", None),
    expected(CompareOp::GreaterOrEqual, ">=", None),
];

const fn expected(
    operation: CompareOp,
    token: &'static str,
    value: Option<bool>,
) -> ExpectedComparison {
    ExpectedComparison {
        operation,
        token,
        value,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceShape {
    Map,
    Null,
}

#[derive(Clone, Copy, Debug)]
struct ComparisonCase {
    report_id: Option<u16>,
    scenario: Option<u8>,
    name: &'static str,
    left_function: ResidentTemporalValueFunction,
    right_function: ResidentTemporalValueFunction,
    left_expression: &'static str,
    right_expression: &'static str,
    left_shape: SourceShape,
    right_shape: SourceShape,
    comparisons: &'static [ExpectedComparison],
}

impl ComparisonCase {
    fn query(self) -> String {
        let expressions = self
            .comparisons
            .iter()
            .map(|comparison| format!("x {} d", comparison.token))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "WITH {} AS x, {} AS d\nRETURN {expressions}",
            self.left_expression, self.right_expression
        )
    }

    fn label(self) -> String {
        match (self.report_id, self.scenario) {
            (Some(id), Some(scenario)) => {
                format!("TCK {id} {FEATURE} [{scenario}] {}", self.name)
            }
            _ => format!("adversarial {}", self.name),
        }
    }

    fn expanded_report_name(self) -> Option<String> {
        let report_id = self.report_id?;
        let scenario = self.scenario?;
        let title = match scenario {
            1 => "Should compare dates",
            2 => "Should compare local times",
            3 => "Should compare times",
            4 => "Should compare local date times",
            5 => "Should compare date times",
            6 => "Should compare durations for equality",
            _ => return None,
        };
        Some(format!(
            "[{scenario}] {title} [{}]",
            usize::from(report_id) + 1
        ))
    }

    fn expected_scalars(self) -> Vec<ScalarValue> {
        self.comparisons
            .iter()
            .map(|comparison| {
                comparison
                    .value
                    .map_or(ScalarValue::Null, ScalarValue::Boolean)
            })
            .collect()
    }

    fn expected_temporal_values(self) -> Vec<ResidentTemporalValue> {
        self.comparisons
            .iter()
            .map(|comparison| {
                comparison
                    .value
                    .map_or(ResidentTemporalValue::Null, ResidentTemporalValue::Boolean)
            })
            .collect()
    }
}

const fn official_ordered(
    report_id: u16,
    scenario: u8,
    name: &'static str,
    function: ResidentTemporalValueFunction,
    left_expression: &'static str,
    right_expression: &'static str,
    comparisons: &'static [ExpectedComparison],
) -> ComparisonCase {
    ComparisonCase {
        report_id: Some(report_id),
        scenario: Some(scenario),
        name,
        left_function: function,
        right_function: function,
        left_expression,
        right_expression,
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons,
    }
}

const fn official_duration(
    report_id: u16,
    name: &'static str,
    right_function: ResidentTemporalValueFunction,
    right_expression: &'static str,
    comparisons: &'static [ExpectedComparison],
) -> ComparisonCase {
    ComparisonCase {
        report_id: Some(report_id),
        scenario: Some(6),
        name,
        left_function: ResidentTemporalValueFunction::Duration,
        right_function,
        left_expression: "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70})",
        right_expression,
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons,
    }
}

const OFFICIAL_CASES: [ComparisonCase; 18] = [
    official_ordered(
        3453,
        1,
        "dates less-than example",
        ResidentTemporalValueFunction::Date,
        "date({year: 1980, month: 12, day: 24})",
        "date({year: 1984, month: 10, day: 11})",
        &LESS_FIVE,
    ),
    official_ordered(
        3454,
        1,
        "dates equality example",
        ResidentTemporalValueFunction::Date,
        "date({year: 1984, month: 10, day: 11})",
        "date({year: 1984, month: 10, day: 11})",
        &EQUAL_FIVE,
    ),
    official_ordered(
        3455,
        2,
        "local times less-than example",
        ResidentTemporalValueFunction::LocalTime,
        "localtime({hour: 10, minute: 35})",
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &LESS_FIVE,
    ),
    official_ordered(
        3456,
        2,
        "local times equality example",
        ResidentTemporalValueFunction::LocalTime,
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &EQUAL_FIVE,
    ),
    official_ordered(
        3457,
        3,
        "times less-than example",
        ResidentTemporalValueFunction::Time,
        "time({hour: 10, minute: 0, timezone: '+01:00'})",
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
        &LESS_FIVE,
    ),
    official_ordered(
        3458,
        3,
        "times equality example",
        ResidentTemporalValueFunction::Time,
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
        &EQUAL_FIVE,
    ),
    official_ordered(
        3459,
        4,
        "local date times less-than example",
        ResidentTemporalValueFunction::LocalDateTime,
        "localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})",
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &LESS_FIVE,
    ),
    official_ordered(
        3460,
        4,
        "local date times equality example",
        ResidentTemporalValueFunction::LocalDateTime,
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &EQUAL_FIVE,
    ),
    official_ordered(
        3461,
        5,
        "date times less-than example",
        ResidentTemporalValueFunction::DateTime,
        "datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'})",
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
        &LESS_FIVE,
    ),
    official_ordered(
        3462,
        5,
        "date times equality example",
        ResidentTemporalValueFunction::DateTime,
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
        &EQUAL_FIVE,
    ),
    official_duration(
        3463,
        "duration versus date",
        ResidentTemporalValueFunction::Date,
        "date({year: 1984, month: 10, day: 11})",
        &EQ_FALSE,
    ),
    official_duration(
        3464,
        "duration versus local time",
        ResidentTemporalValueFunction::LocalTime,
        "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &EQ_FALSE,
    ),
    official_duration(
        3465,
        "duration versus time",
        ResidentTemporalValueFunction::Time,
        "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
        &EQ_FALSE,
    ),
    official_duration(
        3466,
        "duration versus local date time",
        ResidentTemporalValueFunction::LocalDateTime,
        "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        &EQ_FALSE,
    ),
    official_duration(
        3467,
        "duration versus date time",
        ResidentTemporalValueFunction::DateTime,
        "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
        &EQ_FALSE,
    ),
    official_duration(
        3468,
        "identical durations",
        ResidentTemporalValueFunction::Duration,
        "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70})",
        &EQ_TRUE,
    ),
    official_duration(
        3469,
        "normalized equivalent durations",
        ResidentTemporalValueFunction::Duration,
        "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 13, seconds: 10})",
        &EQ_TRUE,
    ),
    official_duration(
        3470,
        "different durations",
        ResidentTemporalValueFunction::Duration,
        "duration({years: 12, months: 5, days: 13, hours: 40, minutes: 13, seconds: 10})",
        &EQ_FALSE,
    ),
];

const OFFICIAL_QUERIES: [&str; 18] = [
    "WITH date({year: 1980, month: 12, day: 24}) AS x, date({year: 1984, month: 10, day: 11}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH date({year: 1984, month: 10, day: 11}) AS x, date({year: 1984, month: 10, day: 11}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH localtime({hour: 10, minute: 35}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH time({hour: 10, minute: 0, timezone: '+01:00'}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\nRETURN x > d, x < d, x >= d, x <= d, x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, date({year: 1984, month: 10, day: 11}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 14, hours: 16, minutes: 13, seconds: 10}) AS d\nRETURN x = d",
    "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 13, hours: 40, minutes: 13, seconds: 10}) AS d\nRETURN x = d",
];

const ADVERSARIAL_CASES: [ComparisonCase; 9] = [
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "all six Date operators",
        left_function: ResidentTemporalValueFunction::Date,
        right_function: ResidentTemporalValueFunction::Date,
        left_expression: "date({year: 1980, month: 12, day: 24})",
        right_expression: "date({year: 1984, month: 10, day: 11})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &LESS_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "all six LocalTime operators",
        left_function: ResidentTemporalValueFunction::LocalTime,
        right_function: ResidentTemporalValueFunction::LocalTime,
        left_expression: "localtime({hour: 10, minute: 35})",
        right_expression: "localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &LESS_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "all six Time operators",
        left_function: ResidentTemporalValueFunction::Time,
        right_function: ResidentTemporalValueFunction::Time,
        left_expression: "time({hour: 10, minute: 0, timezone: '+01:00'})",
        right_expression: "time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &LESS_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "all six LocalDateTime operators",
        left_function: ResidentTemporalValueFunction::LocalDateTime,
        right_function: ResidentTemporalValueFunction::LocalDateTime,
        left_expression: "localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})",
        right_expression: "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &LESS_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "all six DateTime operators",
        left_function: ResidentTemporalValueFunction::DateTime,
        right_function: ResidentTemporalValueFunction::DateTime,
        left_expression: "datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'})",
        right_expression: "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &LESS_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "NULL propagates through all six operators",
        left_function: ResidentTemporalValueFunction::Date,
        right_function: ResidentTemporalValueFunction::Date,
        left_expression: "date(null)",
        right_expression: "date({year: 1984, month: 10, day: 11})",
        left_shape: SourceShape::Null,
        right_shape: SourceShape::Map,
        comparisons: &NULL_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "Date versus LocalDateTime cross-family semantics",
        left_function: ResidentTemporalValueFunction::Date,
        right_function: ResidentTemporalValueFunction::LocalDateTime,
        left_expression: "date({year: 1984, month: 10, day: 11})",
        right_expression: "localdatetime({year: 1984, month: 10, day: 11, hour: 0, minute: 0})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &CROSS_FAMILY_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "Duration versus Date cross-family semantics",
        left_function: ResidentTemporalValueFunction::Duration,
        right_function: ResidentTemporalValueFunction::Date,
        left_expression: "duration({days: 1})",
        right_expression: "date({year: 1984, month: 10, day: 11})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &CROSS_FAMILY_ALL_SIX,
    },
    ComparisonCase {
        report_id: None,
        scenario: None,
        name: "normalized equal Duration values and undefined ordering",
        left_function: ResidentTemporalValueFunction::Duration,
        right_function: ResidentTemporalValueFunction::Duration,
        left_expression: "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70})",
        right_expression: "duration({years: 12, months: 5, days: 14, hours: 16, minutes: 13, seconds: 10})",
        left_shape: SourceShape::Map,
        right_shape: SourceShape::Map,
        comparisons: &EQUIVALENT_DURATION_ALL_SIX,
    },
];

fn all_cases() -> impl Iterator<Item = &'static ComparisonCase> {
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
    project: ProjectId,
    bookmark: Bookmark,
    graph_revision: u64,
    completion: BackendKind,
    request: ResidentTemporalValueProgramRequest,
    result: ResidentTemporalValueProgramResult,
}

#[derive(Default)]
struct TemporalObservations {
    pins: AtomicUsize,
    temporal_value_calls: AtomicUsize,
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
            BackendKind::Metal,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "temporal comparison test did not construct a real Metal backend",
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
            Error::internal("temporal comparison backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal comparison backend has no admitted graph revision")
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
            format!("strict temporal comparison test rejected `{route}` execution"),
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
            format!("pinned temporal comparison generation rejected `{route}` mutation"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal comparison project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal(
                    "replacement temporal comparison project has no resident graph revision",
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
                "pinned temporal comparison generation has wrong backend provenance or fence",
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
        true
    }

    fn execute_temporal_value_program(
        &self,
        request: &ResidentTemporalValueProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        if !self.pinned {
            return self.reject_query_route("execute_temporal_value_program_without_pin");
        }
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal comparison execution escaped its immutable generation or provenance",
            ));
        }
        self.observations
            .temporal_value_calls
            .fetch_add(1, Ordering::SeqCst);
        let result = self
            .inner
            .execute_temporal_value_program(request, cancellation)?;
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal comparison dispatch mutated its pinned generation",
            ));
        }
        self.observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(NativeTemporalReceipt {
                project: PROJECT,
                bookmark: self.expected_bookmark,
                graph_revision: self.expected_graph_revision,
                completion: self.actual_kind,
                request: request.clone(),
                result: result.clone(),
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
    side: &str,
) -> Result<()> {
    match (shape, input) {
        (SourceShape::Null, ResidentTemporalValueInput::Null) => Ok(()),
        (SourceShape::Map, ResidentTemporalValueInput::Map(packet)) => {
            if packet.len() < 3 || packet[0] != TEMPORAL_MAP_WIRE_VERSION || packet[1] == 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "{side} constructor was not dispatched as a non-empty raw temporal map packet: {packet:?}"
                    ),
                ));
            }
            Ok(())
        }
        _ => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{side} temporal source was folded, precomputed, or lowered through the wrong input shape: {input:?}"
            ),
        )),
    }
}

fn assert_native_receipt(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ComparisonCase,
    receipt: &NativeTemporalReceipt,
) -> Result<()> {
    if receipt.project != PROJECT
        || receipt.bookmark != fixture.bookmark
        || receipt.graph_revision != fixture.graph.revision()
        || receipt.completion != backend.actual_kind
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal comparison receipt has wrong project, immutable fence, or backend provenance: {receipt:?}"
            ),
        ));
    }

    let request = &receipt.request;
    let expected_invocations = 2 + case.comparisons.len();
    if request.invocations.len() != expected_invocations {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal comparison request must contain two raw operands plus every comparison; expected {expected_invocations}, got {:?}",
                request.invocations
            ),
        ));
    }
    if request.invocations[0].function != case.left_function
        || request.invocations[1].function != case.right_function
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal comparison source types were changed: {:?}",
                request.invocations
            ),
        ));
    }
    assert_source_input(&request.invocations[0].input, case.left_shape, "left")?;
    assert_source_input(&request.invocations[1].input, case.right_shape, "right")?;

    let expected_outputs = (2..expected_invocations)
        .map(|register| u16::try_from(register).expect("tiny comparison program fits u16"))
        .collect::<Vec<_>>();
    if request.output_registers != expected_outputs {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "host exposed source/precomputed registers instead of device comparison registers: expected {expected_outputs:?}, got {:?}",
                request.output_registers
            ),
        ));
    }

    for (index, comparison) in case.comparisons.iter().enumerate() {
        let expected_invocation = ResidentTemporalValueInvocation {
            function: case.left_function,
            input: ResidentTemporalValueInput::Comparison {
                operation: comparison.operation,
                left: 0,
                right: 1,
            },
        };
        if request.invocations[index + 2] != expected_invocation {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "comparison {} was folded or dispatched with the wrong typed SSA operands: expected {expected_invocation:?}, got {:?}",
                    comparison.token,
                    request.invocations[index + 2]
                ),
            ));
        }
    }

    let expected_values = case.expected_temporal_values();
    if receipt.result.values != expected_values {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "selected backend returned the wrong typed comparison registers: expected {expected_values:?}, got {:?}",
                receipt.result.values
            ),
        ));
    }
    Ok(())
}

fn observe_output(
    output: &ExecutionOutput,
    fixture: &Fixture,
    case: ComparisonCase,
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
            "temporal comparison query produced side effects, truncation, or auxiliary work",
        ));
    }
    let expected_schema = case
        .comparisons
        .iter()
        .map(|comparison| {
            (
                format!("x {} d", comparison.token),
                if comparison.value.is_some() {
                    ColumnType::Boolean
                } else {
                    ColumnType::Null
                },
            )
        })
        .collect::<Vec<_>>();
    if output.result.bookmark != fixture.bookmark
        || output.result.schema != expected_schema
        || output.result.batches.len() != 1
    {
        return Err(Error::internal(format!(
            "temporal comparison result metadata is wrong: {:?}",
            output.result
        )));
    }
    let batch = &output.result.batches[0];
    if batch.row_count != 1 || batch.columns.len() != case.comparisons.len() {
        return Err(Error::internal(
            "temporal comparison result is not exactly one complete row",
        ));
    }
    let mut values = Vec::with_capacity(case.comparisons.len());
    for (column, comparison) in batch.columns.iter().zip(case.comparisons) {
        let expected_name = format!("x {} d", comparison.token);
        let expected_type = if comparison.value.is_some() {
            ColumnType::Boolean
        } else {
            ColumnType::Null
        };
        let expected_value = comparison
            .value
            .map_or(ScalarValue::Null, ScalarValue::Boolean);
        if column.name != expected_name
            || column.value_type != expected_type
            || column.values != vec![ResultValue::Scalar(expected_value.clone())]
        {
            return Err(Error::internal(format!(
                "temporal comparison column is wrong: expected {expected_name:?}/{expected_type:?}/{expected_value:?}, got {column:?}"
            )));
        }
        values.push(expected_value);
    }
    Ok(values)
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ComparisonCase,
) -> std::result::Result<Vec<ScalarValue>, String> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let temporal_before = observations.temporal_value_calls.load(Ordering::SeqCst);
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

    let pinned = backend.pin_project(PROJECT).map_err(|error| {
        format!(
            "immutable native generation pin failed with {:?}: {error}",
            error.code
        )
    })?;
    if pinned.kind() != BackendKind::Metal
        || pinned.resident_bookmark(PROJECT) != Some(fixture.bookmark)
        || pinned.resident_graph_revision(PROJECT) != Some(fixture.graph.revision())
    {
        return Err(
            "strict native generation did not retain its advertised class and exact fence"
                .to_owned(),
        );
    }

    let query = case.query();
    let output = QueryEngine
        .execute(&query, &mut context(fixture, Some(pinned.as_ref())))
        .map_err(|error| {
            format!(
                "query failed with {:?}: {error}; pins={}, temporal_calls={}, legacy_calls={}, unexpected_calls={}, mutation_attempts={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.temporal_value_calls.load(Ordering::SeqCst) - temporal_before,
                observations.legacy_route_calls.load(Ordering::SeqCst) - legacy_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
                observations
                    .generation_mutation_attempts
                    .load(Ordering::SeqCst)
                    - mutations_before,
            )
        })?;

    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations.temporal_value_calls.load(Ordering::SeqCst) != temporal_before + 1 {
        return Err(
            "query did not cross exactly one native temporal comparison boundary".to_owned(),
        );
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
        || pinned.resident_bookmark(PROJECT) != Some(fixture.bookmark)
        || pinned.resident_graph_revision(PROJECT) != Some(fixture.graph.revision())
    {
        return Err("query attempted to mutate or replace its pinned generation".to_owned());
    }

    let receipt = {
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if receipts.len() != receipts_before + 1 {
            return Err("native temporal observer did not retain exactly one receipt".to_owned());
        }
        receipts
            .last()
            .cloned()
            .ok_or_else(|| "native temporal comparison receipt disappeared".to_owned())?
    };
    assert_native_receipt(fixture, backend, case, &receipt)
        .map_err(|error| format!("native request/provenance assertion failed: {error}"))?;
    let values = observe_output(&output, fixture, case)
        .map_err(|error| format!("native output inspection failed: {error}"))?;
    let expected = case.expected_scalars();
    if values != expected {
        return Err(format!(
            "temporal comparison output mismatch: expected {expected:?}, got {values:?}"
        ));
    }
    Ok(values)
}

fn run_native_suite(fixture: &Fixture, backend: &ObservedTemporalBackend) -> Vec<String> {
    let mut failures = Vec::new();
    for case in all_cases() {
        if let Err(error) = execute_native_case(fixture, backend, *case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} native temporal comparison suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_temporal7_ids_queries_types_and_all_six_operators() {
    assert_eq!(OFFICIAL_CASES.len(), 18);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.report_id.expect("official case has report id"))
            .collect::<Vec<_>>(),
        (3453_u16..=3470).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.scenario.expect("official case has scenario"))
            .collect::<Vec<_>>(),
        vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .copied()
            .map(ComparisonCase::query)
            .collect::<Vec<_>>(),
        OFFICIAL_QUERIES
            .iter()
            .map(|query| (*query).to_owned())
            .collect::<Vec<_>>()
    );

    for function in [
        ResidentTemporalValueFunction::Date,
        ResidentTemporalValueFunction::LocalTime,
        ResidentTemporalValueFunction::Time,
        ResidentTemporalValueFunction::LocalDateTime,
        ResidentTemporalValueFunction::DateTime,
    ] {
        assert!(
            OFFICIAL_CASES
                .iter()
                .any(|case| case.left_function == function),
            "official manifest omitted {function:?}"
        );
        assert!(
            ADVERSARIAL_CASES.iter().any(|case| {
                case.left_function == function && case.comparisons.len() == LESS_ALL_SIX.len()
            }),
            "all-six adversarial manifest omitted {function:?}"
        );
    }

    for operation in [
        CompareOp::Eq,
        CompareOp::NotEq,
        CompareOp::Less,
        CompareOp::LessOrEqual,
        CompareOp::Greater,
        CompareOp::GreaterOrEqual,
    ] {
        assert!(
            all_cases().any(|case| {
                case.comparisons
                    .iter()
                    .any(|comparison| comparison.operation == operation)
            }),
            "manifest omitted {operation:?}"
        );
    }
    assert!(
        ADVERSARIAL_CASES
            .iter()
            .any(|case| case.left_shape == SourceShape::Null)
    );
    assert!(ADVERSARIAL_CASES.iter().any(|case| {
        case.left_function != case.right_function
            && case.left_function != ResidentTemporalValueFunction::Duration
    }));
    assert!(ADVERSARIAL_CASES.iter().any(|case| {
        case.left_function == ResidentTemporalValueFunction::Duration
            && case.right_function != ResidentTemporalValueFunction::Duration
    }));
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_18_temporal_comparison_selectors() {
    assert_certified_report_identities(OFFICIAL_CASES.iter().map(|case| {
        (
            usize::from(case.report_id.expect("official case has report id")),
            FEATURE.to_owned(),
            case.expanded_report_name()
                .expect("official case has an expanded report name"),
        )
    }));
}

#[test]
fn strict_cpu_runs_all_18_official_and_9_adversarial_cases_natively() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_no_failures("strict CPU reference", run_native_suite(&fixture, &backend));
    let observations = backend.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 27);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 27);
    assert_eq!(observations.legacy_route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
    assert_eq!(
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
        0
    );
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(receipts.len(), 27);
    assert!(
        receipts
            .iter()
            .all(|receipt| receipt.completion == BackendKind::Cpu)
    );
    Ok(())
}

#[test]
fn strict_cpu_runs_all_five_official_cross_family_duration_equalities() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    let failures = OFFICIAL_CASES[10..15]
        .iter()
        .filter_map(|case| {
            execute_native_case(&fixture, &backend, *case)
                .err()
                .map(|error| format!("{}: {error}", case.label()))
        })
        .collect();
    assert_no_failures("strict CPU cross-family equality", failures);
    let observations = backend.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 5);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 5);
    Ok(())
}

#[test]
fn cpu_backend_cannot_masquerade_as_metal_or_publish_comparison_answers() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.cpu_masquerading_as_metal()?;
    let observations = backend.observations();
    let error = match backend.pin_project(PROJECT) {
        Ok(_) => panic!("a CPU backend advertised as Metal unexpectedly pinned as real Metal"),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 0);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 0);
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
fn pinned_temporal_comparison_generation_rejects_every_mutation_surface() -> Result<()> {
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
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 0);
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
#[ignore = "focused hardware regression gate: requires real Metal"]
fn real_metal_runs_all_five_official_cross_family_duration_equalities() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let metal = fixture.real_metal_backend()?;
    let failures = OFFICIAL_CASES[10..15]
        .iter()
        .filter_map(|case| {
            execute_native_case(&fixture, &metal, *case)
                .err()
                .map(|error| format!("{}: {error}", case.label()))
        })
        .collect();
    assert_no_failures("real Metal cross-family equality", failures);
    let observations = metal.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 5);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 5);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: exact Temporal7 plus adversarial comparison suite requires real Metal"]
fn real_metal_matches_independent_cpu_with_exact_dispatch_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let cpu = fixture.strict_cpu_backend()?;
    let metal = fixture.real_metal_backend()?;
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for case in all_cases() {
        let cpu_result = execute_native_case(&fixture, &cpu, *case);
        let metal_result = execute_native_case(&fixture, &metal, *case);
        match (cpu_result, metal_result) {
            (Ok(cpu_values), Ok(metal_values)) if cpu_values == metal_values => {}
            (Ok(cpu_values), Ok(metal_values)) => failures.push(format!(
                "{}: CPU/Metal mismatch: CPU={cpu_values:?}, Metal={metal_values:?}",
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
    }
    assert_no_failures("strict CPU versus real Metal parity", failures);

    for (name, backend, completion) in [
        ("CPU", &cpu, BackendKind::Cpu),
        ("Metal", &metal, BackendKind::Metal),
    ] {
        let observations = backend.observations();
        assert_eq!(observations.pins.load(Ordering::SeqCst), 27, "{name}");
        assert_eq!(
            observations.temporal_value_calls.load(Ordering::SeqCst),
            27,
            "{name}"
        );
        assert_eq!(
            observations.legacy_route_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        assert_eq!(
            observations.unexpected_query_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        assert_eq!(
            observations
                .generation_mutation_attempts
                .load(Ordering::SeqCst),
            0,
            "{name}"
        );
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(receipts.len(), 27, "{name}");
        assert!(
            receipts.iter().all(|receipt| {
                receipt.completion == completion
                    && receipt.bookmark == fixture.bookmark
                    && receipt.graph_revision == fixture.graph.revision()
            }),
            "{name} receipt provenance or immutable fence changed"
        );
    }
    Ok(())
}
