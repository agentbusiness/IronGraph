// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use chrono::{NaiveDate, TimeZone};
use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentEntityBinding, ResidentGroup, ResidentGroupRequest,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodeBinding, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowOperation,
        ResidentRowProgramRequest, ResidentRowProgramResult, ResidentRowProgramResultParts,
        ResidentRowValueType, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
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
const MAX_RESULT_ROWS: usize = 64;
const FEATURE: &str = "features/clauses/with-orderBy/WithOrderBy2.feature";
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";

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
    let mut selected = BTreeSet::new();
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
struct DurationParts {
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
}

const DATE_DURATION: DurationParts = DurationParts {
    months: 1,
    days: 2,
    seconds: 0,
    nanos: 0,
};
const TIME_DURATION: DurationParts = DurationParts {
    months: 0,
    days: 0,
    seconds: 6 * 60,
    nanos: 0,
};
const DATETIME_DURATION: DurationParts = DurationParts {
    months: 0,
    days: 4,
    seconds: 6 * 60,
    nanos: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direction {
    Default,
    Asc,
    Ascending,
    Desc,
    Descending,
}

impl Direction {
    const fn suffix(self) -> &'static str {
        match self {
            Self::Default => "",
            Self::Asc => " ASC",
            Self::Ascending => " ASCENDING",
            Self::Desc => " DESC",
            Self::Descending => " DESCENDING",
        }
    }

    const fn descending(self) -> bool {
        matches!(self, Self::Desc | Self::Descending)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixtureKind {
    OfficialDate,
    OfficialLocalTime,
    OfficialZonedTime,
    OfficialLocalDateTime,
    OfficialZonedDateTime,
    DateMonthClamp,
    LocalTimeMidnight,
    ZonedTimeOffset,
    LocalDateTimeCarry,
    FixedOffsetZonedDateTime,
    NullPropagation,
    NamedZone,
}

impl FixtureKind {
    const fn property(self) -> &'static str {
        match self {
            Self::OfficialDate => "date",
            Self::OfficialLocalTime | Self::OfficialZonedTime => "time",
            Self::OfficialLocalDateTime | Self::OfficialZonedDateTime => "datetime",
            Self::DateMonthClamp | Self::NullPropagation => "d",
            Self::LocalTimeMidnight => "lt",
            Self::ZonedTimeOffset => "zt",
            Self::LocalDateTimeCarry => "ldt",
            Self::FixedOffsetZonedDateTime | Self::NamedZone => "zdt",
        }
    }

    const fn value_type(self) -> ResidentRowValueType {
        match self {
            Self::OfficialDate | Self::DateMonthClamp | Self::NullPropagation => {
                ResidentRowValueType::Date
            }
            Self::OfficialLocalTime | Self::LocalTimeMidnight => ResidentRowValueType::LocalTime,
            Self::OfficialZonedTime | Self::ZonedTimeOffset => ResidentRowValueType::ZonedTime,
            Self::OfficialLocalDateTime | Self::LocalDateTimeCarry => {
                ResidentRowValueType::LocalDateTime
            }
            Self::OfficialZonedDateTime | Self::FixedOffsetZonedDateTime | Self::NamedZone => {
                ResidentRowValueType::ZonedDateTime
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct DurationCase {
    report_id: Option<u16>,
    scenario: Option<u8>,
    name: &'static str,
    fixture: FixtureKind,
    duration_literal: &'static str,
    duration: DurationParts,
    direction: Direction,
    limit: usize,
    expected_ids: &'static [u64],
    project_shifted: bool,
}

impl DurationCase {
    fn expression(self) -> String {
        format!("a.{} + {}", self.fixture.property(), self.duration_literal)
    }

    fn query(self) -> String {
        let expression = self.expression();
        if self.project_shifted {
            format!(
                "MATCH (a)\nWITH a, {expression} AS shifted\n  ORDER BY shifted{}\n  LIMIT {}\nRETURN a, shifted",
                self.direction.suffix(),
                self.limit
            )
        } else {
            format!(
                "MATCH (a)\nWITH a\n  ORDER BY {expression}{}\n  LIMIT {}\nRETURN a",
                self.direction.suffix(),
                self.limit
            )
        }
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
            11 => "Sort by a date expression in ascending order",
            12 => "Sort by a date expression in descending order",
            13 => "Sort by a local time expression in ascending order",
            14 => "Sort by a local time expression in descending order",
            15 => "Sort by a time expression in ascending order",
            16 => "Sort by a time expression in descending order",
            17 => "Sort by a local date time expression in ascending order",
            18 => "Sort by a local date time expression in descending order",
            19 => "Sort by a date time expression in ascending order",
            20 => "Sort by a date time expression in descending order",
            _ => return None,
        };
        Some(format!(
            "[{scenario}] {title} [{}]",
            usize::from(report_id) + 1
        ))
    }
}

const fn official(
    report_id: u16,
    scenario: u8,
    name: &'static str,
    fixture: FixtureKind,
    duration_literal: &'static str,
    duration: DurationParts,
    direction: Direction,
    limit: usize,
    expected_ids: &'static [u64],
) -> DurationCase {
    DurationCase {
        report_id: Some(report_id),
        scenario: Some(scenario),
        name,
        fixture,
        duration_literal,
        duration,
        direction,
        limit,
        expected_ids,
        project_shifted: false,
    }
}

const OFFICIAL_CASES: [DurationCase; 25] = [
    official(
        1052,
        11,
        "date expression ascending (implicit)",
        FixtureKind::OfficialDate,
        "duration({months: 1, days: 2})",
        DATE_DURATION,
        Direction::Default,
        2,
        &[1, 5],
    ),
    official(
        1053,
        11,
        "date expression ascending (ASC)",
        FixtureKind::OfficialDate,
        "duration({months: 1, days: 2})",
        DATE_DURATION,
        Direction::Asc,
        2,
        &[1, 5],
    ),
    official(
        1054,
        11,
        "date expression ascending (ASCENDING)",
        FixtureKind::OfficialDate,
        "duration({months: 1, days: 2})",
        DATE_DURATION,
        Direction::Ascending,
        2,
        &[1, 5],
    ),
    official(
        1055,
        12,
        "date expression descending (DESC)",
        FixtureKind::OfficialDate,
        "duration({months: 1, days: 2})",
        DATE_DURATION,
        Direction::Desc,
        2,
        &[4, 3],
    ),
    official(
        1056,
        12,
        "date expression descending (DESCENDING)",
        FixtureKind::OfficialDate,
        "duration({months: 1, days: 2})",
        DATE_DURATION,
        Direction::Descending,
        2,
        &[4, 3],
    ),
    official(
        1057,
        13,
        "local-time expression ascending (implicit)",
        FixtureKind::OfficialLocalTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Default,
        3,
        &[1, 4, 2],
    ),
    official(
        1058,
        13,
        "local-time expression ascending (ASC)",
        FixtureKind::OfficialLocalTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Asc,
        3,
        &[1, 4, 2],
    ),
    official(
        1059,
        13,
        "local-time expression ascending (ASCENDING)",
        FixtureKind::OfficialLocalTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Ascending,
        3,
        &[1, 4, 2],
    ),
    official(
        1060,
        14,
        "local-time expression descending (DESC)",
        FixtureKind::OfficialLocalTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Desc,
        3,
        &[5, 3, 2],
    ),
    official(
        1061,
        14,
        "local-time expression descending (DESCENDING)",
        FixtureKind::OfficialLocalTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Descending,
        3,
        &[5, 3, 2],
    ),
    official(
        1062,
        15,
        "zoned-time expression ascending (implicit)",
        FixtureKind::OfficialZonedTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Default,
        3,
        &[4, 5, 2],
    ),
    official(
        1063,
        15,
        "zoned-time expression ascending (ASC)",
        FixtureKind::OfficialZonedTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Asc,
        3,
        &[4, 5, 2],
    ),
    official(
        1064,
        15,
        "zoned-time expression ascending (ASCENDING)",
        FixtureKind::OfficialZonedTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Ascending,
        3,
        &[4, 5, 2],
    ),
    official(
        1065,
        16,
        "zoned-time expression descending (DESC)",
        FixtureKind::OfficialZonedTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Desc,
        3,
        &[1, 3, 2],
    ),
    official(
        1066,
        16,
        "zoned-time expression descending (DESCENDING)",
        FixtureKind::OfficialZonedTime,
        "duration({minutes: 6})",
        TIME_DURATION,
        Direction::Descending,
        3,
        &[1, 3, 2],
    ),
    official(
        1067,
        17,
        "local-datetime expression ascending (implicit)",
        FixtureKind::OfficialLocalDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Default,
        3,
        &[3, 5, 1],
    ),
    official(
        1068,
        17,
        "local-datetime expression ascending (ASC)",
        FixtureKind::OfficialLocalDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Asc,
        3,
        &[3, 5, 1],
    ),
    official(
        1069,
        17,
        "local-datetime expression ascending (ASCENDING)",
        FixtureKind::OfficialLocalDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Ascending,
        3,
        &[3, 5, 1],
    ),
    official(
        1070,
        18,
        "local-datetime expression descending (DESC)",
        FixtureKind::OfficialLocalDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Desc,
        3,
        &[4, 2, 1],
    ),
    official(
        1071,
        18,
        "local-datetime expression descending (DESCENDING)",
        FixtureKind::OfficialLocalDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Descending,
        3,
        &[4, 2, 1],
    ),
    official(
        1072,
        19,
        "zoned-datetime expression ascending (implicit)",
        FixtureKind::OfficialZonedDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Default,
        3,
        &[3, 5, 2],
    ),
    official(
        1073,
        19,
        "zoned-datetime expression ascending (ASC)",
        FixtureKind::OfficialZonedDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Asc,
        3,
        &[3, 5, 2],
    ),
    official(
        1074,
        19,
        "zoned-datetime expression ascending (ASCENDING)",
        FixtureKind::OfficialZonedDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Ascending,
        3,
        &[3, 5, 2],
    ),
    official(
        1075,
        20,
        "zoned-datetime expression descending (DESC)",
        FixtureKind::OfficialZonedDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Desc,
        3,
        &[4, 1, 2],
    ),
    official(
        1076,
        20,
        "zoned-datetime expression descending (DESCENDING)",
        FixtureKind::OfficialZonedDateTime,
        "duration({days: 4, minutes: 6})",
        DATETIME_DURATION,
        Direction::Descending,
        3,
        &[4, 1, 2],
    ),
];

const ADVERSARIAL_CASES: [DurationCase; 6] = [
    DurationCase {
        report_id: None,
        scenario: None,
        name: "month-end clamp distinguishes leap and common February",
        fixture: FixtureKind::DateMonthClamp,
        duration_literal: "duration({months: 1})",
        duration: DurationParts {
            months: 1,
            days: 0,
            seconds: 0,
            nanos: 0,
        },
        direction: Direction::Asc,
        limit: 4,
        expected_ids: &[2, 1, 3, 4],
        project_shifted: true,
    },
    DurationCase {
        report_id: None,
        scenario: None,
        name: "local-time nanoseconds carry through midnight",
        fixture: FixtureKind::LocalTimeMidnight,
        duration_literal: "duration({nanoseconds: 200000000})",
        duration: DurationParts {
            months: 0,
            days: 0,
            seconds: 0,
            nanos: 200_000_000,
        },
        direction: Direction::Asc,
        limit: 4,
        expected_ids: &[2, 1, 3, 4],
        project_shifted: true,
    },
    DurationCase {
        report_id: None,
        scenario: None,
        name: "zoned-time preserves offsets and uses normalized instant plus offset ordering",
        fixture: FixtureKind::ZonedTimeOffset,
        duration_literal: "duration({minutes: 6})",
        duration: TIME_DURATION,
        direction: Direction::Asc,
        limit: 5,
        expected_ids: &[3, 1, 2, 4, 5],
        project_shifted: true,
    },
    DurationCase {
        report_id: None,
        scenario: None,
        name: "local-datetime applies calendar day before nanosecond carry",
        fixture: FixtureKind::LocalDateTimeCarry,
        duration_literal: "duration({days: 1, nanoseconds: 200000000})",
        duration: DurationParts {
            months: 0,
            days: 1,
            seconds: 0,
            nanos: 200_000_000,
        },
        direction: Direction::Asc,
        limit: 4,
        expected_ids: &[3, 1, 2, 4],
        project_shifted: true,
    },
    DurationCase {
        report_id: None,
        scenario: None,
        name: "fixed-offset zoned-datetime applies calendar clamp and nanosecond carry",
        fixture: FixtureKind::FixedOffsetZonedDateTime,
        duration_literal: "duration({months: 1, nanoseconds: 200000000})",
        duration: DurationParts {
            months: 1,
            days: 0,
            seconds: 0,
            nanos: 200_000_000,
        },
        direction: Direction::Asc,
        limit: 4,
        expected_ids: &[2, 1, 3, 4],
        project_shifted: true,
    },
    DurationCase {
        report_id: None,
        scenario: None,
        name: "missing temporal properties propagate null without fabricated payload",
        fixture: FixtureKind::NullPropagation,
        duration_literal: "duration({days: 1})",
        duration: DurationParts {
            months: 0,
            days: 1,
            seconds: 0,
            nanos: 0,
        },
        direction: Direction::Asc,
        limit: 3,
        expected_ids: &[1, 2, 3],
        project_shifted: true,
    },
];

const NAMED_ZONE_CASE: DurationCase = DurationCase {
    report_id: None,
    scenario: None,
    name: "named-zone calendar arithmetic succeeds natively or fails closed on Metal",
    fixture: FixtureKind::NamedZone,
    duration_literal: "duration({months: 1})",
    duration: DurationParts {
        months: 1,
        days: 0,
        seconds: 0,
        nanos: 0,
    },
    direction: Direction::Asc,
    limit: 3,
    expected_ids: &[2, 1, 3],
    project_shifted: true,
};

fn success_cases() -> impl Iterator<Item = &'static DurationCase> {
    OFFICIAL_CASES.iter().chain(ADVERSARIAL_CASES.iter())
}

fn all_cpu_cases() -> impl Iterator<Item = &'static DurationCase> {
    success_cases().chain(std::iter::once(&NAMED_ZONE_CASE))
}

fn epoch_day(year: i32, month: u32, day: u32) -> Result<ScalarValue> {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| Error::internal("duration-order fixture has an invalid date"))?;
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)
        .ok_or_else(|| Error::internal("chrono omitted the Unix epoch"))?;
    Ok(ScalarValue::Date(
        date.signed_duration_since(epoch).num_days(),
    ))
}

fn nanos_of_day(hour: u32, minute: u32, second: u32, nanos: u32) -> i64 {
    i64::from(hour * 3_600 + minute * 60 + second) * 1_000_000_000 + i64::from(nanos)
}

fn local_time(hour: u32, minute: u32, second: u32, nanos: u32) -> ScalarValue {
    ScalarValue::LocalTime(nanos_of_day(hour, minute, second, nanos))
}

fn zoned_time(hour: u32, minute: u32, second: u32, nanos: u32, offset: i32) -> ScalarValue {
    ScalarValue::ZonedTime {
        nanos: nanos_of_day(hour, minute, second, nanos),
        offset_seconds: offset,
    }
}

fn local_datetime(
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
    nanos: u32,
) -> Result<ScalarValue> {
    let value = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("duration-order fixture has an invalid local datetime"))?;
    Ok(ScalarValue::LocalDateTime {
        seconds: value.and_utc().timestamp(),
        nanos,
    })
}

fn fixed_zoned_datetime(
    date_time: (i32, u32, u32, u32, u32, u32, u32),
    offset_seconds: i32,
    timezone: &'static str,
) -> Result<ScalarValue> {
    let (year, month, day, hour, minute, second, nanos) = date_time;
    let local = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("duration-order fixture has an invalid zoned datetime"))?;
    let seconds = local
        .and_utc()
        .timestamp()
        .checked_sub(i64::from(offset_seconds))
        .ok_or_else(|| Error::internal("fixed-offset UTC conversion overflowed"))?;
    Ok(ScalarValue::ZonedDateTime {
        seconds,
        nanos,
        timezone: Arc::from(timezone),
    })
}

fn named_zoned_datetime(
    date_time: (i32, u32, u32, u32, u32, u32, u32),
    timezone: &'static str,
) -> Result<ScalarValue> {
    let (year, month, day, hour, minute, second, nanos) = date_time;
    let local = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("duration-order fixture has an invalid named datetime"))?;
    let zone = timezone
        .parse::<chrono_tz::Tz>()
        .map_err(|_| Error::internal("duration-order fixture has an unknown timezone"))?;
    let instant = zone
        .from_local_datetime(&local)
        .single()
        .ok_or_else(|| Error::internal("duration-order named local time is not unique"))?;
    Ok(ScalarValue::ZonedDateTime {
        seconds: instant.timestamp(),
        nanos,
        timezone: Arc::from(timezone),
    })
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    properties: BTreeMap<String, PropertyId>,
}

impl Fixture {
    fn new(kind: FixtureKind) -> Result<Self> {
        match kind {
            FixtureKind::OfficialDate => Self::from_rows(vec![
                ("A", vec![("date", epoch_day(1910, 5, 6)?)]),
                ("B", vec![("date", epoch_day(1980, 12, 24)?)]),
                ("C", vec![("date", epoch_day(1984, 10, 12)?)]),
                ("D", vec![("date", epoch_day(1985, 5, 6)?)]),
                ("E", vec![("date", epoch_day(1980, 10, 24)?)]),
                ("F", vec![("date", epoch_day(1984, 10, 11)?)]),
            ]),
            FixtureKind::OfficialLocalTime => Self::from_rows(vec![
                ("A", vec![("time", local_time(10, 35, 0, 0))]),
                ("B", vec![("time", local_time(12, 31, 14, 645_876_123))]),
                ("C", vec![("time", local_time(12, 31, 14, 645_876_124))]),
                ("D", vec![("time", local_time(12, 30, 14, 645_876_123))]),
                ("E", vec![("time", local_time(12, 31, 15, 0))]),
            ]),
            FixtureKind::OfficialZonedTime => Self::from_rows(vec![
                ("A", vec![("time", zoned_time(10, 35, 0, 0, -8 * 3_600))]),
                (
                    "B",
                    vec![("time", zoned_time(12, 31, 14, 645_876_123, 3_600))],
                ),
                (
                    "C",
                    vec![("time", zoned_time(12, 31, 14, 645_876_124, 3_600))],
                ),
                ("D", vec![("time", zoned_time(12, 35, 15, 0, 5 * 3_600))]),
                (
                    "E",
                    vec![("time", zoned_time(12, 30, 14, 645_876_123, 3_660))],
                ),
            ]),
            FixtureKind::OfficialLocalDateTime => Self::from_rows(vec![
                (
                    "A",
                    vec![("datetime", local_datetime(1984, 10, 11, 12, 30, 14, 12)?)],
                ),
                (
                    "B",
                    vec![(
                        "datetime",
                        local_datetime(1984, 10, 11, 12, 31, 14, 645_876_123)?,
                    )],
                ),
                (
                    "C",
                    vec![("datetime", local_datetime(1, 1, 1, 1, 1, 1, 1)?)],
                ),
                (
                    "D",
                    vec![(
                        "datetime",
                        local_datetime(9999, 9, 9, 9, 59, 59, 999_999_999)?,
                    )],
                ),
                (
                    "E",
                    vec![("datetime", local_datetime(1980, 12, 11, 12, 31, 14, 0)?)],
                ),
            ]),
            FixtureKind::OfficialZonedDateTime => Self::from_rows(vec![
                (
                    "A",
                    vec![(
                        "datetime",
                        fixed_zoned_datetime((1984, 10, 11, 12, 30, 14, 12), 15 * 60, "+00:15")?,
                    )],
                ),
                (
                    "B",
                    vec![(
                        "datetime",
                        fixed_zoned_datetime(
                            (1984, 10, 11, 12, 31, 14, 645_876_123),
                            17 * 60,
                            "+00:17",
                        )?,
                    )],
                ),
                (
                    "C",
                    vec![(
                        "datetime",
                        fixed_zoned_datetime(
                            (1, 1, 1, 1, 1, 1, 1),
                            -(11 * 3_600 + 59 * 60),
                            "-11:59",
                        )?,
                    )],
                ),
                (
                    "D",
                    vec![(
                        "datetime",
                        fixed_zoned_datetime(
                            (9999, 9, 9, 9, 59, 59, 999_999_999),
                            11 * 3_600 + 59 * 60,
                            "+11:59",
                        )?,
                    )],
                ),
                (
                    "E",
                    vec![(
                        "datetime",
                        fixed_zoned_datetime(
                            (1980, 12, 11, 12, 31, 14, 0),
                            -(11 * 3_600 + 59 * 60),
                            "-11:59",
                        )?,
                    )],
                ),
            ]),
            FixtureKind::DateMonthClamp => Self::from_rows(vec![
                ("A", vec![("d", epoch_day(2020, 1, 31)?)]),
                ("B", vec![("d", epoch_day(2019, 1, 31)?)]),
                ("C", vec![("d", epoch_day(2020, 2, 29)?)]),
                ("D", vec![]),
            ]),
            FixtureKind::LocalTimeMidnight => Self::from_rows(vec![
                ("A", vec![("lt", local_time(23, 59, 59, 900_000_000))]),
                ("B", vec![("lt", local_time(23, 59, 59, 800_000_000))]),
                ("C", vec![("lt", local_time(0, 0, 0, 0))]),
                ("D", vec![]),
            ]),
            FixtureKind::ZonedTimeOffset => Self::from_rows(vec![
                ("A", vec![("zt", zoned_time(12, 0, 0, 0, 0))]),
                ("B", vec![("zt", zoned_time(13, 0, 0, 0, 3_600))]),
                ("C", vec![("zt", zoned_time(11, 0, 0, 0, -3_600))]),
                ("D", vec![("zt", zoned_time(12, 0, 0, 1, 0))]),
                ("E", vec![]),
            ]),
            FixtureKind::LocalDateTimeCarry => Self::from_rows(vec![
                (
                    "A",
                    vec![("ldt", local_datetime(2020, 1, 1, 23, 59, 59, 900_000_000)?)],
                ),
                ("B", vec![("ldt", local_datetime(2020, 1, 2, 0, 0, 0, 0)?)]),
                (
                    "C",
                    vec![("ldt", local_datetime(2020, 1, 1, 23, 59, 59, 700_000_000)?)],
                ),
                ("D", vec![]),
            ]),
            FixtureKind::FixedOffsetZonedDateTime => Self::from_rows(vec![
                (
                    "A",
                    vec![(
                        "zdt",
                        fixed_zoned_datetime(
                            (2020, 1, 31, 23, 59, 59, 900_000_000),
                            5 * 3_600 + 30 * 60,
                            "+05:30",
                        )?,
                    )],
                ),
                (
                    "B",
                    vec![(
                        "zdt",
                        fixed_zoned_datetime(
                            (2019, 1, 31, 23, 59, 59, 900_000_000),
                            5 * 3_600 + 30 * 60,
                            "+05:30",
                        )?,
                    )],
                ),
                (
                    "C",
                    vec![(
                        "zdt",
                        fixed_zoned_datetime(
                            (2020, 2, 29, 0, 0, 0, 0),
                            5 * 3_600 + 30 * 60,
                            "+05:30",
                        )?,
                    )],
                ),
                ("D", vec![]),
            ]),
            FixtureKind::NullPropagation => Self::from_rows(vec![
                ("A", vec![("d", epoch_day(1970, 1, 1)?)]),
                ("B", vec![]),
                ("C", vec![]),
            ]),
            FixtureKind::NamedZone => Self::from_rows(vec![
                (
                    "A",
                    vec![(
                        "zdt",
                        named_zoned_datetime((2020, 2, 29, 12, 0, 0, 0), "America/New_York")?,
                    )],
                ),
                (
                    "B",
                    vec![(
                        "zdt",
                        named_zoned_datetime((2020, 1, 31, 12, 0, 0, 0), "America/New_York")?,
                    )],
                ),
                ("C", vec![]),
            ]),
        }
    }

    fn from_rows(rows: Vec<(&'static str, Vec<(&'static str, ScalarValue)>)>) -> Result<Self> {
        let mut graph = GraphStore::default();
        let mut labels = BTreeMap::new();
        let mut properties = BTreeMap::new();
        for (label, row_properties) in &rows {
            labels.insert(*label, graph.catalog_mut().intern_label(label)?);
            for (name, _) in row_properties {
                properties.insert(
                    (*name).to_owned(),
                    graph.catalog_mut().intern_property(name)?,
                );
            }
        }
        for (offset, (label, row_properties)) in rows.into_iter().enumerate() {
            let id = NodeId(
                u64::try_from(offset + 1)
                    .map_err(|_| Error::internal("duration-order fixture node ID overflowed"))?,
            );
            let row_properties = row_properties
                .into_iter()
                .map(|(name, value)| {
                    properties
                        .get(name)
                        .copied()
                        .map(|property| (property, value))
                        .ok_or_else(|| Error::internal("duration-order property disappeared"))
                })
                .collect::<Result<Vec<_>>>()?;
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision: id.0,
                labels: vec![
                    *labels
                        .get(label)
                        .ok_or_else(|| Error::internal("duration-order label disappeared"))?,
                ],
                properties: row_properties,
            })?;
        }
        let bookmark = Bookmark {
            term: 29,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            properties,
        })
    }

    fn property(&self, name: &str) -> Result<PropertyId> {
        self.properties
            .get(name)
            .copied()
            .ok_or_else(|| Error::internal(format!("fixture omitted property `{name}`")))
    }

    fn value(&self, id: NodeId, property: &str) -> Result<ScalarValue> {
        let node = self
            .graph
            .node(id)
            .ok_or_else(|| Error::internal(format!("fixture omitted node {id}")))?;
        Ok(node
            .property(self.property(property)?)
            .unwrap_or(ScalarValue::Null))
    }

    fn maximum_timezone_bytes(&self, property: &str) -> Result<u32> {
        let property = self.property(property)?;
        self.graph
            .nodes()
            .filter_map(|node| match node.property(property) {
                Some(ScalarValue::ZonedDateTime { timezone, .. }) => Some(timezone.len()),
                _ => None,
            })
            .max()
            .map_or(Ok(0), |bytes| {
                u32::try_from(bytes)
                    .map_err(|_| Error::internal("fixture timezone width exceeds u32"))
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
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(cpu)
    }

    fn strict_cpu_backend(&self) -> Result<ObservedRowBackend> {
        ObservedRowBackend::strict_cpu_reference(self.cpu()?)
    }

    fn wrong_receipt_backend(&self) -> Result<ObservedRowBackend> {
        ObservedRowBackend::wrong_receipt_provenance(self.cpu()?)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedRowBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedRowBackend::real_metal(metal)
    }
}

#[derive(Default)]
struct RowObservations {
    pins: AtomicUsize,
    row_program_calls: AtomicUsize,
    old_node_pipeline_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentRowProgramRequest>>,
    results: Mutex<Vec<ResidentRowProgramResult>>,
}

/// Fail-closed observer around the only accepted query route.
///
/// Before pinning, the strict CPU reference deliberately advertises Metal so the planner cannot
/// select the generic CPU evaluator. The immutable pinned generation then reports its real CPU
/// provenance. Real Metal reports Metal throughout. Every backend route other than one complete
/// typed-row program is rejected and counted.
struct ObservedRowBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RowObservations>,
}

impl ObservedRowBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    fn wrong_receipt_provenance(inner: CpuBackend) -> Result<Self> {
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
                "temporal-duration test did not construct a real Metal backend",
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
            Error::internal("temporal-duration backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal-duration backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RowObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RowObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict temporal-duration test rejected `{route}` execution"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement duration-order project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement duration-order project has no graph revision")
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedRowBackend {
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
                "pinned duration-order generation does not match its immutable fence",
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
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
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
        self.observations
            .old_node_pipeline_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject_query_route("execute_node_pipeline")
    }

    fn execute_row_program(
        &self,
        request: &ResidentRowProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        if !self.pinned {
            return self.reject_query_route("execute_row_program_without_pin");
        }
        if request.project != PROJECT
            || request.expected_bookmark != self.expected_bookmark
            || request.expected_graph_revision != self.expected_graph_revision
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "duration-order row request does not belong to the pinned generation",
            ));
        }
        self.observations
            .row_program_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let result = self.inner.execute_row_program(request, cancellation)?;
        self.observations
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(result.clone());
        Ok(result)
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
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: backend.is_some(),
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 3,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn expected_shifted_values(kind: FixtureKind) -> Result<Vec<ScalarValue>> {
    match kind {
        FixtureKind::DateMonthClamp => Ok(vec![
            epoch_day(2019, 2, 28)?,
            epoch_day(2020, 2, 29)?,
            epoch_day(2020, 3, 29)?,
            ScalarValue::Null,
        ]),
        FixtureKind::LocalTimeMidnight => Ok(vec![
            local_time(0, 0, 0, 0),
            local_time(0, 0, 0, 100_000_000),
            local_time(0, 0, 0, 200_000_000),
            ScalarValue::Null,
        ]),
        FixtureKind::ZonedTimeOffset => Ok(vec![
            zoned_time(11, 6, 0, 0, -3_600),
            zoned_time(12, 6, 0, 0, 0),
            zoned_time(13, 6, 0, 0, 3_600),
            zoned_time(12, 6, 0, 1, 0),
            ScalarValue::Null,
        ]),
        FixtureKind::LocalDateTimeCarry => Ok(vec![
            local_datetime(2020, 1, 2, 23, 59, 59, 900_000_000)?,
            local_datetime(2020, 1, 3, 0, 0, 0, 100_000_000)?,
            local_datetime(2020, 1, 3, 0, 0, 0, 200_000_000)?,
            ScalarValue::Null,
        ]),
        FixtureKind::FixedOffsetZonedDateTime => Ok(vec![
            fixed_zoned_datetime(
                (2019, 3, 1, 0, 0, 0, 100_000_000),
                5 * 3_600 + 30 * 60,
                "+05:30",
            )?,
            fixed_zoned_datetime(
                (2020, 3, 1, 0, 0, 0, 100_000_000),
                5 * 3_600 + 30 * 60,
                "+05:30",
            )?,
            fixed_zoned_datetime(
                (2020, 3, 29, 0, 0, 0, 200_000_000),
                5 * 3_600 + 30 * 60,
                "+05:30",
            )?,
            ScalarValue::Null,
        ]),
        FixtureKind::NullPropagation => Ok(vec![
            epoch_day(1970, 1, 2)?,
            ScalarValue::Null,
            ScalarValue::Null,
        ]),
        FixtureKind::NamedZone => Ok(vec![
            named_zoned_datetime((2020, 2, 29, 12, 0, 0, 0), "America/New_York")?,
            named_zoned_datetime((2020, 3, 29, 12, 0, 0, 0), "America/New_York")?,
            ScalarValue::Null,
        ]),
        _ => Err(Error::internal(
            "official duration-order case does not project its hidden sort expression",
        )),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedRow {
    node: NodeId,
    source: ScalarValue,
    shifted: Option<ScalarValue>,
}

fn observe_output(
    output: &ExecutionOutput,
    fixture: &Fixture,
    case: DurationCase,
) -> Result<Vec<ObservedRow>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(
            "temporal-duration ORDER BY produced side effects or truncation",
        ));
    }
    let expected_schema = if case.project_shifted {
        vec![
            ("a".to_owned(), ColumnType::Node),
            ("shifted".to_owned(), ColumnType::Temporal),
        ]
    } else {
        vec![("a".to_owned(), ColumnType::Node)]
    };
    if output.result.schema != expected_schema {
        return Err(Error::internal(format!(
            "temporal-duration result schema is wrong: {:?}",
            output.result.schema
        )));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        let nodes = batch
            .columns
            .iter()
            .find(|column| column.name == "a")
            .ok_or_else(|| Error::internal("duration-order result omitted node column `a`"))?;
        if nodes.value_type != ColumnType::Node || nodes.values.len() != batch.row_count {
            return Err(Error::internal(
                "duration-order node batch shape is inconsistent",
            ));
        }
        let shifted = if case.project_shifted {
            let column = batch
                .columns
                .iter()
                .find(|column| column.name == "shifted")
                .ok_or_else(|| Error::internal("duration-order result omitted `shifted`"))?;
            if column.value_type != ColumnType::Temporal || column.values.len() != batch.row_count {
                return Err(Error::internal(
                    "duration-order shifted batch shape is inconsistent",
                ));
            }
            Some(column)
        } else {
            None
        };
        if batch.columns.len() != usize::from(case.project_shifted) + 1 {
            return Err(Error::internal(
                "duration-order result contains an unexpected host projection",
            ));
        }
        for row in 0..batch.row_count {
            let ResultValue::Node(node) = &nodes.values[row] else {
                return Err(Error::internal(
                    "duration-order result contains a non-node `a`",
                ));
            };
            let source = node
                .properties
                .get(case.fixture.property())
                .cloned()
                .unwrap_or(ScalarValue::Null);
            if source != fixture.value(node.id, case.fixture.property())? {
                return Err(Error::internal(
                    "duration-order node projection changed its canonical source property",
                ));
            }
            let shifted = shifted
                .map(|column| match &column.values[row] {
                    ResultValue::Scalar(value)
                        if matches!(
                            value,
                            ScalarValue::Null
                                | ScalarValue::Date(_)
                                | ScalarValue::LocalTime(_)
                                | ScalarValue::ZonedTime { .. }
                                | ScalarValue::LocalDateTime { .. }
                                | ScalarValue::ZonedDateTime { .. }
                        ) =>
                    {
                        Ok(value.clone())
                    }
                    _ => Err(Error::internal(
                        "duration-order projected a non-temporal shifted value",
                    )),
                })
                .transpose()?;
            rows.push(ObservedRow {
                node: node.id,
                source,
                shifted,
            });
        }
    }
    Ok(rows)
}

fn expected_output(fixture: &Fixture, case: DurationCase) -> Result<Vec<ObservedRow>> {
    let shifted = if case.project_shifted {
        Some(expected_shifted_values(case.fixture)?)
    } else {
        None
    };
    case.expected_ids
        .iter()
        .copied()
        .enumerate()
        .map(|(position, id)| {
            let node = NodeId(id);
            Ok(ObservedRow {
                node,
                source: fixture.value(node, case.fixture.property())?,
                shifted: shifted.as_ref().map(|values| values[position].clone()),
            })
        })
        .collect()
}

fn run_cpu_oracle_case(fixture: &Fixture, case: DurationCase) -> Result<Vec<ObservedRow>> {
    let query = case.query();
    let output = QueryEngine.execute(&query, &mut context(fixture, None))?;
    let actual = observe_output(&output, fixture, case)?;
    assert_eq!(actual, expected_output(fixture, case)?, "{query}");
    Ok(actual)
}

fn assert_property_loader(
    fixture: &Fixture,
    case: DurationCase,
    operation: &ResidentRowOperation,
) -> Result<()> {
    let expected_binding = ResidentEntityBinding::Node(ResidentNodeBinding::Start);
    let expected_property = fixture.property(case.fixture.property())?;
    match (case.fixture.value_type(), operation) {
        (
            ResidentRowValueType::Date,
            ResidentRowOperation::LoadDateProperty { binding, property },
        )
        | (
            ResidentRowValueType::LocalTime,
            ResidentRowOperation::LoadLocalTimeProperty { binding, property },
        )
        | (
            ResidentRowValueType::ZonedTime,
            ResidentRowOperation::LoadZonedTimeProperty { binding, property },
        )
        | (
            ResidentRowValueType::LocalDateTime,
            ResidentRowOperation::LoadLocalDateTimeProperty { binding, property },
        ) => {
            assert_eq!(*binding, expected_binding);
            assert_eq!(*property, expected_property);
        }
        (
            ResidentRowValueType::ZonedDateTime,
            ResidentRowOperation::LoadZonedDateTimeProperty {
                binding,
                property,
                maximum_timezone_bytes,
            },
        ) => {
            assert_eq!(*binding, expected_binding);
            assert_eq!(*property, expected_property);
            assert_eq!(
                *maximum_timezone_bytes,
                fixture.maximum_timezone_bytes(case.fixture.property())?
            );
        }
        _ => {
            return Err(Error::internal(
                "duration-order program did not begin with its exact temporal property loader",
            ));
        }
    }
    Ok(())
}

fn assert_request(
    fixture: &Fixture,
    case: DurationCase,
    request: &ResidentRowProgramRequest,
) -> Result<()> {
    request.validate()?;
    assert_eq!(request.project, PROJECT);
    assert_eq!(request.expected_bookmark, fixture.bookmark);
    assert_eq!(request.expected_graph_revision, fixture.graph.revision());
    assert_eq!(
        request.expected_layout_version,
        fixture.graph.layout_version()
    );
    assert!(request.execution.high != 0 || request.execution.low != 0);
    assert_eq!(request.input.project, PROJECT);
    assert_eq!(request.input.layers, LayerMask::AUTHORITY);
    assert!(request.input.labels.is_empty());
    assert!(!request.input.initial_optional);
    assert!(request.input.expansion.is_none());
    assert!(request.input.continuations.is_empty());
    assert!(request.input.correlated_optional.is_none());
    assert!(request.input.relationship_null_filter.is_none());
    assert!(request.input.predicates.is_empty());
    assert!(request.input.property_filters.is_empty());
    assert!(request.input.value_matrix.is_none());
    assert!(request.input.mutation.is_none());
    assert!(request.input.orders.is_empty());
    assert_eq!(request.input.offset, 0);
    assert_eq!(request.input.limit, usize::MAX);
    assert!(request.input.integer_projections.is_empty());
    assert!(request.input.property_null_projections.is_empty());
    assert_eq!(
        request.input.max_output_rows,
        fixture.graph.node_slot_count()
    );
    assert_eq!(request.offset, 0);
    assert_eq!(request.limit, case.limit);
    assert_eq!(request.max_output_rows, MAX_RESULT_ROWS);

    assert_eq!(request.program.instructions.len(), 2);
    assert_eq!(
        request.program.instructions[0].output_type,
        case.fixture.value_type()
    );
    assert_property_loader(fixture, case, &request.program.instructions[0].operation)?;
    assert_eq!(
        request.program.instructions[1].output_type,
        case.fixture.value_type()
    );
    assert_eq!(
        request.program.instructions[1].operation,
        ResidentRowOperation::TemporalAddDuration {
            temporal: 0,
            months: case.duration.months,
            days: case.duration.days,
            seconds: case.duration.seconds,
            nanos: case.duration.nanos,
        }
    );
    assert_eq!(request.sort_keys.len(), 1);
    assert_eq!(request.sort_keys[0].register, 1);
    assert_eq!(request.sort_keys[0].descending, case.direction.descending());
    assert_eq!(
        request.sort_keys[0].nulls_first,
        case.direction.descending()
    );
    if case.project_shifted {
        assert_eq!(request.final_registers, vec![1]);
    } else {
        assert!(request.final_registers.is_empty());
    }
    assert_eq!(request.manifest.instruction_obligations.len(), 2);
    assert_eq!(request.obligations().collect::<Vec<_>>().len(), 3);
    assert_eq!(
        request.obligations().collect::<BTreeSet<_>>().len(),
        3,
        "expression and sort receipts must be unique"
    );
    Ok(())
}

fn assert_result(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: DurationCase,
    request: &ResidentRowProgramRequest,
    result: &ResidentRowProgramResultParts,
) -> Result<()> {
    assert_eq!(result.project, PROJECT);
    assert_eq!(result.execution, request.execution);
    assert_eq!(result.bookmark, fixture.bookmark);
    assert_eq!(result.graph_revision, fixture.graph.revision());
    assert_eq!(result.layout_version, fixture.graph.layout_version());
    assert_eq!(result.manifest_fingerprint, request.manifest.fingerprint);
    assert_eq!(result.input_cardinality, fixture.graph.node_count());
    let expected_dense = case
        .expected_ids
        .iter()
        .map(|id| u32::try_from(id - 1).expect("tiny fixture row fits u32"))
        .collect::<Vec<_>>();
    let expected_positions = expected_dense
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(result.source_positions, expected_positions);
    assert_eq!(result.rows.start_rows, expected_dense);
    assert!(result.rows.intermediate_node_rows.is_empty());
    assert!(result.rows.intermediate_edge_rows.is_empty());
    assert!(result.rows.edge_rows.is_empty());
    assert!(result.rows.end_rows.is_empty());
    assert!(result.rows.integer_columns.is_empty());
    assert!(result.rows.boolean_columns.is_empty());
    assert!(result.rows.value_left_indices.is_empty());
    assert!(result.rows.value_right_indices.is_empty());
    assert!(result.rows.mutation.is_none());
    if case.project_shifted {
        assert_eq!(result.projected_columns.len(), 1);
        assert_eq!(result.projected_columns[0].register, 1);
    } else {
        assert!(result.projected_columns.is_empty());
    }
    assert_eq!(
        result.scratch_bytes,
        request.scratch_bytes(result.input_cardinality)?
    );

    let completion = match backend.actual_kind {
        BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
        BackendKind::Metal => ResidentDeviceCompletion::Metal,
        kind => return Err(Error::internal(format!("unexpected backend kind {kind:?}"))),
    };
    let expected_obligations = request.obligations().collect::<BTreeSet<_>>();
    let actual_obligations = result
        .receipts
        .iter()
        .map(|receipt| receipt.obligation)
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_obligations, expected_obligations);
    assert_eq!(result.receipts.len(), 3);
    for receipt in &result.receipts {
        assert_eq!(receipt.execution, request.execution);
        assert_eq!(receipt.completion, completion);
        assert_eq!(receipt.input_cardinality, result.input_cardinality as u64);
        let expected_rows = if receipt.obligation == request.manifest.sort_obligation {
            case.expected_ids.len() as u64
        } else {
            result.input_cardinality as u64
        };
        assert_eq!(receipt.output_cardinality, expected_rows);
    }
    Ok(())
}

struct NativeRun {
    rows: Vec<ObservedRow>,
    request: ResidentRowProgramRequest,
    result: ResidentRowProgramResultParts,
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: DurationCase,
) -> std::result::Result<NativeRun, String> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.row_program_calls.load(Ordering::SeqCst);
    let old_before = observations.old_node_pipeline_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let results_before = observations
        .results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let query = case.query();
    let output = QueryEngine
        .execute(&query, &mut context(fixture, Some(backend)))
        .map_err(|error| {
            format!(
                "query failed with {:?}: {error}; pins={}, row_program_calls={}, old_pipeline_calls={}, unexpected_calls={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.row_program_calls.load(Ordering::SeqCst) - calls_before,
                observations.old_node_pipeline_calls.load(Ordering::SeqCst) - old_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations.row_program_calls.load(Ordering::SeqCst) != calls_before + 1 {
        return Err(
            "query did not cross exactly one complete native row-program boundary".to_owned(),
        );
    }
    if observations.old_node_pipeline_calls.load(Ordering::SeqCst) != old_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered a generic, legacy, CPU-fallback, graph-pipeline, or host route"
                .to_owned(),
        );
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one native row request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "native row request disappeared".to_owned())?
    };
    let result = {
        let results = observations
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if results.len() != results_before + 1 {
            return Err("observer did not retain exactly one native row result".to_owned());
        }
        results
            .last()
            .cloned()
            .ok_or_else(|| "native row result disappeared".to_owned())?
            .into_untrusted_parts()
    };
    assert_request(fixture, case, &request)
        .map_err(|error| format!("native request assertion failed: {error}"))?;
    assert_result(fixture, backend, case, &request, &result)
        .map_err(|error| format!("native result/provenance assertion failed: {error}"))?;
    let rows = observe_output(&output, fixture, case)
        .map_err(|error| format!("native output inspection failed: {error}"))?;
    let expected = expected_output(fixture, case)
        .map_err(|error| format!("cannot build pinned expected rows: {error}"))?;
    if rows != expected {
        return Err(format!(
            "duration-order output mismatch: expected {expected:?}, got {rows:?}"
        ));
    }
    Ok(NativeRun {
        rows,
        request,
        result,
    })
}

fn run_native_suite(
    backend: &mut ObservedRowBackend,
    cases: impl IntoIterator<Item = &'static DurationCase>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for case in cases {
        let fixture = match Fixture::new(case.fixture) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: cannot build fixture: {error}", case.label()));
                continue;
            }
        };
        if let Err(error) = fixture
            .image()
            .and_then(|image| backend.replace_all_projects(vec![image]))
        {
            failures.push(format!(
                "{}: cannot replace resident generation: {error}",
                case.label()
            ));
            continue;
        }
        if let Err(error) = execute_native_case(&fixture, backend, *case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} temporal-property + constant-duration ORDER BY suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_tck_ids_1052_through_1076_and_scenarios_11_through_20() {
    assert_eq!(OFFICIAL_CASES.len(), 25);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.report_id)
            .collect::<Vec<_>>(),
        (1052_u16..=1076).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.scenario)
            .collect::<Vec<_>>(),
        vec![
            11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 15, 16, 16, 17, 17, 17, 18, 18, 19, 19,
            19, 20, 20,
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| (format!("{:?}", case.fixture), case.query()))
            .collect::<BTreeSet<_>>()
            .len(),
        25
    );
    for case in OFFICIAL_CASES {
        let query = case.query();
        assert!(query.starts_with("MATCH (a)\nWITH a\n  ORDER BY a."));
        assert!(query.contains(" + duration({"));
        assert!(query.ends_with("RETURN a"));
        assert!(!query.contains("AS shifted"));
        assert_eq!(case.expected_ids.len(), case.limit);
        assert!(!case.name.is_empty());
    }
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.direction)
            .collect::<Vec<_>>(),
        vec![
            Direction::Default,
            Direction::Asc,
            Direction::Ascending,
            Direction::Desc,
            Direction::Descending,
            Direction::Default,
            Direction::Asc,
            Direction::Ascending,
            Direction::Desc,
            Direction::Descending,
            Direction::Default,
            Direction::Asc,
            Direction::Ascending,
            Direction::Desc,
            Direction::Descending,
            Direction::Default,
            Direction::Asc,
            Direction::Ascending,
            Direction::Desc,
            Direction::Descending,
            Direction::Default,
            Direction::Asc,
            Direction::Ascending,
            Direction::Desc,
            Direction::Descending,
        ]
    );
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_25_duration_order_selectors() {
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
fn cpu_oracle_matches_all_25_pinned_official_duration_order_examples() -> Result<()> {
    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        match Fixture::new(case.fixture)
            .and_then(|fixture| run_cpu_oracle_case(&fixture, case).map(|_| ()))
        {
            Ok(()) => {}
            Err(error) => failures.push(format!("{}: {error}", case.label())),
        }
    }
    assert_no_failures("CPU oracle (official IDs 1053-1077)", failures);
    Ok(())
}

#[test]
fn cpu_oracle_pins_all_requested_temporal_duration_edge_semantics() -> Result<()> {
    let mut failures = Vec::new();
    for case in ADVERSARIAL_CASES
        .iter()
        .chain(std::iter::once(&NAMED_ZONE_CASE))
    {
        match Fixture::new(case.fixture)
            .and_then(|fixture| run_cpu_oracle_case(&fixture, *case).map(|_| ()))
        {
            Ok(()) => {}
            Err(error) => failures.push(format!("{}: {error}", case.label())),
        }
    }
    assert_no_failures("CPU oracle (seven focused edge cases)", failures);
    Ok(())
}

#[test]
fn strict_cpu_reference_runs_all_25_official_and_7_edge_cases_natively() -> Result<()> {
    let first = Fixture::new(FixtureKind::OfficialDate)?;
    let mut backend = first.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_no_failures(
        "strict CPU reference",
        run_native_suite(&mut backend, all_cpu_cases()),
    );
    Ok(())
}

#[test]
fn temporal_duration_receipts_reject_wrong_device_provenance_before_publication() -> Result<()> {
    let fixture = Fixture::new(FixtureKind::OfficialDate)?;
    let case = OFFICIAL_CASES[0];
    let backend = fixture.wrong_receipt_backend()?;
    let observations = backend.observations();
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            &case.query(),
            &mut context(&fixture, Some(&backend)),
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("CPU-reference receipts reported as Metal must not be published");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.row_program_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        observations.old_node_pipeline_calls.load(Ordering::SeqCst),
        0
    );
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
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
fn assert_named_zone_is_native_or_fails_closed(
    fixture: &Fixture,
    cpu_rows: &[ObservedRow],
    metal: &ObservedRowBackend,
) -> Result<()> {
    let case = NAMED_ZONE_CASE;
    let observations = metal.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.row_program_calls.load(Ordering::SeqCst);
    let old_before = observations.old_node_pipeline_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let results_before = observations
        .results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let mut emitted = 0_usize;
    let execution = QueryEngine.execute_streaming(
        &case.query(),
        &mut context(fixture, Some(metal)),
        &mut |_| {
            emitted += 1;
            Ok(())
        },
    );

    assert_eq!(observations.pins.load(Ordering::SeqCst), pins_before + 1);
    assert_eq!(
        observations.row_program_calls.load(Ordering::SeqCst),
        calls_before + 1
    );
    assert_eq!(
        observations.old_node_pipeline_calls.load(Ordering::SeqCst),
        old_before
    );
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        unexpected_before
    );
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(requests.len(), requests_before + 1);
        requests
            .last()
            .cloned()
            .ok_or_else(|| Error::internal("named-zone Metal request disappeared"))?
    };
    assert_request(fixture, case, &request)?;

    match execution {
        Err(error) => {
            assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
            assert!(
                error.to_string().contains("fixed-offset timezone"),
                "named-zone rejection must explain the native limitation: {error}"
            );
            assert_eq!(emitted, 0, "failed native Metal work published stream data");
            assert_eq!(
                observations
                    .results
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len(),
                results_before,
                "failed Metal work fabricated a completed receipt packet"
            );
        }
        Ok(_) => {
            assert!(emitted >= 2, "successful streaming omitted schema or rows");
            let result = {
                let results = observations
                    .results
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                assert_eq!(results.len(), results_before + 1);
                results
                    .last()
                    .cloned()
                    .ok_or_else(|| Error::internal("named-zone Metal result disappeared"))?
                    .into_untrusted_parts()
            };
            assert_result(fixture, metal, case, &request, &result)?;
            let native = execute_native_case(fixture, metal, case)
                .map_err(|error| Error::internal(format!("named-zone Metal parity: {error}")))?;
            assert_eq!(native.rows, cpu_rows);
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: 25 official and 6 adversarial cases require real Metal"]
fn real_metal_matches_strict_cpu_and_named_zones_never_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = Fixture::new(FixtureKind::OfficialDate)?;
    let mut cpu = first.strict_cpu_backend()?;
    let mut metal = first.real_metal_backend()?;
    assert_eq!(cpu.kind(), BackendKind::Metal);
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.kind(), BackendKind::Metal);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for case in success_cases() {
        let fixture = match Fixture::new(case.fixture) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: cannot build fixture: {error}", case.label()));
                continue;
            }
        };
        let image = fixture.image()?;
        cpu.replace_all_projects(vec![image.clone()])?;
        metal.replace_all_projects(vec![image])?;
        let cpu_run = execute_native_case(&fixture, &cpu, *case);
        let metal_run = execute_native_case(&fixture, &metal, *case);
        match (cpu_run, metal_run) {
            (Ok(cpu_run), Ok(metal_run)) => {
                if cpu_run.rows != metal_run.rows
                    || cpu_run.request.program != metal_run.request.program
                    || cpu_run.request.sort_keys != metal_run.request.sort_keys
                    || cpu_run.request.final_registers != metal_run.request.final_registers
                    || cpu_run.result.source_positions != metal_run.result.source_positions
                    || cpu_run.result.projected_columns != metal_run.result.projected_columns
                {
                    failures.push(format!(
                        "{}: strict CPU and real Metal native artifacts differ",
                        case.label()
                    ));
                }
            }
            (Err(cpu_error), Ok(_)) => failures.push(format!(
                "{}: strict CPU failed while Metal succeeded: {cpu_error}",
                case.label()
            )),
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

    let fixture = Fixture::new(FixtureKind::NamedZone)?;
    let image = fixture.image()?;
    cpu.replace_all_projects(vec![image.clone()])?;
    metal.replace_all_projects(vec![image])?;
    let cpu_named = execute_native_case(&fixture, &cpu, NAMED_ZONE_CASE)
        .map_err(|error| Error::internal(format!("named-zone strict CPU failed: {error}")))?;
    assert_named_zone_is_native_or_fails_closed(&fixture, &cpu_named.rows, &metal)
}
