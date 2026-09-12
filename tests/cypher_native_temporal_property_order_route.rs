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

use chrono::NaiveDate;
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
        ResidentRowProgramResultParts, ResidentSortRequest, ResidentSortResult,
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
const MAX_RESULT_ROWS: usize = 64;
const FEATURE: &str = "features/clauses/with-orderBy/WithOrderBy1.feature";
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
enum FixtureKind {
    Date,
    LocalTime,
    ZonedTime,
    LocalDateTime,
    ZonedDateTime,
    Adversarial,
}

impl FixtureKind {
    const ALL: [Self; 6] = [
        Self::Date,
        Self::LocalTime,
        Self::ZonedTime,
        Self::LocalDateTime,
        Self::ZonedDateTime,
        Self::Adversarial,
    ];
}

#[derive(Clone, Copy, Debug)]
struct TemporalCase {
    tck_id: Option<u16>,
    scenario: Option<u8>,
    name: &'static str,
    fixture: FixtureKind,
    property: &'static str,
    alias: &'static str,
    sort: &'static str,
    limit: usize,
    expected_ids: &'static [u64],
}

impl TemporalCase {
    fn query(self) -> String {
        format!(
            "MATCH (a) WITH a, a.{} AS {} WITH a, {} ORDER BY {} LIMIT {} RETURN a, {}",
            self.property, self.alias, self.alias, self.sort, self.limit, self.alias
        )
    }

    fn descending(self) -> bool {
        self.sort.contains("DESC")
    }

    fn label(self) -> String {
        match (self.tck_id, self.scenario) {
            (Some(id), Some(scenario)) => format!(
                "TCK {id} {FEATURE} [{scenario}] {} ({})",
                self.name, self.sort
            ),
            _ => format!("adversarial {} ({})", self.name, self.sort),
        }
    }

    fn expanded_report_name(self) -> Option<String> {
        let report_id = self.tck_id?;
        let scenario = self.scenario?;
        Some(format!(
            "[{scenario}] {} [{}]",
            self.name,
            usize::from(report_id) + 1
        ))
    }
}

const OFFICIAL_CASES: [TemporalCase; 25] = [
    TemporalCase {
        tck_id: Some(978),
        scenario: Some(33),
        name: "Sort by a date variable projected from a node property in ascending order",
        fixture: FixtureKind::Date,
        property: "date",
        alias: "date",
        sort: "date",
        limit: 2,
        expected_ids: &[1, 5],
    },
    TemporalCase {
        tck_id: Some(979),
        scenario: Some(33),
        name: "Sort by a date variable projected from a node property in ascending order",
        fixture: FixtureKind::Date,
        property: "date",
        alias: "date",
        sort: "date ASC",
        limit: 2,
        expected_ids: &[1, 5],
    },
    TemporalCase {
        tck_id: Some(980),
        scenario: Some(33),
        name: "Sort by a date variable projected from a node property in ascending order",
        fixture: FixtureKind::Date,
        property: "date",
        alias: "date",
        sort: "date ASCENDING",
        limit: 2,
        expected_ids: &[1, 5],
    },
    TemporalCase {
        tck_id: Some(981),
        scenario: Some(34),
        name: "Sort by a date variable projected from a node property in descending order",
        fixture: FixtureKind::Date,
        property: "date",
        alias: "date",
        sort: "date DESC",
        limit: 2,
        expected_ids: &[4, 3],
    },
    TemporalCase {
        tck_id: Some(982),
        scenario: Some(34),
        name: "Sort by a date variable projected from a node property in descending order",
        fixture: FixtureKind::Date,
        property: "date",
        alias: "date",
        sort: "date DESCENDING",
        limit: 2,
        expected_ids: &[4, 3],
    },
    TemporalCase {
        tck_id: Some(983),
        scenario: Some(35),
        name: "Sort by a local time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalTime,
        property: "time",
        alias: "time",
        sort: "time",
        limit: 3,
        expected_ids: &[1, 4, 2],
    },
    TemporalCase {
        tck_id: Some(984),
        scenario: Some(35),
        name: "Sort by a local time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalTime,
        property: "time",
        alias: "time",
        sort: "time ASC",
        limit: 3,
        expected_ids: &[1, 4, 2],
    },
    TemporalCase {
        tck_id: Some(985),
        scenario: Some(35),
        name: "Sort by a local time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalTime,
        property: "time",
        alias: "time",
        sort: "time ASCENDING",
        limit: 3,
        expected_ids: &[1, 4, 2],
    },
    TemporalCase {
        tck_id: Some(986),
        scenario: Some(36),
        name: "Sort by a local time variable projected from a node property in descending order",
        fixture: FixtureKind::LocalTime,
        property: "time",
        alias: "time",
        sort: "time DESC",
        limit: 3,
        expected_ids: &[5, 3, 2],
    },
    TemporalCase {
        tck_id: Some(987),
        scenario: Some(36),
        name: "Sort by a local time variable projected from a node property in descending order",
        fixture: FixtureKind::LocalTime,
        property: "time",
        alias: "time",
        sort: "time DESCENDING",
        limit: 3,
        expected_ids: &[5, 3, 2],
    },
    TemporalCase {
        tck_id: Some(988),
        scenario: Some(37),
        name: "Sort by a time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedTime,
        property: "time",
        alias: "time",
        sort: "time",
        limit: 3,
        expected_ids: &[4, 5, 2],
    },
    TemporalCase {
        tck_id: Some(989),
        scenario: Some(37),
        name: "Sort by a time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedTime,
        property: "time",
        alias: "time",
        sort: "time ASC",
        limit: 3,
        expected_ids: &[4, 5, 2],
    },
    TemporalCase {
        tck_id: Some(990),
        scenario: Some(37),
        name: "Sort by a time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedTime,
        property: "time",
        alias: "time",
        sort: "time ASCENDING",
        limit: 3,
        expected_ids: &[4, 5, 2],
    },
    TemporalCase {
        tck_id: Some(991),
        scenario: Some(38),
        name: "Sort by a time variable projected from a node property in descending order",
        fixture: FixtureKind::ZonedTime,
        property: "time",
        alias: "time",
        sort: "time DESC",
        limit: 3,
        expected_ids: &[1, 3, 2],
    },
    TemporalCase {
        tck_id: Some(992),
        scenario: Some(38),
        name: "Sort by a time variable projected from a node property in descending order",
        fixture: FixtureKind::ZonedTime,
        property: "time",
        alias: "time",
        sort: "time DESCENDING",
        limit: 3,
        expected_ids: &[1, 3, 2],
    },
    TemporalCase {
        tck_id: Some(993),
        scenario: Some(39),
        name: "Sort by a local date time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime",
        limit: 3,
        expected_ids: &[3, 5, 1],
    },
    TemporalCase {
        tck_id: Some(994),
        scenario: Some(39),
        name: "Sort by a local date time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime ASC",
        limit: 3,
        expected_ids: &[3, 5, 1],
    },
    TemporalCase {
        tck_id: Some(995),
        scenario: Some(39),
        name: "Sort by a local date time variable projected from a node property in ascending order",
        fixture: FixtureKind::LocalDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime ASCENDING",
        limit: 3,
        expected_ids: &[3, 5, 1],
    },
    TemporalCase {
        tck_id: Some(996),
        scenario: Some(40),
        name: "Sort by a local date time variable projected from a node property in descending order",
        fixture: FixtureKind::LocalDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime DESC",
        limit: 3,
        expected_ids: &[4, 2, 1],
    },
    TemporalCase {
        tck_id: Some(997),
        scenario: Some(40),
        name: "Sort by a local date time variable projected from a node property in descending order",
        fixture: FixtureKind::LocalDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime DESCENDING",
        limit: 3,
        expected_ids: &[4, 2, 1],
    },
    TemporalCase {
        tck_id: Some(998),
        scenario: Some(41),
        name: "Sort by a date time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime",
        limit: 3,
        expected_ids: &[3, 5, 2],
    },
    TemporalCase {
        tck_id: Some(999),
        scenario: Some(41),
        name: "Sort by a date time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime ASC",
        limit: 3,
        expected_ids: &[3, 5, 2],
    },
    TemporalCase {
        tck_id: Some(1000),
        scenario: Some(41),
        name: "Sort by a date time variable projected from a node property in ascending order",
        fixture: FixtureKind::ZonedDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime ASCENDING",
        limit: 3,
        expected_ids: &[3, 5, 2],
    },
    TemporalCase {
        tck_id: Some(1001),
        scenario: Some(42),
        name: "Sort by a date time variable projected from a node property in descending order",
        fixture: FixtureKind::ZonedDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime DESC",
        limit: 3,
        expected_ids: &[4, 1, 2],
    },
    TemporalCase {
        tck_id: Some(1002),
        scenario: Some(42),
        name: "Sort by a date time variable projected from a node property in descending order",
        fixture: FixtureKind::ZonedDateTime,
        property: "datetime",
        alias: "datetime",
        sort: "datetime DESCENDING",
        limit: 3,
        expected_ids: &[4, 1, 2],
    },
];

const ADVERSARIAL_CASES: [TemporalCase; 10] = [
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "date null placement and stable ties ascending",
        fixture: FixtureKind::Adversarial,
        property: "d",
        alias: "value",
        sort: "value ASC",
        limit: 8,
        expected_ids: &[1, 6, 2, 3, 4, 8, 5, 7],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "date null placement and stable ties descending",
        fixture: FixtureKind::Adversarial,
        property: "d",
        alias: "value",
        sort: "value DESC",
        limit: 8,
        expected_ids: &[5, 7, 4, 8, 2, 3, 1, 6],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "local-time nanoseconds and stable ties ascending",
        fixture: FixtureKind::Adversarial,
        property: "lt",
        alias: "value",
        sort: "value ASC",
        limit: 8,
        expected_ids: &[1, 7, 2, 4, 3, 5, 6, 8],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "local-time nanoseconds and stable ties descending",
        fixture: FixtureKind::Adversarial,
        property: "lt",
        alias: "value",
        sort: "value DESC",
        limit: 8,
        expected_ids: &[6, 8, 5, 3, 2, 4, 7, 1],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "zoned-time normalized instant and offset tie-break ascending",
        fixture: FixtureKind::Adversarial,
        property: "zt",
        alias: "value",
        sort: "value ASC",
        limit: 8,
        expected_ids: &[5, 3, 1, 7, 2, 4, 6, 8],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "zoned-time normalized instant and offset tie-break descending",
        fixture: FixtureKind::Adversarial,
        property: "zt",
        alias: "value",
        sort: "value DESC",
        limit: 8,
        expected_ids: &[6, 8, 4, 2, 1, 7, 3, 5],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "local-datetime nanoseconds and stable ties ascending",
        fixture: FixtureKind::Adversarial,
        property: "ldt",
        alias: "value",
        sort: "value ASC",
        limit: 8,
        expected_ids: &[5, 1, 2, 3, 7, 4, 6, 8],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "local-datetime nanoseconds and stable ties descending",
        fixture: FixtureKind::Adversarial,
        property: "ldt",
        alias: "value",
        sort: "value DESC",
        limit: 8,
        expected_ids: &[6, 8, 4, 7, 2, 3, 1, 5],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "zoned-datetime timezone and nanosecond tie-break ascending",
        fixture: FixtureKind::Adversarial,
        property: "zdt",
        alias: "value",
        sort: "value ASC",
        limit: 8,
        expected_ids: &[5, 1, 2, 3, 7, 4, 6, 8],
    },
    TemporalCase {
        tck_id: None,
        scenario: None,
        name: "zoned-datetime timezone and nanosecond tie-break descending",
        fixture: FixtureKind::Adversarial,
        property: "zdt",
        alias: "value",
        sort: "value DESC",
        limit: 8,
        expected_ids: &[6, 8, 4, 3, 7, 2, 1, 5],
    },
];

fn all_cases() -> impl Iterator<Item = &'static TemporalCase> {
    OFFICIAL_CASES.iter().chain(ADVERSARIAL_CASES.iter())
}

fn epoch_day(year: i32, month: u32, day: u32) -> Result<ScalarValue> {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| Error::internal("temporal route fixture has an invalid date"))?;
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

fn zoned_time(hour: u32, minute: u32, second: u32, nanos: u32, offset_seconds: i32) -> ScalarValue {
    ScalarValue::ZonedTime {
        nanos: nanos_of_day(hour, minute, second, nanos),
        offset_seconds,
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
    let datetime = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("temporal route fixture has an invalid local datetime"))?;
    Ok(ScalarValue::LocalDateTime {
        seconds: datetime.and_utc().timestamp(),
        nanos,
    })
}

fn zoned_datetime(
    date_time: (i32, u32, u32, u32, u32, u32, u32),
    offset_seconds: i32,
    timezone: &'static str,
) -> Result<ScalarValue> {
    let (year, month, day, hour, minute, second, nanos) = date_time;
    let datetime = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("temporal route fixture has an invalid zoned datetime"))?;
    let seconds = datetime
        .and_utc()
        .timestamp()
        .checked_sub(i64::from(offset_seconds))
        .ok_or_else(|| Error::internal("temporal route fixture UTC conversion overflowed"))?;
    Ok(ScalarValue::ZonedDateTime {
        seconds,
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
            FixtureKind::Date => Self::from_rows(vec![
                ("A", vec![("date", epoch_day(1910, 5, 6)?)]),
                ("B", vec![("date", epoch_day(1980, 12, 24)?)]),
                ("C", vec![("date", epoch_day(1984, 10, 12)?)]),
                ("D", vec![("date", epoch_day(1985, 5, 6)?)]),
                ("E", vec![("date", epoch_day(1980, 10, 24)?)]),
                ("F", vec![("date", epoch_day(1984, 10, 11)?)]),
            ]),
            FixtureKind::LocalTime => Self::from_rows(vec![
                ("A", vec![("time", local_time(10, 35, 0, 0))]),
                ("B", vec![("time", local_time(12, 31, 14, 645_876_123))]),
                ("C", vec![("time", local_time(12, 31, 14, 645_876_124))]),
                ("D", vec![("time", local_time(12, 30, 14, 645_876_123))]),
                ("E", vec![("time", local_time(12, 31, 15, 0))]),
            ]),
            FixtureKind::ZonedTime => Self::from_rows(vec![
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
            FixtureKind::LocalDateTime => Self::from_rows(vec![
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
            FixtureKind::ZonedDateTime => Self::from_rows(vec![
                (
                    "A",
                    vec![(
                        "datetime",
                        zoned_datetime((1984, 10, 11, 12, 30, 14, 12), 15 * 60, "+00:15")?,
                    )],
                ),
                (
                    "B",
                    vec![(
                        "datetime",
                        zoned_datetime((1984, 10, 11, 12, 31, 14, 645_876_123), 17 * 60, "+00:17")?,
                    )],
                ),
                (
                    "C",
                    vec![(
                        "datetime",
                        zoned_datetime((1, 1, 1, 1, 1, 1, 1), -(11 * 3_600 + 59 * 60), "-11:59")?,
                    )],
                ),
                (
                    "D",
                    vec![(
                        "datetime",
                        zoned_datetime(
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
                        zoned_datetime(
                            (1980, 12, 11, 12, 31, 14, 0),
                            -(11 * 3_600 + 59 * 60),
                            "-11:59",
                        )?,
                    )],
                ),
            ]),
            FixtureKind::Adversarial => Self::adversarial(),
        }
    }

    fn adversarial() -> Result<Self> {
        let day_end = 86_399_999_999_999_i64;
        let noon = 12 * 3_600 * 1_000_000_000_i64;
        let local_tie = 45_074_645_876_123_i64;
        let zdt = |seconds, nanos, timezone| ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone: Arc::from(timezone),
        };
        Self::from_rows(vec![
            (
                "A",
                vec![
                    ("d", ScalarValue::Date(-1)),
                    ("lt", ScalarValue::LocalTime(0)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon,
                            offset_seconds: 0,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: 0,
                            nanos: 0,
                        },
                    ),
                    ("zdt", zdt(0, 0, "+00:00")),
                ],
            ),
            (
                "B",
                vec![
                    ("d", ScalarValue::Date(0)),
                    ("lt", ScalarValue::LocalTime(local_tie)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon + 3_600_000_000_000,
                            offset_seconds: 3_600,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: 0,
                            nanos: 1,
                        },
                    ),
                    ("zdt", zdt(0, 0, "Europe/London")),
                ],
            ),
            (
                "C",
                vec![
                    ("d", ScalarValue::Date(0)),
                    ("lt", ScalarValue::LocalTime(local_tie + 1)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon - 3_600_000_000_000,
                            offset_seconds: -3_600,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: 0,
                            nanos: 1,
                        },
                    ),
                    ("zdt", zdt(0, 0, "UTC")),
                ],
            ),
            (
                "D",
                vec![
                    ("d", ScalarValue::Date(1)),
                    ("lt", ScalarValue::LocalTime(local_tie)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon + 1,
                            offset_seconds: 0,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: 1,
                            nanos: 0,
                        },
                    ),
                    ("zdt", zdt(0, 1, "UTC")),
                ],
            ),
            (
                "E",
                vec![
                    ("lt", ScalarValue::LocalTime(day_end)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon - 1,
                            offset_seconds: 0,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: -1,
                            nanos: 999_999_999,
                        },
                    ),
                    ("zdt", zdt(-1, 999_999_999, "UTC")),
                ],
            ),
            ("F", vec![("d", ScalarValue::Date(-1))]),
            (
                "G",
                vec![
                    ("lt", ScalarValue::LocalTime(1)),
                    (
                        "zt",
                        ScalarValue::ZonedTime {
                            nanos: noon,
                            offset_seconds: 0,
                        },
                    ),
                    (
                        "ldt",
                        ScalarValue::LocalDateTime {
                            seconds: 0,
                            nanos: 999_999_999,
                        },
                    ),
                    ("zdt", zdt(0, 0, "UTC")),
                ],
            ),
            ("H", vec![("d", ScalarValue::Date(1))]),
        ])
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
                    .map_err(|_| Error::internal("temporal route fixture node ID overflowed"))?,
            );
            let row_properties = row_properties
                .into_iter()
                .map(|(name, value)| {
                    properties
                        .get(name)
                        .copied()
                        .map(|property| (property, value))
                        .ok_or_else(|| Error::internal("temporal fixture property disappeared"))
                })
                .collect::<Result<Vec<_>>>()?;
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision: id.0,
                labels: vec![
                    *labels
                        .get(label)
                        .ok_or_else(|| Error::internal("temporal fixture label disappeared"))?,
                ],
                properties: row_properties,
            })?;
        }
        let bookmark = Bookmark {
            term: 23,
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
            .ok_or_else(|| Error::internal(format!("temporal fixture omitted property `{name}`")))
    }

    fn value(&self, id: NodeId, property: &str) -> Result<ScalarValue> {
        let node = self
            .graph
            .node(id)
            .ok_or_else(|| Error::internal(format!("temporal fixture omitted node {id}")))?;
        Ok(node
            .property(self.property(property)?)
            .unwrap_or(ScalarValue::Null))
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

/// Fail-closed observer around the one acceptable typed-row boundary.
///
/// The strict CPU reference advertises Metal until the immutable project generation is pinned, so
/// the query engine cannot enter its generic CPU evaluator. The pinned wrapper then exposes honest
/// CPU receipt provenance. Real Metal reports Metal throughout. Every backend route except one
/// complete `execute_row_program` call is rejected.
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
        // The inner CPU still emits CpuReference receipts. Claiming that the pinned backend is
        // Metal must make publication reject those receipts before schema or rows are emitted.
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
                "temporal property route test did not construct a real Metal backend",
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
            Error::internal("temporal route backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal route backend has no admitted graph revision")
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
            format!("strict temporal property test rejected `{route}` execution"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal project has no resident bookmark")
        })?;
        self.expected_graph_revision = self
            .inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("replacement temporal project has no graph revision"))?;
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
                "pinned temporal generation does not match the admitted immutable fence",
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
                "temporal row request does not belong to the pinned graph generation",
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

#[derive(Clone, Debug, PartialEq, Eq)]
struct ObservedValue {
    node: NodeId,
    value: ScalarValue,
}

fn observe_output(output: &ExecutionOutput, case: TemporalCase) -> Result<Vec<ObservedValue>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(
            "temporal property read produced side effects or truncation",
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        let nodes = batch
            .columns
            .iter()
            .find(|column| column.name == "a")
            .ok_or_else(|| Error::internal("temporal result omitted node column `a`"))?;
        let values = batch
            .columns
            .iter()
            .find(|column| column.name == case.alias)
            .ok_or_else(|| Error::internal(format!("temporal result omitted `{}`", case.alias)))?;
        if nodes.value_type != ColumnType::Node
            || values.value_type != ColumnType::Temporal
            || nodes.values.len() != batch.row_count
            || values.values.len() != batch.row_count
        {
            return Err(Error::internal(
                "temporal result schema or batch cardinality is wrong",
            ));
        }
        for (node, value) in nodes.values.iter().zip(&values.values) {
            let ResultValue::Node(node) = node else {
                return Err(Error::internal("temporal result contains a non-node `a`"));
            };
            let ResultValue::Scalar(value) = value else {
                return Err(Error::internal(
                    "temporal projection contains a non-scalar value",
                ));
            };
            if !matches!(
                value,
                ScalarValue::Null
                    | ScalarValue::Date(_)
                    | ScalarValue::LocalTime(_)
                    | ScalarValue::ZonedTime { .. }
                    | ScalarValue::LocalDateTime { .. }
                    | ScalarValue::ZonedDateTime { .. }
            ) {
                return Err(Error::internal(
                    "temporal projection published a non-temporal scalar",
                ));
            }
            rows.push(ObservedValue {
                node: node.id,
                value: value.clone(),
            });
        }
    }
    Ok(rows)
}

fn expected_output(fixture: &Fixture, case: TemporalCase) -> Result<Vec<ObservedValue>> {
    case.expected_ids
        .iter()
        .copied()
        .map(|id| {
            let node = NodeId(id);
            Ok(ObservedValue {
                node,
                value: fixture.value(node, case.property)?,
            })
        })
        .collect()
}

fn run_cpu_oracle_case(fixture: &Fixture, case: TemporalCase) -> Result<()> {
    let query = case.query();
    let output = QueryEngine.execute(&query, &mut context(fixture, None))?;
    assert_eq!(
        observe_output(&output, case)?,
        expected_output(fixture, case)?,
        "{query}"
    );
    Ok(())
}

fn assert_request_and_result(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: TemporalCase,
    request: &ResidentRowProgramRequest,
    result: &ResidentRowProgramResultParts,
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
    assert!(request.input.expansion.is_none());
    assert!(request.input.continuations.is_empty());
    assert!(request.input.correlated_optional.is_none());
    assert!(request.input.predicates.is_empty());
    assert!(request.input.property_filters.is_empty());
    assert!(request.input.value_matrix.is_none());
    assert!(request.input.mutation.is_none());
    assert!(request.input.orders.is_empty());
    assert_eq!(request.input.offset, 0);
    assert_eq!(request.input.limit, usize::MAX);
    assert!(request.input.integer_projections.is_empty());
    assert!(request.input.property_null_projections.is_empty());
    assert_eq!(request.offset, 0);
    assert_eq!(request.limit, case.limit);
    assert_eq!(request.max_output_rows, MAX_RESULT_ROWS);
    assert_eq!(request.sort_keys.len(), 1);
    assert_eq!(request.sort_keys[0].descending, case.descending());
    assert_eq!(request.sort_keys[0].nulls_first, case.descending());
    assert_eq!(request.final_registers, vec![request.sort_keys[0].register]);
    assert_eq!(
        request.manifest.instruction_obligations.len(),
        request.program.instructions.len()
    );
    assert_eq!(
        request.obligations().collect::<Vec<_>>().len(),
        request.program.instructions.len() + 1
    );

    assert_eq!(result.project, PROJECT);
    assert_eq!(result.execution, request.execution);
    assert_eq!(result.bookmark, fixture.bookmark);
    assert_eq!(result.graph_revision, fixture.graph.revision());
    assert_eq!(result.layout_version, fixture.graph.layout_version());
    assert_eq!(result.manifest_fingerprint, request.manifest.fingerprint);
    assert_eq!(result.source_positions.len(), case.expected_ids.len());
    assert_eq!(result.rows.start_rows.len(), case.expected_ids.len());
    assert!(result.rows.intermediate_node_rows.is_empty());
    assert!(result.rows.intermediate_edge_rows.is_empty());
    assert!(result.rows.edge_rows.is_empty());
    assert!(result.rows.end_rows.is_empty());
    assert_eq!(result.projected_columns.len(), 1);
    assert_eq!(
        result.projected_columns[0].register,
        request.final_registers[0]
    );
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
    assert_eq!(result.receipts.len(), expected_obligations.len());
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

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: TemporalCase,
) -> std::result::Result<(), String> {
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
        return Err("query did not cross exactly one typed-row boundary".to_owned());
    }
    if observations.old_node_pipeline_calls.load(Ordering::SeqCst) != old_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered an obsolete, generic, or host-oriented backend route".to_owned(),
        );
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one row request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "resident row request disappeared".to_owned())?
    };
    let result = {
        let results = observations
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if results.len() != results_before + 1 {
            return Err("observer did not retain exactly one row result".to_owned());
        }
        results
            .last()
            .cloned()
            .ok_or_else(|| "resident row result disappeared".to_owned())?
            .into_untrusted_parts()
    };
    assert_request_and_result(fixture, backend, case, &request, &result)
        .map_err(|error| format!("native request/provenance assertion failed: {error}"))?;
    let actual = observe_output(&output, case)
        .map_err(|error| format!("native output inspection failed: {error}"))?;
    let expected = expected_output(fixture, case)
        .map_err(|error| format!("cannot build expected output: {error}"))?;
    if actual != expected {
        return Err(format!(
            "stable temporal output mismatch: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(())
}

fn run_native_suite(backend: &mut ObservedRowBackend) -> Vec<String> {
    let mut failures = Vec::new();
    for fixture_kind in FixtureKind::ALL {
        let fixture = match Fixture::new(fixture_kind) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{fixture_kind:?}: cannot build fixture: {error}"));
                continue;
            }
        };
        if let Err(error) = fixture
            .image()
            .and_then(|image| backend.replace_all_projects(vec![image]))
        {
            failures.push(format!("{fixture_kind:?}: cannot replace project: {error}"));
            continue;
        }
        for case in all_cases().filter(|case| case.fixture == fixture_kind) {
            if let Err(error) = execute_native_case(&fixture, backend, *case) {
                failures.push(format!("{}: {error}", case.label()));
            }
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} native temporal-property suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_is_exactly_pinned_report_ids_978_through_1002() {
    assert_eq!(OFFICIAL_CASES.len(), 25);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.tck_id)
            .collect::<Vec<_>>(),
        (978_u16..=1002).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.scenario)
            .collect::<Vec<_>>(),
        vec![
            33, 33, 33, 34, 34, 35, 35, 35, 36, 36, 37, 37, 37, 38, 38, 39, 39, 39, 40, 40, 41, 41,
            41, 42, 42,
        ]
    );
    assert!(OFFICIAL_CASES.iter().all(|case| {
        !case.name.is_empty()
            && case.query().starts_with("MATCH (a) WITH a, a.")
            && case.query().ends_with(&format!("RETURN a, {}", case.alias))
    }));
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_25_temporal_property_order_selectors() {
    assert_certified_report_identities(OFFICIAL_CASES.iter().map(|case| {
        (
            usize::from(case.tck_id.expect("official case has report id")),
            FEATURE.to_owned(),
            case.expanded_report_name()
                .expect("official case has an expanded report name"),
        )
    }));
}

#[test]
fn cpu_oracle_matches_all_25_pinned_temporal_scenarios() -> Result<()> {
    let mut failures = Vec::new();
    for fixture_kind in [
        FixtureKind::Date,
        FixtureKind::LocalTime,
        FixtureKind::ZonedTime,
        FixtureKind::LocalDateTime,
        FixtureKind::ZonedDateTime,
    ] {
        let fixture = Fixture::new(fixture_kind)?;
        for case in OFFICIAL_CASES
            .iter()
            .filter(|case| case.fixture == fixture_kind)
        {
            if let Err(error) = run_cpu_oracle_case(&fixture, *case) {
                failures.push(format!("{}: {error}", case.label()));
            }
        }
    }
    assert_no_failures("CPU oracle (official IDs 979-1003)", failures);
    Ok(())
}

#[test]
fn cpu_oracle_proves_temporal_null_tie_offset_timezone_and_nanosecond_order() -> Result<()> {
    let fixture = Fixture::new(FixtureKind::Adversarial)?;
    let mut failures = Vec::new();
    for case in ADVERSARIAL_CASES {
        if let Err(error) = run_cpu_oracle_case(&fixture, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert_no_failures("CPU oracle (adversarial temporal order)", failures);
    Ok(())
}

#[test]
fn strict_cpu_reference_runs_all_temporal_cases_through_one_pinned_boundary() -> Result<()> {
    let first = Fixture::new(FixtureKind::Date)?;
    let mut backend = first.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_no_failures("strict CPU reference", run_native_suite(&mut backend));
    Ok(())
}

#[test]
fn temporal_receipts_reject_wrong_device_provenance_before_publication() -> Result<()> {
    let fixture = Fixture::new(FixtureKind::Date)?;
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
#[test]
#[ignore = "red acceptance gate: all 25 pinned and 10 adversarial cases must run on real Metal"]
fn real_metal_matches_cpu_for_all_temporal_cases_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = Fixture::new(FixtureKind::Date)?;
    let mut backend = first.real_metal_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    assert_no_failures("real Metal", run_native_suite(&mut backend));
    Ok(())
}
