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

use chrono::{FixedOffset, NaiveDate, TimeZone};
use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentProjectImage, ResidentRowColumn, ResidentRowOperation, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentRowProgramResultParts, ResidentRowValueType,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
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
const MAX_RESULT_ROWS: usize = 64;
const FEATURE: &str = "features/clauses/with-orderBy/WithOrderBy1.feature";

const OFFICIAL_DATE_ASC: &str = "UNWIND [date({year: 1910, month: 5, day: 6}),\n\
              date({year: 1980, month: 12, day: 24}),\n\
              date({year: 1984, month: 10, day: 12}),\n\
              date({year: 1985, month: 5, day: 6}),\n\
              date({year: 1980, month: 10, day: 24}),\n\
              date({year: 1984, month: 10, day: 11})] AS dates\n\
WITH dates\n\
  ORDER BY dates\n\
  LIMIT 2\n\
RETURN dates";

const OFFICIAL_DATE_DESC: &str = "UNWIND [date({year: 1910, month: 5, day: 6}),\n\
              date({year: 1980, month: 12, day: 24}),\n\
              date({year: 1984, month: 10, day: 12}),\n\
              date({year: 1985, month: 5, day: 6}),\n\
              date({year: 1980, month: 10, day: 24}),\n\
              date({year: 1984, month: 10, day: 11})] AS dates\n\
WITH dates\n\
  ORDER BY dates DESC\n\
  LIMIT 2\n\
RETURN dates";

const OFFICIAL_LOCAL_TIME_ASC: &str = "UNWIND [localtime({hour: 10, minute: 35}),\n\
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n\
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n\
              localtime({hour: 12, minute: 35, second: 13}),\n\
              localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\n\
WITH localtimes\n\
  ORDER BY localtimes\n\
  LIMIT 3\n\
RETURN localtimes";

const OFFICIAL_LOCAL_TIME_DESC: &str = "UNWIND [localtime({hour: 10, minute: 35}),\n\
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n\
              localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n\
              localtime({hour: 12, minute: 35, second: 13}),\n\
              localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\n\
WITH localtimes\n\
  ORDER BY localtimes DESC\n\
  LIMIT 3\n\
RETURN localtimes";

const OFFICIAL_ZONED_TIME_ASC: &str = "UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n\
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n\
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n\
              time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n\
              time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\n\
WITH times\n\
  ORDER BY times\n\
  LIMIT 3\n\
RETURN times";

const OFFICIAL_ZONED_TIME_DESC: &str = "UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n\
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n\
              time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n\
              time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n\
              time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\n\
WITH times\n\
  ORDER BY times DESC\n\
  LIMIT 3\n\
RETURN times";

const OFFICIAL_LOCAL_DATETIME_ASC: &str = "UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n\
              localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n\
              localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n\
              localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n\
              localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\n\
WITH localdatetimes\n\
  ORDER BY localdatetimes\n\
  LIMIT 3\n\
RETURN localdatetimes";

const OFFICIAL_LOCAL_DATETIME_DESC: &str = "UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n\
              localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n\
              localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n\
              localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n\
              localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\n\
WITH localdatetimes\n\
  ORDER BY localdatetimes DESC\n\
  LIMIT 3\n\
RETURN localdatetimes";

const OFFICIAL_ZONED_DATETIME_ASC: &str = "UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n\
              datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n\
              datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n\
              datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n\
              datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\n\
WITH datetimes\n\
  ORDER BY datetimes\n\
  LIMIT 3\n\
RETURN datetimes";

const OFFICIAL_ZONED_DATETIME_DESC: &str = "UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n\
              datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n\
              datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n\
              datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n\
              datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\n\
WITH datetimes\n\
  ORDER BY datetimes DESC\n\
  LIMIT 3\n\
RETURN datetimes";

const ADVERSARIAL_DATE_ASC: &str = "UNWIND [date({year: 1970, month: 1, day: 1}), null, date({year: 1969, month: 12, day: 31}), date({year: 1970, month: 1, day: 1})] AS value WITH value ORDER BY value ASC LIMIT 4 RETURN value";
const ADVERSARIAL_DATE_DESC: &str = "UNWIND [date({year: 1970, month: 1, day: 1}), null, date({year: 1969, month: 12, day: 31}), date({year: 1970, month: 1, day: 1})] AS value WITH value ORDER BY value DESC LIMIT 4 RETURN value";
const ADVERSARIAL_LOCAL_TIME_ASC: &str = "UNWIND [localtime({hour: 0}), null, localtime({hour: 0, nanosecond: 1}), localtime({hour: 0, nanosecond: 1}), localtime({hour: 23, minute: 59, second: 59, nanosecond: 999999999})] AS value WITH value ORDER BY value ASC LIMIT 5 RETURN value";
const ADVERSARIAL_LOCAL_TIME_DESC: &str = "UNWIND [localtime({hour: 0}), null, localtime({hour: 0, nanosecond: 1}), localtime({hour: 0, nanosecond: 1}), localtime({hour: 23, minute: 59, second: 59, nanosecond: 999999999})] AS value WITH value ORDER BY value DESC LIMIT 5 RETURN value";
const ADVERSARIAL_ZONED_TIME_ASC: &str = "UNWIND [time({hour: 12, timezone: '+00:00'}), null, time({hour: 13, timezone: '+01:00'}), time({hour: 11, timezone: '-01:00'}), time({hour: 12, nanosecond: 1, timezone: '+00:00'}), time({hour: 12, timezone: '+00:00'})] AS value WITH value ORDER BY value ASC LIMIT 6 RETURN value";
const ADVERSARIAL_ZONED_TIME_DESC: &str = "UNWIND [time({hour: 12, timezone: '+00:00'}), null, time({hour: 13, timezone: '+01:00'}), time({hour: 11, timezone: '-01:00'}), time({hour: 12, nanosecond: 1, timezone: '+00:00'}), time({hour: 12, timezone: '+00:00'})] AS value WITH value ORDER BY value DESC LIMIT 6 RETURN value";
const ADVERSARIAL_LOCAL_DATETIME_ASC: &str = "UNWIND [localdatetime({year: 1970, month: 1, day: 1}), null, localdatetime({year: 1970, month: 1, day: 1, nanosecond: 1}), localdatetime({year: 1970, month: 1, day: 1, nanosecond: 1}), localdatetime({year: 1970, month: 1, day: 1, second: 1})] AS value WITH value ORDER BY value ASC LIMIT 5 RETURN value";
const ADVERSARIAL_LOCAL_DATETIME_DESC: &str = "UNWIND [localdatetime({year: 1970, month: 1, day: 1}), null, localdatetime({year: 1970, month: 1, day: 1, nanosecond: 1}), localdatetime({year: 1970, month: 1, day: 1, nanosecond: 1}), localdatetime({year: 1970, month: 1, day: 1, second: 1})] AS value WITH value ORDER BY value DESC LIMIT 5 RETURN value";
const ADVERSARIAL_ZONED_DATETIME_ASC: &str = "UNWIND [datetime({year: 2020, month: 1, day: 1, timezone: 'UTC'}), null, datetime({year: 2020, month: 1, day: 1, hour: 1, timezone: '+01:00'}), datetime({year: 2019, month: 12, day: 31, hour: 19, timezone: 'America/New_York'}), datetime({year: 2020, month: 1, day: 1, hour: 1, timezone: 'Europe/Stockholm'}), datetime({year: 2020, month: 1, day: 1, nanosecond: 1, timezone: 'UTC'}), datetime({year: 2020, month: 1, day: 1, timezone: 'UTC'})] AS value WITH value ORDER BY value ASC LIMIT 7 RETURN value";
const ADVERSARIAL_ZONED_DATETIME_DESC: &str = "UNWIND [datetime({year: 2020, month: 1, day: 1, timezone: 'UTC'}), null, datetime({year: 2020, month: 1, day: 1, hour: 1, timezone: '+01:00'}), datetime({year: 2019, month: 12, day: 31, hour: 19, timezone: 'America/New_York'}), datetime({year: 2020, month: 1, day: 1, hour: 1, timezone: 'Europe/Stockholm'}), datetime({year: 2020, month: 1, day: 1, nanosecond: 1, timezone: 'UTC'}), datetime({year: 2020, month: 1, day: 1, timezone: 'UTC'})] AS value WITH value ORDER BY value DESC LIMIT 7 RETURN value";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceKind {
    OfficialDate,
    OfficialLocalTime,
    OfficialZonedTime,
    OfficialLocalDateTime,
    OfficialZonedDateTime,
    AdversarialDate,
    AdversarialLocalTime,
    AdversarialZonedTime,
    AdversarialLocalDateTime,
    AdversarialZonedDateTime,
}

impl SourceKind {
    const fn value_type(self) -> ResidentRowValueType {
        match self {
            Self::OfficialDate | Self::AdversarialDate => ResidentRowValueType::Date,
            Self::OfficialLocalTime | Self::AdversarialLocalTime => ResidentRowValueType::LocalTime,
            Self::OfficialZonedTime | Self::AdversarialZonedTime => ResidentRowValueType::ZonedTime,
            Self::OfficialLocalDateTime | Self::AdversarialLocalDateTime => {
                ResidentRowValueType::LocalDateTime
            }
            Self::OfficialZonedDateTime | Self::AdversarialZonedDateTime => {
                ResidentRowValueType::ZonedDateTime
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TemporalCase {
    report_id: Option<u16>,
    scenario: Option<u8>,
    name: &'static str,
    source: SourceKind,
    alias: &'static str,
    query: &'static str,
    descending: bool,
    limit: usize,
    expected_positions: &'static [usize],
    expected_tck_rows: &'static [&'static str],
}

impl TemporalCase {
    fn label(self) -> String {
        match (self.report_id, self.scenario) {
            (Some(id), Some(scenario)) => {
                format!("TCK {id} {FEATURE} [{scenario}] {}", self.name)
            }
            _ => format!("adversarial {}", self.name),
        }
    }
}

const OFFICIAL_CASES: [TemporalCase; 10] = [
    TemporalCase {
        report_id: Some(942),
        scenario: Some(11),
        name: "Sort dates in ascending order",
        source: SourceKind::OfficialDate,
        alias: "dates",
        query: OFFICIAL_DATE_ASC,
        descending: false,
        limit: 2,
        expected_positions: &[0, 4],
        expected_tck_rows: &["'1910-05-06'", "'1980-10-24'"],
    },
    TemporalCase {
        report_id: Some(943),
        scenario: Some(12),
        name: "Sort dates in descending order",
        source: SourceKind::OfficialDate,
        alias: "dates",
        query: OFFICIAL_DATE_DESC,
        descending: true,
        limit: 2,
        expected_positions: &[3, 2],
        expected_tck_rows: &["'1985-05-06'", "'1984-10-12'"],
    },
    TemporalCase {
        report_id: Some(944),
        scenario: Some(13),
        name: "Sort local times in ascending order",
        source: SourceKind::OfficialLocalTime,
        alias: "localtimes",
        query: OFFICIAL_LOCAL_TIME_ASC,
        descending: false,
        limit: 3,
        expected_positions: &[0, 4, 1],
        expected_tck_rows: &["'10:35'", "'12:30:14.645876123'", "'12:31:14.645876123'"],
    },
    TemporalCase {
        report_id: Some(945),
        scenario: Some(14),
        name: "Sort local times in descending order",
        source: SourceKind::OfficialLocalTime,
        alias: "localtimes",
        query: OFFICIAL_LOCAL_TIME_DESC,
        descending: true,
        limit: 3,
        expected_positions: &[3, 2, 1],
        expected_tck_rows: &["'12:35:13'", "'12:31:14.645876124'", "'12:31:14.645876123'"],
    },
    TemporalCase {
        report_id: Some(946),
        scenario: Some(15),
        name: "Sort times in ascending order",
        source: SourceKind::OfficialZonedTime,
        alias: "times",
        query: OFFICIAL_ZONED_TIME_ASC,
        descending: false,
        limit: 3,
        expected_positions: &[3, 4, 1],
        expected_tck_rows: &[
            "'12:35:15+05:00'",
            "'12:30:14.645876123+01:01'",
            "'12:31:14.645876123+01:00'",
        ],
    },
    TemporalCase {
        report_id: Some(947),
        scenario: Some(16),
        name: "Sort times in descending order",
        source: SourceKind::OfficialZonedTime,
        alias: "times",
        query: OFFICIAL_ZONED_TIME_DESC,
        descending: true,
        limit: 3,
        expected_positions: &[0, 2, 1],
        expected_tck_rows: &[
            "'10:35-08:00'",
            "'12:31:14.645876124+01:00'",
            "'12:31:14.645876123+01:00'",
        ],
    },
    TemporalCase {
        report_id: Some(948),
        scenario: Some(17),
        name: "Sort local date times in ascending order",
        source: SourceKind::OfficialLocalDateTime,
        alias: "localdatetimes",
        query: OFFICIAL_LOCAL_DATETIME_ASC,
        descending: false,
        limit: 3,
        expected_positions: &[2, 4, 0],
        expected_tck_rows: &[
            "'0001-01-01T01:01:01.000000001'",
            "'1980-12-11T12:31:14'",
            "'1984-10-11T12:30:14.000000012'",
        ],
    },
    TemporalCase {
        report_id: Some(949),
        scenario: Some(18),
        name: "Sort local date times in descending order",
        source: SourceKind::OfficialLocalDateTime,
        alias: "localdatetimes",
        query: OFFICIAL_LOCAL_DATETIME_DESC,
        descending: true,
        limit: 3,
        expected_positions: &[3, 1, 0],
        expected_tck_rows: &[
            "'9999-09-09T09:59:59.999999999'",
            "'1984-10-11T12:31:14.645876123'",
            "'1984-10-11T12:30:14.000000012'",
        ],
    },
    TemporalCase {
        report_id: Some(950),
        scenario: Some(19),
        name: "Sort date times in ascending order",
        source: SourceKind::OfficialZonedDateTime,
        alias: "datetimes",
        query: OFFICIAL_ZONED_DATETIME_ASC,
        descending: false,
        limit: 3,
        expected_positions: &[2, 4, 1],
        expected_tck_rows: &[
            "'0001-01-01T01:01:01.000000001-11:59'",
            "'1980-12-11T12:31:14-11:59'",
            "'1984-10-11T12:31:14.645876123+00:17'",
        ],
    },
    TemporalCase {
        report_id: Some(951),
        scenario: Some(20),
        name: "Sort date times in descending order",
        source: SourceKind::OfficialZonedDateTime,
        alias: "datetimes",
        query: OFFICIAL_ZONED_DATETIME_DESC,
        descending: true,
        limit: 3,
        expected_positions: &[3, 0, 1],
        expected_tck_rows: &[
            "'9999-09-09T09:59:59.999999999+11:59'",
            "'1984-10-11T12:30:14.000000012+00:15'",
            "'1984-10-11T12:31:14.645876123+00:17'",
        ],
    },
];

const ADVERSARIAL_CASES: [TemporalCase; 10] = [
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "date NULL placement and stable ties ascending",
        source: SourceKind::AdversarialDate,
        alias: "value",
        query: ADVERSARIAL_DATE_ASC,
        descending: false,
        limit: 4,
        expected_positions: &[2, 0, 3, 1],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "date NULL placement and stable ties descending",
        source: SourceKind::AdversarialDate,
        alias: "value",
        query: ADVERSARIAL_DATE_DESC,
        descending: true,
        limit: 4,
        expected_positions: &[1, 0, 3, 2],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "local-time nanoseconds and NULL ascending",
        source: SourceKind::AdversarialLocalTime,
        alias: "value",
        query: ADVERSARIAL_LOCAL_TIME_ASC,
        descending: false,
        limit: 5,
        expected_positions: &[0, 2, 3, 4, 1],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "local-time nanoseconds and NULL descending",
        source: SourceKind::AdversarialLocalTime,
        alias: "value",
        query: ADVERSARIAL_LOCAL_TIME_DESC,
        descending: true,
        limit: 5,
        expected_positions: &[1, 4, 2, 3, 0],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "zoned-time equal instants, offset ties, nanoseconds, and NULL ascending",
        source: SourceKind::AdversarialZonedTime,
        alias: "value",
        query: ADVERSARIAL_ZONED_TIME_ASC,
        descending: false,
        limit: 6,
        expected_positions: &[3, 0, 5, 2, 4, 1],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "zoned-time equal instants, offset ties, nanoseconds, and NULL descending",
        source: SourceKind::AdversarialZonedTime,
        alias: "value",
        query: ADVERSARIAL_ZONED_TIME_DESC,
        descending: true,
        limit: 6,
        expected_positions: &[1, 4, 2, 0, 5, 3],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "local-datetime nanoseconds, stable ties, and NULL ascending",
        source: SourceKind::AdversarialLocalDateTime,
        alias: "value",
        query: ADVERSARIAL_LOCAL_DATETIME_ASC,
        descending: false,
        limit: 5,
        expected_positions: &[0, 2, 3, 4, 1],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "local-datetime nanoseconds, stable ties, and NULL descending",
        source: SourceKind::AdversarialLocalDateTime,
        alias: "value",
        query: ADVERSARIAL_LOCAL_DATETIME_DESC,
        descending: true,
        limit: 5,
        expected_positions: &[1, 4, 2, 3, 0],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "zoned-datetime exact timezone UTF-8, nanoseconds, stable ties, and NULL ascending",
        source: SourceKind::AdversarialZonedDateTime,
        alias: "value",
        query: ADVERSARIAL_ZONED_DATETIME_ASC,
        descending: false,
        limit: 7,
        expected_positions: &[2, 3, 4, 0, 6, 5, 1],
        expected_tck_rows: &[],
    },
    TemporalCase {
        report_id: None,
        scenario: None,
        name: "zoned-datetime exact timezone UTF-8, nanoseconds, stable ties, and NULL descending",
        source: SourceKind::AdversarialZonedDateTime,
        alias: "value",
        query: ADVERSARIAL_ZONED_DATETIME_DESC,
        descending: true,
        limit: 7,
        expected_positions: &[1, 5, 0, 6, 4, 3, 2],
        expected_tck_rows: &[],
    },
];

fn all_cases() -> impl Iterator<Item = &'static TemporalCase> {
    OFFICIAL_CASES.iter().chain(ADVERSARIAL_CASES.iter())
}

fn epoch_day(year: i32, month: u32, day: u32) -> Result<ScalarValue> {
    let date = NaiveDate::from_ymd_opt(year, month, day)
        .ok_or_else(|| Error::internal("temporal UNWIND fixture has an invalid date"))?;
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
        .ok_or_else(|| Error::internal("temporal UNWIND fixture has an invalid local datetime"))?;
    Ok(ScalarValue::LocalDateTime {
        seconds: value.and_utc().timestamp(),
        nanos,
    })
}

fn zoned_datetime(
    date_time: (i32, u32, u32, u32, u32, u32, u32),
    timezone: &'static str,
) -> Result<ScalarValue> {
    let (year, month, day, hour, minute, second, nanos) = date_time;
    let local = NaiveDate::from_ymd_opt(year, month, day)
        .and_then(|date| date.and_hms_nano_opt(hour, minute, second, nanos))
        .ok_or_else(|| Error::internal("temporal UNWIND fixture has an invalid zoned datetime"))?;
    let seconds = if timezone.eq_ignore_ascii_case("UTC") || timezone == "Z" {
        local.and_utc().timestamp()
    } else if let Ok(offset) = timezone.parse::<FixedOffset>() {
        offset
            .from_local_datetime(&local)
            .single()
            .ok_or_else(|| Error::internal("fixed-offset fixture datetime is not unique"))?
            .timestamp()
    } else {
        let zone = timezone
            .parse::<chrono_tz::Tz>()
            .map_err(|_| Error::internal("temporal UNWIND fixture has an unknown timezone"))?;
        zone.from_local_datetime(&local)
            .single()
            .ok_or_else(|| Error::internal("named-zone fixture datetime is not unique"))?
            .timestamp()
    };
    Ok(ScalarValue::ZonedDateTime {
        seconds,
        nanos,
        timezone: Arc::from(timezone),
    })
}

fn source_values(source: SourceKind) -> Result<Vec<ScalarValue>> {
    match source {
        SourceKind::OfficialDate => Ok(vec![
            epoch_day(1910, 5, 6)?,
            epoch_day(1980, 12, 24)?,
            epoch_day(1984, 10, 12)?,
            epoch_day(1985, 5, 6)?,
            epoch_day(1980, 10, 24)?,
            epoch_day(1984, 10, 11)?,
        ]),
        SourceKind::OfficialLocalTime => Ok(vec![
            local_time(10, 35, 0, 0),
            local_time(12, 31, 14, 645_876_123),
            local_time(12, 31, 14, 645_876_124),
            local_time(12, 35, 13, 0),
            local_time(12, 30, 14, 645_876_123),
        ]),
        SourceKind::OfficialZonedTime => Ok(vec![
            zoned_time(10, 35, 0, 0, -8 * 3_600),
            zoned_time(12, 31, 14, 645_876_123, 3_600),
            zoned_time(12, 31, 14, 645_876_124, 3_600),
            zoned_time(12, 35, 15, 0, 5 * 3_600),
            zoned_time(12, 30, 14, 645_876_123, 3_660),
        ]),
        SourceKind::OfficialLocalDateTime => Ok(vec![
            local_datetime(1984, 10, 11, 12, 30, 14, 12)?,
            local_datetime(1984, 10, 11, 12, 31, 14, 645_876_123)?,
            local_datetime(1, 1, 1, 1, 1, 1, 1)?,
            local_datetime(9999, 9, 9, 9, 59, 59, 999_999_999)?,
            local_datetime(1980, 12, 11, 12, 31, 14, 0)?,
        ]),
        SourceKind::OfficialZonedDateTime => Ok(vec![
            zoned_datetime((1984, 10, 11, 12, 30, 14, 12), "+00:15")?,
            zoned_datetime((1984, 10, 11, 12, 31, 14, 645_876_123), "+00:17")?,
            zoned_datetime((1, 1, 1, 1, 1, 1, 1), "-11:59")?,
            zoned_datetime((9999, 9, 9, 9, 59, 59, 999_999_999), "+11:59")?,
            zoned_datetime((1980, 12, 11, 12, 31, 14, 0), "-11:59")?,
        ]),
        SourceKind::AdversarialDate => Ok(vec![
            epoch_day(1970, 1, 1)?,
            ScalarValue::Null,
            epoch_day(1969, 12, 31)?,
            epoch_day(1970, 1, 1)?,
        ]),
        SourceKind::AdversarialLocalTime => Ok(vec![
            local_time(0, 0, 0, 0),
            ScalarValue::Null,
            local_time(0, 0, 0, 1),
            local_time(0, 0, 0, 1),
            local_time(23, 59, 59, 999_999_999),
        ]),
        SourceKind::AdversarialZonedTime => Ok(vec![
            zoned_time(12, 0, 0, 0, 0),
            ScalarValue::Null,
            zoned_time(13, 0, 0, 0, 3_600),
            zoned_time(11, 0, 0, 0, -3_600),
            zoned_time(12, 0, 0, 1, 0),
            zoned_time(12, 0, 0, 0, 0),
        ]),
        SourceKind::AdversarialLocalDateTime => Ok(vec![
            local_datetime(1970, 1, 1, 0, 0, 0, 0)?,
            ScalarValue::Null,
            local_datetime(1970, 1, 1, 0, 0, 0, 1)?,
            local_datetime(1970, 1, 1, 0, 0, 0, 1)?,
            local_datetime(1970, 1, 1, 0, 0, 1, 0)?,
        ]),
        SourceKind::AdversarialZonedDateTime => Ok(vec![
            zoned_datetime((2020, 1, 1, 0, 0, 0, 0), "UTC")?,
            ScalarValue::Null,
            zoned_datetime((2020, 1, 1, 1, 0, 0, 0), "+01:00")?,
            zoned_datetime((2019, 12, 31, 19, 0, 0, 0), "America/New_York")?,
            zoned_datetime((2020, 1, 1, 1, 0, 0, 0), "Europe/Stockholm")?,
            zoned_datetime((2020, 1, 1, 0, 0, 0, 1), "UTC")?,
            zoned_datetime((2020, 1, 1, 0, 0, 0, 0), "UTC")?,
        ]),
    }
}

fn column_from_values(
    value_type: ResidentRowValueType,
    values: &[ScalarValue],
) -> Result<ResidentRowColumn> {
    let validity = values
        .iter()
        .map(|value| u8::from(!matches!(value, ScalarValue::Null)))
        .collect::<Vec<_>>();
    let column = match value_type {
        ResidentRowValueType::Date => ResidentRowColumn::Date {
            days: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Date(days) => *days,
                    ScalarValue::Null => 0,
                    _ => unreachable!("date source changed type"),
                })
                .collect(),
            validity,
        },
        ResidentRowValueType::LocalTime => ResidentRowColumn::LocalTime {
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalTime(nanos) => *nanos,
                    ScalarValue::Null => 0,
                    _ => unreachable!("local-time source changed type"),
                })
                .collect(),
            validity,
        },
        ResidentRowValueType::ZonedTime => ResidentRowColumn::ZonedTime {
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::ZonedTime { nanos, .. } => *nanos,
                    ScalarValue::Null => 0,
                    _ => unreachable!("zoned-time source changed type"),
                })
                .collect(),
            offset_seconds: values
                .iter()
                .map(|value| match value {
                    ScalarValue::ZonedTime { offset_seconds, .. } => *offset_seconds,
                    ScalarValue::Null => 0,
                    _ => unreachable!("zoned-time source changed type"),
                })
                .collect(),
            validity,
        },
        ResidentRowValueType::LocalDateTime => ResidentRowColumn::LocalDateTime {
            seconds: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalDateTime { seconds, .. } => *seconds,
                    ScalarValue::Null => 0,
                    _ => unreachable!("local-datetime source changed type"),
                })
                .collect(),
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalDateTime { nanos, .. } => *nanos,
                    ScalarValue::Null => 0,
                    _ => unreachable!("local-datetime source changed type"),
                })
                .collect(),
            validity,
        },
        ResidentRowValueType::ZonedDateTime => {
            let mut timezone_offsets = Vec::with_capacity(values.len() + 1);
            let mut timezone_bytes = Vec::new();
            timezone_offsets.push(0);
            for value in values {
                if let ScalarValue::ZonedDateTime { timezone, .. } = value {
                    timezone_bytes.extend_from_slice(timezone.as_bytes());
                }
                timezone_offsets.push(
                    u32::try_from(timezone_bytes.len())
                        .map_err(|_| Error::internal("test timezone byte arena exceeds u32"))?,
                );
            }
            ResidentRowColumn::ZonedDateTime {
                seconds: values
                    .iter()
                    .map(|value| match value {
                        ScalarValue::ZonedDateTime { seconds, .. } => *seconds,
                        ScalarValue::Null => 0,
                        _ => unreachable!("zoned-datetime source changed type"),
                    })
                    .collect(),
                nanos: values
                    .iter()
                    .map(|value| match value {
                        ScalarValue::ZonedDateTime { nanos, .. } => *nanos,
                        ScalarValue::Null => 0,
                        _ => unreachable!("zoned-datetime source changed type"),
                    })
                    .collect(),
                timezone_offsets,
                timezone_bytes,
                validity,
            }
        }
        other => {
            return Err(Error::internal(format!(
                "temporal source unexpectedly requested {other:?}"
            )));
        }
    };
    Ok(column)
}

fn expected_values(case: TemporalCase) -> Result<Vec<ScalarValue>> {
    let source = source_values(case.source)?;
    case.expected_positions
        .iter()
        .map(|position| {
            source.get(*position).cloned().ok_or_else(|| {
                Error::internal("pinned temporal expected position is outside its source")
            })
        })
        .collect()
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
                term: 29,
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

/// Fail-closed observer around the only acceptable execution boundary.
///
/// The strict CPU reference advertises Metal until the immutable project generation is pinned,
/// preventing the query engine from choosing its generic CPU evaluator. The pinned wrapper then
/// exposes honest CPU receipt provenance. Real Metal reports Metal throughout. Every backend
/// query route except one complete `execute_row_program` call is rejected.
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
                "temporal UNWIND test did not construct a real Metal backend",
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
            Error::internal("temporal UNWIND backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal UNWIND backend has no admitted graph revision")
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
            format!("strict temporal UNWIND test rejected `{route}` execution"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal UNWIND project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement temporal UNWIND project has no graph revision")
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
                "pinned temporal UNWIND generation does not match the admitted immutable fence",
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
                "temporal UNWIND request does not belong to the pinned graph generation",
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

fn observe_output(output: &ExecutionOutput, case: TemporalCase) -> Result<Vec<ScalarValue>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(
            "temporal constructor/UNWIND read produced side effects or truncation",
        ));
    }
    if output.result.schema != vec![(case.alias.to_owned(), ColumnType::Temporal)] {
        return Err(Error::internal(format!(
            "temporal constructor/UNWIND schema is wrong: {:?}",
            output.result.schema
        )));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if batch.columns.len() != 1
            || batch.columns[0].name != case.alias
            || batch.columns[0].value_type != ColumnType::Temporal
            || batch.columns[0].values.len() != batch.row_count
        {
            return Err(Error::internal(
                "temporal constructor/UNWIND batch shape is wrong",
            ));
        }
        for value in &batch.columns[0].values {
            let ResultValue::Scalar(value) = value else {
                return Err(Error::internal(
                    "temporal constructor/UNWIND returned a non-scalar value",
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
                    "temporal constructor/UNWIND returned a non-temporal scalar",
                ));
            }
            rows.push(value.clone());
        }
    }
    Ok(rows)
}

fn run_cpu_oracle_case(fixture: &Fixture, case: TemporalCase) -> Result<Vec<ScalarValue>> {
    let output = QueryEngine.execute(case.query, &mut context(fixture, None))?;
    let actual = observe_output(&output, case)?;
    assert_eq!(actual, expected_values(case)?, "{}", case.query);
    Ok(actual)
}

fn assert_request_and_result(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: TemporalCase,
    request: &ResidentRowProgramRequest,
    result: &ResidentRowProgramResultParts,
) -> Result<()> {
    let source_values = source_values(case.source)?;
    let expected_source = column_from_values(case.source.value_type(), &source_values)?;
    let expected_projected = column_from_values(case.source.value_type(), &expected_values(case)?)?;

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
    assert_eq!(request.input.max_output_rows, source_values.len());

    assert_eq!(request.program.instructions.len(), 1);
    assert_eq!(
        request.program.instructions[0].output_type,
        case.source.value_type()
    );
    let ResidentRowOperation::InputColumn(source_column) =
        &request.program.instructions[0].operation
    else {
        return Err(Error::internal(
            "temporal constructor/UNWIND source was not one raw typed input column",
        ));
    };
    assert_eq!(source_column, &expected_source);
    assert_eq!(request.offset, 0);
    assert_eq!(request.limit, case.limit);
    assert_eq!(request.max_output_rows, MAX_RESULT_ROWS);
    assert_eq!(request.sort_keys.len(), 1);
    assert_eq!(request.sort_keys[0].register, 0);
    assert_eq!(request.sort_keys[0].descending, case.descending);
    assert_eq!(request.sort_keys[0].nulls_first, case.descending);
    assert_eq!(request.final_registers, vec![0]);
    assert_eq!(request.manifest.instruction_obligations.len(), 1);
    assert_eq!(request.obligations().collect::<Vec<_>>().len(), 2);

    assert_eq!(result.project, PROJECT);
    assert_eq!(result.execution, request.execution);
    assert_eq!(result.bookmark, fixture.bookmark);
    assert_eq!(result.graph_revision, fixture.graph.revision());
    assert_eq!(result.layout_version, fixture.graph.layout_version());
    assert_eq!(result.manifest_fingerprint, request.manifest.fingerprint);
    assert_eq!(result.input_cardinality, source_values.len());
    assert_eq!(
        result.source_positions,
        case.expected_positions
            .iter()
            .map(|position| u64::try_from(*position).expect("tiny pinned position fits u64"))
            .collect::<Vec<_>>()
    );
    assert!(result.rows.start_rows.is_empty());
    assert!(result.rows.intermediate_node_rows.is_empty());
    assert!(result.rows.intermediate_edge_rows.is_empty());
    assert!(result.rows.edge_rows.is_empty());
    assert!(result.rows.end_rows.is_empty());
    assert!(result.rows.integer_columns.is_empty());
    assert!(result.rows.boolean_columns.is_empty());
    assert!(result.rows.value_left_indices.is_empty());
    assert!(result.rows.value_right_indices.is_empty());
    assert!(result.rows.mutation.is_none());
    assert_eq!(result.projected_columns.len(), 1);
    assert_eq!(result.projected_columns[0].register, 0);
    assert_eq!(result.projected_columns[0].column, expected_projected);
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
    assert_eq!(result.receipts.len(), 2);
    for receipt in &result.receipts {
        assert_eq!(receipt.execution, request.execution);
        assert_eq!(receipt.completion, completion);
        assert_eq!(receipt.input_cardinality, source_values.len() as u64);
        let expected_rows = if receipt.obligation == request.manifest.sort_obligation {
            case.expected_positions.len() as u64
        } else {
            source_values.len() as u64
        };
        assert_eq!(receipt.output_cardinality, expected_rows);
    }
    Ok(())
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    case: TemporalCase,
) -> std::result::Result<Vec<ScalarValue>, String> {
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
    let output = QueryEngine
        .execute(case.query, &mut context(fixture, Some(backend)))
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
        return Err("query did not cross exactly one complete typed-row boundary".to_owned());
    }
    if observations.old_node_pipeline_calls.load(Ordering::SeqCst) != old_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered an obsolete, generic, graph, or host-oriented backend route".to_owned(),
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
    let expected = expected_values(case)
        .map_err(|error| format!("cannot build pinned expected output: {error}"))?;
    if actual != expected {
        return Err(format!(
            "stable temporal output mismatch: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(actual)
}

fn run_native_suite(
    fixture: &Fixture,
    backend: &ObservedRowBackend,
    cases: impl IntoIterator<Item = &'static TemporalCase>,
) -> Vec<String> {
    let mut failures = Vec::new();
    for case in cases {
        if let Err(error) = execute_native_case(fixture, backend, *case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} temporal constructor/UNWIND suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_official_ids_942_through_951_queries_and_rows() {
    assert_eq!(OFFICIAL_CASES.len(), 10);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.report_id)
            .collect::<Vec<_>>(),
        (942_u16..=951).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .filter_map(|case| case.scenario)
            .collect::<Vec<_>>(),
        (11_u8..=20).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.name)
            .collect::<Vec<_>>(),
        vec![
            "Sort dates in ascending order",
            "Sort dates in descending order",
            "Sort local times in ascending order",
            "Sort local times in descending order",
            "Sort times in ascending order",
            "Sort times in descending order",
            "Sort local date times in ascending order",
            "Sort local date times in descending order",
            "Sort date times in ascending order",
            "Sort date times in descending order",
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.query)
            .collect::<BTreeSet<_>>()
            .len(),
        10
    );
    for case in OFFICIAL_CASES {
        assert!(case.query.starts_with("UNWIND ["));
        assert!(case.query.contains("WITH "));
        assert!(case.query.contains("ORDER BY "));
        assert!(case.query.ends_with(&format!("RETURN {}", case.alias)));
        assert!(!case.query.contains("MATCH"));
        assert_eq!(case.expected_tck_rows.len(), case.expected_positions.len());
        assert_eq!(case.expected_positions.len(), case.limit);
        assert!(
            case.expected_tck_rows
                .iter()
                .all(|row| row.starts_with('\'') && row.ends_with('\''))
        );
    }
}

#[test]
fn cpu_oracle_matches_all_10_pinned_official_temporal_unwind_scenarios() -> Result<()> {
    let fixture = Fixture::new();
    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        if let Err(error) = run_cpu_oracle_case(&fixture, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert_no_failures("CPU oracle (official IDs 942-951)", failures);
    Ok(())
}

#[test]
fn cpu_oracle_proves_timezone_bytes_nulls_offset_ties_and_nanoseconds() -> Result<()> {
    let fixture = Fixture::new();
    let mut failures = Vec::new();
    for case in ADVERSARIAL_CASES {
        if let Err(error) = run_cpu_oracle_case(&fixture, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert_no_failures("CPU oracle (10 adversarial temporal cases)", failures);
    Ok(())
}

#[test]
fn strict_cpu_reference_runs_all_20_cases_through_one_complete_native_boundary() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_no_failures(
        "strict CPU reference",
        run_native_suite(&fixture, &backend, all_cases()),
    );
    Ok(())
}

#[test]
fn temporal_unwind_receipts_reject_wrong_device_provenance_before_publication() -> Result<()> {
    let fixture = Fixture::new();
    let case = OFFICIAL_CASES[0];
    let backend = fixture.wrong_receipt_backend()?;
    let observations = backend.observations();
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            case.query,
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
#[ignore = "hardware acceptance gate: 10 official and 10 adversarial cases require real Metal"]
fn real_metal_exactly_matches_strict_cpu_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let cpu = fixture.strict_cpu_backend()?;
    let metal = fixture.real_metal_backend()?;
    assert_eq!(cpu.kind(), BackendKind::Metal);
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.kind(), BackendKind::Metal);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for case in all_cases() {
        let cpu_result = execute_native_case(&fixture, &cpu, *case);
        let metal_result = execute_native_case(&fixture, &metal, *case);
        match (cpu_result, metal_result) {
            (Ok(cpu_rows), Ok(metal_rows)) if cpu_rows == metal_rows => {}
            (Ok(cpu_rows), Ok(metal_rows)) => failures.push(format!(
                "{}: CPU/Metal mismatch: CPU={cpu_rows:?}, Metal={metal_rows:?}",
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
    Ok(())
}
