//! Backend-neutral temporal value ABI and canonical scalar semantics.

use std::mem::size_of;

use chrono::{Datelike, Duration, NaiveDate, NaiveDateTime, Offset, TimeZone, Timelike};
use irongraph_types::{Error, ErrorCode, Result, ScalarValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    NotEq,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

/// Physical type of one SSA register. Nullability is carried by a separate validity column.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentRowValueType {
    Boolean = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Date = 5,
    LocalTime = 6,
    ZonedTime = 7,
    LocalDateTime = 8,
    ZonedDateTime = 9,
    /// A bounded homogeneous list of INTEGER values. The row frame stores one packed
    /// `(offset, length)` reference; flattened elements live in the admitted list arena.
    List = 10,
}

impl ResidentRowValueType {
    pub const fn payload_bytes(self) -> usize {
        match self {
            Self::Boolean => size_of::<u8>(),
            Self::Integer
            | Self::Float
            | Self::String
            | Self::Date
            | Self::LocalTime
            | Self::List => size_of::<u64>(),
            Self::ZonedTime | Self::LocalDateTime => size_of::<i64>() + size_of::<u32>(),
            // Zoned datetimes retain seconds/nanos plus one packed reference into the register's
            // timezone UTF-8 arena. The bytes themselves are admitted separately below.
            Self::ZonedDateTime => size_of::<i64>() + size_of::<u32>() + size_of::<u64>(),
        }
    }

    pub const fn is_numeric(self) -> bool {
        matches!(self, Self::Integer | Self::Float)
    }

    pub const fn is_temporal(self) -> bool {
        matches!(
            self,
            Self::Date
                | Self::LocalTime
                | Self::ZonedTime
                | Self::LocalDateTime
                | Self::ZonedDateTime
        )
    }
}

/// One native scalar field projected from an already-typed temporal value. The discriminants are
/// part of the Metal row-program ABI; append-only changes require a manifest-version bump.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ResidentTemporalAccessor {
    Year = 1,
    Quarter = 2,
    Month = 3,
    Week = 4,
    WeekYear = 5,
    Day = 6,
    OrdinalDay = 7,
    WeekDay = 8,
    DayOfQuarter = 9,
    Hour = 10,
    Minute = 11,
    Second = 12,
    Millisecond = 13,
    Microsecond = 14,
    Nanosecond = 15,
    Timezone = 16,
    Offset = 17,
    OffsetMinutes = 18,
    OffsetSeconds = 19,
    EpochSeconds = 20,
    EpochMillis = 21,
    Years = 22,
    Quarters = 23,
    Months = 24,
    Weeks = 25,
    Days = 26,
    Hours = 27,
    Minutes = 28,
    Seconds = 29,
    Milliseconds = 30,
    Microseconds = 31,
    Nanoseconds = 32,
    QuartersOfYear = 33,
    MonthsOfQuarter = 34,
    MonthsOfYear = 35,
    DaysOfWeek = 36,
    MinutesOfHour = 37,
    SecondsOfMinute = 38,
    MillisecondsOfSecond = 39,
    MicrosecondsOfSecond = 40,
    NanosecondsOfSecond = 41,
}

impl ResidentTemporalAccessor {
    #[must_use]
    pub fn from_property_name(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "year" => Self::Year,
            "quarter" => Self::Quarter,
            "month" => Self::Month,
            "week" => Self::Week,
            "weekyear" => Self::WeekYear,
            "day" => Self::Day,
            "ordinalday" => Self::OrdinalDay,
            "weekday" | "dayofweek" => Self::WeekDay,
            "dayofquarter" => Self::DayOfQuarter,
            "hour" => Self::Hour,
            "minute" => Self::Minute,
            "second" => Self::Second,
            "millisecond" => Self::Millisecond,
            "microsecond" => Self::Microsecond,
            "nanosecond" => Self::Nanosecond,
            "timezone" => Self::Timezone,
            "offset" => Self::Offset,
            "offsetminutes" => Self::OffsetMinutes,
            "offsetseconds" => Self::OffsetSeconds,
            "epochseconds" => Self::EpochSeconds,
            "epochmillis" => Self::EpochMillis,
            "years" => Self::Years,
            "quarters" => Self::Quarters,
            "months" => Self::Months,
            "weeks" => Self::Weeks,
            "days" => Self::Days,
            "hours" => Self::Hours,
            "minutes" => Self::Minutes,
            "seconds" => Self::Seconds,
            "milliseconds" => Self::Milliseconds,
            "microseconds" => Self::Microseconds,
            "nanoseconds" => Self::Nanoseconds,
            "quartersofyear" => Self::QuartersOfYear,
            "monthsofquarter" => Self::MonthsOfQuarter,
            "monthsofyear" => Self::MonthsOfYear,
            "daysofweek" => Self::DaysOfWeek,
            "minutesofhour" => Self::MinutesOfHour,
            "secondsofminute" => Self::SecondsOfMinute,
            "millisecondsofsecond" => Self::MillisecondsOfSecond,
            "microsecondsofsecond" => Self::MicrosecondsOfSecond,
            "nanosecondsofsecond" => Self::NanosecondsOfSecond,
            _ => return None,
        })
    }

    #[must_use]
    pub const fn property_name(self) -> &'static str {
        match self {
            Self::Year => "year",
            Self::Quarter => "quarter",
            Self::Month => "month",
            Self::Week => "week",
            Self::WeekYear => "weekyear",
            Self::Day => "day",
            Self::OrdinalDay => "ordinalday",
            Self::WeekDay => "weekday",
            Self::DayOfQuarter => "dayofquarter",
            Self::Hour => "hour",
            Self::Minute => "minute",
            Self::Second => "second",
            Self::Millisecond => "millisecond",
            Self::Microsecond => "microsecond",
            Self::Nanosecond => "nanosecond",
            Self::Timezone => "timezone",
            Self::Offset => "offset",
            Self::OffsetMinutes => "offsetminutes",
            Self::OffsetSeconds => "offsetseconds",
            Self::EpochSeconds => "epochseconds",
            Self::EpochMillis => "epochmillis",
            Self::Years => "years",
            Self::Quarters => "quarters",
            Self::Months => "months",
            Self::Weeks => "weeks",
            Self::Days => "days",
            Self::Hours => "hours",
            Self::Minutes => "minutes",
            Self::Seconds => "seconds",
            Self::Milliseconds => "milliseconds",
            Self::Microseconds => "microseconds",
            Self::Nanoseconds => "nanoseconds",
            Self::QuartersOfYear => "quartersofyear",
            Self::MonthsOfQuarter => "monthsofquarter",
            Self::MonthsOfYear => "monthsofyear",
            Self::DaysOfWeek => "daysofweek",
            Self::MinutesOfHour => "minutesofhour",
            Self::SecondsOfMinute => "secondsofminute",
            Self::MillisecondsOfSecond => "millisecondsofsecond",
            Self::MicrosecondsOfSecond => "microsecondsofsecond",
            Self::NanosecondsOfSecond => "nanosecondsofsecond",
        }
    }

    #[must_use]
    pub const fn output_type(self) -> ResidentRowValueType {
        if matches!(self, Self::Timezone | Self::Offset) {
            ResidentRowValueType::String
        } else {
            ResidentRowValueType::Integer
        }
    }

    pub const fn is_duration(self) -> bool {
        matches!(
            self,
            Self::Years
                | Self::Quarters
                | Self::Months
                | Self::Weeks
                | Self::Days
                | Self::Hours
                | Self::Minutes
                | Self::Seconds
                | Self::Milliseconds
                | Self::Microseconds
                | Self::Nanoseconds
                | Self::QuartersOfYear
                | Self::MonthsOfQuarter
                | Self::MonthsOfYear
                | Self::DaysOfWeek
                | Self::MinutesOfHour
                | Self::SecondsOfMinute
                | Self::MillisecondsOfSecond
                | Self::MicrosecondsOfSecond
                | Self::NanosecondsOfSecond
        )
    }

    pub const fn supports(self, value_type: ResidentRowValueType) -> bool {
        match self {
            Self::Year
            | Self::Quarter
            | Self::Month
            | Self::Week
            | Self::WeekYear
            | Self::Day
            | Self::OrdinalDay
            | Self::WeekDay
            | Self::DayOfQuarter => matches!(
                value_type,
                ResidentRowValueType::Date
                    | ResidentRowValueType::LocalDateTime
                    | ResidentRowValueType::ZonedDateTime
            ),
            Self::Hour
            | Self::Minute
            | Self::Second
            | Self::Millisecond
            | Self::Microsecond
            | Self::Nanosecond => matches!(
                value_type,
                ResidentRowValueType::LocalTime
                    | ResidentRowValueType::ZonedTime
                    | ResidentRowValueType::LocalDateTime
                    | ResidentRowValueType::ZonedDateTime
            ),
            Self::Timezone | Self::Offset | Self::OffsetMinutes | Self::OffsetSeconds => matches!(
                value_type,
                ResidentRowValueType::ZonedTime | ResidentRowValueType::ZonedDateTime
            ),
            Self::EpochSeconds | Self::EpochMillis => {
                matches!(value_type, ResidentRowValueType::ZonedDateTime)
            }
            _ => false,
        }
    }
}

const NANOS_PER_SECOND: i128 = 1_000_000_000;
const NANOS_PER_DAY: i128 = 86_400 * NANOS_PER_SECOND;

pub fn evaluate_temporal_accessor(
    value: &ScalarValue,
    accessor: ResidentTemporalAccessor,
) -> Result<ScalarValue> {
    use ResidentTemporalAccessor as A;
    let integer = |value| Ok(ScalarValue::Integer(value));
    match value {
        ScalarValue::Date(days) => date_accessor(date_from_epoch_day(*days)?, accessor, integer),
        ScalarValue::LocalTime(nanos) => time_accessor(*nanos, accessor, integer),
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => match accessor {
            A::Timezone | A::Offset => {
                Ok(ScalarValue::String(offset_text(*offset_seconds)?.into()))
            }
            A::OffsetMinutes => integer(i64::from(*offset_seconds) / 60),
            A::OffsetSeconds => integer(i64::from(*offset_seconds)),
            _ => time_accessor(*nanos, accessor, integer),
        },
        ScalarValue::LocalDateTime { seconds, nanos } => {
            let value = datetime_from_stored(*seconds, *nanos)?;
            date_accessor(value.date(), accessor, integer).or_else(|_| {
                time_accessor(
                    i64::from(value.time().num_seconds_from_midnight()) * 1_000_000_000
                        + i64::from(value.nanosecond()),
                    accessor,
                    integer,
                )
            })
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            if accessor == A::Timezone {
                return Ok(ScalarValue::String(timezone.clone()));
            }
            if accessor == A::EpochSeconds {
                return integer(*seconds);
            }
            if accessor == A::EpochMillis {
                return integer(
                    seconds
                        .checked_mul(1_000)
                        .and_then(|v| v.checked_add(i64::from(*nanos / 1_000_000)))
                        .ok_or_else(|| temporal_range("datetime epochMillis overflow"))?,
                );
            }
            let (local, offset) = zoned_local(*seconds, *nanos, timezone)?;
            match accessor {
                A::Offset => Ok(ScalarValue::String(offset_text(offset)?.into())),
                A::OffsetMinutes => integer(i64::from(offset) / 60),
                A::OffsetSeconds => integer(i64::from(offset)),
                _ => date_accessor(local.date(), accessor, integer).or_else(|_| {
                    time_accessor(
                        i64::from(local.time().num_seconds_from_midnight()) * 1_000_000_000
                            + i64::from(local.nanosecond()),
                        accessor,
                        integer,
                    )
                }),
            }
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => duration_accessor(*months, *days, *seconds, *nanos, accessor),
        _ => Err(Error::new(
            ErrorCode::QueryType,
            "temporal accessor requires a temporal or duration value",
        )),
    }
}

fn date_accessor<F>(
    date: NaiveDate,
    accessor: ResidentTemporalAccessor,
    integer: F,
) -> Result<ScalarValue>
where
    F: Fn(i64) -> Result<ScalarValue>,
{
    use ResidentTemporalAccessor as A;
    let value = match accessor {
        A::Year => i64::from(date.year()),
        A::Quarter => i64::from((date.month0() / 3) + 1),
        A::Month => i64::from(date.month()),
        A::Week => i64::from(date.iso_week().week()),
        A::WeekYear => i64::from(date.iso_week().year()),
        A::Day => i64::from(date.day()),
        A::OrdinalDay => i64::from(date.ordinal()),
        A::WeekDay => i64::from(date.weekday().number_from_monday()),
        A::DayOfQuarter => i64::from(
            date.ordinal()
                - NaiveDate::from_ymd_opt(date.year(), (date.month0() / 3) * 3 + 1, 1)
                    .ok_or_else(|| temporal_range("invalid quarter"))?
                    .ordinal()
                + 1,
        ),
        _ => return Err(unsupported_accessor(accessor)),
    };
    integer(value)
}

fn time_accessor<F>(
    nanos: i64,
    accessor: ResidentTemporalAccessor,
    integer: F,
) -> Result<ScalarValue>
where
    F: Fn(i64) -> Result<ScalarValue>,
{
    if !(0..86_400_000_000_000).contains(&nanos) {
        return Err(temporal_range("time is outside one day"));
    }
    use ResidentTemporalAccessor as A;
    let seconds = nanos / 1_000_000_000;
    let sub = nanos % 1_000_000_000;
    integer(match accessor {
        A::Hour => seconds / 3_600,
        A::Minute => (seconds / 60) % 60,
        A::Second => seconds % 60,
        A::Millisecond => sub / 1_000_000,
        A::Microsecond => sub / 1_000,
        A::Nanosecond => sub,
        _ => return Err(unsupported_accessor(accessor)),
    })
}

fn duration_accessor(
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
    accessor: ResidentTemporalAccessor,
) -> Result<ScalarValue> {
    use ResidentTemporalAccessor as A;
    let value = match accessor {
        A::Years => months / 12,
        A::Quarters => months / 3,
        A::Months => months,
        A::Weeks => days / 7,
        A::Days => days,
        A::Hours => seconds / 3_600,
        A::Minutes => seconds / 60,
        A::Seconds => seconds,
        A::Milliseconds => seconds
            .checked_mul(1_000)
            .and_then(|v| v.checked_add(i64::from(nanos) / 1_000_000))
            .ok_or_else(|| temporal_range("duration milliseconds overflow"))?,
        A::Microseconds => seconds
            .checked_mul(1_000_000)
            .and_then(|v| v.checked_add(i64::from(nanos) / 1_000))
            .ok_or_else(|| temporal_range("duration microseconds overflow"))?,
        A::Nanoseconds => seconds
            .checked_mul(1_000_000_000)
            .and_then(|v| v.checked_add(i64::from(nanos)))
            .ok_or_else(|| temporal_range("duration nanoseconds overflow"))?,
        A::QuartersOfYear => (months / 3) % 4,
        A::MonthsOfQuarter => months % 3,
        A::MonthsOfYear => months % 12,
        A::DaysOfWeek => days % 7,
        A::MinutesOfHour => (seconds / 60) % 60,
        A::SecondsOfMinute => seconds % 60,
        A::MillisecondsOfSecond => i64::from(nanos) / 1_000_000,
        A::MicrosecondsOfSecond => i64::from(nanos) / 1_000,
        A::NanosecondsOfSecond => i64::from(nanos),
        _ => return Err(unsupported_accessor(accessor)),
    };
    Ok(ScalarValue::Integer(value))
}

pub fn compare_temporal_scalars(
    left: &ScalarValue,
    right: &ScalarValue,
    operation: CompareOp,
) -> Result<ScalarValue> {
    if matches!(left, ScalarValue::Null) || matches!(right, ScalarValue::Null) {
        return Ok(ScalarValue::Null);
    }
    ensure_temporal(left)?;
    ensure_temporal(right)?;
    if matches!(operation, CompareOp::Eq | CompareOp::NotEq) {
        let equal = temporal_family(left) == temporal_family(right) && left == right;
        return Ok(ScalarValue::Boolean(if operation == CompareOp::Eq {
            equal
        } else {
            !equal
        }));
    }
    if temporal_family(left) != temporal_family(right)
        || matches!(left, ScalarValue::Duration { .. })
    {
        return Ok(ScalarValue::Null);
    }
    let ordering = temporal_key(left)?.cmp(&temporal_key(right)?);
    let value = match operation {
        CompareOp::Eq => ordering.is_eq(),
        CompareOp::NotEq => !ordering.is_eq(),
        CompareOp::Less => ordering.is_lt(),
        CompareOp::LessOrEqual => !ordering.is_gt(),
        CompareOp::Greater => ordering.is_gt(),
        CompareOp::GreaterOrEqual => !ordering.is_lt(),
    };
    Ok(ScalarValue::Boolean(value))
}

fn ensure_temporal(value: &ScalarValue) -> Result<()> {
    temporal_family(value).map(|_| ()).ok_or_else(|| {
        Error::new(
            ErrorCode::QueryType,
            "temporal comparison requires temporal values",
        )
    })
}

fn temporal_family(value: &ScalarValue) -> Option<u8> {
    Some(match value {
        ScalarValue::Date(_) => 0,
        ScalarValue::LocalTime(_) => 1,
        ScalarValue::ZonedTime { .. } => 2,
        ScalarValue::LocalDateTime { .. } => 3,
        ScalarValue::ZonedDateTime { .. } => 4,
        ScalarValue::Duration { .. } => 5,
        _ => return None,
    })
}

fn temporal_key(value: &ScalarValue) -> Result<(u8, i128, i128, String)> {
    Ok(match value {
        ScalarValue::Date(days) => (0, i128::from(*days), 0, String::new()),
        ScalarValue::LocalTime(nanos) => (1, i128::from(*nanos), 0, String::new()),
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => (
            2,
            i128::from(*nanos) - i128::from(*offset_seconds) * NANOS_PER_SECOND,
            i128::from(*offset_seconds),
            String::new(),
        ),
        ScalarValue::LocalDateTime { seconds, nanos } => {
            (3, i128::from(*seconds), i128::from(*nanos), String::new())
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => (
            4,
            i128::from(*seconds),
            i128::from(*nanos),
            timezone.to_string(),
        ),
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => (
            5,
            i128::from(*months) * 2_629_746_000_000_000
                + i128::from(*days) * 86_400_000_000_000
                + i128::from(*seconds) * NANOS_PER_SECOND
                + i128::from(*nanos),
            0,
            String::new(),
        ),
        _ => {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal comparison requires temporal values",
            ));
        }
    })
}

pub fn apply_duration_to_temporal_scalar(
    temporal: &ScalarValue,
    months: i64,
    days: i64,
    seconds: i64,
    nanos: i32,
) -> Result<ScalarValue> {
    match temporal {
        ScalarValue::Date(epoch_day) => {
            let date = add_months(date_from_epoch_day(*epoch_day)?, months)?;
            let elapsed_days = seconds / 86_400;
            let date = date
                .checked_add_signed(Duration::days(
                    days.checked_add(elapsed_days)
                        .ok_or_else(|| temporal_range("date duration overflow"))?,
                ))
                .ok_or_else(|| temporal_range("date duration overflow"))?;
            Ok(ScalarValue::Date(epoch_day_from_date(date)?))
        }
        ScalarValue::LocalTime(value) => {
            Ok(ScalarValue::LocalTime(add_clock(*value, seconds, nanos)?))
        }
        ScalarValue::ZonedTime {
            nanos: value,
            offset_seconds,
        } => Ok(ScalarValue::ZonedTime {
            nanos: add_clock(*value, seconds, nanos)?,
            offset_seconds: *offset_seconds,
        }),
        ScalarValue::LocalDateTime {
            seconds: value,
            nanos: sub,
        } => {
            let local = add_calendar(datetime_from_stored(*value, *sub)?, months, days)?;
            let (seconds, nanos) = add_elapsed(
                local.and_utc().timestamp(),
                local.nanosecond(),
                seconds,
                nanos,
            )?;
            datetime_from_stored(seconds, nanos)?;
            Ok(ScalarValue::LocalDateTime { seconds, nanos })
        }
        ScalarValue::ZonedDateTime {
            seconds: value,
            nanos: sub,
            timezone,
        } => {
            let (base_seconds, base_nanos) = if months == 0 && days == 0 {
                (*value, *sub)
            } else {
                let (local, _) = zoned_local(*value, *sub, timezone)?;
                let local = add_calendar(local, months, days)?;
                local_to_instant(local, timezone)?
            };
            let (seconds, nanos) = add_elapsed(base_seconds, base_nanos, seconds, nanos)?;
            zoned_local(seconds, nanos, timezone)?;
            Ok(ScalarValue::ZonedDateTime {
                seconds,
                nanos,
                timezone: timezone.clone(),
            })
        }
        _ => Err(Error::new(
            ErrorCode::QueryType,
            "duration arithmetic requires a temporal value",
        )),
    }
}

pub fn format_temporal_scalar(value: &ScalarValue) -> Result<Option<String>> {
    Ok(Some(match value {
        ScalarValue::Null => return Ok(None),
        ScalarValue::Date(days) => date_from_epoch_day(*days)?.format("%Y-%m-%d").to_string(),
        ScalarValue::LocalTime(nanos) => format_time(*nanos)?,
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => format!("{}{}", format_time(*nanos)?, offset_text(*offset_seconds)?),
        ScalarValue::LocalDateTime { seconds, nanos } => datetime_from_stored(*seconds, *nanos)?
            .format("%Y-%m-%dT%H:%M:%S%.f")
            .to_string(),
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            let utc = chrono::DateTime::<chrono::Utc>::from_timestamp(*seconds, *nanos)
                .ok_or_else(|| temporal_range("datetime is outside the format range"))?;
            if timezone.as_ref() == "UTC" {
                utc.to_rfc3339()
            } else if let Ok(zone) = timezone.parse::<chrono_tz::Tz>() {
                format!("{}[{timezone}]", utc.with_timezone(&zone).to_rfc3339())
            } else if let Ok(offset) = timezone.parse::<chrono::FixedOffset>() {
                utc.with_timezone(&offset).to_rfc3339()
            } else {
                return Err(temporal_range("datetime timezone is invalid"));
            }
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => format_duration(*months, *days, *seconds, *nanos),
        _ => {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal serialization requires a temporal value",
            ));
        }
    }))
}

fn date_from_epoch_day(days: i64) -> Result<NaiveDate> {
    NaiveDate::from_ymd_opt(1970, 1, 1)
        .and_then(|date| date.checked_add_signed(Duration::days(days)))
        .ok_or_else(|| temporal_range("date is outside the supported range"))
}
fn epoch_day_from_date(date: NaiveDate) -> Result<i64> {
    Ok(i64::from(
        date.signed_duration_since(
            NaiveDate::from_ymd_opt(1970, 1, 1).ok_or_else(|| temporal_range("invalid epoch"))?,
        )
        .num_days(),
    ))
}
fn datetime_from_stored(seconds: i64, nanos: u32) -> Result<NaiveDateTime> {
    chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanos)
        .map(|value| value.naive_utc())
        .ok_or_else(|| temporal_range("datetime is outside the supported range"))
}
fn add_months(date: NaiveDate, months: i64) -> Result<NaiveDate> {
    let total = i64::from(date.year())
        .checked_mul(12)
        .and_then(|v| v.checked_add(i64::from(date.month0())))
        .and_then(|v| v.checked_add(months))
        .ok_or_else(|| temporal_range("calendar month overflow"))?;
    let year = i32::try_from(total.div_euclid(12))
        .map_err(|_| temporal_range("calendar year overflow"))?;
    let month = u32::try_from(total.rem_euclid(12) + 1)
        .map_err(|_| temporal_range("calendar month overflow"))?;
    let mut day = date.day();
    loop {
        if let Some(value) = NaiveDate::from_ymd_opt(year, month, day) {
            return Ok(value);
        }
        day = day
            .checked_sub(1)
            .ok_or_else(|| temporal_range("calendar date overflow"))?;
    }
}
fn add_calendar(value: NaiveDateTime, months: i64, days: i64) -> Result<NaiveDateTime> {
    let date = add_months(value.date(), months)?
        .checked_add_signed(Duration::days(days))
        .ok_or_else(|| temporal_range("calendar day overflow"))?;
    Ok(date.and_time(value.time()))
}
fn add_clock(value: i64, seconds: i64, nanos: i32) -> Result<i64> {
    let elapsed = i128::from(seconds)
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|v| v.checked_add(i128::from(nanos)))
        .ok_or_else(|| temporal_range("time duration overflow"))?;
    i64::try_from((i128::from(value) + elapsed).rem_euclid(NANOS_PER_DAY))
        .map_err(|_| temporal_range("time duration overflow"))
}
fn add_elapsed(base_seconds: i64, base_nanos: u32, seconds: i64, nanos: i32) -> Result<(i64, u32)> {
    let value = i128::from(base_seconds)
        .checked_mul(NANOS_PER_SECOND)
        .and_then(|v| v.checked_add(i128::from(base_nanos)))
        .and_then(|v| v.checked_add(i128::from(seconds) * NANOS_PER_SECOND + i128::from(nanos)))
        .ok_or_else(|| temporal_range("datetime duration overflow"))?;
    let seconds = i64::try_from(value.div_euclid(NANOS_PER_SECOND))
        .map_err(|_| temporal_range("datetime duration overflow"))?;
    let nanos = u32::try_from(value.rem_euclid(NANOS_PER_SECOND))
        .map_err(|_| temporal_range("datetime duration overflow"))?;
    Ok((seconds, nanos))
}
fn zoned_local(seconds: i64, nanos: u32, timezone: &str) -> Result<(NaiveDateTime, i32)> {
    let utc = chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanos)
        .ok_or_else(|| temporal_range("datetime is outside the supported range"))?;
    if let Ok(zone) = timezone.parse::<chrono_tz::Tz>() {
        let local = utc.with_timezone(&zone);
        Ok((local.naive_local(), local.offset().fix().local_minus_utc()))
    } else if let Ok(offset) = timezone.parse::<chrono::FixedOffset>() {
        Ok((
            utc.with_timezone(&offset).naive_local(),
            offset.local_minus_utc(),
        ))
    } else {
        Err(temporal_range("datetime timezone is invalid"))
    }
}
fn local_to_instant(local: NaiveDateTime, timezone: &str) -> Result<(i64, u32)> {
    if let Ok(zone) = timezone.parse::<chrono_tz::Tz>() {
        let value = zone
            .from_local_datetime(&local)
            .single()
            .ok_or_else(|| temporal_range("local datetime is ambiguous or nonexistent"))?;
        Ok((value.timestamp(), value.timestamp_subsec_nanos()))
    } else if let Ok(offset) = timezone.parse::<chrono::FixedOffset>() {
        let value = offset
            .from_local_datetime(&local)
            .single()
            .ok_or_else(|| temporal_range("local datetime is invalid"))?;
        Ok((value.timestamp(), value.timestamp_subsec_nanos()))
    } else {
        Err(temporal_range("datetime timezone is invalid"))
    }
}
fn offset_text(offset: i32) -> Result<String> {
    chrono::FixedOffset::east_opt(offset)
        .map(|v| v.to_string())
        .ok_or_else(|| temporal_range("time offset is invalid"))
}
fn format_time(nanos: i64) -> Result<String> {
    if !(0..86_400_000_000_000).contains(&nanos) {
        return Err(temporal_range("local time is outside one day"));
    }
    let seconds =
        u32::try_from(nanos / 1_000_000_000).map_err(|_| temporal_range("time overflow"))?;
    let sub = u32::try_from(nanos % 1_000_000_000).map_err(|_| temporal_range("time overflow"))?;
    Ok(
        chrono::NaiveTime::from_num_seconds_from_midnight_opt(seconds, sub)
            .ok_or_else(|| temporal_range("local time is invalid"))?
            .format("%H:%M:%S%.f")
            .to_string(),
    )
}
fn format_duration(months: i64, days: i64, seconds: i64, nanos: i32) -> String {
    const NANOS_PER_SECOND: i128 = 1_000_000_000;

    let mut rendered = String::from("P");
    let years = months / 12;
    let remaining_months = months % 12;
    if years != 0 {
        rendered.push_str(&format!("{years}Y"));
    }
    if remaining_months != 0 {
        rendered.push_str(&format!("{remaining_months}M"));
    }
    if days != 0 {
        rendered.push_str(&format!("{days}D"));
    }

    let total_nanos = i128::from(seconds) * NANOS_PER_SECOND + i128::from(nanos);
    if total_nanos != 0 {
        rendered.push('T');
        let whole_seconds = total_nanos / NANOS_PER_SECOND;
        let fractional_nanos = total_nanos % NANOS_PER_SECOND;
        let hours = whole_seconds / 3_600;
        let after_hours = whole_seconds % 3_600;
        let minutes = after_hours / 60;
        let second_nanos = (after_hours % 60) * NANOS_PER_SECOND + fractional_nanos;
        if hours != 0 {
            rendered.push_str(&format!("{hours}H"));
        }
        if minutes != 0 {
            rendered.push_str(&format!("{minutes}M"));
        }
        if second_nanos != 0 {
            let sign = if second_nanos < 0 { "-" } else { "" };
            let absolute = second_nanos.abs();
            let component_seconds = absolute / NANOS_PER_SECOND;
            let component_nanos = absolute % NANOS_PER_SECOND;
            if component_nanos == 0 {
                rendered.push_str(&format!("{sign}{component_seconds}S"));
            } else {
                let fraction = format!("{component_nanos:09}");
                rendered.push_str(&format!(
                    "{sign}{component_seconds}.{}S",
                    fraction.trim_end_matches('0')
                ));
            }
        }
    } else if years == 0 && remaining_months == 0 && days == 0 {
        rendered.push_str("T0S");
    }
    rendered
}
fn unsupported_accessor(accessor: ResidentTemporalAccessor) -> Error {
    Error::new(
        ErrorCode::QueryType,
        format!("unsupported temporal field `{}`", accessor.property_name()),
    )
}
fn temporal_range(message: &'static str) -> Error {
    Error::new(ErrorCode::TemporalRange, message)
}
