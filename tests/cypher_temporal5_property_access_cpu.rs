// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

fn context(graph: &GraphStore) -> ExecutionContext<'_> {
    ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn row(query: &str) -> Result<BTreeMap<String, ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("query returned no batch"))?;
    batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("query returned an empty column"))
        })
        .collect()
}

fn assert_integers(values: &BTreeMap<String, ResultValue>, expected: &[(&str, i64)]) -> Result<()> {
    for (name, expected) in expected {
        let Some(ResultValue::Scalar(ScalarValue::Integer(actual))) = values.get(*name) else {
            return Err(irongraph::Error::internal(format!(
                "column {name} did not return INTEGER"
            )));
        };
        assert_eq!(actual, expected, "column: {name}");
    }
    Ok(())
}

fn assert_string(values: &BTreeMap<String, ResultValue>, name: &str, expected: &str) -> Result<()> {
    let Some(ResultValue::Scalar(ScalarValue::String(actual))) = values.get(name) else {
        return Err(irongraph::Error::internal(format!(
            "column {name} did not return STRING"
        )));
    };
    assert_eq!(actual.as_ref(), expected, "column: {name}");
    Ok(())
}

#[test]
fn date_properties_follow_calendar_iso_week_and_quarter_rules() -> Result<()> {
    let values = row("WITH date({year: 1984, month: 10, day: 11}) AS d \
         RETURN d.year AS year, d.quarter AS quarter, d.month AS month, d.week AS week, \
                d.weekYear AS week_year, d.day AS day, d.ordinalDay AS ordinal_day, \
                d.weekDay AS week_day, d.dayOfWeek AS day_of_week, \
                d.dayOfQuarter AS day_of_quarter")?;
    assert_integers(
        &values,
        &[
            ("year", 1984),
            ("quarter", 4),
            ("month", 10),
            ("week", 41),
            ("week_year", 1984),
            ("day", 11),
            ("ordinal_day", 285),
            ("week_day", 4),
            ("day_of_week", 4),
            ("day_of_quarter", 11),
        ],
    )?;

    let boundary = row("WITH date({year: 1984, month: 1, day: 1}) AS d \
         RETURN d.year AS year, d.weekYear AS week_year, d.week AS week, d.weekDay AS week_day")?;
    assert_integers(
        &boundary,
        &[
            ("year", 1984),
            ("week_year", 1983),
            ("week", 52),
            ("week_day", 7),
        ],
    )
}

#[test]
fn local_time_properties_split_the_normalized_clock() -> Result<()> {
    let values = row(
        "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d \
         RETURN d.hour AS hour, d.minute AS minute, d.second AS second, \
                d.millisecond AS millisecond, d.microsecond AS microsecond, \
                d.nanosecond AS nanosecond",
    )?;
    assert_integers(
        &values,
        &[
            ("hour", 12),
            ("minute", 31),
            ("second", 14),
            ("millisecond", 645),
            ("microsecond", 645_876),
            ("nanosecond", 645_876_123),
        ],
    )
}

#[test]
fn zoned_time_properties_include_the_fixed_offset() -> Result<()> {
    let values = row(
        "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, \
                   timezone: '+01:00'}) AS d \
         RETURN d.hour AS hour, d.minute AS minute, d.second AS second, \
                d.millisecond AS millisecond, d.microsecond AS microsecond, \
                d.nanosecond AS nanosecond, d.timezone AS timezone, d.offset AS offset, \
                d.offsetMinutes AS offset_minutes, d.offsetSeconds AS offset_seconds",
    )?;
    assert_integers(
        &values,
        &[
            ("hour", 12),
            ("minute", 31),
            ("second", 14),
            ("millisecond", 645),
            ("microsecond", 645_876),
            ("nanosecond", 645_876_123),
            ("offset_minutes", 60),
            ("offset_seconds", 3_600),
        ],
    )?;
    assert_string(&values, "timezone", "+01:00")?;
    assert_string(&values, "offset", "+01:00")
}

#[test]
fn local_datetime_properties_combine_local_date_and_clock_fields() -> Result<()> {
    let values = row(
        "WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, \
                            second: 14, nanosecond: 645876123}) AS d \
         RETURN d.year AS year, d.quarter AS quarter, d.month AS month, d.week AS week, \
                d.weekYear AS week_year, d.day AS day, d.ordinalDay AS ordinal_day, \
                d.weekDay AS week_day, d.dayOfQuarter AS day_of_quarter, d.hour AS hour, \
                d.minute AS minute, d.second AS second, d.millisecond AS millisecond, \
                d.microsecond AS microsecond, d.nanosecond AS nanosecond",
    )?;
    assert_integers(
        &values,
        &[
            ("year", 1984),
            ("quarter", 4),
            ("month", 11),
            ("week", 45),
            ("week_year", 1984),
            ("day", 11),
            ("ordinal_day", 316),
            ("week_day", 7),
            ("day_of_quarter", 42),
            ("hour", 12),
            ("minute", 31),
            ("second", 14),
            ("millisecond", 645),
            ("microsecond", 645_876),
            ("nanosecond", 645_876_123),
        ],
    )
}

#[test]
fn zoned_datetime_properties_use_local_zone_fields_and_instant_epoch() -> Result<()> {
    let values = row(
        "WITH datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, \
                       nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d \
         RETURN d.year AS year, d.quarter AS quarter, d.month AS month, d.week AS week, \
                d.weekYear AS week_year, d.day AS day, d.ordinalDay AS ordinal_day, \
                d.weekDay AS week_day, d.dayOfQuarter AS day_of_quarter, d.hour AS hour, \
                d.minute AS minute, d.second AS second, d.millisecond AS millisecond, \
                d.microsecond AS microsecond, d.nanosecond AS nanosecond, \
                d.timezone AS timezone, d.offset AS offset, d.offsetMinutes AS offset_minutes, \
                d.offsetSeconds AS offset_seconds, d.epochSeconds AS epoch_seconds, \
                d.epochMillis AS epoch_millis",
    )?;
    assert_integers(
        &values,
        &[
            ("year", 1984),
            ("quarter", 4),
            ("month", 11),
            ("week", 45),
            ("week_year", 1984),
            ("day", 11),
            ("ordinal_day", 316),
            ("week_day", 7),
            ("day_of_quarter", 42),
            ("hour", 12),
            ("minute", 31),
            ("second", 14),
            ("millisecond", 645),
            ("microsecond", 645_876),
            ("nanosecond", 645_876_123),
            ("offset_minutes", 60),
            ("offset_seconds", 3_600),
            ("epoch_seconds", 469_020_674),
            ("epoch_millis", 469_020_674_645),
        ],
    )?;
    assert_string(&values, "timezone", "Europe/Stockholm")?;
    assert_string(&values, "offset", "+01:00")
}

#[test]
fn duration_properties_continue_to_use_normalized_duration_accessors() -> Result<()> {
    let values = row(
        "WITH duration({years: 1, months: 4, days: 10, hours: 1, minutes: 1, seconds: 1, \
                       nanoseconds: 111111111}) AS d \
         RETURN d.years AS years, d.quarters AS quarters, d.months AS months, \
                d.weeks AS weeks, d.days AS days, d.hours AS hours, d.minutes AS minutes, \
                d.seconds AS seconds, d.milliseconds AS milliseconds, \
                d.microseconds AS microseconds, d.nanoseconds AS nanoseconds, \
                d.quartersOfYear AS quarters_of_year, d.monthsOfQuarter AS months_of_quarter, \
                d.monthsOfYear AS months_of_year, d.daysOfWeek AS days_of_week, \
                d.minutesOfHour AS minutes_of_hour, d.secondsOfMinute AS seconds_of_minute, \
                d.millisecondsOfSecond AS milliseconds_of_second, \
                d.microsecondsOfSecond AS microseconds_of_second, \
                d.nanosecondsOfSecond AS nanoseconds_of_second",
    )?;
    assert_integers(
        &values,
        &[
            ("years", 1),
            ("quarters", 5),
            ("months", 16),
            ("weeks", 1),
            ("days", 10),
            ("hours", 1),
            ("minutes", 61),
            ("seconds", 3_661),
            ("milliseconds", 3_661_111),
            ("microseconds", 3_661_111_111),
            ("nanoseconds", 3_661_111_111_111),
            ("quarters_of_year", 1),
            ("months_of_quarter", 1),
            ("months_of_year", 4),
            ("days_of_week", 3),
            ("minutes_of_hour", 1),
            ("seconds_of_minute", 1),
            ("milliseconds_of_second", 111),
            ("microseconds_of_second", 111_111),
            ("nanoseconds_of_second", 111_111_111),
        ],
    )
}

#[test]
fn temporal_property_null_map_and_invalid_field_semantics_are_preserved() -> Result<()> {
    let values = row("WITH null AS missing, {year: 99} AS map \
         RETURN missing.year AS null_year, map.year AS map_year")?;
    assert_eq!(
        values.get("null_year"),
        Some(&ResultValue::Scalar(ScalarValue::Null))
    );
    assert_integers(&values, &[("map_year", 99)])?;

    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "WITH date('1984-10-11') AS d RETURN d.hour",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| irongraph::Error::internal("invalid temporal property succeeded"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("UnsupportedTemporalUnit"));
    Ok(())
}
