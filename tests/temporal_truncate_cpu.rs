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

fn value(query: &str) -> Result<ResultValue> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first())
        .cloned()
        .ok_or_else(|| irongraph::Error::internal("query returned no value"))
}

fn string_value(query: &str) -> Result<String> {
    match value(query)? {
        ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.to_string()),
        _ => Err(irongraph::Error::internal("query did not return a STRING")),
    }
}

#[test]
fn date_truncation_covers_calendar_and_iso_boundaries() -> Result<()> {
    let cases = [
        (
            "RETURN toString(date.truncate('millennium', date({year: 2017, month: 10, day: 11}), {day: 2}))",
            "2000-01-02",
        ),
        (
            "RETURN toString(date.truncate('century', date({year: 1984, month: 10, day: 11}), {}))",
            "1900-01-01",
        ),
        (
            "RETURN toString(date.truncate('decade', date({year: 1984, month: 10, day: 11}), {}))",
            "1980-01-01",
        ),
        (
            "RETURN toString(date.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {day: 5}))",
            "1984-01-05",
        ),
        (
            "RETURN toString(date.truncate('quarter', date({year: 1984, month: 11, day: 11}), {day: 2}))",
            "1984-10-02",
        ),
        (
            "RETURN toString(date.truncate('week', date({year: 1984, month: 10, day: 11}), {dayOfWeek: 2}))",
            "1984-10-09",
        ),
        (
            "RETURN toString(date.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12}), {}))",
            "1984-10-11",
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(string_value(query)?, expected);
    }
    Ok(())
}

#[test]
fn local_datetime_truncation_composes_smaller_subsecond_fields() -> Result<()> {
    let input = "localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})";
    let cases = [
        (
            format!("RETURN toString(localdatetime.truncate('hour', {input}, {{nanosecond: 2}}))"),
            "1984-10-11T12:00:00.000000002",
        ),
        (
            format!("RETURN toString(localdatetime.truncate('millisecond', {input}, {{}}))"),
            "1984-10-11T12:31:14.645",
        ),
        (
            format!(
                "RETURN toString(localdatetime.truncate('millisecond', {input}, {{nanosecond: 2}}))"
            ),
            "1984-10-11T12:31:14.645000002",
        ),
        (
            format!(
                "RETURN toString(localdatetime.truncate('microsecond', {input}, {{nanosecond: 2}}))"
            ),
            "1984-10-11T12:31:14.645876002",
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(string_value(&query)?, expected);
    }
    Ok(())
}

#[test]
fn zoned_truncation_preserves_or_replaces_wall_clock_zone() -> Result<()> {
    let input = "datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'})";
    assert_eq!(
        string_value(&format!(
            "RETURN toString(datetime.truncate('hour', {input}, {{}}))"
        ))?,
        "1984-10-11T12:00:00-01:00"
    );
    let named = value(&format!(
        "RETURN datetime.truncate('hour', {input}, {{timezone: 'Europe/Stockholm'}})"
    ))?;
    let ResultValue::Scalar(ScalarValue::ZonedDateTime {
        seconds,
        nanos,
        timezone,
    }) = named
    else {
        return Err(irongraph::Error::internal(
            "named truncation did not return DATETIME",
        ));
    };
    assert_eq!(timezone.as_ref(), "Europe/Stockholm");
    assert_eq!(nanos, 0);
    let instant = chrono::DateTime::<chrono::Utc>::from_timestamp(seconds, nanos)
        .ok_or_else(|| irongraph::Error::internal("invalid returned timestamp"))?;
    let zone: chrono_tz::Tz = "Europe/Stockholm"
        .parse()
        .map_err(|_| irongraph::Error::internal("test timezone is unavailable"))?;
    assert_eq!(
        instant
            .with_timezone(&zone)
            .naive_local()
            .format("%Y-%m-%dT%H:%M:%S")
            .to_string(),
        "1984-10-11T12:00:00"
    );
    Ok(())
}

#[test]
fn time_truncation_preserves_offset_and_defaults_local_inputs_to_utc() -> Result<()> {
    let zoned = value(
        "RETURN time.truncate('minute', time({hour: 12, minute: 31, second: 14, timezone: '-01:00'}), {})",
    )?;
    assert_eq!(
        zoned,
        ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: (12 * 3_600 + 31 * 60) * 1_000_000_000,
            offset_seconds: -3_600,
        })
    );
    let local = value(
        "RETURN time.truncate('hour', localtime({hour: 12, minute: 31}), {timezone: '+01:00'})",
    )?;
    assert_eq!(
        local,
        ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: 12 * 3_600 * 1_000_000_000,
            offset_seconds: 3_600,
        })
    );
    assert_eq!(
        value("RETURN localtime.truncate('day', time({hour: 12, timezone: '+01:00'}), {})")?,
        ResultValue::Scalar(ScalarValue::LocalTime(0))
    );
    Ok(())
}

#[test]
fn truncation_nulls_and_invalid_arguments_are_not_silently_coerced() -> Result<()> {
    assert_eq!(
        value("RETURN date.truncate('day', null, {})")?,
        ResultValue::Scalar(ScalarValue::Null)
    );

    let invalid = [
        "RETURN date.truncate('nanosecond', date({year: 1984}), {})",
        "RETURN date.truncate('hour', date({year: 1984}), {})",
        "RETURN date.truncate('month', date({year: 1984}), {month: 2})",
        "RETURN localdatetime.truncate('hour', date({year: 1984}), {})",
        "RETURN time.truncate('hour', localtime({hour: 12}), {timezone: 'Europe/Stockholm'})",
        "RETURN date.truncate('day', date({year: 1984}), 1)",
    ];
    for query in invalid {
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid truncation succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
    }
    Ok(())
}
