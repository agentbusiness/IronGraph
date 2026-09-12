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
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{GraphStore, IndexCatalog, TemporalStore},
};
use tokio_util::sync::CancellationToken;

fn context(graph: &GraphStore) -> ExecutionContext<'_> {
    context_with_backend(graph, None)
}

fn context_with_backend<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
) -> ExecutionContext<'a> {
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
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

#[test]
fn exact_numeric_tck_regressions_bypass_temporal_arithmetic_and_preserve_cpu_results() -> Result<()>
{
    let graph = GraphStore::default();
    let bookmark = Bookmark { term: 1, index: 0 };
    let image = ResidentProjectImage::build(
        ProjectId(uuid::Uuid::nil()),
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 4 * 1024 * 1024);
    cpu.admit_project(image)?;
    let cases = [
        (1956_u16, "RETURN 12 / 4 * 3 - 2 * 4"),
        (1957, "RETURN 12 / 4 * (3 - 2 * 4)"),
        (
            2131,
            "RETURN 4 * 2 + 3 * 2 AS a, 4 * 2 + (3 * 2) AS b, 4 * (2 + 3) * 2 AS c",
        ),
        (
            2132,
            "RETURN 4 * 2 + 3 / 2 AS a, 4 * 2 + (3 / 2) AS b, 4 * (2 + 3) / 2 AS c",
        ),
        (
            2134,
            "RETURN 4 * 2 - 3 * 2 AS a, 4 * 2 - (3 * 2) AS b, 4 * (2 - 3) * 2 AS c",
        ),
        (
            2135,
            "RETURN 4 * 2 - 3 / 2 AS a, 4 * 2 - (3 / 2) AS b, 4 * (2 - 3) / 2 AS c",
        ),
        (
            2137,
            "RETURN 4 / 2 + 3 * 2 AS a, 4 / 2 + (3 * 2) AS b, 4 / (2 + 3) * 2 AS c",
        ),
        (
            2138,
            "RETURN 4 / 2 + 3 / 2 AS a, 4 / 2 + (3 / 2) AS b, 4 / (2 + 3) / 2 AS c",
        ),
        (
            2140,
            "RETURN 4 / 2 - 3 * 2 AS a, 4 / 2 - (3 * 2) AS b, 4 / (2 - 3) * 2 AS c",
        ),
        (
            2141,
            "RETURN 4 / 2 - 3 / 2 AS a, 4 / 2 - (3 / 2) AS b, 4 / (2 - 3) / 2 AS c",
        ),
    ];
    for (report_id, query) in cases {
        let expected = QueryEngine.execute(query, &mut context(&graph))?;
        let actual = QueryEngine.execute(query, &mut context_with_backend(&graph, Some(&cpu)))?;
        assert_eq!(
            actual.result, expected.result,
            "numeric TCK report {report_id} changed under the resident CPU backend: {query}"
        );
    }
    Ok(())
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

fn duration(months: i64, days: i64, seconds: i64, nanos: i32) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Duration {
        months,
        days,
        seconds,
        nanos,
    })
}

fn string(values: &BTreeMap<String, ResultValue>, name: &str) -> Result<String> {
    let Some(ResultValue::Scalar(ScalarValue::String(value))) = values.get(name) else {
        return Err(irongraph::Error::internal(format!(
            "column {name} did not return STRING"
        )));
    };
    Ok(value.to_string())
}

#[test]
fn duration_map_builds_integral_fractional_and_subsecond_components() -> Result<()> {
    let values = row("RETURN \
         duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2}) AS integral, \
         duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3}) AS fractional, \
         duration({weeks: 2.5}) AS weeks, \
         duration({milliseconds: 1, microseconds: 2, nanoseconds: 3}) AS subsecond, \
         duration({seconds: -1, nanoseconds: -1}) AS normalized_negative")?;
    assert_eq!(values.get("integral"), Some(&duration(149, 14, 58_390, 2)));
    assert_eq!(
        values.get("fractional"),
        Some(&duration(155, 29, 122_293, 500_000_003))
    );
    assert_eq!(values.get("weeks"), Some(&duration(0, 17, 43_200, 0)));
    assert_eq!(values.get("subsecond"), Some(&duration(0, 0, 0, 1_002_003)));
    assert_eq!(
        values.get("normalized_negative"),
        Some(&duration(0, 0, -2, 999_999_999))
    );
    Ok(())
}

#[test]
fn temporal_values_apply_calendar_then_clock_duration_components() -> Result<()> {
    let values = row(
        "WITH duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3}) AS span, \
         date({year: 1984, month: 10, day: 11}) AS d, \
         localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS lt, \
         time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS t, \
         localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS ldt, \
         datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS dt \
         RETURN toString(d + span) AS date_sum, toString(d - span) AS date_diff, \
                toString(lt + span) AS local_time_sum, toString(lt - span) AS local_time_diff, \
                toString(t + span) AS time_sum, toString(t - span) AS time_diff, \
                toString(ldt + span) AS local_datetime_sum, toString(ldt - span) AS local_datetime_diff, \
                toString(dt + span) AS datetime_sum, toString(dt - span) AS datetime_diff, \
                toString(span + d) AS reverse_date_sum",
    )?;
    let expected = [
        ("date_sum", "1997-10-11"),
        ("date_diff", "1971-10-12"),
        ("local_time_sum", "22:29:27.500000004"),
        ("local_time_diff", "02:33:00.499999998"),
        ("time_sum", "22:29:27.500000004+01:00"),
        ("time_diff", "02:33:00.499999998+01:00"),
        ("local_datetime_sum", "1997-10-11T22:29:27.500000004"),
        ("local_datetime_diff", "1971-10-12T02:33:00.499999998"),
        ("datetime_sum", "1997-10-11T22:29:27.500000004+01:00"),
        ("datetime_diff", "1971-10-12T02:33:00.499999998+01:00"),
        ("reverse_date_sum", "1997-10-11"),
    ];
    for (name, expected) in expected {
        assert_eq!(string(&values, name)?, expected, "column: {name}");
    }
    Ok(())
}

#[test]
fn durations_add_subtract_and_normalize_componentwise() -> Result<()> {
    let values = row(
        "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}) AS a, \
              duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70}) AS b, \
              duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3}) AS c \
         RETURN a + b AS ab_sum, a - b AS ab_diff, b - a AS ba_diff, \
                a + c AS ac_sum, a - c AS ac_diff, c - a AS ca_diff, a - a AS zero",
    )?;
    let expected = [
        ("ab_sum", duration(150, 0, 115_340, 1)),
        ("ab_diff", duration(148, 28, 1_440, 1)),
        ("ba_diff", duration(-148, -28, -1_441, 999_999_999)),
        ("ac_sum", duration(304, 43, 180_683, 500_000_004)),
        ("ac_diff", duration(-6, -15, -63_904, 499_999_998)),
        ("ca_diff", duration(6, 15, 63_903, 500_000_002)),
        ("zero", duration(0, 0, 0, 0)),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_multiplication_and_division_use_exact_or_approximate_paths() -> Result<()> {
    let values = row(
        "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}) AS d \
         RETURN d * 2 AS twice, 2 * d AS reverse_twice, d / 2 AS half, \
                d * 0.5 AS float_half, d / 0.5 AS float_twice, d * -1 AS negative",
    )?;
    let expected = [
        ("twice", duration(298, 28, 116_780, 2)),
        ("reverse_twice", duration(298, 28, 116_780, 2)),
        ("half", duration(74, 22, 48_068, 0)),
        ("float_half", duration(74, 22, 48_068, 0)),
        ("float_twice", duration(298, 28, 116_780, 2)),
        ("negative", duration(-149, -14, -58_391, 999_999_999)),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_construction_and_division_reject_invalid_values() -> Result<()> {
    let invalid = [
        "RETURN duration({day: 1})",
        "RETURN duration({days: 'one'})",
        "RETURN duration({seconds: 0.0 / 0.0})",
        "RETURN duration({seconds: 1}) / 0",
        "RETURN duration({seconds: 1}) / 0.0",
    ];
    for query in invalid {
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid duration query succeeded"))?;
        assert!(
            matches!(error.code, ErrorCode::QueryType | ErrorCode::TemporalRange),
            "query: {query}; error: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn negative_iso_duration_is_stored_in_canonical_nanosecond_form() -> Result<()> {
    let values = row("RETURN duration('-PT23H59M59.9S') AS value")?;
    assert_eq!(
        values.get("value"),
        Some(&duration(0, 0, -86_400, 100_000_000))
    );
    Ok(())
}

#[test]
fn duration_between_splits_calendar_and_clock_boundaries() -> Result<()> {
    let values = row("WITH duration.between(\
             localdatetime('2018-01-01T10:00:00.2'), \
             localdatetime('2018-01-02T10:00:00.1')) AS forward, \
              duration.between(\
             localdatetime('2018-01-02T10:00:00.1'), \
             localdatetime('2018-01-01T10:00:00.2')) AS reverse \
         RETURN forward, forward.days AS forward_days, forward.seconds AS forward_seconds, \
                forward.nanosecondsOfSecond AS forward_nanos, reverse, \
                reverse.days AS reverse_days, reverse.seconds AS reverse_seconds, \
                reverse.nanosecondsOfSecond AS reverse_nanos")?;
    let expected = [
        ("forward", duration(0, 0, 86_399, 900_000_000)),
        ("forward_days", ResultValue::Scalar(ScalarValue::Integer(0))),
        (
            "forward_seconds",
            ResultValue::Scalar(ScalarValue::Integer(86_399)),
        ),
        (
            "forward_nanos",
            ResultValue::Scalar(ScalarValue::Integer(900_000_000)),
        ),
        ("reverse", duration(0, 0, -86_400, 100_000_000)),
        ("reverse_days", ResultValue::Scalar(ScalarValue::Integer(0))),
        (
            "reverse_seconds",
            ResultValue::Scalar(ScalarValue::Integer(-86_400)),
        ),
        (
            "reverse_nanos",
            ResultValue::Scalar(ScalarValue::Integer(100_000_000)),
        ),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_between_units_follow_complete_calendar_and_elapsed_time_rules() -> Result<()> {
    let values = row("RETURN \
         duration.between(date('1984-10-11'), date('2015-06-24')) AS between_dates, \
         duration.inMonths(date('1984-10-11'), date('2015-06-24')) AS months, \
         duration.inDays(date('1984-10-11'), date('2015-06-24')) AS days, \
         duration.inSeconds(date('1984-10-11'), date('2015-06-24')) AS seconds, \
         duration.between(date('1984-10-11'), \
             localdatetime('2016-07-21T21:45:22.142')) AS mixed_between, \
         duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), \
             localdatetime('2016-07-21T21:45:22.142')) AS complete_months, \
         duration.inDays(localdatetime('2015-07-21T21:40:32.142'), \
             date('2015-06-24')) AS negative_days, \
         duration.inSeconds(date('1984-10-11'), localtime('16:30')) AS attached_clock")?;
    let expected = [
        ("between_dates", duration(368, 13, 0, 0)),
        ("months", duration(368, 0, 0, 0)),
        ("days", duration(0, 11_213, 0, 0)),
        ("seconds", duration(0, 0, 968_803_200, 0)),
        ("mixed_between", duration(381, 10, 78_322, 142_000_000)),
        ("complete_months", duration(12, 0, 0, 0)),
        ("negative_days", duration(0, -27, 0, 0)),
        ("attached_clock", duration(0, 0, 59_400, 0)),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_between_respects_offsets_named_zones_and_daylight_saving() -> Result<()> {
    let values = row("RETURN \
         duration.inSeconds(time('14:30'), time('16:30+0100')) AS offset_times, \
         duration.between(\
             datetime('2017-10-28T23:00+02:00[Europe/Stockholm]'), \
             datetime('2017-10-29T04:00+01:00[Europe/Stockholm]')) AS bracketed_zone, \
         duration.inSeconds(\
             datetime({year: 2017, month: 10, day: 29, hour: 0, \
                 timezone: 'Europe/Stockholm'}), \
             localdatetime({year: 2017, month: 10, day: 29, hour: 4})) AS dst_mixed, \
         duration.inSeconds(\
             datetime({year: 2017, month: 10, day: 29, hour: 0, \
                 timezone: 'Europe/Stockholm'}), \
             date({year: 2017, month: 10, day: 30})) AS dst_day")?;
    let expected = [
        ("offset_times", duration(0, 0, 3_600, 0)),
        ("bracketed_zone", duration(0, 0, 21_600, 0)),
        ("dst_mixed", duration(0, 0, 18_000, 0)),
        ("dst_day", duration(0, 0, 90_000, 0)),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_between_preserves_subsecond_signs_and_supports_wide_local_datetimes() -> Result<()> {
    let values = row("RETURN \
         duration.inSeconds(localtime('12:34:54.7'), localtime('12:34:54.3')) AS negative_fraction, \
         duration.inSeconds(localtime('12:44:54.7'), localtime('12:34:55.3')) AS mixed_sign, \
         duration.inSeconds(localdatetime('-999999999-01-01'), \
             localdatetime('+999999999-12-31T23:59:59')) AS wide, \
         duration.between(date('-999999999-01-01'), \
             date('+999999999-12-31')) AS wide_date, \
         toString(date('-999999999-01-01')) AS wide_date_text")?;
    let expected = [
        ("negative_fraction", duration(0, 0, -1, 600_000_000)),
        ("mixed_sign", duration(0, 0, -600, 600_000_000)),
        ("wide", duration(0, 0, 63_113_903_968_377_599, 0)),
        ("wide_date", duration(23_999_999_987, 30, 0, 0)),
        (
            "wide_date_text",
            ResultValue::Scalar(ScalarValue::String("-999999999-01-01".into())),
        ),
    ];
    for (name, expected) in expected {
        assert_eq!(values.get(name), Some(&expected), "column: {name}");
    }
    Ok(())
}

#[test]
fn duration_between_propagates_either_null_operand_and_rejects_non_temporals() -> Result<()> {
    let values = row(
        "RETURN duration.between(null, date('2015-06-24')) AS between_left, \
                duration.inMonths(date('2015-06-24'), null) AS months_right, \
                duration.inDays(null, null) AS days_both, \
                duration.inSeconds(localtime('12:00'), null) AS seconds_right",
    )?;
    for name in ["between_left", "months_right", "days_both", "seconds_right"] {
        assert_eq!(
            values.get(name),
            Some(&ResultValue::Scalar(ScalarValue::Null)),
            "column: {name}"
        );
    }

    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "RETURN duration.between(1, date('2015-06-24'))",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| irongraph::Error::internal("non-temporal duration call succeeded"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    Ok(())
}
