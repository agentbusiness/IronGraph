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
    gpu::{CpuBackend, ExecutionBackend},
    graph::GraphStore,
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

fn row_with_backend(
    query: &str,
    backend: &dyn ExecutionBackend,
) -> Result<BTreeMap<String, ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context_with_backend(&graph, Some(backend)))?;
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

fn schema_names(query: &str) -> Result<Vec<String>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    Ok(output
        .result
        .schema
        .into_iter()
        .map(|(name, _)| name)
        .collect())
}

fn string(values: &BTreeMap<String, ResultValue>, name: &str) -> Result<String> {
    let Some(ResultValue::Scalar(ScalarValue::String(value))) = values.get(name) else {
        return Err(irongraph::Error::internal(format!(
            "column {name} did not return STRING"
        )));
    };
    Ok(value.to_string())
}

fn boolean(values: &BTreeMap<String, ResultValue>, name: &str) -> Result<bool> {
    let Some(ResultValue::Scalar(ScalarValue::Boolean(value))) = values.get(name) else {
        return Err(irongraph::Error::internal(format!(
            "column {name} did not return BOOLEAN"
        )));
    };
    Ok(*value)
}

#[test]
fn temporal6_duration_examples_render_canonically_and_round_trip() -> Result<()> {
    let cases = [
        (
            "{years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}",
            "P12Y5M14DT16H13M10.000000001S",
        ),
        (
            "{years: 12, months: 5, days: -14, hours: 16}",
            "P12Y5M-14DT16H",
        ),
        ("{minutes: 12, seconds: -60}", "PT11M"),
        ("{seconds: 2, milliseconds: -1}", "PT1.999S"),
        ("{seconds: -2, milliseconds: 1}", "PT-1.999S"),
        ("{seconds: -2, milliseconds: -1}", "PT-2.001S"),
        ("{days: 1, milliseconds: 1}", "P1DT0.001S"),
        ("{days: 1, milliseconds: -1}", "P1DT-0.001S"),
        ("{seconds: 60, milliseconds: -1}", "PT59.999S"),
        ("{seconds: -60, milliseconds: 1}", "PT-59.999S"),
        ("{seconds: -60, milliseconds: -1}", "PT-1M-0.001S"),
    ];

    for (map, expected) in cases {
        let values = row(&format!(
            "WITH duration({map}) AS d \
             RETURN toString(d) AS rendered, duration(toString(d)) = d AS round_trip"
        ))?;
        assert_eq!(string(&values, "rendered")?, expected, "map: {map}");
        assert!(boolean(&values, "round_trip")?, "map: {map}");
    }
    Ok(())
}

#[test]
fn signed_and_fractional_iso_components_normalize_without_precision_loss() -> Result<()> {
    let cases = [
        ("P5M1.5D", "P5M1DT12H"),
        ("P0.75M", "P22DT19H51M49.5S"),
        ("PT0.75M", "PT45S"),
        ("P2.5W", "P17DT12H"),
        ("P1.5Y", "P1Y6M"),
        ("-PT23H59M59.9S", "PT-23H-59M-59.9S"),
        ("P12Y5M-14DT16H", "P12Y5M-14DT16H"),
        ("+PT1.000000001S", "PT1.000000001S"),
    ];

    for (input, expected) in cases {
        let values = row(&format!(
            "WITH duration('{input}') AS d \
             RETURN toString(d) AS rendered, duration(toString(d)) = d AS round_trip"
        ))?;
        assert_eq!(string(&values, "rendered")?, expected, "input: {input}");
        assert!(boolean(&values, "round_trip")?, "input: {input}");
    }
    Ok(())
}

#[test]
fn zero_and_null_duration_rendering_preserves_cypher_semantics() -> Result<()> {
    let values = row("WITH duration({}) AS zero \
         RETURN toString(zero) AS rendered, duration(toString(zero)) = zero AS round_trip, \
                duration(null) AS duration_null, toString(duration(null)) AS rendered_null")?;
    assert_eq!(string(&values, "rendered")?, "PT0S");
    assert!(boolean(&values, "round_trip")?);
    assert_eq!(
        values.get("duration_null"),
        Some(&ResultValue::Scalar(ScalarValue::Null))
    );
    assert_eq!(
        values.get("rendered_null"),
        Some(&ResultValue::Scalar(ScalarValue::Null))
    );
    Ok(())
}

#[test]
fn named_timezone_rendering_keeps_the_zone_identity() -> Result<()> {
    let values = row(
        "WITH datetime({year: 2017, month: 8, day: 8, hour: 12, minute: 31, second: 14, \
                        nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d \
         RETURN toString(d) AS rendered",
    )?;
    assert_eq!(
        string(&values, "rendered")?,
        "2017-08-08T12:31:14.645876123+02:00[Europe/Stockholm]"
    );
    Ok(())
}

#[test]
fn cpu_output_names_keep_exact_unaliased_projection_source() -> Result<()> {
    assert_eq!(
        schema_names("RETURN ( 1   +  2 ), {  first : 1,   second: 2  }, 3 + 4 AS total")?,
        [
            "( 1   +  2 )".to_owned(),
            "{  first : 1,   second: 2  }".to_owned(),
            "total".to_owned(),
        ]
    );
    assert_eq!(
        schema_names("UNWIND [1, 2] AS x RETURN cOuNt( * ), sum( x )")?,
        ["cOuNt( * )".to_owned(), "sum( x )".to_owned()]
    );
    Ok(())
}

#[test]
fn cpu_time_string_defaults_to_utc_and_preserves_explicit_offsets() -> Result<()> {
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let offsetless = row_with_backend("RETURN time('14:30') AS value", &cpu)?;
    assert_eq!(
        offsetless.get("value"),
        Some(&ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: 52_200_000_000_000,
            offset_seconds: 0,
        }))
    );
    for hour_only in [
        row("RETURN time('14') AS value")?,
        row_with_backend("RETURN time('14') AS value", &cpu)?,
    ] {
        assert_eq!(
            hour_only.get("value"),
            Some(&ResultValue::Scalar(ScalarValue::ZonedTime {
                nanos: 50_400_000_000_000,
                offset_seconds: 0,
            }))
        );
    }

    let explicit = row_with_backend("RETURN time('16:30+0100') AS value", &cpu)?;
    assert_eq!(
        explicit.get("value"),
        Some(&ResultValue::Scalar(ScalarValue::ZonedTime {
            nanos: 59_400_000_000_000,
            offset_seconds: 3_600,
        }))
    );

    let differences = row_with_backend(
        "RETURN duration.between(time('14:30'), date('2015-06-24')) AS date_difference, \
                duration.between(time('14:30'), time('16:30+0100')) AS offset_difference, \
                duration.inSeconds(time('14:30'), \
                    datetime('2015-07-21T21:40:32.142+0100')) AS datetime_difference",
        &cpu,
    )?;
    assert_eq!(
        differences.get("date_difference"),
        Some(&ResultValue::Scalar(ScalarValue::Duration {
            months: 0,
            days: 0,
            seconds: -52_200,
            nanos: 0,
        }))
    );
    assert_eq!(
        differences.get("offset_difference"),
        Some(&ResultValue::Scalar(ScalarValue::Duration {
            months: 0,
            days: 0,
            seconds: 3_600,
            nanos: 0,
        }))
    );
    assert_eq!(
        differences.get("datetime_difference"),
        Some(&ResultValue::Scalar(ScalarValue::Duration {
            months: 0,
            days: 0,
            seconds: 22_232,
            nanos: 142_000_000,
        }))
    );

    for query in ["RETURN time('14:30+')", "RETURN time('not-a-time')"] {
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut context_with_backend(&graph, Some(&cpu)))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid time string succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
    }
    Ok(())
}
