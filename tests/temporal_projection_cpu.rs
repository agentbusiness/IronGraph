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
fn direct_projection_extracts_each_available_temporal_component() -> Result<()> {
    let cases = [
        (
            "RETURN toString(date(localdatetime({year: 1984, month: 11, day: 11, hour: 12})))",
            "1984-11-11",
        ),
        (
            "RETURN toString(localtime(datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})))",
            "12:00:00",
        ),
        (
            "RETURN toString(time(datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'})))",
            "12:00:00+01:00",
        ),
        (
            "RETURN toString(localdatetime(datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'})))",
            "1984-10-11T12:00:00",
        ),
        (
            "RETURN toString(datetime(localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, millisecond: 645})))",
            "1984-03-07T12:00:00.645+00:00",
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(string_value(query)?, expected, "query: {query}");
    }
    Ok(())
}

#[test]
fn projection_maps_inherit_components_before_applying_overrides() -> Result<()> {
    let cases = [
        (
            "WITH date({year: 1984, month: 11, day: 11}) AS other RETURN toString(date({date: other, week: 1}))",
            "1984-01-08",
        ),
        (
            "WITH date({year: 1984, month: 11, day: 11}) AS other RETURN toString(date({date: other, quarter: 3}))",
            "1984-08-11",
        ),
        (
            "WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other RETURN toString(localtime({time: other, second: 42}))",
            "12:31:42.645876",
        ),
        (
            "WITH date({year: 1984, month: 10, day: 11}) AS d, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS t RETURN toString(localdatetime({date: d, time: t, day: 28, second: 42}))",
            "1984-10-28T12:31:42.645876123",
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other RETURN toString(localdatetime({datetime: other, day: 28, second: 42}))",
            "1984-10-28T12:00:42",
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(string_value(query)?, expected, "query: {query}");
    }
    Ok(())
}

#[test]
fn zoned_projection_converts_instants_only_when_the_source_is_zoned() -> Result<()> {
    let cases = [
        (
            "WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other RETURN toString(time({time: other, timezone: '+05:00'}))",
            "16:31:14.645876+05:00",
        ),
        (
            "WITH localtime({hour: 12, minute: 31, second: 14}) AS other RETURN toString(time({time: other, timezone: '+05:00'}))",
            "12:31:14+05:00",
        ),
        (
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other RETURN toString(datetime({datetime: other, timezone: '+05:00'}))",
            "1984-10-11T16:00:00+05:00",
        ),
        (
            "WITH localdatetime({year: 1984, month: 3, day: 7}) AS d, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS t RETURN toString(datetime({date: d, time: t, day: 28, second: 42, timezone: 'Pacific/Honolulu'}))",
            "1984-03-28T00:00:42-10:00[Pacific/Honolulu]",
        ),
    ];
    for (query, expected) in cases {
        assert_eq!(string_value(query)?, expected, "query: {query}");
    }
    let named = value(
        "WITH localdatetime({year: 1984, month: 3, day: 7}) AS d, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS t RETURN datetime({date: d, time: t, day: 28, second: 42, timezone: 'Pacific/Honolulu'})",
    )?;
    let ResultValue::Scalar(ScalarValue::ZonedDateTime { timezone, .. }) = named else {
        return Err(irongraph::Error::internal(
            "named temporal projection did not return DATETIME",
        ));
    };
    assert_eq!(timezone.as_ref(), "Pacific/Honolulu");
    Ok(())
}

#[test]
fn temporal_projection_rejects_incompatible_sources_and_conflicts() -> Result<()> {
    let invalid = [
        "RETURN date({date: localtime({hour: 12})})",
        "RETURN localtime({time: date({year: 1984})})",
        "RETURN datetime({datetime: localdatetime({year: 1984}), date: date({year: 1984})})",
        "RETURN time({time: time({hour: 12}), timezone: 'Europe/Stockholm'})",
        "RETURN datetime({epochSeconds: 0, epochMillis: 0})",
    ];
    for query in invalid {
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid temporal projection succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
    }
    Ok(())
}
