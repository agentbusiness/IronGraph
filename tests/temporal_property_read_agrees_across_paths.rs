// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Reading a declared temporal property must not depend on how the plan reaches the node.
//!
//! A declared temporal property has two stored representations. The canonical property column holds
//! whatever was written before the declaration and is not written again afterwards; the sample
//! series holds every write from the declaration onward and is the current value. A plan that
//! projects the canonical column therefore answers with a frozen value, while the same read
//! resolved through the series answers correctly — and which one a query got depended on whether
//! its plan filtered to specific nodes or scanned the label. The same expression returned two
//! different numbers for the same node, with no error and no warning.
//!
//! These tests pin the contract: every access path answers from the series, and a query that
//! carries its own time still resolves against that time.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::types::EntityKind;
use irongraph::{
    Bookmark, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{
        GraphStore, IndexCatalog, NodeInput, TemporalDeclaration, TemporalSample, TemporalStore,
        TemporalType,
    },
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

const DECLARED_AT: i64 = 1_000;

/// Three instruments, each carrying a canonical price written before the declaration.
fn priced_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let instrument = graph.catalog_mut().intern_label("Instrument")?;
    let symbol = graph.catalog_mut().intern_property("symbol")?;
    let price = graph.catalog_mut().intern_property("price")?;
    for (id, code) in [(1_u64, "AAA"), (2, "BBB"), (3, "CCC")] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![instrument],
            properties: vec![
                (symbol, ScalarValue::String(Arc::from(code))),
                // The value the canonical column keeps for ever once the declaration exists.
                (price, ScalarValue::Float(OrderedFloat(0.0))),
            ],
        })?;
    }
    Ok(graph)
}

/// Declares `Instrument.price` temporal and records a rising series for every instrument.
fn priced_history(graph: &GraphStore) -> Result<TemporalStore> {
    let instrument = graph
        .catalog()
        .label("Instrument")
        .ok_or_else(|| irongraph::Error::internal("Instrument label missing"))?;
    let price = graph
        .catalog()
        .property("price")
        .ok_or_else(|| irongraph::Error::internal("price property missing"))?;
    let mut temporal = TemporalStore::default();
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: instrument.0,
            property: price,
            value_type: TemporalType::Float,
            retention_nanos: 1_000_000,
        },
        DECLARED_AT,
    )?;
    let mut sequence_index = 0_u64;
    for entity_id in 1_u64..=3 {
        for (time, value) in [(2_000_i64, 10.0_f64), (3_000, 20.0), (4_000, 30.0)] {
            sequence_index += 1;
            temporal.append(
                EntityKind::Node,
                instrument.0,
                TemporalSample {
                    entity_id,
                    property: price,
                    event_time_nanos: time,
                    sequence_index,
                    value: ScalarValue::Float(OrderedFloat(value * entity_id as f64)),
                },
                5_000,
            )?;
        }
    }
    Ok(temporal)
}

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark {
    term: 1,
    index: 1_000,
};

/// A backend holding the graph and its history resident.
///
/// Without one, every query falls to the host reference path — which always resolved the series
/// correctly — and these tests would pass no matter what the resident compilers did.
fn admitted_backend(graph: &GraphStore, temporal: &TemporalStore) -> Result<CpuBackend> {
    let image =
        ResidentProjectImage::build(PROJECT, BOOKMARK, graph, temporal, &IndexCatalog::default())?;
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    backend.admit_project(image)?;
    Ok(backend)
}

fn context<'a>(
    graph: &'a GraphStore,
    temporal: &'a TemporalStore,
    backend: &'a CpuBackend,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: Some(temporal),
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: BOOKMARK,
        mutation_revision: 1_001,
        resolved_time_nanos: 5_000,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn floats(query: &str, graph: &GraphStore, temporal: &TemporalStore) -> Result<Vec<Option<f64>>> {
    let backend = admitted_backend(graph, temporal)?;
    let output = QueryEngine.execute(query, &mut context(graph, temporal, &backend))?;
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let Some(column) = batch.columns.first() else {
            continue;
        };
        for value in &column.values {
            values.push(match value {
                ResultValue::Scalar(ScalarValue::Float(price)) => Some(price.into_inner()),
                ResultValue::Scalar(ScalarValue::Null) => None,
                other => panic!("expected a FLOAT or NULL price, got {other:?}"),
            });
        }
    }
    Ok(values)
}

#[test]
fn every_access_path_reads_the_current_sample() -> Result<()> {
    let graph = priced_graph()?;
    let temporal = priced_history(&graph)?;

    // 30.0 is the latest sample for instrument 1. The canonical column still holds 0.0, and a plan
    // that projects it directly would answer with that instead.
    let expected = vec![Some(30.0)];

    let by_property = floats(
        "MATCH (i:Instrument {symbol: 'AAA'}) RETURN i.price",
        &graph,
        &temporal,
    )?;
    let by_filter = floats(
        "MATCH (i:Instrument) WHERE i.symbol = 'AAA' RETURN i.price",
        &graph,
        &temporal,
    )?;
    let by_scan_limited = floats(
        "MATCH (i:Instrument) RETURN i.price ORDER BY i.symbol LIMIT 1",
        &graph,
        &temporal,
    )?;

    assert_eq!(by_property, expected, "property-match read");
    assert_eq!(by_filter, expected, "filtered scan read");
    assert_eq!(by_scan_limited, expected, "ordered scan read");
    Ok(())
}

#[test]
fn a_scan_over_the_label_reads_every_current_sample() -> Result<()> {
    let graph = priced_graph()?;
    let temporal = priced_history(&graph)?;

    let scanned = floats(
        "MATCH (i:Instrument) RETURN i.price ORDER BY i.symbol",
        &graph,
        &temporal,
    )?;
    assert_eq!(
        scanned,
        vec![Some(30.0), Some(60.0), Some(90.0)],
        "a scan must read the series for every node, not the canonical column"
    );
    Ok(())
}

#[test]
fn an_aggregate_over_the_label_summarises_the_series() -> Result<()> {
    let graph = priced_graph()?;
    let temporal = priced_history(&graph)?;

    // Every canonical price is 0.0, so an aggregate that reached the canonical column would
    // average to zero rather than to the mean of 30, 60 and 90.
    let mean = floats(
        "MATCH (i:Instrument) RETURN avg(i.price)",
        &graph,
        &temporal,
    )?;
    assert_eq!(mean, vec![Some(60.0)]);
    Ok(())
}

#[test]
fn a_query_carrying_its_own_time_still_resolves_against_that_time() -> Result<()> {
    let graph = priced_graph()?;
    let temporal = priced_history(&graph)?;

    // At 3,500 the sample in effect for instrument 1 is the 3,000 one, worth 20.0.
    let filtered = floats(
        "AT TIME 3500 MATCH (i:Instrument {symbol: 'AAA'}) RETURN i.price",
        &graph,
        &temporal,
    )?;
    let scanned = floats(
        "AT TIME 3500 MATCH (i:Instrument) RETURN i.price ORDER BY i.symbol LIMIT 1",
        &graph,
        &temporal,
    )?;
    assert_eq!(filtered, vec![Some(20.0)], "time-travelling property match");
    assert_eq!(scanned, vec![Some(20.0)], "time-travelling scan");
    Ok(())
}

#[test]
fn a_read_before_the_first_sample_is_null_on_every_path() -> Result<()> {
    let graph = priced_graph()?;
    let temporal = priced_history(&graph)?;

    let filtered = floats(
        "AT TIME 1500 MATCH (i:Instrument {symbol: 'AAA'}) RETURN i.price",
        &graph,
        &temporal,
    )?;
    let scanned = floats(
        "AT TIME 1500 MATCH (i:Instrument) RETURN i.price ORDER BY i.symbol LIMIT 1",
        &graph,
        &temporal,
    )?;
    assert_eq!(
        filtered,
        vec![None],
        "before the first sample, property match"
    );
    assert_eq!(scanned, vec![None], "before the first sample, scan");
    Ok(())
}
