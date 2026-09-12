// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! An aggregate over the result of an aggregate must answer, not fail.
//!
//! Group, then count the groups: it is how nearly every distribution question is written, and it is
//! ordinary Cypher. The resident segmented compiler emits one program per plan and the accelerator
//! can lower only one aggregating stage inside it, so a second stage was rejected once the program
//! reached the backend — as a hard error rather than as a decline, which meant the query failed
//! instead of falling through to the host path that can answer it. The failure was structural: it
//! did not depend on how many groups either stage produced.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{GraphStore, IndexCatalog, NodeInput, TemporalStore},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 1, index: 64 };

/// Twelve readings across three stations, so an inner grouping has several groups and an outer one
/// has something to count.
fn station_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let reading = graph.catalog_mut().intern_label("Reading")?;
    let station = graph.catalog_mut().intern_property("station")?;
    let celsius = graph.catalog_mut().intern_property("celsius")?;
    let mut id = 0_u64;
    for (name, values) in [
        ("north", [1_i64, 2, 3, 4].as_slice()),
        ("south", [5, 6, 7].as_slice()),
        ("west", [8, 9, 10, 11, 12].as_slice()),
    ] {
        for value in values {
            id += 1;
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![reading],
                properties: vec![
                    (station, ScalarValue::String(Arc::from(name))),
                    (celsius, ScalarValue::Integer(*value)),
                ],
            })?;
        }
    }
    Ok(graph)
}

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
        mutation_revision: 65,
        resolved_time_nanos: 0,
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

fn integers(query: &str) -> Result<Vec<i64>> {
    let graph = station_graph()?;
    let temporal = TemporalStore::default();
    let backend = admitted_backend(&graph, &temporal)?;
    let output = QueryEngine.execute(query, &mut context(&graph, &temporal, &backend))?;
    // Row-major, so an expectation reads the way the result prints.
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let rows = batch
            .columns
            .first()
            .map_or(0, |column| column.values.len());
        for row in 0..rows {
            for column in &batch.columns {
                if let Some(ResultValue::Scalar(ScalarValue::Integer(number))) =
                    column.values.get(row)
                {
                    values.push(*number);
                }
            }
        }
    }
    Ok(values)
}

#[test]
fn counting_the_groups_of_a_grouping_answers() -> Result<()> {
    assert_eq!(
        integers(
            "MATCH (r:Reading) WITH r.station AS station, count(*) AS readings \
             RETURN count(*) AS stations"
        )?,
        vec![3],
    );
    Ok(())
}

#[test]
fn an_outer_aggregate_summarises_the_inner_one() -> Result<()> {
    // Three stations with four, three and five readings: twelve in total, the largest group five.
    assert_eq!(
        integers(
            "MATCH (r:Reading) WITH r.station AS station, count(*) AS readings \
             RETURN count(*) AS stations, sum(readings) AS total, max(readings) AS largest"
        )?,
        vec![3, 12, 5],
    );
    Ok(())
}

#[test]
fn the_outer_grouping_may_group_again() -> Result<()> {
    // Bucketing the per-station counts re-groups an already aggregated stream. Four and three fall
    // in the 'under five' bucket, five does not.
    assert_eq!(
        integers(
            "MATCH (r:Reading) WITH r.station AS station, count(*) AS readings \
             WITH CASE WHEN readings < 5 THEN 0 ELSE 1 END AS bucket, readings \
             RETURN bucket, count(*) AS stations ORDER BY bucket"
        )?,
        vec![0, 2, 1, 1],
    );
    Ok(())
}

#[test]
fn a_single_aggregating_stage_is_unaffected() -> Result<()> {
    assert_eq!(
        integers("MATCH (r:Reading) RETURN count(*) AS readings, sum(r.celsius) AS total")?,
        vec![12, 78],
    );
    Ok(())
}
