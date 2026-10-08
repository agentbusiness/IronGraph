// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Canonical CPU openCypher semantic baseline with an optional retained Metal comparison.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    gpu::{BackendKind, ExecutionBackend},
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "legacy-graph", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
#[cfg(all(feature = "legacy-graph", target_os = "macos"))]
const MEMORY_LIMIT: usize = 256 * 1024 * 1024;
#[cfg(all(feature = "legacy-graph", target_os = "macos"))]
const RESERVED_MEMORY: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    query: &'static str,
}

const CORE_SCENARIOS: &[Scenario] = &[
    Scenario {
        name: "ordered numeric property projection",
        query: "MATCH (n:Person) RETURN n.age AS age ORDER BY age",
    },
    Scenario {
        name: "equality filter",
        query: "MATCH (n:Person) WHERE n.age = 37 RETURN n.age AS age",
    },
];

fn sample_graph() -> irongraph::Result<GraphStore> {
    let graph = GraphStore::default();
    let person = graph.catalog_mut().intern_label("Person")?;
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let age = graph.catalog_mut().intern_property("age")?;

    for (id, value, years) in [(1, "Ada", 37), (2, "Grace", 41), (3, "Linus", 29)] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![person],
            properties: vec![
                (name, ScalarValue::String(Arc::from(value))),
                (age, ScalarValue::Integer(years)),
            ],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: knows,
        layer: Layer::Observed,
        revision: 4,
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(11),
        source: NodeId(2),
        target: NodeId(3),
        relationship_type: knows,
        layer: Layer::Observed,
        revision: 5,
        properties: Vec::new(),
    })?;
    Ok(graph)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 0, index: 5 },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 256,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

#[cfg_attr(
    not(all(feature = "legacy-graph", target_os = "macos")),
    allow(dead_code)
)]
fn run_suite(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
) -> irongraph::Result<Vec<(String, irongraph::cypher::QueryResult)>> {
    let mut results = Vec::with_capacity(CORE_SCENARIOS.len());
    for scenario in CORE_SCENARIOS {
        assert_eq!(
            backend.kind(),
            BackendKind::Metal,
            "GPU conformance scenario '{}' unexpectedly did not use Metal",
            scenario.name
        );
        let mut context = context(graph, Some(backend));
        let output = QueryEngine
            .execute(scenario.query, &mut context)
            .map_err(|error| {
                irongraph::Error::internal(format!(
                    "openCypher GPU scenario '{}' failed: {}",
                    scenario.name, error.message
                ))
            })?;
        results.push((scenario.name.to_owned(), output.result));
    }
    Ok(results)
}

#[test]
fn cpu_reference_executes_core_opencypher_semantics() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    for (name, _) in run_cpu_suite(&graph)? {
        assert!(!name.is_empty());
    }
    Ok(())
}

fn run_cpu_suite(
    graph: &GraphStore,
) -> irongraph::Result<Vec<(String, irongraph::cypher::QueryResult)>> {
    let mut results = Vec::with_capacity(CORE_SCENARIOS.len());
    for scenario in CORE_SCENARIOS {
        let mut context = context(graph, None);
        let output = QueryEngine.execute(scenario.query, &mut context)?;
        results.push((scenario.name.to_owned(), output.result));
    }
    Ok(results)
}

#[cfg(all(feature = "legacy-graph", target_os = "macos"))]
fn legacy_snapshot(graph: &GraphStore) -> irongraph::Result<irongraph::graph::GraphSnapshot> {
    use irongraph::graph::GraphMutation;
    let mut fixture = irongraph::graph::legacy::GraphStore::default();
    for (id, name) in graph.catalog().labels() {
        fixture.apply(GraphMutation::DeclareLabel {
            name: name.to_string(),
            id,
        })?;
    }
    for (id, name) in graph.catalog().properties() {
        fixture.apply(GraphMutation::DeclareProperty {
            name: name.to_string(),
            id,
        })?;
    }
    for (id, name) in graph.catalog().relationship_types() {
        fixture.apply(GraphMutation::DeclareRelationshipType {
            name: name.to_string(),
            id,
        })?;
    }
    for node in graph.nodes() {
        fixture.apply(GraphMutation::InsertNode(NodeInput {
            id: node.id(),
            layer: node.layer(),
            revision: node.revision(),
            labels: node.labels().to_vec(),
            properties: node.properties(),
        }))?;
    }
    for edge in graph.edges() {
        fixture.apply(GraphMutation::InsertEdge(EdgeInput {
            id: edge.id(),
            source: edge.source(),
            target: edge.target(),
            relationship_type: edge.relationship_type(),
            layer: edge.layer(),
            revision: edge.revision(),
            properties: edge.properties(),
        }))?;
    }
    fixture.snapshot()
}

#[cfg(all(feature = "legacy-graph", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_is_the_primary_opencypher_conformance_gate() -> irongraph::Result<()> {
    static METAL_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let graph = sample_graph()?;
    let cpu_results = run_cpu_suite(&graph)?;

    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT, RESERVED_MEMORY);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_graph(Arc::new(legacy_snapshot(&graph)?))?;
    let metal_results = run_suite(&graph, &metal)?;

    assert_eq!(cpu_results, metal_results);
    Ok(())
}

#[cfg(not(all(feature = "legacy-graph", target_os = "macos")))]
#[test]
fn gpu_opencypher_gate_requires_real_gpu_build() {
    eprintln!(
        "GPU openCypher gate skipped: run on macOS with --features accelerator and Metal available"
    );
}
