// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! GPU-first openCypher semantic baseline.
//!
//! These scenarios intentionally exercise the public query engine rather than calling individual
//! operators. The CPU run is an oracle for diagnosis; the Metal run is the production gate.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    gpu::{BackendKind, CpuBackend, ExecutionBackend},
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT: usize = 256 * 1024 * 1024;
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
    let mut graph = GraphStore::default();
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

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

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
        let mut context = context(graph, backend);
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

fn make_cpu(graph: &GraphStore) -> irongraph::Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    Ok(backend)
}

#[test]
fn cpu_reference_executes_core_opencypher_semantics() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let cpu = make_cpu(&graph)?;
    for (name, _) in run_cpu_suite(&graph, &cpu)? {
        assert!(!name.is_empty());
    }
    Ok(())
}

fn run_cpu_suite(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
) -> irongraph::Result<Vec<(String, irongraph::cypher::QueryResult)>> {
    let mut results = Vec::with_capacity(CORE_SCENARIOS.len());
    for scenario in CORE_SCENARIOS {
        let mut context = context_without_gpu(graph, backend);
        let output = QueryEngine.execute(scenario.query, &mut context)?;
        results.push((scenario.name.to_owned(), output.result));
    }
    Ok(results)
}

fn context_without_gpu<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
) -> ExecutionContext<'a> {
    context(graph, backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_is_the_primary_opencypher_conformance_gate() -> irongraph::Result<()> {
    static METAL_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let graph = sample_graph()?;
    let cpu = make_cpu(&graph)?;
    let cpu_results = run_cpu_suite(&graph, &cpu)?;

    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT, RESERVED_MEMORY);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_graph(Arc::new(graph.snapshot()?))?;
    let metal_results = run_suite(&graph, &metal)?;

    assert_eq!(cpu_results, metal_results);
    Ok(())
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
fn gpu_opencypher_gate_requires_real_gpu_build() {
    eprintln!(
        "GPU openCypher gate skipped: run on macOS with --features accelerator and Metal available"
    );
}
