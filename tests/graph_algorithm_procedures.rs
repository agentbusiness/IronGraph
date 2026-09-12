// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::Arc;

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::{ExecutionBackend, MetalBackend};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::graph::GraphMutation;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_backend() -> Result<Option<MetalBackend>> {
    match MetalBackend::new(0, 128 * 1024 * 1024, 1024 * 1024) {
        Ok(backend) => Ok(Some(backend)),
        Err(error)
            if error.code == irongraph::ErrorCode::GpuAdmissionFailure
                && error.message.contains("unavailable") =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let node = graph.catalog_mut().intern_label("Node")?;
    let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    for id in 1..=6 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![node],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, value) in [
        (11, 1, 2, 1),
        (12, 2, 3, 2),
        (13, 3, 1, 4),
        (14, 3, 4, 1),
        (15, 4, 5, 1),
        (16, 5, 4, 1),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: id,
            properties: vec![(weight, ScalarValue::Integer(value))],
        })?;
    }
    Ok(graph)
}

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
        bookmark: Bookmark {
            // `ExecutionBackend::admit_graph` publishes a graph-only image at term zero. The
            // resident bookmark must match exactly or `current_backend` deliberately disables
            // the accelerator and this differential test silently exercises the CPU path.
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 10_000,
        max_batch_rows: 4_096,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn integer_column(output: &irongraph::cypher::ExecutionOutput, name: &str) -> Result<Vec<i64>> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|column| column.name == name)
        .ok_or_else(|| irongraph::Error::internal(format!("column {name} is absent")))?
        .values
        .iter()
        .map(|value| match value {
            ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(*value),
            _ => Err(irongraph::Error::internal(format!(
                "column {name} is not integer"
            ))),
        })
        .collect()
}

#[test]
fn built_in_algorithms_execute_through_call_yield() -> Result<()> {
    let graph = graph()?;
    for query in [
        "CALL graph.degree() YIELD node, degree RETURN id(node) AS id, degree ORDER BY id",
        "CALL graph.bfs(1) YIELD node, distance RETURN id(node) AS id, distance ORDER BY id",
        "CALL graph.dfs(1) YIELD node, order RETURN id(node) AS id, order ORDER BY order",
        "CALL graph.wcc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "CALL graph.scc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "CALL graph.kcore() YIELD node, core RETURN id(node) AS id, core ORDER BY id",
        "CALL graph.louvain() YIELD node, community RETURN id(node) AS id, community ORDER BY id",
        "CALL graph.clusteringcoefficient() YIELD node, coefficient RETURN id(node) AS id, coefficient ORDER BY id",
        "CALL graph.pagerank() YIELD node, score RETURN id(node) AS id, score ORDER BY id",
    ] {
        let output = QueryEngine.execute(query, &mut context(&graph))?;
        assert!(!output.result.batches.is_empty(), "query: {query}");
    }

    let bfs = QueryEngine.execute(
        "CALL graph.bfs(1) YIELD node, distance RETURN id(node) AS id, distance ORDER BY id",
        &mut context(&graph),
    )?;
    assert_eq!(integer_column(&bfs, "distance")?, vec![0, 1, 2, 3, 4]);

    let triangles = QueryEngine.execute(
        "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        &mut context(&graph),
    )?;
    assert_eq!(integer_column(&triangles, "triangleCount")?, vec![1]);
    Ok(())
}

#[test]
fn weighted_and_unweighted_shortest_paths_are_typed() -> Result<()> {
    let graph = graph()?;
    let path = QueryEngine.execute(
        "CALL graph.shortestpath(1, 5) YIELD path, cost RETURN path, cost",
        &mut context(&graph),
    )?;
    assert_eq!(integer_column(&path, "cost")?, vec![4]);
    let value = path.result.batches[0].columns[0].values[0].clone();
    let ResultValue::Path {
        nodes,
        relationships,
    } = value
    else {
        return Err(irongraph::Error::internal(
            "shortest path did not return a path",
        ));
    };
    assert_eq!(nodes.len(), 5);
    assert_eq!(relationships.len(), 4);

    let dijkstra = QueryEngine.execute(
        "CALL graph.dijkstra(1, 'weight') YIELD node, cost RETURN id(node) AS id, cost ORDER BY id",
        &mut context(&graph),
    )?;
    let costs = &dijkstra.result.batches[0].columns[1].values;
    assert_eq!(
        costs[4],
        ResultValue::Scalar(ScalarValue::Float(5.0.into()))
    );
    Ok(())
}

#[test]
fn procedure_signatures_reject_unknown_outputs_and_wrong_arity() -> Result<()> {
    let graph = graph()?;
    for query in [
        "CALL graph.bfs() YIELD node RETURN node",
        "CALL graph.wcc() YIELD score RETURN score",
        "CALL graph.leiden() YIELD node RETURN node",
    ] {
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid procedure call succeeded"))?;
        assert_eq!(error.code, irongraph::ErrorCode::QueryType);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
fn metal_degree_and_bfs_match_the_cpu_reference() -> Result<()> {
    let graph = graph()?;
    let Some(mut metal) = metal_backend()? else {
        return Ok(());
    };
    metal.admit_graph(Arc::new(graph.snapshot()?))?;
    for query in [
        "CALL graph.degree() YIELD node, outDegree, inDegree, degree RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
        "CALL graph.bfs(1) YIELD node, distance RETURN id(node) AS id, distance ORDER BY id",
    ] {
        let reference = QueryEngine.execute(query, &mut context(&graph))?;
        let mut device_context = context(&graph);
        device_context.backend = Some(&metal);
        let device = QueryEngine.execute(query, &mut device_context)?;
        assert_eq!(device.result.schema, reference.result.schema);
        assert_eq!(device.result.batches, reference.result.batches);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
fn prior_transaction_insert_node_and_edge_are_visible_to_metal_degree_and_bfs() -> Result<()> {
    let graph = graph()?;
    let node = graph
        .catalog()
        .label("Node")
        .ok_or_else(|| irongraph::Error::internal("Node label is absent"))?;
    let connected = graph
        .catalog()
        .relationship_type("CONNECTED")
        .ok_or_else(|| irongraph::Error::internal("CONNECTED type is absent"))?;
    let overlay_revision = graph
        .revision()
        .checked_add(1)
        .ok_or_else(|| irongraph::Error::internal("overlay fixture revision space is exhausted"))?;
    let prior = vec![
        GraphMutation::InsertNode(NodeInput {
            id: NodeId(7),
            layer: Layer::Observed,
            revision: overlay_revision,
            labels: vec![node],
            properties: Vec::new(),
        }),
        GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(17),
            source: NodeId(5),
            target: NodeId(7),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: overlay_revision,
            properties: Vec::new(),
        }),
    ];
    let Some(mut metal) = metal_backend()? else {
        return Ok(());
    };
    metal.admit_graph(Arc::new(graph.snapshot()?))?;

    for query in [
        "CALL graph.degree() YIELD node, outDegree, inDegree, degree \
         RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
        "CALL graph.bfs(1) YIELD node, distance \
         RETURN id(node) AS id, distance ORDER BY id",
    ] {
        let mut reference_context = context(&graph);
        reference_context.prior_graph_mutations = &prior;
        reference_context.mutation_revision = overlay_revision.checked_add(1).ok_or_else(|| {
            irongraph::Error::internal("overlay fixture statement revision is exhausted")
        })?;
        let reference = QueryEngine.execute(query, &mut reference_context)?;

        let mut device_context = context(&graph);
        device_context.prior_graph_mutations = &prior;
        device_context.mutation_revision = overlay_revision.checked_add(1).ok_or_else(|| {
            irongraph::Error::internal("overlay fixture statement revision is exhausted")
        })?;
        device_context.backend = Some(&metal);
        let device = QueryEngine.execute(query, &mut device_context)?;
        assert_eq!(
            device.result.schema, reference.result.schema,
            "query: {query}"
        );
        assert_eq!(
            device.result.batches, reference.result.batches,
            "query: {query}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
fn layer_subset_budget_uses_visible_cardinality_for_cpu_and_metal() -> Result<()> {
    let mut graph = graph()?;
    let node = graph
        .catalog()
        .label("Node")
        .ok_or_else(|| irongraph::Error::internal("Node label is absent"))?;
    let connected = graph
        .catalog()
        .relationship_type("CONNECTED")
        .ok_or_else(|| irongraph::Error::internal("CONNECTED type is absent"))?;
    let first_revision = graph
        .revision()
        .checked_add(1)
        .ok_or_else(|| irongraph::Error::internal("layer fixture revision space is exhausted"))?;
    graph.insert_node(NodeInput {
        id: NodeId(70),
        layer: Layer::Knowledge,
        revision: first_revision,
        labels: vec![node],
        properties: Vec::new(),
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(80),
        layer: Layer::Knowledge,
        revision: first_revision + 1,
        labels: vec![node],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(170),
        source: NodeId(70),
        target: NodeId(80),
        relationship_type: connected,
        layer: Layer::Knowledge,
        revision: first_revision + 2,
        properties: Vec::new(),
    })?;
    let Some(mut metal) = metal_backend()? else {
        return Ok(());
    };
    metal.admit_graph(Arc::new(graph.snapshot()?))?;
    let query = "USE LAYER KNOWLEDGE\nWRITE LAYER KNOWLEDGE\n\
                 CALL graph.degree() YIELD node, degree \
                 RETURN id(node) AS id, degree ORDER BY id";

    let mut reference_context = context(&graph);
    reference_context.capabilities.knowledge_write = true;
    reference_context.max_result_rows = 2;
    let reference = QueryEngine.execute(query, &mut reference_context)?;
    assert_eq!(integer_column(&reference, "id")?, vec![70, 80]);

    let mut device_context = context(&graph);
    device_context.capabilities.knowledge_write = true;
    device_context.max_result_rows = 2;
    device_context.backend = Some(&metal);
    let device = QueryEngine.execute(query, &mut device_context)?;
    assert_eq!(device.result.schema, reference.result.schema);
    assert_eq!(device.result.batches, reference.result.batches);

    for backend in [None, Some(&metal as &dyn ExecutionBackend)] {
        let mut bounded = context(&graph);
        bounded.capabilities.knowledge_write = true;
        bounded.max_result_rows = 1;
        bounded.backend = backend;
        let error = QueryEngine
            .execute(query, &mut bounded)
            .expect_err("one row below the visible-layer cardinality must exceed the budget");
        assert_eq!(error.code, irongraph::ErrorCode::ResultBudgetExceeded);
    }
    Ok(())
}
