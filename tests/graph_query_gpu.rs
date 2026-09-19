// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::cypher::VectorSearchSource;
use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionStreamItem, QueryEngine, ResultValue, parse,
    },
    gpu::{
        CompareOp, CpuBackend, ExecutionBackend, ResidentAggregate, ResidentDirection,
        ResidentExpansion, ResidentGroupRequest, ResidentI64Predicate, ResidentI64Projection,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodeBinding, ResidentNodeOrder,
        ResidentNodePipelineRequest, ResidentProjectDelta, ResidentProjectImage, ResidentSortKey,
        ResidentSortRequest, ResidentSortSource, ResidentTemporalDelta,
        ResidentTemporalPipelineRequest, ResidentValueMatrixProgram, ResidentValueMatrixValue,
    },
    graph::{
        EdgeInput, EqualityIndex, GraphIndexDefinition, GraphIndexKind, GraphMutation, GraphStore,
        IndexCatalog, IndexKey, IvfPqConfig, IvfPqIndex, LayerMask, NodeInput, PageRankConfig,
        RangeIndex, ResolvedVectorMutation, Similarity, TemporalDeclaration, TemporalSample,
        TemporalStore, TemporalType, TextIndex, VectorIndex, WindowSpec, bfs, page_rank,
        triangle_count,
    },
    types::{EntityKind, PropertyId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::{
    gpu::{DeviceMemoryGovernor, MetalBackend, ResidentVectorQuery},
    graph::{EmbeddingDType, EmbeddingIndexDefinition, EmbeddingProfile},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static METAL_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    match METAL_TEST.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

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
        layer: Layer::Knowledge,
        revision: 5,
        properties: Vec::new(),
    })?;
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn optional_social_graph() -> irongraph::Result<GraphStore> {
    let mut graph = GraphStore::default();
    let person = graph.catalog_mut().intern_label("Person")?;
    let post = graph.catalog_mut().intern_label("Post")?;
    let comment = graph.catalog_mut().intern_label("Comment")?;
    let authored = graph.catalog_mut().intern_relationship_type("AUTHORED")?;
    let liked = graph.catalog_mut().intern_relationship_type("LIKED")?;
    let uid = graph.catalog_mut().intern_property("uid")?;
    let rank = graph.catalog_mut().intern_property("rank")?;
    let title = graph.catalog_mut().intern_property("title")?;
    for (id, value) in [(1, 1001), (2, 1002), (3, 1003)] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![person],
            properties: vec![(uid, ScalarValue::Integer(value))],
        })?;
    }
    for (id, value, text, layer) in [
        (11, 1, "First", Layer::Observed),
        (12, 2, "Second", Layer::Observed),
        (13, 3, "Private", Layer::Knowledge),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer,
            revision: id,
            labels: vec![post],
            properties: vec![
                (rank, ScalarValue::Integer(value)),
                (title, ScalarValue::String(Arc::from(text))),
            ],
        })?;
    }
    graph.insert_node(NodeInput {
        id: NodeId(14),
        layer: Layer::Observed,
        revision: 14,
        labels: vec![comment],
        properties: vec![(rank, ScalarValue::Integer(4))],
    })?;
    for (id, source, target, relationship_type, layer) in [
        (101, 1, 11, authored, Layer::Observed),
        (102, 1, 12, authored, Layer::Observed),
        (103, 2, 13, authored, Layer::Knowledge),
        (104, 2, 14, authored, Layer::Observed),
        (105, 2, 11, liked, Layer::Observed),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn execute_resident_query(
    graph: &GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
    backend: Option<&dyn ExecutionBackend>,
    source: &str,
) -> irongraph::Result<irongraph::cypher::QueryResult> {
    execute_resident_query_with_native_requirement(graph, project, bookmark, backend, source, false)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn execute_resident_query_with_native_requirement(
    graph: &GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
    backend: Option<&dyn ExecutionBackend>,
    source: &str,
    require_native_execution: bool,
) -> irongraph::Result<irongraph::cypher::QueryResult> {
    let mut context = ExecutionContext {
        project_id: project,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1_000,
        next_edge_id: 1_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 100,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    Ok(QueryEngine.execute(source, &mut context)?.result)
}

fn integer_operator_graph() -> irongraph::Result<(GraphStore, PropertyId, PropertyId)> {
    let mut graph = GraphStore::default();
    let row = graph.catalog_mut().intern_label("Row")?;
    let key = graph.catalog_mut().intern_property("key")?;
    let value = graph.catalog_mut().intern_property("value")?;
    let records = [
        (Some(2), Some(10)),
        (Some(1), None),
        (Some(2), Some(-3)),
        (None, Some(7)),
        (Some(1), Some(5)),
    ];
    for (position, (key_value, aggregate_value)) in records.into_iter().enumerate() {
        let mut properties = Vec::new();
        if let Some(key_value) = key_value {
            properties.push((key, ScalarValue::Integer(key_value)));
        }
        if let Some(aggregate_value) = aggregate_value {
            properties.push((value, ScalarValue::Integer(aggregate_value)));
        }
        graph.insert_node(NodeInput {
            id: NodeId(position as u64 + 1),
            layer: Layer::Observed,
            revision: position as u64 + 1,
            labels: vec![row],
            properties,
        })?;
    }
    Ok((graph, key, value))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn scalable_integer_operator_graph(
    row_count: usize,
) -> irongraph::Result<(GraphStore, PropertyId, PropertyId)> {
    let mut graph = GraphStore::default();
    let row = graph.catalog_mut().intern_label("ScalableRow")?;
    let key = graph.catalog_mut().intern_property("scalable_key")?;
    let value = graph.catalog_mut().intern_property("scalable_value")?;
    for position in 0..row_count {
        let mut properties = Vec::with_capacity(2);
        if position % 257 != 0 {
            properties.push((key, ScalarValue::Integer((position % 2048) as i64 - 1024)));
        }
        if position % 89 != 0 {
            properties.push((value, ScalarValue::Integer((position % 101) as i64 - 50)));
        }
        graph.insert_node(NodeInput {
            id: NodeId(position as u64 + 1),
            layer: Layer::Observed,
            revision: position as u64 + 1,
            labels: vec![row],
            properties,
        })?;
    }
    Ok((graph, key, value))
}

#[test]
fn graph_preserves_stable_ids_layers_and_merged_adjacency() -> irongraph::Result<()> {
    let mut graph = sample_graph()?;
    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
    let observed = graph.expand_out(NodeId(1), None, LayerMask::OBSERVED)?;
    assert_eq!(observed.len(), 1);
    assert_eq!(observed[0].1.id(), NodeId(2));
    assert!(
        graph
            .expand_out(NodeId(2), None, LayerMask::OBSERVED)?
            .is_empty()
    );
    let authority = graph.expand_out(NodeId(2), None, LayerMask::AUTHORITY)?;
    assert_eq!(authority.len(), 1);
    graph.compact_adjacency()?;
    let incoming = graph.expand_in(NodeId(3), None, LayerMask::AUTHORITY)?;
    let source = incoming
        .first()
        .ok_or_else(|| irongraph::Error::internal("expected incoming relationship"))?;
    assert_eq!(source.1.id(), NodeId(2));
    graph.delete_node(NodeId(2), true, 6)?;
    let remap = graph.compact()?;
    assert_eq!(graph.node_count(), 2);
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(remap.nodes.get(&NodeId(1)), Some(&0));
    assert_eq!(remap.nodes.get(&NodeId(3)), Some(&1));
    Ok(())
}

#[test]
fn device_delta_contains_only_touched_rows_and_detach_adjacency() -> irongraph::Result<()> {
    let mut graph = sample_graph()?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property is absent"))?;
    graph.set_node_property(NodeId(1), age, ScalarValue::Integer(38), 6)?;
    let update = graph.device_delta(6)?;
    assert_eq!(update.nodes.len(), 1);
    assert_eq!(update.nodes[0].dense, 0);
    assert!(update.edges.is_empty());
    assert!(update.outgoing.is_empty());

    graph.delete_node(NodeId(2), true, 7)?;
    let detach = graph.device_delta(7)?;
    assert_eq!(detach.nodes.len(), 1);
    assert_eq!(detach.nodes[0].dense, 1);
    assert!(!detach.nodes[0].active);
    assert_eq!(detach.edges.len(), 2);
    assert!(detach.edges.iter().all(|edge| !edge.active));
    assert_eq!(
        detach
            .outgoing
            .iter()
            .map(|row| row.dense)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(
        detach
            .incoming
            .iter()
            .map(|row| row.dense)
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    Ok(())
}

#[test]
fn cpu_resident_delta_is_published_once_and_failure_keeps_the_previous_image()
-> irongraph::Result<()> {
    let mut graph = sample_graph()?;
    let project = ProjectId(uuid::Uuid::nil());
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property is absent"))?;
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;

    graph.set_node_property(NodeId(1), age, ScalarValue::Integer(38), 6)?;
    backend.apply_project_delta(ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 6 },
        graph: graph.device_delta(6)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;
    let cancellation = CancellationToken::new();
    assert_eq!(
        backend.filter_node_i64(project, age, CompareOp::Eq, 38, &cancellation)?,
        vec![0]
    );

    graph.set_node_property(NodeId(1), age, ScalarValue::Integer(39), 7)?;
    let failure = backend.apply_project_delta(ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 7 },
        graph: graph.device_delta(7)?,
        temporal: Vec::new(),
        vectors: vec![ResolvedVectorMutation::Upsert {
            property: PropertyId(999),
            entity_id: 1,
            coordinates: vec![0],
            revision: 7,
        }],
        invalidate_derived: true,
    });
    assert!(failure.is_err());
    assert_eq!(
        backend.filter_node_i64(project, age, CompareOp::Eq, 38, &cancellation)?,
        vec![0]
    );
    assert_eq!(
        backend.resident_bookmark(project),
        Some(Bookmark { term: 1, index: 6 })
    );
    Ok(())
}

#[test]
fn temporal_late_samples_and_half_open_windows_are_deterministic() -> irongraph::Result<()> {
    let property = PropertyId(7);
    let mut temporal = TemporalStore::default();
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: 11,
            property,
            value_type: TemporalType::Float,
            retention_nanos: 10_000,
        },
        500,
    )?;
    for (time, index, value) in [(100, 1, 1.0), (300, 2, 3.0), (200, 3, 2.0), (300, 4, 4.0)] {
        temporal.append(
            EntityKind::Node,
            11,
            TemporalSample {
                entity_id: 1,
                property,
                event_time_nanos: time,
                sequence_index: index,
                value: ScalarValue::Float(OrderedFloat(value)),
            },
            500,
        )?;
    }
    assert_eq!(
        temporal
            .current(EntityKind::Node, 11, 1, property)
            .map(|sample| sample.sequence_index),
        Some(4)
    );
    assert_eq!(
        temporal
            .at_time(EntityKind::Node, 11, 1, property, 250, 4)
            .map(|sample| sample.event_time_nanos),
        Some(200)
    );
    let history = temporal.history(EntityKind::Node, 11, 1, property, 100, 300, 4)?;
    assert_eq!(history.len(), 2);
    let buckets = TemporalStore::window(&history, 100, 300, &WindowSpec::tumbling(100))?;
    assert_eq!(buckets.len(), 2);
    assert_eq!(buckets.first().map(|bucket| bucket.start_nanos), Some(100));
    assert_eq!(buckets.get(1).map(|bucket| bucket.start_nanos), Some(200));
    Ok(())
}

#[test]
fn cypher_history_window_uses_the_complete_resident_pipeline() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label missing"))?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property missing"))?;
    let mut temporal = TemporalStore::default();
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: person.0,
            property: age,
            value_type: TemporalType::Integer,
            retention_nanos: 10_000,
        },
        300,
    )?;
    for (entity_id, time, index, value) in [
        (1, 25, 1, 35),
        (1, 75, 2, 36),
        (2, 125, 3, 40),
        (2, 175, 4, 41),
    ] {
        temporal.append(
            EntityKind::Node,
            person.0,
            TemporalSample {
                entity_id,
                property: age,
                event_time_nanos: time,
                sequence_index: index,
                value: ScalarValue::Integer(value),
            },
            300,
        )?;
    }
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &temporal,
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    backend.admit_project(image)?;
    let parameters = BTreeMap::from([
        (
            "from".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(0)),
        ),
        (
            "to".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(200)),
        ),
    ]);
    let source = "MATCH (p:Person)\n\
                  HISTORY p.age FROM $from TO $to AS sample\n\
                  WINDOW HOPPING 100 EVERY 50 ON sample.time AS bucket\n\
                  RETURN p.name AS name, sample.time AS time, sample.value AS value, \
                         bucket.start AS start, bucket.end AS end";
    let mut resident_context = ExecutionContext {
        project_id: project,
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: Some(&temporal),
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: parameters.clone(),
        bookmark,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 300,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: Some(&backend),
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let resident = QueryEngine.execute(source, &mut resident_context)?;
    let mut reference_context = ExecutionContext {
        project_id: project,
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: Some(&temporal),
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters,
        bookmark,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 300,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let reference = QueryEngine.execute(source, &mut reference_context)?;
    assert_eq!(resident.result.schema, reference.result.schema);
    assert_eq!(resident.result.batches, reference.result.batches);
    Ok(())
}

#[test]
fn deterministic_ivf_pq_reranks_against_exact_vectors() -> irongraph::Result<()> {
    let mut vectors = VectorIndex::new(4, Similarity::Euclidean)?;
    for id in 0..64_u64 {
        let x = id as f32 / 64.0;
        vectors.upsert(id, &[x, x * x, 1.0 - x, 0.5 * x], id + 1)?;
    }
    let config = IvfPqConfig {
        size_class_version: irongraph::graph::IVF_PQ_SIZE_CLASS_VERSION,
        coarse_centroids: 8,
        subquantizers: 2,
        bits_per_code: 4,
        probes: 8,
        candidate_budget: 64,
        iterations: 8,
        seed: 19,
    };
    let first = IvfPqIndex::build(&vectors, config)?;
    let second = IvfPqIndex::build(&vectors, config)?;
    let query = [0.49, 0.49 * 0.49, 0.51, 0.245];
    let exact = vectors.exact_search(&query, 10)?;
    let approximate = first.search(&vectors, &query, 10)?;
    let repeated = second.search(&vectors, &query, 10)?;
    assert_eq!(approximate, repeated);
    assert_eq!(approximate, exact);
    vectors.upsert(63, &query, 1_000)?;
    let updated = first.search(&vectors, &query, 1)?;
    assert_eq!(updated.first().map(|hit| hit.entity_id), Some(63));
    Ok(())
}

#[test]
fn scalar_range_and_text_indexes_have_identical_fallback_semantics() -> irongraph::Result<()> {
    let mut equality = EqualityIndex::default();
    let mut range = RangeIndex::default();
    for (row, value) in [10_i64, 20, 20, 30].into_iter().enumerate() {
        let value = ScalarValue::Integer(value);
        equality.insert(&value, row as u32)?;
        range.insert(&value, row as u32)?;
    }
    let twenty = IndexKey::try_from(&ScalarValue::Integer(20))?;
    assert_eq!(
        equality
            .get(&twenty)
            .map(|rows| rows.iter().collect::<Vec<_>>()),
        Some(vec![1, 2])
    );
    let lower = IndexKey::try_from(&ScalarValue::Integer(15))?;
    let upper = IndexKey::try_from(&ScalarValue::Integer(30))?;
    assert_eq!(
        range
            .between(Some((&lower, true)), Some((&upper, false)))
            .iter()
            .collect::<Vec<_>>(),
        vec![1, 2]
    );
    let mut text = TextIndex::default();
    text.upsert(1, "GPU-native graph memory");
    text.upsert(2, "Graph storage");
    assert_eq!(
        text.search("GRAPH MEMORY").iter().collect::<Vec<_>>(),
        vec![1]
    );
    Ok(())
}

#[test]
fn multiline_parser_preserves_layer_temporal_and_window_clauses() -> irongraph::Result<()> {
    let query = parse(
        "/* query\ncomment */ USE health\nUSE LAYER OBSERVED, KNOWLEDGE\nAT TIME $when\n\
         MATCH (s:Sensor {id: $id})\n\
         HISTORY s.temperature FROM $from TO $to AS sample\n\
         WINDOW HOPPING duration('PT1H') EVERY duration('PT5M') ON sample.time\n\
         ALIGN TO $anchor TIME ZONE 'UTC' EMIT EMPTY AS bucket\n\
         RETURN bucket.start, avg(sample.value)",
    )?;
    assert_eq!(query.project.as_deref(), Some("health"));
    assert!(query.read_layers.contains(LayerMask::OBSERVED));
    assert!(query.read_layers.contains(LayerMask::KNOWLEDGE));
    let irongraph::cypher::Statement::Query(body) = query.statement else {
        return Err(irongraph::Error::internal("expected query statement"));
    };
    assert_eq!(body.clauses.len(), 4);
    Ok(())
}

#[test]
fn query_engine_executes_filters_projection_sort_and_records_dependencies() -> irongraph::Result<()>
{
    let graph = sample_graph()?;
    let mut parameters = BTreeMap::new();
    parameters.insert(
        "minimum".to_owned(),
        ResultValue::Scalar(ScalarValue::Integer(35)),
    );
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters,
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "MATCH (p:Person) WHERE p.age >= $minimum RETURN p.name AS name ORDER BY name",
        &mut context,
    )?;
    assert_eq!(
        output.result.batches.first().map(|batch| batch.row_count),
        Some(2)
    );
    assert!(!output.dependencies.entities.is_empty());
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn query_engine_streaming_yields_bounded_batches_without_retaining_them() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 1,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let mut schema_count = 0;
    let mut batch_rows = Vec::new();
    let output = QueryEngine.execute_streaming(
        "MATCH (p:Person) RETURN p.name AS name ORDER BY name",
        &mut context,
        &mut |item| {
            match item {
                ExecutionStreamItem::Schema(schema) => {
                    schema_count += 1;
                    assert_eq!(schema.len(), 1);
                }
                ExecutionStreamItem::Batch(batch) => batch_rows.push(batch.row_count),
            }
            Ok(())
        },
    )?;
    assert_eq!(schema_count, 1);
    assert_eq!(batch_rows, vec![1, 1, 1]);
    assert!(output.result.batches.is_empty());
    assert_eq!(output.result.schema.len(), 1);
    Ok(())
}

#[test]
fn query_engine_dispatches_integer_top_k_to_resident_backend() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let mut backend = CpuBackend::new(32 * 1024 * 1024, 1024);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: Some(&backend),
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "MATCH (p:Person) WITH p ORDER BY p.age DESC LIMIT 2 RETURN p.age AS age",
        &mut context,
    )?;
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter().cloned())
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            ResultValue::Scalar(ScalarValue::Integer(41)),
            ResultValue::Scalar(ScalarValue::Integer(37)),
        ]
    );
    let incoming = QueryEngine.execute(
        "MATCH (p:Person {name: 'Grace'})<-[:KNOWS]-(source:Person) RETURN source.name AS name",
        &mut context,
    )?;
    assert_eq!(
        incoming.result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::String(Arc::from("Ada")))]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_query_engine_materialized_multikey_sort_matches_reference() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let governor = DeviceMemoryGovernor::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::with_governor(0, governor.clone())?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: Some(&metal),
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let source = "UNWIND [{name: 'Ada', rank: 38}, {name: 'Grace', rank: 42},\n\
                          {name: 'Linus', rank: 30}, {name: 'Grace', rank: 41}] AS row\n\
                  WITH row.name AS name, row.rank AS rank\n\
                  ORDER BY name DESC, rank ASC\n\
                  RETURN name, rank";
    let metal_output = QueryEngine.execute(source, &mut context)?;
    context.backend = Some(&cpu);
    let cpu_output = QueryEngine.execute(source, &mut context)?;
    assert_eq!(metal_output.result.schema, cpu_output.result.schema);
    assert_eq!(metal_output.result.batches, cpu_output.result.batches);
    assert_eq!(governor.snapshot().scratch_bytes, 0);
    assert_eq!(
        metal_output.result.batches[0].columns[0].values,
        vec![
            ResultValue::Scalar(ScalarValue::String(Arc::from("Linus"))),
            ResultValue::Scalar(ScalarValue::String(Arc::from("Grace"))),
            ResultValue::Scalar(ScalarValue::String(Arc::from("Grace"))),
            ResultValue::Scalar(ScalarValue::String(Arc::from("Ada"))),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_native_string_property_order_uses_utf8_values_not_dictionary_ids() -> irongraph::Result<()>
{
    let _metal_test = metal_test_guard();
    let project = ProjectId(uuid::Uuid::nil());
    let mut graph = GraphStore::default();
    let row = graph.catalog_mut().intern_label("Row")?;
    let name = graph.catalog_mut().intern_property("name")?;
    // Deliberately intern the dictionary in the opposite of lexical order. The resident GPU path
    // must compare UTF-8 bytes, not use these insertion IDs as sort keys.
    for (id, value) in [
        (1, Some("zebra")),
        (2, Some("apple")),
        (3, Some("applepie")),
        (4, None),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![row],
            properties: value
                .map(|value| vec![(name, ScalarValue::String(Arc::from(value)))])
                .unwrap_or_default(),
        })?;
    }
    let snapshot = Arc::new(graph.snapshot()?);
    let bookmark = Bookmark {
        term: 0,
        index: graph.revision(),
    };
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    for (source, expected) in [
        (
            "MATCH (n:Row) WITH n, n.name AS name ORDER BY name RETURN name",
            vec!["apple", "applepie", "zebra", "<null>"],
        ),
        (
            "MATCH (n:Row) WITH n, n.name AS name ORDER BY name DESC RETURN name",
            vec!["<null>", "zebra", "applepie", "apple"],
        ),
    ] {
        let cpu_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            true,
        )?;
        let metal_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )?;
        assert_eq!(metal_result, cpu_result);
        let actual = metal_result.batches[0].columns[0]
            .values
            .iter()
            .map(|value| match value {
                ResultValue::Scalar(ScalarValue::String(value)) => value.to_string(),
                ResultValue::Scalar(ScalarValue::Null) => "<null>".to_owned(),
                other => panic!("unexpected native string-sort value: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_native_float_property_order_matches_ordered_float_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let project = ProjectId(uuid::Uuid::nil());
    let mut graph = GraphStore::default();
    let row = graph.catalog_mut().intern_label("Row")?;
    let score = graph.catalog_mut().intern_property("score")?;
    let tag = graph.catalog_mut().intern_property("tag")?;
    // Exercise the semantic edge cases of the reference `OrderedFloat` ordering. In particular,
    // signed zero must remain one stable tie and every NaN must sort after finite values.
    for (id, score_value, tag_value) in [
        (1, Some(f64::NEG_INFINITY), "negative-infinity"),
        (2, Some(-1.5), "negative"),
        (3, Some(-0.0), "negative-zero"),
        (4, Some(0.0), "positive-zero"),
        (5, Some(0.25), "positive"),
        (6, Some(f64::INFINITY), "infinity"),
        (7, Some(f64::NAN), "nan"),
        (8, None, "null"),
    ] {
        let mut properties = vec![(tag, ScalarValue::String(Arc::from(tag_value)))];
        if let Some(score_value) = score_value {
            properties.push((score, ScalarValue::Float(OrderedFloat(score_value))));
        }
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![row],
            properties,
        })?;
    }
    let snapshot = Arc::new(graph.snapshot()?);
    let bookmark = Bookmark {
        term: 0,
        index: graph.revision(),
    };
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    for (source, expected) in [
        (
            "MATCH (n:Row) WITH n, n.score AS score, n.tag AS tag ORDER BY score RETURN tag",
            vec![
                "negative-infinity",
                "negative",
                "negative-zero",
                "positive-zero",
                "positive",
                "infinity",
                "nan",
                "null",
            ],
        ),
        (
            "MATCH (n:Row) WITH n, n.score AS score, n.tag AS tag ORDER BY score DESC RETURN tag",
            vec![
                "null",
                "nan",
                "infinity",
                "positive",
                "negative-zero",
                "positive-zero",
                "negative",
                "negative-infinity",
            ],
        ),
    ] {
        let cpu_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            true,
        )?;
        let metal_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )?;
        assert_eq!(metal_result, cpu_result);
        let actual = metal_result.batches[0].columns[0]
            .values
            .iter()
            .map(|value| match value {
                ResultValue::Scalar(ScalarValue::String(value)) => value.to_string(),
                other => panic!("unexpected native float-sort value: {other:?}"),
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn resident_pipeline_keeps_expansion_filter_order_and_projection_columnar() -> irongraph::Result<()>
{
    let graph = sample_graph()?;
    let project = ProjectId(uuid::Uuid::nil());
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label missing"))?;
    let knows = graph
        .catalog()
        .relationship_type("KNOWS")
        .ok_or_else(|| irongraph::Error::internal("KNOWS type missing"))?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property missing"))?;
    let mut backend = CpuBackend::new(32 * 1024 * 1024, 1024);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    let request = ResidentNodePipelineRequest {
        project,
        labels: vec![person],
        layers: LayerMask::AUTHORITY,
        initial_optional: false,
        expansion: Some(ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![knows],
            end_labels: vec![person],
            end_equals_start: false,
            optional: false,
            end_predicates: Vec::new(),
        }),
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: vec![ResidentI64Predicate {
            binding: ResidentNodeBinding::End,
            property: age,
            operation: CompareOp::GreaterOrEqual,
            operand: 20,
        }],
        property_filters: Vec::new(),
        value_matrix: None,
        mutation: None,
        orders: vec![ResidentNodeOrder {
            binding: ResidentNodeBinding::End,
            property: age,
            descending: false,
            nulls_first: false,
        }],
        offset: 0,
        limit: 10,
        integer_projections: vec![
            ResidentI64Projection {
                binding: ResidentNodeBinding::Start,
                property: age,
            },
            ResidentI64Projection {
                binding: ResidentNodeBinding::End,
                property: age,
            },
        ],
        property_null_projections: Vec::new(),
        max_output_rows: 10,
    };
    let result = backend.execute_node_pipeline(&request, &CancellationToken::new())?;
    assert_eq!(result.start_rows, vec![1, 0]);
    assert_eq!(result.edge_rows, vec![1, 0]);
    assert_eq!(result.end_rows, vec![2, 1]);
    assert_eq!(result.integer_columns[0].values, vec![41, 37]);
    assert_eq!(result.integer_columns[1].values, vec![29, 41]);
    assert!(
        result
            .integer_columns
            .iter()
            .all(|column| column.validity == vec![1, 1])
    );

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = backend
        .execute_node_pipeline(&request, &cancellation)
        .expect_err("cancelled resident pipeline must not execute");
    assert_eq!(error.code, irongraph::ErrorCode::Cancelled);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_to_boolean_scalar_pipeline_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "UNWIND [true, false, 'TRUE', 'false', 'not-a-boolean', null] AS value \
                  RETURN toBoolean(value) AS converted";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(result.schema, cpu_result.schema);
    assert_eq!(result.batches, cpu_result.batches);
    assert_eq!(result.schema.len(), 1);
    let values = result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns[0].values)
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        values,
        vec![
            ResultValue::Scalar(ScalarValue::Boolean(true)),
            ResultValue::Scalar(ScalarValue::Boolean(false)),
            ResultValue::Scalar(ScalarValue::Boolean(true)),
            ResultValue::Scalar(ScalarValue::Boolean(false)),
            ResultValue::Scalar(ScalarValue::Null),
            ResultValue::Scalar(ScalarValue::Null),
        ]
    );
    let error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN toBoolean(1.0)",
        true,
    )
    .expect_err("native toBoolean must report invalid non-string/non-boolean values");
    assert_eq!(error.code, irongraph::ErrorCode::QueryType);
    assert!(error.message.contains("InvalidArgumentValue"));
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_date_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: vec![
            irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Date,
                input: irongraph::gpu::ResidentTemporalValueInput::String("2015-07-21".to_owned()),
            },
            irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Date,
                input: irongraph::gpu::ResidentTemporalValueInput::String("2000-02-29".to_owned()),
            },
            irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Date,
                input: irongraph::gpu::ResidentTemporalValueInput::String("2015-02-29".to_owned()),
            },
            irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Date,
                input: irongraph::gpu::ResidentTemporalValueInput::Null,
            },
        ],
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        vec![
            irongraph::gpu::ResidentTemporalValue::Date(16_637),
            irongraph::gpu::ResidentTemporalValue::Date(11_016),
            irongraph::gpu::ResidentTemporalValue::InvalidArgument,
            irongraph::gpu::ResidentTemporalValue::Null,
        ]
    );
    for (input, expected_days) in [
        ("20150721", 16_637),
        ("2015-07", 16_617),
        ("201507", 16_617),
        ("2015-W30-2", 16_637),
        ("2015W302", 16_637),
        ("2015-W30", 16_636),
        ("2015W30", 16_636),
        ("2015-202", 16_637),
        ("2015202", 16_637),
        ("2015", 16_436),
    ] {
        let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
            invocations: vec![irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Date,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            }],
            output_registers: Vec::new(),
        };
        let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
        let metal_direct =
            metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
        assert_eq!(metal_direct, cpu_direct, "native date form {input}");
        assert_eq!(
            metal_direct.values,
            vec![irongraph::gpu::ResidentTemporalValue::Date(expected_days)],
            "native date form {input}"
        );
    }

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN date('2015-07-21') AS date, date('2000-02-29') AS leap";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Date(16_637))]
    );
    assert_eq!(
        metal_result.batches[0].columns[1].values,
        vec![ResultValue::Scalar(ScalarValue::Date(11_016))]
    );

    for source in ["RETURN date('2015-02-29')", "RETURN date(42)"] {
        let cpu_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            false,
        )
        .expect_err("CPU reference must reject invalid date arguments");
        let metal_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )
        .expect_err("native date program must reject invalid date arguments");
        assert_eq!(metal_error.code, cpu_error.code);
        assert_eq!(metal_error.message, cpu_error.message);
    }
    let null_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN date(null) AS date",
        true,
    )?;
    assert_eq!(
        null_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_dependent_temporal_projection_uses_device_registers_and_matches_cpu()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "WITH date({year: 1984, month: 11, day: 11}) AS other \
                  RETURN date({date: other, day: 28}) AS result";

    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;

    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Date(5_445))]
    );

    let named_source = "WITH date({year: 1984, month: 10, day: 11}) AS other \
                        RETURN datetime({date: other, day: 28, hour: 10, minute: 10, \
                                         second: 10, timezone: 'Pacific/Honolulu'}) AS result";
    let cpu_named = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        named_source,
        false,
    )?;
    let metal_named = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        named_source,
        true,
    )?;
    assert_eq!(metal_named.schema, cpu_named.schema);
    assert_eq!(metal_named.batches, cpu_named.batches);
    assert_eq!(
        metal_named.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::ZonedDateTime {
            seconds: 467_842_210,
            nanos: 0,
            timezone: Arc::from("Pacific/Honolulu"),
        })]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_local_time_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            ("21:40:32.142", 78_032_142_000_000_i64),
            ("214032.142", 78_032_142_000_000_i64),
            ("21:40:32", 78_032_000_000_000_i64),
            ("214032", 78_032_000_000_000_i64),
            ("21:40", 78_000_000_000_000_i64),
            ("2140", 78_000_000_000_000_i64),
            ("21", 75_600_000_000_000_i64),
        ]
        .into_iter()
        .map(
            |(input, _)| irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::LocalTime,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            },
        )
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        vec![
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_032_142_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_032_142_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_032_000_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_032_000_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_000_000_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(78_000_000_000_000),
            irongraph::gpu::ResidentTemporalValue::LocalTime(75_600_000_000_000),
        ]
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN localtime('21:40:32.142') AS time";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    let cpu_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        "RETURN localtime('24:00')",
        false,
    )
    .expect_err("CPU reference must reject invalid local times");
    let metal_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN localtime('24:00')",
        true,
    )
    .expect_err("native localtime program must reject invalid local times");
    assert_eq!(metal_error.code, cpu_error.code);
    assert_eq!(metal_error.message, cpu_error.message);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_time_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            ("14:30", 52_200_000_000_000_i64, 0_i32),
            ("14:30+0100", 52_200_000_000_000_i64, 3_600_i32),
            ("21:40:32.142+0100", 78_032_142_000_000_i64, 3_600_i32),
            ("214032.142Z", 78_032_142_000_000_i64, 0_i32),
            ("21:40:32+01:00", 78_032_000_000_000_i64, 3_600_i32),
            ("214032-0100", 78_032_000_000_000_i64, -3_600_i32),
            ("21:40-01:30", 78_000_000_000_000_i64, -5_400_i32),
            ("2140-00:00", 78_000_000_000_000_i64, 0_i32),
            ("2140-02", 78_000_000_000_000_i64, -7_200_i32),
            ("22+18:00", 79_200_000_000_000_i64, 64_800_i32),
        ]
        .into_iter()
        .map(
            |(input, _, _)| irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Time,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            },
        )
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        [
            (52_200_000_000_000_i64, 0_i32),
            (52_200_000_000_000_i64, 3_600_i32),
            (78_032_142_000_000_i64, 3_600_i32),
            (78_032_142_000_000_i64, 0_i32),
            (78_032_000_000_000_i64, 3_600_i32),
            (78_032_000_000_000_i64, -3_600_i32),
            (78_000_000_000_000_i64, -5_400_i32),
            (78_000_000_000_000_i64, 0_i32),
            (78_000_000_000_000_i64, -7_200_i32),
            (79_200_000_000_000_i64, 64_800_i32),
        ]
        .into_iter()
        .map(
            |(nanos, offset_seconds)| irongraph::gpu::ResidentTemporalValue::ZonedTime {
                nanos,
                offset_seconds,
            },
        )
        .collect::<Vec<_>>()
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN time('14:30') AS time, time('14:30+0100') AS offset_time";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            ResultValue::Scalar(ScalarValue::ZonedTime {
                nanos: 52_200_000_000_000,
                offset_seconds: 0,
            }),
            ResultValue::Scalar(ScalarValue::ZonedTime {
                nanos: 52_200_000_000_000,
                offset_seconds: 3_600,
            }),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_local_datetime_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            (
                "2015-07-21T21:40:32.142",
                1_437_514_832_i64,
                142_000_000_u32,
            ),
            ("2015-W30-2T214032.142", 1_437_514_832_i64, 142_000_000_u32),
            ("2015-202T21:40:32", 1_437_514_832_i64, 0_u32),
            ("2015T214032", 1_420_148_432_i64, 0_u32),
            ("20150721T21:40", 1_437_514_800_i64, 0_u32),
            ("2015-W30T2140", 1_437_428_400_i64, 0_u32),
            ("2015202T21", 1_437_512_400_i64, 0_u32),
        ]
        .into_iter()
        .map(
            |(input, _, _)| irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::LocalDateTime,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            },
        )
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        [
            (1_437_514_832_i64, 142_000_000_u32),
            (1_437_514_832_i64, 142_000_000_u32),
            (1_437_514_832_i64, 0_u32),
            (1_420_148_432_i64, 0_u32),
            (1_437_514_800_i64, 0_u32),
            (1_437_428_400_i64, 0_u32),
            (1_437_512_400_i64, 0_u32),
        ]
        .into_iter()
        .map(
            |(seconds, nanos)| irongraph::gpu::ResidentTemporalValue::LocalDateTime {
                seconds,
                nanos,
            },
        )
        .collect::<Vec<_>>()
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN localdatetime('2015-W30-2T214032.142') AS datetime";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    let cpu_date_only = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        "RETURN localdatetime('2015-07-21')",
        false,
    )?;
    let metal_date_only = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN localdatetime('2015-07-21')",
        true,
    )?;
    assert_eq!(metal_date_only.schema, cpu_date_only.schema);
    assert_eq!(metal_date_only.batches, cpu_date_only.batches);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_datetime_fixed_offset_constructor_is_native_and_matches_cpu_semantics()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            (
                "2015-07-21T21:40:32.142+0100",
                1_437_511_232_i64,
                142_000_000_u32,
                3_600_i32,
            ),
            (
                "2015-W30-2T214032.142Z",
                1_437_514_832_i64,
                142_000_000_u32,
                0_i32,
            ),
            (
                "2015-202T21:40:32+01:00",
                1_437_511_232_i64,
                0_u32,
                3_600_i32,
            ),
            ("2015T214032-0100", 1_420_152_032_i64, 0_u32, -3_600_i32),
            ("20150721T21:40-01:30", 1_437_520_200_i64, 0_u32, -5_400_i32),
            ("2015-W30T2140-00:00", 1_437_428_400_i64, 0_u32, 0_i32),
            ("2015-W30T2140-02", 1_437_435_600_i64, 0_u32, -7_200_i32),
            ("2015202T21+18:00", 1_437_447_600_i64, 0_u32, 64_800_i32),
        ]
        .into_iter()
        .map(
            |(input, _, _, _)| irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::DateTime,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            },
        )
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        [
            (1_437_511_232_i64, 142_000_000_u32, 3_600_i32),
            (1_437_514_832_i64, 142_000_000_u32, 0_i32),
            (1_437_511_232_i64, 0_u32, 3_600_i32),
            (1_420_152_032_i64, 0_u32, -3_600_i32),
            (1_437_520_200_i64, 0_u32, -5_400_i32),
            (1_437_428_400_i64, 0_u32, 0_i32),
            (1_437_435_600_i64, 0_u32, -7_200_i32),
            (1_437_447_600_i64, 0_u32, 64_800_i32),
        ]
        .into_iter()
        .map(|(seconds, nanos, offset_seconds)| {
            irongraph::gpu::ResidentTemporalValue::ZonedDateTimeFixedOffset {
                seconds,
                nanos,
                offset_seconds,
            }
        })
        .collect::<Vec<_>>()
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN datetime('20150721T21:40-01:30') AS datetime";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    let cpu_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        "RETURN datetime('2015-07-21T21:40')",
        false,
    )
    .expect_err("CPU reference must reject a datetime without an offset");
    let metal_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN datetime('2015-07-21T21:40')",
        true,
    )
    .expect_err("native datetime program must reject a datetime without an offset");
    assert_eq!(metal_error.code, cpu_error.code);
    assert_eq!(metal_error.message, cpu_error.message);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_named_datetime_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            "2015-07-21T21:40:32.142+02:00[Europe/Stockholm]",
            "2015-07-21T21:40:32.142+0845[Australia/Eucla]",
            "2015-07-21T21:40:32.142-04[America/New_York]",
            "2015-07-21T21:40:32.142[Europe/London]",
            "1818-07-21T21:40:32.142[Europe/Stockholm]",
        ]
        .into_iter()
        .map(|input| irongraph::gpu::ResidentTemporalValueInvocation {
            function: irongraph::gpu::ResidentTemporalValueFunction::DateTime,
            input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
        })
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        [
            (1_437_507_632_i64, 142_000_000_u32, "Europe/Stockholm"),
            (1_437_483_332_i64, 142_000_000_u32, "Australia/Eucla"),
            (1_437_529_232_i64, 142_000_000_u32, "America/New_York"),
            (1_437_511_232_i64, 142_000_000_u32, "Europe/London"),
            (-4_779_227_576_i64, 142_000_000_u32, "Europe/Stockholm"),
        ]
        .into_iter()
        .map(|(seconds, nanos, timezone)| {
            irongraph::gpu::ResidentTemporalValue::ZonedDateTimeNamed {
                seconds,
                nanos,
                timezone: timezone.to_owned(),
            }
        })
        .collect::<Vec<_>>()
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN datetime('1818-07-21T21:40:32.142[Europe/Stockholm]') AS datetime";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::ZonedDateTime {
            seconds: -4_779_227_576,
            nanos: 142_000_000,
            timezone: Arc::from("Europe/Stockholm"),
        })]
    );

    for source in [
        "RETURN datetime('2015-07-21T21:40:32+01:00[Europe/Stockholm]')",
        "RETURN datetime('1799-07-21T21:40:32[Europe/Stockholm]')",
    ] {
        let cpu_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            false,
        )
        .expect_err("CPU reference must reject an invalid named datetime");
        let metal_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )
        .expect_err("native named datetime program must reject an invalid named datetime");
        assert_eq!(metal_error.code, cpu_error.code);
        assert_eq!(metal_error.message, cpu_error.message);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_literal_temporal_map_constructors_are_native_and_match_cpu_semantics()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN \
        date({year: 1984, month: 10, day: 11}) AS calendar, \
        date({year: 1817, week: 30}) AS week, \
        date({year: 1984, quarter: 3, dayOfQuarter: 45}) AS quarter, \
        date({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS inherited, \
        localtime({hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456}) AS local_time, \
        time({hour: 12, minute: 34, second: 56, timezone: '+02:05:59'}) AS zoned_time, \
        localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS local_datetime, \
        datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, timezone: '+01:00'}) AS datetime, \
        datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}) AS named_datetime, \
        datetime({epochMillis: -1}) AS epoch";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            ResultValue::Scalar(ScalarValue::Date(5_397)),
            ResultValue::Scalar(ScalarValue::Date(-55_681)),
            ResultValue::Scalar(ScalarValue::Date(5_339)),
            ResultValue::Scalar(ScalarValue::Date(-55_875)),
            ResultValue::Scalar(ScalarValue::LocalTime(45_074_123_456_789)),
            ResultValue::Scalar(ScalarValue::ZonedTime {
                nanos: 45_296_000_000_000,
                offset_seconds: 7_559,
            }),
            ResultValue::Scalar(ScalarValue::LocalDateTime {
                seconds: 459_174_674,
                nanos: 645_876_000,
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 447_505_200,
                nanos: 0,
                timezone: Arc::from("+01:00"),
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 459_167_474,
                nanos: 645_876_000,
                timezone: Arc::from("Europe/Stockholm"),
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: -1,
                nanos: 999_000_000,
                timezone: Arc::from("UTC"),
            }),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_temporal_truncation_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN \
        date.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {day: 5}) AS date, \
        localtime.truncate('microsecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS local_time, \
        time.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: '+01:00'}) AS time, \
        localdatetime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS local_datetime, \
        datetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14}), {timezone: 'Europe/Stockholm'}) AS datetime, \
        datetime.truncate('hour', datetime('2015-07-21T21:40:32.142+02:00[Europe/Stockholm]'), {}) AS named_source";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            ResultValue::Scalar(ScalarValue::Date(5_117)),
            ResultValue::Scalar(ScalarValue::LocalTime(45_074_645_876_002)),
            ResultValue::Scalar(ScalarValue::ZonedTime {
                nanos: 43_200_000_000_000,
                offset_seconds: 3_600,
            }),
            ResultValue::Scalar(ScalarValue::LocalDateTime {
                seconds: 466_345_874,
                nanos: 645_000_002,
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 946_681_200,
                nanos: 0,
                timezone: Arc::from("Europe/Stockholm"),
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 1_437_505_200,
                nanos: 0,
                timezone: Arc::from("Europe/Stockholm"),
            }),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_literal_duration_map_constructors_are_native_and_match_cpu_semantics()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN \
        duration({days: 14, hours: 16, minutes: 12}) AS d1, \
        duration({months: 5, days: 1.5}) AS d2, \
        duration({months: 0.75}) AS d3, \
        duration({weeks: 2.5}) AS d4, \
        duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS d5, \
        duration({days: 14, seconds: 70, milliseconds: 1}) AS d6, \
        duration({days: 14, seconds: 70, microseconds: 1}) AS d7, \
        duration({days: 14, seconds: 70, nanoseconds: 1}) AS d8, \
        duration({minutes: 1.5, seconds: 1}) AS d9";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            (0_i64, 14_i64, 58_320_i64, 0_i32),
            (5_i64, 1_i64, 43_200_i64, 0_i32),
            (0_i64, 22_i64, 71_509_i64, 500_000_000_i32),
            (0_i64, 17_i64, 43_200_i64, 0_i32),
            (149_i64, 14_i64, 58_390_i64, 0_i32),
            (0_i64, 14_i64, 70_i64, 1_000_000_i32),
            (0_i64, 14_i64, 70_i64, 1_000_i32),
            (0_i64, 14_i64, 70_i64, 1_i32),
            (0_i64, 0_i64, 91_i64, 0_i32),
        ]
        .into_iter()
        .map(|(months, days, seconds, nanos)| {
            ResultValue::Scalar(ScalarValue::Duration {
                months,
                days,
                seconds,
                nanos,
            })
        })
        .collect::<Vec<_>>(),
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_datetime_epoch_helpers_are_native_and_match_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN datetime.fromepoch(416779, 999999999) AS d1, \
        datetime.fromepochmillis(237821673987) AS d2";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 416_779,
                nanos: 999_999_999,
                timezone: Arc::from("UTC"),
            }),
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 237_821_673,
                nanos: 987_000_000,
                timezone: Arc::from("UTC"),
            }),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_duration_constructor_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentTemporalValueProgramRequest {
        invocations: [
            ("P14DT16H12M", 0_i64, 14_i64, 58_320_i64, 0_i32),
            ("P5M1.5D", 5_i64, 1_i64, 43_200_i64, 0_i32),
            ("P0.75M", 0_i64, 22_i64, 71_509_i64, 500_000_000_i32),
            ("PT0.75M", 0_i64, 0_i64, 45_i64, 0_i32),
            ("P2.5W", 0_i64, 17_i64, 43_200_i64, 0_i32),
            ("P12Y5M14DT16H12M70S", 149_i64, 14_i64, 58_390_i64, 0_i32),
            (
                "P2012-02-02T14:37:21.545",
                24_146_i64,
                2_i64,
                52_641_i64,
                545_000_000_i32,
            ),
        ]
        .into_iter()
        .map(
            |(input, _, _, _, _)| irongraph::gpu::ResidentTemporalValueInvocation {
                function: irongraph::gpu::ResidentTemporalValueFunction::Duration,
                input: irongraph::gpu::ResidentTemporalValueInput::String(input.to_owned()),
            },
        )
        .collect(),
        output_registers: Vec::new(),
    };
    let cpu_direct = cpu.execute_temporal_value_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_temporal_value_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(
        metal_direct.values,
        [
            (0_i64, 14_i64, 58_320_i64, 0_i32),
            (5_i64, 1_i64, 43_200_i64, 0_i32),
            (0_i64, 22_i64, 71_509_i64, 500_000_000_i32),
            (0_i64, 0_i64, 45_i64, 0_i32),
            (0_i64, 17_i64, 43_200_i64, 0_i32),
            (149_i64, 14_i64, 58_390_i64, 0_i32),
            (24_146_i64, 2_i64, 52_641_i64, 545_000_000_i32),
        ]
        .into_iter()
        .map(
            |(months, days, seconds, nanos)| irongraph::gpu::ResidentTemporalValue::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        )
        .collect::<Vec<_>>()
    );

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN duration('P0.75M') AS duration";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    let cpu_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        "RETURN duration('P')",
        false,
    )
    .expect_err("CPU reference must reject an empty duration");
    let metal_error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        "RETURN duration('P')",
        true,
    )
    .expect_err("native duration program must reject an empty duration");
    assert_eq!(metal_error.code, cpu_error.code);
    assert_eq!(metal_error.message, cpu_error.message);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_range_scalar_program_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentRangeProgramRequest {
        start: -3,
        end: 5,
        step: 2,
        integer_operands: [true; 3],
        max_values: 16,
    };
    let cpu_direct = cpu.execute_range_program(&request, &CancellationToken::new())?;
    let metal_direct = metal.execute_range_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_direct, cpu_direct);
    assert_eq!(metal_direct.values, vec![-3, -1, 1, 3, 5]);

    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "RETURN range(-3, 5, 2) AS values";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);

    for source in ["RETURN range(0, 1, 0)", "RETURN range(true, 1, 1)"] {
        let cpu_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            false,
        )
        .expect_err("CPU reference must reject invalid range arguments");
        let metal_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )
        .expect_err("native range program must reject invalid range arguments");
        assert_eq!(metal_error.code, cpu_error.code);
        assert_eq!(metal_error.message, cpu_error.message);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_literal_sort_pipeline_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let project = ProjectId(uuid::Uuid::nil());
    let bookmark = Bookmark { term: 0, index: 0 };
    let source = "UNWIND [[1, null], [], [1, 'a'], ['a'], [null, 2]] AS value\n\
                  WITH value ORDER BY value DESC LIMIT 3\n\
                  RETURN value";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![
            ResultValue::List(vec![
                ResultValue::Scalar(ScalarValue::Null),
                ResultValue::Scalar(ScalarValue::Integer(2)),
            ]),
            ResultValue::List(vec![
                ResultValue::Scalar(ScalarValue::Integer(1)),
                ResultValue::Scalar(ScalarValue::Null),
            ]),
            ResultValue::List(vec![
                ResultValue::Scalar(ScalarValue::Integer(1)),
                ResultValue::Scalar(ScalarValue::String(Arc::from("a"))),
            ]),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_scalar_value_program_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let request = irongraph::gpu::ResidentScalarProgramRequest {
        // 6 = {outer: [{a: [1, null]}, "ß"], a: 1}. Every compound child precedes its
        // owner, exercising both nested list and nested map materialization.
        scalar_cells: vec![
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Integer,
                payload: 1,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::String,
                payload: 1,
                auxiliary: 2,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::List,
                payload: 0,
                auxiliary: 2,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Map,
                payload: 0,
                auxiliary: 1,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::List,
                payload: 2,
                auxiliary: 2,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Map,
                payload: 1,
                auxiliary: 2,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Integer,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Boolean,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Boolean,
                payload: 1,
                auxiliary: 0,
            },
            // One backend-owned destination cell per scalar SSA instruction.
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
        ],
        map_entries: vec![
            irongraph::gpu::ResidentScalarMapEntry { key: 0, value: 3 },
            irongraph::gpu::ResidentScalarMapEntry { key: 1, value: 5 },
            irongraph::gpu::ResidentScalarMapEntry { key: 0, value: 0 },
        ],
        scalar_list_entries: vec![
            irongraph::gpu::ResidentScalarListEntry { value: 0 },
            irongraph::gpu::ResidentScalarListEntry { value: 1 },
            irongraph::gpu::ResidentScalarListEntry { value: 4 },
            irongraph::gpu::ResidentScalarListEntry { value: 2 },
        ],
        string_offsets: vec![0, 2, 34, 66, 98, 130, 162],
        string_bytes: {
            let mut bytes = "ß".as_bytes().to_vec();
            bytes.resize(162, 0);
            bytes
        },
        null_cell: 1,
        false_cell: Some(8),
        true_cell: Some(9),
        instructions: vec![
            irongraph::gpu::ResidentScalarProgramInstruction {
                opcode: irongraph::gpu::ResidentScalarProgramOpcode::ListIndex,
                left: irongraph::gpu::ResidentScalarProgramOperand::Cell(5),
                right: irongraph::gpu::ResidentScalarProgramOperand::Cell(0),
                third: None,
                fourth: None,
            },
            irongraph::gpu::ResidentScalarProgramInstruction {
                opcode: irongraph::gpu::ResidentScalarProgramOpcode::ToInteger,
                left: irongraph::gpu::ResidentScalarProgramOperand::Cell(0),
                right: irongraph::gpu::ResidentScalarProgramOperand::Cell(1),
                third: None,
                fourth: None,
            },
            irongraph::gpu::ResidentScalarProgramInstruction {
                opcode: irongraph::gpu::ResidentScalarProgramOpcode::ListIndex,
                left: irongraph::gpu::ResidentScalarProgramOperand::Cell(3),
                right: irongraph::gpu::ResidentScalarProgramOperand::Register(1),
                third: None,
                fourth: None,
            },
            irongraph::gpu::ResidentScalarProgramInstruction {
                opcode: irongraph::gpu::ResidentScalarProgramOpcode::ListMembership,
                left: irongraph::gpu::ResidentScalarProgramOperand::Cell(0),
                right: irongraph::gpu::ResidentScalarProgramOperand::Cell(3),
                third: None,
                fourth: None,
            },
            irongraph::gpu::ResidentScalarProgramInstruction {
                opcode: irongraph::gpu::ResidentScalarProgramOpcode::ListSliceMembership,
                left: irongraph::gpu::ResidentScalarProgramOperand::Cell(0),
                right: irongraph::gpu::ResidentScalarProgramOperand::Cell(3),
                third: Some(irongraph::gpu::ResidentScalarProgramOperand::Cell(7)),
                fourth: Some(irongraph::gpu::ResidentScalarProgramOperand::Cell(0)),
            },
        ],
        output_values: vec![
            irongraph::gpu::ResidentScalarProgramOperand::Cell(6),
            irongraph::gpu::ResidentScalarProgramOperand::Register(0),
            irongraph::gpu::ResidentScalarProgramOperand::Register(2),
            irongraph::gpu::ResidentScalarProgramOperand::Register(3),
            irongraph::gpu::ResidentScalarProgramOperand::Register(4),
        ],
    };
    let cpu_frame = cpu.execute_scalar_program(&request, &CancellationToken::new())?;
    let metal_frame = metal.execute_scalar_program(&request, &CancellationToken::new())?;
    assert_eq!(metal_frame, cpu_frame);
    assert_eq!(metal_frame.output_cells, vec![6, 2, 1, 9, 9]);

    let graph = GraphStore::default();
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let source = "WITH {outer: [{a: [1, null]}, 'ß']} AS value\n\
                  RETURN value AS literal, [[], {key: -0.0}] AS list";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        true,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);

    // These conversions create new INTEGER cells. A strict Metal success therefore proves that
    // the scalar VM materialized register values on device rather than selecting a host-built
    // answer cell.
    let conversions = "WITH 82.9 AS weight
                       RETURN toInteger(weight) AS integer,
                              toInteger(true) AS one,
                              toInteger(false) AS zero,
                              toInteger(-82.9) AS negative";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        conversions,
        true,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        conversions,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0]
            .columns
            .iter()
            .map(|column| column.values[0].clone())
            .collect::<Vec<_>>(),
        vec![
            ResultValue::Scalar(ScalarValue::Integer(82)),
            ResultValue::Scalar(ScalarValue::Integer(1)),
            ResultValue::Scalar(ScalarValue::Integer(0)),
            ResultValue::Scalar(ScalarValue::Integer(-82)),
        ]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_scalar_arithmetic_nan_and_dynamic_map_index_are_native() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };

    for source in [
        "RETURN 4 * 2 + 3 / 2 AS integer, 4 ^ 3 * 2 ^ 3 AS promoted, -(3 ^ 2) AS signed",
        "RETURN 0.0 / 0.0 = 1 AS integer_eq, 0.0 / 0.0 <> 1 AS integer_ne,\
                0.0 / 0.0 = 'a' AS string_eq, 0.0 / 0.0 <> 'a' AS string_ne",
        "WITH {name: 'Mats', Name: 'Pontus'} AS value, 'Name' AS key RETURN value[key] AS present",
        "WITH {name: 'Mats', Name: 'Pontus'} AS value RETURN value['nAMe'] AS absent",
        "WITH {name: 'Mats'} AS value, null AS missing RETURN value[missing] AS null_key",
    ] {
        let cpu_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            true,
        )?;
        let metal_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )?;
        assert_eq!(metal_result.schema, cpu_result.schema);
        assert_eq!(metal_result.batches, cpu_result.batches);
    }

    for source in [
        "WITH {name: 'Mats'} AS value RETURN value[1]",
        "WITH 100 AS value RETURN value[0]",
    ] {
        let cpu_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            source,
            true,
        )
        .err()
        .ok_or_else(|| {
            irongraph::Error::internal(
                "CPU scalar reference did not preserve the dynamic-index type error",
            )
        })?;
        let metal_error = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            source,
            true,
        )
        .err()
        .ok_or_else(|| {
            irongraph::Error::internal(
                "Metal scalar program did not produce the dynamic-index type error",
            )
        })?;
        assert_eq!(metal_error.code, cpu_error.code);
        assert_eq!(metal_error.message, cpu_error.message);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_boolean_scalar_program_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    let project = ProjectId::random();
    let bookmark = Bookmark { term: 1, index: 0 };
    let direct = metal.execute_boolean_program(
        &irongraph::gpu::ResidentBooleanProgramRequest {
            row_count: 3,
            input_count: 1,
            inputs: vec![
                irongraph::gpu::ResidentBooleanValue::Boolean(true),
                irongraph::gpu::ResidentBooleanValue::Boolean(false),
                irongraph::gpu::ResidentBooleanValue::Null,
            ],
            list_input_count: 0,
            list_inputs: Vec::new(),
            instructions: Vec::new(),
            scalar_cells: Vec::new(),
            map_entries: Vec::new(),
            scalar_list_entries: Vec::new(),
            scalar_predicates: Vec::new(),
            scalar_expressions: Vec::new(),
            reductions: Vec::new(),
            output_registers: vec![0],
            filter_registers: Vec::new(),
        },
        &CancellationToken::new(),
    )?;
    assert_eq!(
        direct.rows,
        vec![
            vec![irongraph::gpu::ResidentBooleanValue::Boolean(true)],
            vec![irongraph::gpu::ResidentBooleanValue::Boolean(false)],
            vec![irongraph::gpu::ResidentBooleanValue::Null],
        ]
    );
    let membership = irongraph::gpu::ResidentBooleanProgramRequest {
        row_count: 3,
        input_count: 1,
        inputs: vec![
            irongraph::gpu::ResidentBooleanValue::Boolean(true),
            irongraph::gpu::ResidentBooleanValue::Boolean(false),
            irongraph::gpu::ResidentBooleanValue::Null,
        ],
        list_input_count: 1,
        list_inputs: vec![
            irongraph::gpu::ResidentBooleanListInput {
                values: vec![
                    irongraph::gpu::ResidentBooleanValue::Boolean(true),
                    irongraph::gpu::ResidentBooleanValue::Boolean(false),
                ],
            },
            irongraph::gpu::ResidentBooleanListInput {
                values: vec![irongraph::gpu::ResidentBooleanValue::Boolean(false)],
            },
            irongraph::gpu::ResidentBooleanListInput {
                values: vec![irongraph::gpu::ResidentBooleanValue::Null],
            },
        ],
        instructions: vec![irongraph::gpu::ResidentBooleanInstruction {
            opcode: irongraph::gpu::ResidentBooleanOpcode::InputMembership,
            left: 0,
            right: 0,
        }],
        scalar_cells: Vec::new(),
        map_entries: Vec::new(),
        scalar_list_entries: Vec::new(),
        scalar_predicates: Vec::new(),
        scalar_expressions: Vec::new(),
        reductions: Vec::new(),
        output_registers: vec![1],
        filter_registers: Vec::new(),
    };
    let cpu_membership = cpu.execute_boolean_program(&membership, &CancellationToken::new())?;
    let metal_membership = metal.execute_boolean_program(&membership, &CancellationToken::new())?;
    assert_eq!(metal_membership, cpu_membership);
    assert_eq!(
        metal_membership.rows,
        vec![
            vec![irongraph::gpu::ResidentBooleanValue::Boolean(true)],
            vec![irongraph::gpu::ResidentBooleanValue::Boolean(true)],
            vec![irongraph::gpu::ResidentBooleanValue::Null],
        ]
    );
    let aggregate = irongraph::gpu::ResidentBooleanAggregateProgramRequest {
        row_program: irongraph::gpu::ResidentBooleanProgramRequest {
            row_count: 3,
            input_count: 1,
            inputs: vec![
                irongraph::gpu::ResidentBooleanValue::Boolean(true),
                irongraph::gpu::ResidentBooleanValue::Null,
                irongraph::gpu::ResidentBooleanValue::Boolean(false),
            ],
            list_input_count: 0,
            list_inputs: Vec::new(),
            instructions: Vec::new(),
            scalar_cells: Vec::new(),
            map_entries: Vec::new(),
            scalar_list_entries: Vec::new(),
            scalar_predicates: Vec::new(),
            scalar_expressions: Vec::new(),
            reductions: Vec::new(),
            output_registers: vec![0],
            filter_registers: Vec::new(),
        },
        reductions: vec![
            irongraph::gpu::ResidentBooleanAggregateReduction {
                kind: irongraph::gpu::ResidentBooleanAggregateKind::All,
                source_output: 0,
            },
            irongraph::gpu::ResidentBooleanAggregateReduction {
                kind: irongraph::gpu::ResidentBooleanAggregateKind::Any,
                source_output: 0,
            },
            irongraph::gpu::ResidentBooleanAggregateReduction {
                kind: irongraph::gpu::ResidentBooleanAggregateKind::None,
                source_output: 0,
            },
            irongraph::gpu::ResidentBooleanAggregateReduction {
                kind: irongraph::gpu::ResidentBooleanAggregateKind::Single,
                source_output: 0,
            },
        ],
        instructions: Vec::new(),
        output_registers: vec![0, 1, 2, 3],
    };
    let cpu_aggregate =
        cpu.execute_boolean_aggregate_program(&aggregate, &CancellationToken::new())?;
    let metal_aggregate =
        metal.execute_boolean_aggregate_program(&aggregate, &CancellationToken::new())?;
    assert_eq!(metal_aggregate, cpu_aggregate);
    assert_eq!(
        metal_aggregate.rows,
        vec![vec![
            irongraph::gpu::ResidentBooleanValue::Boolean(false),
            irongraph::gpu::ResidentBooleanValue::Boolean(true),
            irongraph::gpu::ResidentBooleanValue::Boolean(false),
            irongraph::gpu::ResidentBooleanValue::Boolean(true),
        ]]
    );
    let arithmetic = irongraph::gpu::ResidentBooleanProgramRequest {
        row_count: 1,
        input_count: 0,
        inputs: Vec::new(),
        list_input_count: 0,
        list_inputs: Vec::new(),
        instructions: vec![irongraph::gpu::ResidentBooleanInstruction {
            opcode: irongraph::gpu::ResidentBooleanOpcode::ScalarPredicate,
            left: 0,
            right: 0,
        }],
        scalar_cells: vec![
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Integer,
                payload: 5,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Integer,
                payload: 3,
                auxiliary: 0,
            },
            irongraph::gpu::ResidentScalarCell {
                tag: irongraph::gpu::ResidentScalarCellTag::Integer,
                payload: 8,
                auxiliary: 0,
            },
        ],
        map_entries: Vec::new(),
        scalar_list_entries: Vec::new(),
        scalar_predicates: vec![irongraph::gpu::ResidentScalarPredicate {
            kind: irongraph::gpu::ResidentScalarPredicateKind::ExpressionCompare,
            left: 0,
            right: 1,
            comparison: CompareOp::Eq,
            property_key: None,
        }],
        scalar_expressions: vec![
            irongraph::gpu::ResidentScalarExpression {
                instructions: vec![
                    irongraph::gpu::ResidentScalarExpressionInstruction {
                        opcode: irongraph::gpu::ResidentScalarExpressionOpcode::Cell,
                        left: 0,
                        right: 0,
                    },
                    irongraph::gpu::ResidentScalarExpressionInstruction {
                        opcode: irongraph::gpu::ResidentScalarExpressionOpcode::Cell,
                        left: 1,
                        right: 0,
                    },
                    irongraph::gpu::ResidentScalarExpressionInstruction {
                        opcode: irongraph::gpu::ResidentScalarExpressionOpcode::Add,
                        left: 0,
                        right: 1,
                    },
                ],
                output: 2,
            },
            irongraph::gpu::ResidentScalarExpression {
                instructions: vec![irongraph::gpu::ResidentScalarExpressionInstruction {
                    opcode: irongraph::gpu::ResidentScalarExpressionOpcode::Cell,
                    left: 2,
                    right: 0,
                }],
                output: 0,
            },
        ],
        reductions: Vec::new(),
        output_registers: vec![0],
        filter_registers: Vec::new(),
    };
    let cpu_arithmetic = cpu.execute_boolean_program(&arithmetic, &CancellationToken::new())?;
    let metal_arithmetic = metal.execute_boolean_program(&arithmetic, &CancellationToken::new())?;
    assert_eq!(metal_arithmetic, cpu_arithmetic);
    assert_eq!(
        metal_arithmetic.rows,
        vec![vec![irongraph::gpu::ResidentBooleanValue::Boolean(true)]]
    );
    let filtered_count = irongraph::gpu::ResidentBooleanProgramRequest {
        row_count: 1,
        input_count: 0,
        inputs: Vec::new(),
        list_input_count: 0,
        list_inputs: Vec::new(),
        instructions: vec![
            irongraph::gpu::ResidentBooleanInstruction {
                opcode: irongraph::gpu::ResidentBooleanOpcode::Constant,
                left: 2,
                right: 0,
            },
            irongraph::gpu::ResidentBooleanInstruction {
                opcode: irongraph::gpu::ResidentBooleanOpcode::Constant,
                left: 1,
                right: 0,
            },
            irongraph::gpu::ResidentBooleanInstruction {
                opcode: irongraph::gpu::ResidentBooleanOpcode::Constant,
                left: 0,
                right: 0,
            },
            irongraph::gpu::ResidentBooleanInstruction {
                opcode: irongraph::gpu::ResidentBooleanOpcode::ScalarPredicate,
                left: 0,
                right: 0,
            },
        ],
        scalar_cells: vec![irongraph::gpu::ResidentScalarCell {
            tag: irongraph::gpu::ResidentScalarCellTag::Integer,
            payload: 1,
            auxiliary: 0,
        }],
        map_entries: Vec::new(),
        scalar_list_entries: Vec::new(),
        scalar_predicates: vec![irongraph::gpu::ResidentScalarPredicate {
            kind: irongraph::gpu::ResidentScalarPredicateKind::FilteredListCountCompare,
            left: 0,
            right: 0,
            comparison: CompareOp::Eq,
            property_key: None,
        }],
        scalar_expressions: Vec::new(),
        reductions: vec![irongraph::gpu::ResidentBooleanReduction {
            kind: irongraph::gpu::ResidentBooleanReductionKind::Count,
            registers: vec![0, 1, 2],
        }],
        output_registers: vec![3],
        filter_registers: Vec::new(),
    };
    let cpu_filtered_count =
        cpu.execute_boolean_program(&filtered_count, &CancellationToken::new())?;
    let metal_filtered_count =
        metal.execute_boolean_program(&filtered_count, &CancellationToken::new())?;
    assert_eq!(metal_filtered_count, cpu_filtered_count);
    assert_eq!(
        metal_filtered_count.rows,
        vec![vec![irongraph::gpu::ResidentBooleanValue::Boolean(true)]]
    );
    for operation in ["AND", "OR", "XOR"] {
        let source = format!(
            "UNWIND [true, false, null] AS a \
             UNWIND [true, false, null] AS b \
             WITH a, b WHERE a IS NULL OR b IS NULL \
             RETURN a, b, (a {operation} b) IS NULL = \
                    (b {operation} a) IS NULL AS result"
        );
        let cpu_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&cpu),
            &source,
            false,
        )?;
        let metal_result = execute_resident_query_with_native_requirement(
            &graph,
            project,
            bookmark,
            Some(&metal),
            &source,
            true,
        )?;
        assert_eq!(metal_result.schema, cpu_result.schema, "{operation}");
        assert_eq!(metal_result.batches, cpu_result.batches, "{operation}");
        assert_eq!(
            metal_result
                .batches
                .iter()
                .map(|batch| batch.row_count)
                .sum::<usize>(),
            5,
            "{operation}"
        );
        assert!(
            metal_result
                .batches
                .iter()
                .flat_map(|batch| &batch.columns[2].values)
                .all(|value| value == &ResultValue::Scalar(ScalarValue::Boolean(true))),
            "{operation}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_fixed_two_hop_pipeline_is_native_and_matches_cpu_semantics() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let source = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
                  RETURN a.name AS start, b.name AS middle, c.name AS finish";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::String(Arc::from("Ada")))]
    );
    assert_eq!(
        metal_result.batches[0].columns[1].values,
        vec![ResultValue::Scalar(ScalarValue::String(Arc::from("Grace")))]
    );
    assert_eq!(
        metal_result.batches[0].columns[2].values,
        vec![ResultValue::Scalar(ScalarValue::String(Arc::from("Linus")))]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_count_distinct_relationships_is_native_and_matches_cpu_semantics() -> irongraph::Result<()>
{
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let source = "MATCH ()-[r]-() RETURN count(DISTINCT r) AS relationships";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Integer(2))]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_correlated_optional_after_two_hops_is_native_and_matches_cpu_semantics()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let source = "MATCH (a:Person)-[:KNOWS]->(b:Person)-[:KNOWS]->(c:Person) \
                  OPTIONAL MATCH (a)-[r:KNOWS]->(c) \
                  WITH c WHERE r IS NULL \
                  RETURN c.name AS finish";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        false,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);
    assert_eq!(metal_result.batches, cpu_result.batches);
    assert_eq!(
        metal_result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::String(Arc::from("Linus")))]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_correlated_optional_one_hop_matches_cpu_for_hit_miss_and_mixed_rows()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = optional_social_graph()?;
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;

    let cases = [
        (
            "USE LAYER OBSERVED\n\
             MATCH (p:Person {uid: 1001})\n\
             OPTIONAL MATCH (p)-[relationship:AUTHORED]->(post:Post)\n\
             RETURN p, relationship, post",
            2,
            0,
        ),
        (
            "USE LAYER OBSERVED\n\
             MATCH (p:Person {uid: 1003})\n\
             OPTIONAL MATCH (p)-[relationship:AUTHORED]->(post:Post)\n\
             RETURN p, relationship, post",
            1,
            1,
        ),
        (
            "USE LAYER OBSERVED\n\
             MATCH (p:Person)\n\
             OPTIONAL MATCH (p)-[relationship:AUTHORED]->(post:Post)\n\
             RETURN p, relationship, post",
            4,
            2,
        ),
        (
            "USE LAYER OBSERVED\n\
             MATCH (p:Person)\n\
             OPTIONAL MATCH (p)-[relationship:AUTHORED]->(post:Post {rank: 1})\n\
             RETURN p, relationship, post",
            3,
            2,
        ),
        (
            "USE LAYER OBSERVED\n\
             MATCH (post:Post {rank: 1})\n\
             OPTIONAL MATCH (post)<-[relationship:AUTHORED]-(p:Person)\n\
             RETURN post, relationship, p LIMIT 1",
            1,
            0,
        ),
    ];
    for (source, expected_rows, expected_null_relationships) in cases {
        let reference = execute_resident_query(&graph, project, bookmark, None, source)?;
        let cpu_result = execute_resident_query(&graph, project, bookmark, Some(&cpu), source)?;
        let metal_result = execute_resident_query(&graph, project, bookmark, Some(&metal), source)?;
        assert_eq!(cpu_result.schema, reference.schema, "query: {source}");
        assert_eq!(cpu_result.batches, reference.batches, "query: {source}");
        assert_eq!(metal_result.schema, cpu_result.schema, "query: {source}");
        assert_eq!(metal_result.batches, cpu_result.batches, "query: {source}");
        let row_count = metal_result
            .batches
            .iter()
            .map(|batch| batch.row_count)
            .sum::<usize>();
        assert_eq!(row_count, expected_rows, "query: {source}");
        let null_relationships = metal_result
            .batches
            .iter()
            .flat_map(|batch| &batch.columns[1].values)
            .filter(|value| **value == ResultValue::Scalar(ScalarValue::Null))
            .count();
        assert_eq!(
            null_relationships, expected_null_relationships,
            "query: {source}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_optional_undirected_expansion_matches_cpu_and_emits_each_self_loop_once()
-> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let node_label = graph.catalog_mut().intern_label("OptionalUndirected")?;
    let relationship_type = graph
        .catalog_mut()
        .intern_relationship_type("OPTIONAL_UNDIRECTED")?;
    let identifier = graph.catalog_mut().intern_property("identifier")?;
    for id in 1_u64..=3 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![node_label],
            properties: vec![(identifier, ScalarValue::Integer(id as i64))],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(11),
        source: NodeId(2),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 11,
        properties: Vec::new(),
    })?;

    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let source = "MATCH (a:OptionalUndirected) \
                  OPTIONAL MATCH (a)-[:OPTIONAL_UNDIRECTED]-(b:OptionalUndirected) \
                  RETURN a.identifier AS a, b.identifier AS b";
    let cpu_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&cpu),
        source,
        true,
    )?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        source,
        true,
    )?;
    assert_eq!(metal_result.schema, cpu_result.schema);

    let rows = |result: &irongraph::cypher::QueryResult| {
        let mut rows = result
            .batches
            .iter()
            .flat_map(|batch| {
                (0..batch.row_count).map(|row| {
                    let a = match &batch.columns[0].values[row] {
                        ResultValue::Scalar(ScalarValue::Integer(value)) => *value,
                        value => panic!("unexpected OPTIONAL source value: {value:?}"),
                    };
                    let b = match &batch.columns[1].values[row] {
                        ResultValue::Scalar(ScalarValue::Integer(value)) => Some(*value),
                        ResultValue::Scalar(ScalarValue::Null) => None,
                        value => panic!("unexpected OPTIONAL endpoint value: {value:?}"),
                    };
                    (a, b)
                })
            })
            .collect::<Vec<_>>();
        rows.sort();
        rows
    };
    let expected = vec![(1, Some(2)), (2, Some(1)), (2, Some(2)), (3, None)];
    assert_eq!(rows(&cpu_result), expected);
    assert_eq!(rows(&metal_result), expected);
    Ok(())
}

#[test]
fn query_engine_fuses_integer_grouping_into_the_resident_pipeline() -> irongraph::Result<()> {
    let (graph, _, _) = integer_operator_graph()?;
    let mut backend = CpuBackend::new(32 * 1024 * 1024, 1024);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 100,
        max_batch_rows: 2,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: Some(&backend),
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "MATCH (r:Row) RETURN r.key AS key, sum(r.value) AS total ORDER BY key ASC",
        &mut context,
    )?;
    let rows = output
        .result
        .batches
        .iter()
        .flat_map(|batch| {
            (0..batch.row_count).map(|row| {
                (
                    batch.columns[0].values[row].clone(),
                    batch.columns[1].values[row].clone(),
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(
        rows,
        vec![
            (
                ResultValue::Scalar(ScalarValue::Integer(1)),
                ResultValue::Scalar(ScalarValue::Integer(5)),
            ),
            (
                ResultValue::Scalar(ScalarValue::Integer(2)),
                ResultValue::Scalar(ScalarValue::Integer(7)),
            ),
            (
                ResultValue::Scalar(ScalarValue::Null),
                ResultValue::Scalar(ScalarValue::Integer(7)),
            ),
        ]
    );
    Ok(())
}

#[test]
fn query_engine_uses_online_scalar_index_without_changing_results() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label missing"))?;
    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| irongraph::Error::internal("name property missing"))?;
    let mut indexes = IndexCatalog::default();
    indexes.create(
        &graph,
        GraphIndexDefinition {
            name: "person_name".to_owned(),
            kind: GraphIndexKind::Equality,
            label: person,
            properties: vec![name],
            unique: false,
        },
    )?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: Some(&indexes),
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 1,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "MATCH (p:Person {name: 'Ada'}) RETURN p.age AS age",
        &mut context,
    )?;
    assert_eq!(
        output.result.batches[0].columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Integer(37))]
    );
    Ok(())
}

#[test]
fn query_engine_sparse_overlay_preserves_multistatement_read_own_writes() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let mut first = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let created = QueryEngine.execute(
        "CREATE (n:TransactionOnly {name: 'new'}) RETURN n",
        &mut first,
    )?;
    assert!(!created.graph_mutations.is_empty());
    assert_eq!(graph.node_count(), 3);

    let mut catalog = graph.catalog().clone();
    for mutation in &created.graph_mutations {
        match mutation {
            GraphMutation::DeclareLabel { name, id } => {
                catalog.declare_label(name.clone(), *id)?;
            }
            GraphMutation::DeclareProperty { name, id } => {
                catalog.declare_property(name.clone(), *id)?;
            }
            GraphMutation::DeclareRelationshipType { name, id } => {
                catalog.declare_relationship_type(name.clone(), *id)?;
            }
            GraphMutation::InsertNode(_)
            | GraphMutation::InsertEdge(_)
            | GraphMutation::SetNodeProperty { .. }
            | GraphMutation::AddNodeLabels { .. }
            | GraphMutation::RemoveNodeLabels { .. }
            | GraphMutation::SetEdgeProperty { .. }
            | GraphMutation::DeleteNode { .. }
            | GraphMutation::DeleteEdge { .. } => {}
        }
    }
    let mut second = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: &catalog,
        prior_graph_mutations: &created.graph_mutations,
        temporal: None,
        prior_temporal_mutations: &created.temporal_mutations,
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 0,
        next_node_id: 101,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let read = QueryEngine.execute(
        "MATCH (n:TransactionOnly) RETURN n.name AS name",
        &mut second,
    )?;
    let value = read
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first());
    assert_eq!(
        value,
        Some(&ResultValue::Scalar(ScalarValue::String(Arc::from("new"))))
    );
    assert!(read.dependencies.entities.is_empty());
    assert_eq!(graph.node_count(), 3);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_match_reads_statement_local_gpu_overlay_before_creating_edges() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = GraphStore::default();
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(128 * 1024 * 1024, 1024 * 1024);
    let mut metal = irongraph::gpu::MetalBackend::with_governor(0, governor)?;
    metal.admit_graph(Arc::new(graph.snapshot()?))?;
    let output = execute_resident_query(
        &graph,
        ProjectId(uuid::Uuid::nil()),
        Bookmark { term: 1, index: 0 },
        Some(&metal),
        "UNWIND range(0, 2) AS slot
CREATE (n:Person {slot: slot})
WITH collect(n) AS people
UNWIND range(0, 1) AS edge
MATCH (source:Person {slot: edge}), (target:Person {slot: edge + 1})
CREATE (source)-[:KNOWS]->(target)
RETURN size(people) AS people_created",
    )?;
    assert_eq!(
        output
            .batches
            .first()
            .and_then(|batch| batch.columns.first())
            .and_then(|column| column.values.first()),
        Some(&ResultValue::Scalar(ScalarValue::Integer(3)))
    );
    Ok(())
}

#[test]
fn at_time_projects_declared_scalar_history_without_resurrecting_topology() -> irongraph::Result<()>
{
    let graph = sample_graph()?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property missing"))?;
    let mut temporal = TemporalStore::default();
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label missing"))?;
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: person.0,
            property: age,
            value_type: TemporalType::Integer,
            retention_nanos: 10_000,
        },
        300,
    )?;
    for (time, index, value) in [(100, 4, 20), (200, 5, 37)] {
        temporal.append(
            EntityKind::Node,
            person.0,
            TemporalSample {
                entity_id: 1,
                property: age,
                event_time_nanos: time,
                sequence_index: index,
                value: ScalarValue::Integer(value),
            },
            300,
        )?;
    }
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: Some(&temporal),
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::from([(
            "when".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(150)),
        )]),
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 300,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 100,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "AT TIME $when MATCH (p:Person {name: 'Ada'}) RETURN p.age AS age",
        &mut context,
    )?;
    let value = output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first());
    assert_eq!(value, Some(&ResultValue::Scalar(ScalarValue::Integer(20))));
    Ok(())
}

#[test]
fn query_engine_prepares_resolved_writes_without_mutating_authority() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let mut context = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 5 },
        mutation_revision: 6,
        resolved_time_nanos: 500,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 100,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let output = QueryEngine.execute(
        "CREATE (p:Person {name: 'Margaret', age: 32}) RETURN p",
        &mut context,
    )?;
    assert_eq!(graph.node_count(), 3);
    assert!(
        output
            .graph_mutations
            .iter()
            .any(|mutation| matches!(mutation, irongraph::graph::GraphMutation::InsertNode(_)))
    );
    assert_eq!(output.result.statistics.nodes_created, 1);
    Ok(())
}

#[test]
fn cpu_backend_uses_complete_snapshot_and_honors_cancellation() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let mut backend = CpuBackend::new(32 * 1024 * 1024, 1024);
    backend.admit_graph(snapshot)?;
    let token = CancellationToken::new();
    let selected = backend.filter_i64(
        &[1, 4, 7, 2],
        &[true, true, false, true],
        CompareOp::Greater,
        2,
        &token,
    )?;
    assert_eq!(selected, vec![1]);
    let distances = backend.exact_l2(&[0.0, 0.0, 1.0, 1.0], 2, 2, &[0.0, 1.0], &token)?;
    assert_eq!(distances.distances, vec![1.0, 1.0]);
    token.cancel();
    assert!(backend.expand_out(&[0], &token).is_err());
    Ok(())
}

fn assert_integer_operator_semantics(
    backend: &dyn ExecutionBackend,
    project: ProjectId,
    key: PropertyId,
    value: PropertyId,
) -> irongraph::Result<()> {
    let cancellation = CancellationToken::new();
    let rows = vec![0, 1, 2, 3, 4, 2];
    assert_eq!(
        backend
            .sort_rows(
                &resident_property_sort(project, rows.clone(), key, false, false, None),
                &cancellation,
            )?
            .positions,
        vec![1, 4, 0, 2, 5, 3]
    );
    assert_eq!(
        backend
            .sort_rows(
                &resident_property_sort(project, rows, key, true, true, Some(4)),
                &cancellation,
            )?
            .positions,
        vec![3, 0, 2, 5]
    );

    let join = ResidentJoinRequest {
        project,
        left_rows: vec![0, 1, 3, 2],
        right_rows: vec![4, 2, 0, 3],
        left_property: key,
        right_property: key,
        max_pairs: 5,
    };
    assert_eq!(
        backend.join_node_i64(&join, &cancellation)?,
        vec![
            ResidentJoinPair {
                left_position: 0,
                right_position: 1,
            },
            ResidentJoinPair {
                left_position: 0,
                right_position: 2,
            },
            ResidentJoinPair {
                left_position: 1,
                right_position: 0,
            },
            ResidentJoinPair {
                left_position: 3,
                right_position: 1,
            },
            ResidentJoinPair {
                left_position: 3,
                right_position: 2,
            },
        ]
    );
    assert!(
        backend
            .join_node_i64(
                &ResidentJoinRequest {
                    max_pairs: 4,
                    ..join
                },
                &cancellation,
            )
            .is_err()
    );

    let expected = [
        (
            ResidentAggregate::CountAll,
            vec![
                (None, 1, Some(ScalarValue::Integer(1))),
                (Some(1), 2, Some(ScalarValue::Integer(2))),
                (Some(2), 2, Some(ScalarValue::Integer(2))),
            ],
        ),
        (
            ResidentAggregate::CountValue,
            vec![
                (None, 1, Some(ScalarValue::Integer(1))),
                (Some(1), 2, Some(ScalarValue::Integer(1))),
                (Some(2), 2, Some(ScalarValue::Integer(2))),
            ],
        ),
        (
            ResidentAggregate::Sum,
            vec![
                (None, 1, Some(ScalarValue::Integer(7))),
                (Some(1), 2, Some(ScalarValue::Integer(5))),
                (Some(2), 2, Some(ScalarValue::Integer(7))),
            ],
        ),
        (
            ResidentAggregate::Average,
            vec![
                (None, 1, Some(ScalarValue::Float(OrderedFloat(7.0)))),
                (Some(1), 2, Some(ScalarValue::Float(OrderedFloat(5.0)))),
                (Some(2), 2, Some(ScalarValue::Float(OrderedFloat(3.5)))),
            ],
        ),
        (
            ResidentAggregate::Minimum,
            vec![
                (None, 1, Some(ScalarValue::Integer(7))),
                (Some(1), 2, Some(ScalarValue::Integer(5))),
                (Some(2), 2, Some(ScalarValue::Integer(-3))),
            ],
        ),
        (
            ResidentAggregate::Maximum,
            vec![
                (None, 1, Some(ScalarValue::Integer(7))),
                (Some(1), 2, Some(ScalarValue::Integer(5))),
                (Some(2), 2, Some(ScalarValue::Integer(10))),
            ],
        ),
    ];
    for (aggregate, expected) in expected {
        let actual = backend.group_node_i64(
            &ResidentGroupRequest {
                project,
                rows: vec![0, 1, 2, 3, 4],
                key_property: key,
                value_property: value,
                aggregate,
            },
            &cancellation,
        )?;
        assert_eq!(
            actual
                .into_iter()
                .map(|group| (group.key, group.rows, group.value))
                .collect::<Vec<_>>(),
            expected
        );
    }
    Ok(())
}

fn resident_property_sort(
    project: ProjectId,
    rows: Vec<u32>,
    property: PropertyId,
    descending: bool,
    nulls_first: bool,
    limit: Option<usize>,
) -> ResidentSortRequest {
    ResidentSortRequest {
        project,
        row_count: rows.len(),
        keys: vec![ResidentSortKey {
            source: ResidentSortSource::NodeProperty { rows, property },
            descending,
            nulls_first,
        }],
        limit,
    }
}

#[test]
fn cpu_resident_integer_sort_top_k_join_and_group_are_exact() -> irongraph::Result<()> {
    let (graph, key, value) = integer_operator_graph()?;
    let project = ProjectId(uuid::Uuid::nil());
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 1024);
    cpu.admit_graph(Arc::new(graph.snapshot()?))?;
    assert_integer_operator_semantics(&cpu, project, key, value)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_resident_integer_sort_top_k_join_and_group_match_cpu() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let (graph, key, value) = integer_operator_graph()?;
    let project = ProjectId(uuid::Uuid::nil());
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(128 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 128 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    assert_integer_operator_semantics(&cpu, project, key, value)?;
    assert_integer_operator_semantics(&metal, project, key, value)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_value_matrix_returns_device_selected_pair_indices() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(3),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 3,
        properties: Vec::new(),
    })?;
    let project = ProjectId(uuid::Uuid::nil());
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let request = ResidentNodePipelineRequest {
        project,
        labels: Vec::new(),
        layers: LayerMask::AUTHORITY,
        initial_optional: false,
        expansion: Some(ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![relationship_type],
            end_labels: Vec::new(),
            end_equals_start: false,
            optional: false,
            end_predicates: Vec::new(),
        }),
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: Vec::new(),
        property_filters: Vec::new(),
        value_matrix: Some(ResidentValueMatrixProgram {
            values: vec![
                ResidentValueMatrixValue::Node,
                ResidentValueMatrixValue::Relationship,
                ResidentValueMatrixValue::Path,
                ResidentValueMatrixValue::String(String::new()),
                ResidentValueMatrixValue::Integer(1),
                ResidentValueMatrixValue::Float(3.14_f64.to_bits()),
                ResidentValueMatrixValue::Boolean(true),
                ResidentValueMatrixValue::Null,
                ResidentValueMatrixValue::List(Vec::new()),
                ResidentValueMatrixValue::Map,
            ],
            comparison: CompareOp::Less,
            exclude_same_index: true,
        }),
        mutation: None,
        orders: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        integer_projections: Vec::new(),
        property_null_projections: Vec::new(),
        max_output_rows: 16,
    };
    let cancellation = CancellationToken::new();
    let cpu_result = cpu.execute_node_pipeline(&request, &cancellation)?;
    let metal_result = metal.execute_node_pipeline(&request, &cancellation)?;
    assert_eq!(metal_result, cpu_result);
    assert_eq!(metal_result.value_left_indices, vec![4]);
    assert_eq!(metal_result.value_right_indices, vec![5]);
    assert_eq!(metal_result.start_rows, vec![0]);
    assert_eq!(metal_result.edge_rows, vec![0]);
    assert_eq!(metal_result.end_rows, vec![1]);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_integer_sort_and_join_cover_full_i64_domain() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("IntegerDomain")?;
    let key = graph.catalog_mut().intern_property("domain_key")?;
    for (position, value) in [
        Some(i64::MAX),
        Some(i64::MIN),
        Some(0),
        Some(i64::MIN),
        None,
        Some(i64::MAX),
    ]
    .into_iter()
    .enumerate()
    {
        graph.insert_node(NodeInput {
            id: NodeId(position as u64 + 1),
            layer: Layer::Observed,
            revision: position as u64 + 1,
            labels: vec![label],
            properties: value
                .map(|value| vec![(key, ScalarValue::Integer(value))])
                .unwrap_or_default(),
        })?;
    }
    let project = ProjectId(uuid::Uuid::nil());
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 32 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let cancellation = CancellationToken::new();
    for (descending, nulls_first) in [(false, false), (true, true)] {
        let request = resident_property_sort(
            project,
            vec![0, 1, 2, 3, 4, 5, 1, 0],
            key,
            descending,
            nulls_first,
            None,
        );
        assert_eq!(
            metal.sort_rows(&request, &cancellation)?,
            cpu.sort_rows(&request, &cancellation)?
        );
    }
    let join = ResidentJoinRequest {
        project,
        left_rows: vec![0, 1, 2, 3, 4, 5],
        right_rows: vec![5, 4, 3, 2, 1, 0],
        left_property: key,
        right_property: key,
        max_pairs: 10,
    };
    assert_eq!(
        metal.join_node_i64(&join, &cancellation)?,
        cpu.join_node_i64(&join, &cancellation)?
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_large_integer_operators_are_linear_memory_and_match_cpu() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    const ROW_COUNT: usize = 4097;
    let (graph, key, value) = scalable_integer_operator_graph(ROW_COUNT)?;
    let project = ProjectId(uuid::Uuid::nil());
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let rows = (0..ROW_COUNT).map(|row| row as u32).collect::<Vec<_>>();
    let cancellation = CancellationToken::new();

    let sort = resident_property_sort(project, rows.clone(), key, false, false, None);
    assert_eq!(
        metal.sort_rows(&sort, &cancellation)?,
        cpu.sort_rows(&sort, &cancellation)?
    );

    let mut right_rows = rows.clone();
    right_rows.reverse();
    let join = ResidentJoinRequest {
        project,
        left_rows: rows.clone(),
        right_rows,
        left_property: key,
        right_property: key,
        max_pairs: 16_384,
    };
    assert_eq!(
        metal.join_node_i64(&join, &cancellation)?,
        cpu.join_node_i64(&join, &cancellation)?
    );

    let group = ResidentGroupRequest {
        project,
        rows,
        key_property: key,
        value_property: value,
        aggregate: ResidentAggregate::Sum,
    };
    assert_eq!(
        metal.group_node_i64(&group, &cancellation)?,
        cpu.group_node_i64(&group, &cancellation)?
    );

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    assert!(metal.group_node_i64(&group, &cancelled).is_err());
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_exact_operator_matches_cpu_reference_on_resident_graph() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 1024);
    let mut metal = MetalBackend::new(0, 32 * 1024 * 1024, 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let matrix = [0.0_f32, 1.0, 2.0, 3.0, 1.0, 1.0];
    let query = [1.0_f32, 1.0];
    let cancellation = CancellationToken::new();
    let expected = cpu.exact_l2(&matrix, 3, 2, &query, &cancellation)?;
    let actual = metal.exact_l2(&matrix, 3, 2, &query, &cancellation)?;
    assert_eq!(actual, expected);
    let project = ProjectId(uuid::Uuid::nil());
    assert_eq!(
        metal.expand_project_in(project, &[1, 2], &cancellation)?,
        cpu.expand_project_in(project, &[1, 2], &cancellation)?
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_specializes_only_type_safe_boolean_filter_identities() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let graph = sample_graph()?;
    let project = ProjectId::random();
    let bookmark = Bookmark {
        term: 1,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        project,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;

    // `name` is a resident STRING column, so equality against a STRING literal is known to
    // return BOOLEAN-or-NULL. The three-valued `AND false` identity is therefore sound and the
    // strict Metal execution can remain a native scan without materializing `name` on the host.
    let safe = "MATCH (n) WHERE NOT(n.name = 'nobody' AND false) RETURN n";
    let cpu_result = execute_resident_query(&graph, project, bookmark, Some(&cpu), safe)?;
    let metal_result = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        safe,
        true,
    )?;
    assert_eq!(metal_result, cpu_result);

    // The same identity must not hide an incompatible INTEGER-vs-STRING equality. It remains
    // outside the native subset instead of returning an unsound successful result.
    let unsafe_query = "MATCH (n) WHERE NOT(n.age = 'nobody' AND false) RETURN n";
    let error = execute_resident_query_with_native_requirement(
        &graph,
        project,
        bookmark,
        Some(&metal),
        unsafe_query,
        true,
    )
    .expect_err("incompatible equality must not be folded into a native success");
    assert_eq!(error.code, irongraph::ErrorCode::GpuAdmissionFailure);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_shared_adjacency_first_run_extends_empty_reserved_offsets() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let identity = graph.catalog_mut().intern_label("Identity")?;
    let address = graph.catalog_mut().intern_property("address")?;
    let project = ProjectId(uuid::Uuid::nil());
    let initial = Arc::new(graph.snapshot()?);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    metal.admit_graph(initial)?;

    let pinned_empty = metal
        .shared_project_backing(project)
        .ok_or_else(|| irongraph::Error::internal("empty Metal graph is not shared"))?
        .graph;
    assert_eq!(pinned_empty.outgoing.offsets(), &[0]);
    assert_eq!(pinned_empty.incoming.offsets(), &[0]);
    // Do not retain the old generation across publication. The production failure required the
    // old resident owner to drop so Candle's pool could attempt to recycle the allocation; the
    // newly extended canonical view must itself keep the pool-tracked owner alive.
    drop(pinned_empty);

    for revision in 1_u64..=2 {
        graph.insert_node(NodeInput {
            id: NodeId(revision),
            layer: Layer::Knowledge,
            revision,
            labels: vec![identity],
            properties: vec![(
                address,
                ScalarValue::String(Arc::from(format!("identity-{revision}@example.test"))),
            )],
        })?;
        metal.apply_project_delta(ResidentProjectDelta {
            project,
            bookmark: Bookmark {
                term: 1,
                index: revision,
            },
            graph: graph.device_delta(revision)?,
            temporal: Vec::new(),
            vectors: Vec::new(),
            invalidate_derived: true,
        })?;

        let current = metal
            .shared_project_backing(project)
            .ok_or_else(|| irongraph::Error::internal("updated Metal graph is not shared"))?
            .graph;
        let expected = vec![0; revision as usize + 1];
        assert_eq!(current.outgoing.offsets(), expected);
        assert_eq!(current.incoming.offsets(), expected);
        assert_eq!(current.node_ids.len(), revision as usize);
    }

    // Source bootstrap may create endpoints and their first relationship in one publication, then
    // append an isolated Identity in the next. The topology delta must not detach the canonical
    // CSR offsets from the Metal tensor's reserved allocation before that second publication.
    let mut bootstrap = GraphStore::default();
    let contact = bootstrap.catalog_mut().intern_label("Contact")?;
    let sent = bootstrap.catalog_mut().intern_relationship_type("SENT")?;
    bootstrap.insert_node(NodeInput {
        id: NodeId(11),
        layer: Layer::Knowledge,
        revision: 1,
        labels: vec![contact],
        properties: Vec::new(),
    })?;
    bootstrap.insert_node(NodeInput {
        id: NodeId(12),
        layer: Layer::Knowledge,
        revision: 1,
        labels: vec![contact],
        properties: Vec::new(),
    })?;
    bootstrap.insert_edge(EdgeInput {
        id: EdgeId(21),
        source: NodeId(11),
        target: NodeId(12),
        relationship_type: sent,
        layer: Layer::Knowledge,
        revision: 1,
        properties: Vec::new(),
    })?;
    let bootstrap_project = ProjectId::random();
    let mut bootstrap_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    bootstrap_metal.admit_project(ResidentProjectImage::build(
        bootstrap_project,
        Bookmark { term: 1, index: 0 },
        &GraphStore::default(),
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?)?;
    let bootstrap_delta = bootstrap.device_delta(1)?;
    assert_eq!(bootstrap_delta.nodes.len(), 2);
    assert_eq!(bootstrap_delta.edges.len(), 1);
    assert!(!bootstrap_delta.outgoing.is_empty());
    assert!(!bootstrap_delta.incoming.is_empty());
    bootstrap_metal.apply_project_delta(ResidentProjectDelta {
        project: bootstrap_project,
        bookmark: Bookmark { term: 1, index: 1 },
        graph: bootstrap_delta,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;
    bootstrap.insert_node(NodeInput {
        id: NodeId(13),
        layer: Layer::Knowledge,
        revision: 2,
        labels: vec![contact],
        properties: Vec::new(),
    })?;
    bootstrap_metal.apply_project_delta(ResidentProjectDelta {
        project: bootstrap_project,
        bookmark: Bookmark { term: 1, index: 2 },
        graph: bootstrap.device_delta(2)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;

    // Reproduce the production shape: an edge-bearing delta advances canonical nodes through the
    // overlay while the cold CSR offset prefix legitimately stays shorter.
    // A later isolated-node append must fill the entire missing offset gap in-place, not assume
    // that the cold prefix already equals canonical node count + 1 and not rebuild the graph.
    let overlay_project = ProjectId::random();
    let mut overlay_graph = GraphStore::default();
    let overlay_label = overlay_graph.catalog_mut().intern_label("OverlayNode")?;
    let overlay_type = overlay_graph
        .catalog_mut()
        .intern_relationship_type("OVERLAY_EDGE")?;
    for id in 1_u64..=300 {
        overlay_graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Knowledge,
            revision: 1,
            labels: vec![overlay_label],
            properties: Vec::new(),
        })?;
    }
    let mut overlay_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    overlay_metal.admit_project(ResidentProjectImage::build(
        overlay_project,
        Bookmark { term: 1, index: 1 },
        &overlay_graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?)?;
    for id in 301_u64..=305 {
        overlay_graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Knowledge,
            revision: 2,
            labels: vec![overlay_label],
            properties: Vec::new(),
        })?;
    }
    overlay_graph.insert_edge(EdgeInput {
        id: EdgeId(1_000),
        source: NodeId(301),
        target: NodeId(305),
        relationship_type: overlay_type,
        layer: Layer::Knowledge,
        revision: 2,
        properties: Vec::new(),
    })?;
    overlay_metal.apply_project_delta(ResidentProjectDelta {
        project: overlay_project,
        bookmark: Bookmark { term: 1, index: 2 },
        graph: overlay_graph.device_delta(2)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;
    let overlay_generation = overlay_metal
        .shared_project_backing(overlay_project)
        .ok_or_else(|| irongraph::Error::internal("overlay Metal graph is not shared"))?
        .graph;
    assert_eq!(overlay_generation.node_ids.len(), 305);
    assert_eq!(overlay_generation.outgoing.offsets().len(), 306);
    drop(overlay_generation);

    for id in 306_u64..=308 {
        overlay_graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Workspace,
            revision: 3,
            labels: vec![overlay_label],
            properties: Vec::new(),
        })?;
    }
    overlay_metal.apply_project_delta(ResidentProjectDelta {
        project: overlay_project,
        bookmark: Bookmark { term: 1, index: 3 },
        graph: overlay_graph.device_delta(3)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;
    let completed = overlay_metal
        .shared_project_backing(overlay_project)
        .ok_or_else(|| irongraph::Error::internal("completed overlay graph is not shared"))?
        .graph;
    assert_eq!(completed.node_ids.len(), 308);
    assert_eq!(completed.outgoing.offsets().len(), 309);
    assert_eq!(completed.incoming.offsets().len(), 309);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_shared_fixed_column_rebuild_batches_keep_host_graph_coherent() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let record = graph.catalog_mut().intern_label("Record")?;
    let linked = graph.catalog_mut().intern_relationship_type("LINKED")?;
    let project = ProjectId::random();

    // Match the reported first publication: a populated node graph with no relationships.
    for id in 1_u64..=445 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![record],
            properties: Vec::new(),
        })?;
    }
    let mut metal = MetalBackend::new(0, 128 * 1024 * 1024, 1024 * 1024)?;
    metal.admit_project(ResidentProjectImage::build(
        project,
        Bookmark { term: 1, index: 1 },
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?)?;

    let mut next_node = 446_u64;
    let mut next_edge = 1_u64;
    for revision in 2_u64..=5 {
        // Sixty-four is the exact relationship batch shape in the crash log. It crosses the
        // resident fixed-column growth boundaries at 64, 128 and 192 rows.
        for offset in 0_u64..64 {
            let node = next_node + offset;
            graph.insert_node(NodeInput {
                id: NodeId(node),
                layer: Layer::Observed,
                revision,
                labels: vec![record],
                properties: Vec::new(),
            })?;
            graph.insert_edge(EdgeInput {
                id: EdgeId(next_edge + offset),
                source: NodeId(offset + 1),
                target: NodeId(node),
                relationship_type: linked,
                layer: Layer::Observed,
                revision,
                properties: Vec::new(),
            })?;
        }
        next_node += 64;
        next_edge += 64;
        metal.apply_project_delta(ResidentProjectDelta {
            project,
            bookmark: Bookmark {
                term: 1,
                index: revision,
            },
            graph: graph.device_delta(revision)?,
            temporal: Vec::new(),
            vectors: Vec::new(),
            invalidate_derived: true,
        })?;

        // This is the production step absent from the older adjacency test: the accelerator's
        // shared generation becomes the canonical graph immediately, and source maintenance may
        // traverse it before another device query introduces an implicit synchronization point.
        let published = metal
            .shared_project_backing(project)
            .ok_or_else(|| irongraph::Error::internal("published Metal graph is absent"))?
            .graph;
        graph.rebind_shared(published)?;
        let endpoints = graph
            .edges()
            .map(|edge| (edge.source(), edge.target(), edge.relationship_type()))
            .collect::<Vec<_>>();
        assert_eq!(endpoints.len(), (revision as usize - 1) * 64);
        assert!(endpoints.iter().all(|(_, _, kind)| *kind == linked));
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_first_relationship_property_append_is_immediately_queryable() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = GraphStore::default();
    let record = graph.catalog_mut().intern_label("Record")?;
    let linked = graph.catalog_mut().intern_relationship_type("LINKED")?;
    let kind = graph.catalog_mut().intern_property("relationship_kind")?;
    for id in 1_u64..=2 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![record],
            properties: Vec::new(),
        })?;
    }
    let project = ProjectId::random();
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    metal.admit_project(ResidentProjectImage::build(
        project,
        Bookmark { term: 1, index: 1 },
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?)?;

    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(2),
        target: NodeId(1),
        relationship_type: linked,
        layer: Layer::Observed,
        revision: 2,
        properties: vec![(kind, ScalarValue::String("officer_of".into()))],
    })?;
    let bookmark = Bookmark { term: 1, index: 2 };
    metal.apply_project_delta(ResidentProjectDelta {
        project,
        bookmark,
        graph: graph.device_delta(2)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    })?;

    let query = "MATCH ()-[r:LINKED]->() RETURN r.relationship_kind";
    let expected = execute_resident_query(&graph, project, bookmark, None, query)?;
    let actual = execute_resident_query(&graph, project, bookmark, Some(&metal), query)?;
    assert_eq!(actual, expected);
    graph.validate_structure()
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_incremental_publication_matches_cpu_and_failure_is_atomic() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = sample_graph()?;
    let project = ProjectId(uuid::Uuid::nil());
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property is absent"))?;
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let initial_resident_bytes = metal
        .resident_project_bytes(project)
        .ok_or_else(|| irongraph::Error::internal("Metal project is not resident"))?;

    graph.set_node_property(NodeId(1), age, ScalarValue::Integer(38), 6)?;
    let delta = ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 6 },
        graph: graph.device_delta(6)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    };
    cpu.apply_project_delta(delta.clone())?;
    metal.apply_project_delta(delta)?;
    let cancellation = CancellationToken::new();
    assert_eq!(
        metal.filter_node_i64(project, age, CompareOp::Eq, 38, &cancellation)?,
        cpu.filter_node_i64(project, age, CompareOp::Eq, 38, &cancellation)?
    );
    assert_eq!(
        metal.resident_project_bytes(project),
        Some(initial_resident_bytes),
        "same-shape replacement must not become permanent mutation staging"
    );

    for revision in 7_u64..15 {
        let value = 32_i64 + revision as i64;
        graph.set_node_property(NodeId(1), age, ScalarValue::Integer(value), revision)?;
        let delta = ResidentProjectDelta {
            project,
            bookmark: Bookmark {
                term: 1,
                index: revision,
            },
            graph: graph.device_delta(revision)?,
            temporal: Vec::new(),
            vectors: Vec::new(),
            invalidate_derived: true,
        };
        cpu.apply_project_delta(delta.clone())?;
        metal.apply_project_delta(delta)?;
        assert_eq!(
            metal.filter_node_i64(project, age, CompareOp::Eq, value, &cancellation)?,
            cpu.filter_node_i64(project, age, CompareOp::Eq, value, &cancellation)?
        );
        assert_eq!(
            metal.resident_project_bytes(project),
            Some(initial_resident_bytes),
            "resident bytes drifted after same-shape revision {revision}"
        );
    }

    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label is absent"))?;
    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| irongraph::Error::internal("name property is absent"))?;
    graph.insert_node(NodeInput {
        id: NodeId(4),
        layer: Layer::Observed,
        revision: 15,
        labels: vec![person],
        properties: vec![
            (name, ScalarValue::String(Arc::from("Appended"))),
            (age, ScalarValue::Integer(25)),
        ],
    })?;
    let append = ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 15 },
        graph: graph.device_delta(15)?,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: true,
    };
    cpu.apply_project_delta(append.clone())?;
    metal.apply_project_delta(append)?;
    assert_eq!(
        metal.filter_node_i64(project, age, CompareOp::Eq, 25, &cancellation)?,
        cpu.filter_node_i64(project, age, CompareOp::Eq, 25, &cancellation)?
    );

    graph.set_node_property(NodeId(1), age, ScalarValue::Integer(48), 16)?;
    let invalid = ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 16 },
        graph: graph.device_delta(16)?,
        temporal: Vec::new(),
        vectors: vec![ResolvedVectorMutation::Upsert {
            property: PropertyId(999),
            entity_id: 1,
            coordinates: vec![0],
            revision: 16,
        }],
        invalidate_derived: true,
    };
    assert!(metal.apply_project_delta(invalid).is_err());
    assert_eq!(
        metal.filter_node_i64(project, age, CompareOp::Eq, 46, &cancellation)?,
        vec![0]
    );
    assert_eq!(
        metal.resident_bookmark(project),
        Some(Bookmark { term: 1, index: 15 })
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_admits_temporal_vector_and_ann_state_and_matches_cpu_search() -> irongraph::Result<()> {
    let _metal_test = metal_test_guard();
    let mut graph = sample_graph()?;
    let project = ProjectId::random();
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| irongraph::Error::internal("Person label is absent"))?;
    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| irongraph::Error::internal("name property is absent"))?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property is absent"))?;
    let nullable_age = graph.catalog_mut().intern_property("nullable_age")?;
    let embedding = graph.catalog_mut().intern_property("embedding")?;
    let mut temporal = TemporalStore::default();
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: person.0,
            property: age,
            value_type: TemporalType::Integer,
            retention_nanos: 10_000,
        },
        1_000,
    )?;
    temporal.append(
        EntityKind::Node,
        person.0,
        TemporalSample {
            entity_id: 1,
            property: age,
            event_time_nanos: 100,
            sequence_index: 5,
            value: ScalarValue::Integer(37),
        },
        1_000,
    )?;
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: person.0,
            property: nullable_age,
            value_type: TemporalType::Integer,
            retention_nanos: 10_000,
        },
        1_000,
    )?;
    temporal.append(
        EntityKind::Node,
        person.0,
        TemporalSample {
            entity_id: 1,
            property: nullable_age,
            event_time_nanos: 75,
            sequence_index: 5,
            value: ScalarValue::Null,
        },
        1_000,
    )?;

    let profile = EmbeddingProfile::new(
        [7; 32],
        [9; 32],
        4,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )?;
    let rows = [
        (1, [0.8, 0.6, 0.0, 0.0]),
        (2, [0.0, 1.0, 0.0, 0.0]),
        (3, [0.0, 0.0, 1.0, 0.0]),
    ]
    .into_iter()
    .map(|(entity, vector)| Ok((entity, profile.quantize(&vector)?, 5)))
    .collect::<irongraph::Result<Vec<_>>>()?;
    let mut indexes = IndexCatalog::default();
    indexes.create_embedding(
        &graph,
        EmbeddingIndexDefinition {
            name: "semantic".to_owned(),
            label: person,
            source_property: name,
            target_property: embedding,
            model: "default".to_owned(),
        },
        profile,
        rows,
    )?;
    let bookmark = Bookmark { term: 1, index: 5 };
    let image = ResidentProjectImage::build(project, bookmark, &graph, &temporal, &indexes)?;
    let mut cpu = CpuBackend::new(128 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 128 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let request = ResidentVectorQuery {
        project,
        property: embedding,
        layers: LayerMask::AUTHORITY,
        selection: None,
        allowed_entities: None,
        queries: vec![1.0, 0.0, 0.0, 0.0],
        query_count: 1,
        limit: 3,
        access: irongraph::gpu::ResidentVectorAccess::IvfPq,
    };
    let cancellation = CancellationToken::new();
    let metal_result = metal.search_vectors(&request, &cancellation)?;
    let mut exact_request = request.clone();
    exact_request.access = irongraph::gpu::ResidentVectorAccess::Exact;
    let cpu_result = cpu.search_vectors(&exact_request, &cancellation)?;
    assert_eq!(metal_result.project, cpu_result.project);
    assert_eq!(metal_result.bookmark, cpu_result.bookmark);
    assert_eq!(metal_result.profile, cpu_result.profile);
    assert_eq!(metal_result.hits, cpu_result.hits);
    assert!(metal_result.approximate_candidates);

    let (exact, approximate) = indexes
        .vector_search_source("semantic")
        .ok_or_else(|| irongraph::Error::internal("semantic vector index is not online"))?;
    let vector_indexes = BTreeMap::from([(
        "semantic".to_owned(),
        VectorSearchSource {
            property: embedding,
            exact,
            approximate,
            profile_hash: indexes
                .profile()
                .map_or([0_u8; 32], |profile| profile.profile_hash),
        },
    )]);
    let mut context = ExecutionContext {
        project_id: project,
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: Some(&temporal),
        prior_temporal_mutations: &[],
        vector_indexes,
        scalar_indexes: Some(&indexes),
        text_embedding: None,
        parameters: BTreeMap::from([(
            "query".to_owned(),
            ResultValue::Vector(vec![1.0, 0.0, 0.0, 0.0]),
        )]),
        bookmark,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 100,
        max_batch_rows: 64,
        optimizer_statistics: None,
        resolved_query_at_time_nanos: None,
        backend: Some(&metal),
        cancellation: CancellationToken::new(),
        deadline: None,
    };
    let query = "MATCH (d:Person)\n\
                 SEARCH d IN (EMBEDDING INDEX semantic FOR VECTOR $query LIMIT 2) SCORE AS score\n\
                 RETURN d, score";
    let output = QueryEngine.execute(query, &mut context)?;
    assert_eq!(
        output
            .result
            .batches
            .iter()
            .map(|batch| batch.row_count)
            .sum::<usize>(),
        2
    );
    let first = output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first());
    assert!(matches!(first, Some(ResultValue::Node(node)) if node.id == NodeId(1)));
    context.backend = Some(&cpu);
    let cpu_output = QueryEngine.execute(query, &mut context)?;
    assert_eq!(output.result.schema, cpu_output.result.schema);
    assert_eq!(output.result.batches, cpu_output.result.batches);

    let temporal_sample = TemporalSample {
        entity_id: 1,
        property: age,
        event_time_nanos: 50,
        sequence_index: 6,
        value: ScalarValue::Integer(36),
    };
    let delta = ResidentProjectDelta {
        project,
        bookmark: Bookmark { term: 1, index: 6 },
        graph: graph.device_delta(graph.revision())?,
        temporal: vec![ResidentTemporalDelta {
            entity_kind: EntityKind::Node,
            target: person.0,
            sample: temporal_sample,
        }],
        vectors: Vec::new(),
        invalidate_derived: false,
    };
    cpu.apply_project_delta(delta.clone())?;
    metal.apply_project_delta(delta)?;
    let temporal_request = ResidentTemporalPipelineRequest {
        input: ResidentNodePipelineRequest {
            project,
            labels: vec![person],
            layers: LayerMask::AUTHORITY,
            initial_optional: false,
            expansion: None,
            continuations: Vec::new(),
            correlated_optional: None,
            relationship_null_filter: None,
            predicates: Vec::new(),
            property_filters: Vec::new(),
            value_matrix: None,
            mutation: None,
            orders: Vec::new(),
            offset: 0,
            limit: 10,
            integer_projections: Vec::new(),
            property_null_projections: Vec::new(),
            max_output_rows: 10,
        },
        binding: ResidentNodeBinding::Start,
        target: person.0,
        property: age,
        from_nanos: 0,
        to_nanos: 1_000,
        bookmark_index: 6,
        window: None,
        max_output_rows: 10,
    };
    let metal_temporal = metal.execute_temporal_pipeline(&temporal_request, &cancellation)?;
    let cpu_temporal = cpu.execute_temporal_pipeline(&temporal_request, &cancellation)?;
    assert_eq!(metal_temporal, cpu_temporal);
    assert_eq!(metal_temporal.event_times_nanos, vec![50, 100]);
    assert_eq!(metal_temporal.values, vec![36, 37]);
    let nullable_request = ResidentTemporalPipelineRequest {
        property: nullable_age,
        bookmark_index: 6,
        ..temporal_request
    };
    let metal_nullable = metal.execute_temporal_pipeline(&nullable_request, &cancellation)?;
    let cpu_nullable = cpu.execute_temporal_pipeline(&nullable_request, &cancellation)?;
    assert_eq!(metal_nullable, cpu_nullable);
    assert_eq!(metal_nullable.event_times_nanos, vec![75]);
    assert_eq!(metal_nullable.values, vec![0]);
    assert_eq!(metal_nullable.validity, vec![0]);
    Ok(())
}

#[test]
fn graph_algorithm_references_are_deterministic() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let snapshot = graph.snapshot()?;
    assert_eq!(bfs(&snapshot.outgoing, 0)?, vec![Some(0), Some(1), Some(2)]);
    assert_eq!(triangle_count(&snapshot.outgoing, &snapshot.incoming)?, 0);
    let first = page_rank(&snapshot.outgoing, PageRankConfig::default())?;
    let second = page_rank(&snapshot.outgoing, PageRankConfig::default())?;
    assert_eq!(first, second);
    Ok(())
}
