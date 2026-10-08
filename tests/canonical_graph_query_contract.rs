//! Canonical CPU contracts retained from the resident comparison fixtures.
use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionStreamItem, QueryEngine, ResultValue, parse,
    },
    graph::{
        AdjacencyRead, EdgeInput, EqualityIndex, GraphStore, IndexKey, IvfPqConfig, IvfPqIndex,
        LayerMask, NodeInput, PageRankConfig, RangeIndex, Similarity, TemporalDeclaration,
        TemporalSample, TemporalStore, TemporalType, TextIndex, VectorIndex, WindowSpec, bfs,
        page_rank, triangle_count,
    },
    types::{EntityKind, PropertyId},
};
use ordered_float::OrderedFloat;
use std::{collections::BTreeMap, sync::Arc};
use tokio_util::sync::CancellationToken;

struct CanonicalAdjacency<'a> {
    graph: &'a GraphStore,
    outgoing: bool,
}

impl AdjacencyRead for CanonicalAdjacency<'_> {
    type Row<'a>
        = std::vec::IntoIter<(u32, u32)>
    where
        Self: 'a;

    fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    fn row(&self, node: u32) -> Option<Self::Row<'_>> {
        let node = self.graph.node_dense(node)?;
        let edges = if self.outgoing {
            self.graph.expand_out(node.id(), None, LayerMask::ALL)
        } else {
            self.graph.expand_in(node.id(), None, LayerMask::ALL)
        }
        .ok()?;
        Some(
            edges
                .into_iter()
                .map(|(edge, neighbor)| (neighbor.dense(), edge.dense()))
                .collect::<Vec<_>>()
                .into_iter(),
        )
    }
}

#[test]
fn graph_algorithm_references_are_deterministic() -> irongraph::Result<()> {
    let graph = sample_graph()?;
    let outgoing = CanonicalAdjacency {
        graph: &graph,
        outgoing: true,
    };
    let incoming = CanonicalAdjacency {
        graph: &graph,
        outgoing: false,
    };
    assert_eq!(bfs(&outgoing, 0)?, vec![Some(0), Some(1), Some(2)]);
    assert_eq!(triangle_count(&outgoing, &incoming)?, 0);
    assert_eq!(
        page_rank(&outgoing, PageRankConfig::default())?,
        page_rank(&outgoing, PageRankConfig::default())?
    );
    Ok(())
}
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
        layer: Layer::Knowledge,
        revision: 5,
        properties: Vec::new(),
    })?;
    Ok(graph)
}

#[test]
fn graph_preserves_stable_ids_layers_and_merged_adjacency() -> irongraph::Result<()> {
    let graph = sample_graph()?;
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
    assert_eq!(remap.nodes.get(&NodeId(3)), Some(&2));
    Ok(())
}

#[test]
fn temporal_late_samples_and_half_open_windows_are_deterministic() -> irongraph::Result<()> {
    let property = PropertyId(7);
    let temporal = TemporalStore::default();
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
fn deterministic_ivf_pq_reranks_against_exact_vectors() -> irongraph::Result<()> {
    let vectors = VectorIndex::new(4, Similarity::Euclidean)?;
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
    let equality = EqualityIndex::default();
    let range = RangeIndex::default();
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
    let text = TextIndex::default();
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

    assert!(graph.catalog().label("TransactionOnly").is_none());
    let mut second = ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph: &graph,
        binding_catalog: graph.catalog(),
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

#[test]
fn at_time_projects_declared_scalar_history_without_resurrecting_topology() -> irongraph::Result<()>
{
    let graph = sample_graph()?;
    let age = graph
        .catalog()
        .property("age")
        .ok_or_else(|| irongraph::Error::internal("age property missing"))?;
    let temporal = TemporalStore::default();
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
