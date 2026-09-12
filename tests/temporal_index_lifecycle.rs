// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::sync::Arc;

use irongraph::{
    Error, Layer, NodeId, Result, ScalarValue,
    graph::{
        AggregateSet, DerivedIndexState, EmbeddingDType, EmbeddingIndexDefinition,
        EmbeddingProfile, GraphIndexDefinition, GraphIndexKind, GraphMutation, GraphStore,
        IndexCatalog, NodeInput, ResolvedVectorMutation, Similarity, TemporalDeclaration,
        TemporalRollupDefinition, TemporalSample, TemporalStore, TemporalType, WindowSpec,
    },
    types::{EntityKind, LabelId, PropertyId},
};
use ordered_float::OrderedFloat;

fn indexed_graph() -> Result<(GraphStore, LabelId, PropertyId, PropertyId)> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("Document")?;
    let source = graph.catalog_mut().intern_property("text")?;
    let target = graph.catalog_mut().intern_property("embedding")?;
    graph.insert_node(NodeInput {
        id: NodeId(7),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![label],
        properties: vec![(source, ScalarValue::String(Arc::from("first")))],
    })?;
    Ok((graph, label, source, target))
}

#[test]
fn quantized_vector_mutation_replays_identical_bits() -> Result<()> {
    let (graph, label, source, target) = indexed_graph()?;
    let profile = EmbeddingProfile::new(
        [11; 32],
        [17; 32],
        4,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )?;
    let definition = EmbeddingIndexDefinition {
        name: "semantic".to_owned(),
        label,
        source_property: source,
        target_property: target,
        model: "default".to_owned(),
    };
    let initial = profile.quantize(&[1.0, 0.0, 0.0, 0.0])?;
    let mut first = IndexCatalog::default();
    first.create_embedding(
        &graph,
        definition.clone(),
        profile.clone(),
        vec![(7, initial.clone(), 1)],
    )?;
    let mut second = IndexCatalog::default();
    second.create_embedding(&graph, definition, profile.clone(), vec![(7, initial, 1)])?;

    let bits = profile.quantize(&[0.25, -0.5, 0.75, 1.0])?;
    let mutation = ResolvedVectorMutation::Upsert {
        property: target,
        entity_id: 7,
        coordinates: bits.clone(),
        revision: 41,
    };
    let encoded = postcard::to_stdvec(&mutation)
        .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
    let replayed: ResolvedVectorMutation = postcard::from_bytes(&encoded)
        .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
    let ResolvedVectorMutation::Upsert { coordinates, .. } = &replayed else {
        return Err(Error::internal("upsert changed kind during replay"));
    };
    assert_eq!(coordinates, &bits);
    first.apply_vector_mutation(&mutation)?;
    second.apply_vector_mutation(&replayed)?;
    assert_eq!(
        postcard::to_stdvec(&first)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?,
        postcard::to_stdvec(&second)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?
    );
    Ok(())
}

#[test]
fn index_lifecycle_survives_checkpoint_roundtrip() -> Result<()> {
    let (mut graph, label, source, _) = indexed_graph()?;
    let mut indexes = IndexCatalog::default();
    indexes.create(
        &graph,
        GraphIndexDefinition {
            name: "text_lookup".to_owned(),
            kind: GraphIndexKind::Text,
            label,
            properties: vec![source],
            unique: false,
        },
    )?;
    assert_eq!(
        indexes.statuses().next().map(|status| status.state),
        Some(DerivedIndexState::Online)
    );
    let encoded = postcard::to_stdvec(&indexes)
        .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
    let mut restored: IndexCatalog = postcard::from_bytes(&encoded)
        .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
    let mutation = GraphMutation::SetNodeProperty {
        node: NodeId(7),
        property: source,
        value: ScalarValue::String(Arc::from("updated graph memory")),
        revision: 2,
    };
    restored.before_graph_apply(&graph, &mutation)?;
    graph.apply(mutation.clone())?;
    restored.after_graph_apply(&graph, &mutation)?;
    restored.rebuild(&graph, "text_lookup")?;
    assert_eq!(
        restored.statuses().next().map(|status| status.state),
        Some(DerivedIndexState::Online)
    );
    restored.drop_index("text_lookup")?;
    assert!(restored.statuses().next().is_none());
    Ok(())
}

#[test]
fn late_rollup_repair_matches_rebuild_and_restart() -> Result<()> {
    let property = PropertyId(3);
    let mut store = TemporalStore::default();
    store.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: 12,
            property,
            value_type: TemporalType::Float,
            retention_nanos: 100_000,
        },
        1_000,
    )?;
    store.create_rollup(TemporalRollupDefinition {
        name: "ten_nanos".to_owned(),
        entity_kind: EntityKind::Node,
        target: 12,
        property,
        window: WindowSpec::tumbling(10),
        aggregates: AggregateSet::ALL,
    })?;
    for (time, revision, value) in [(12, 1, 2.0), (28, 2, 8.0), (16, 3, 4.0)] {
        store.append(
            EntityKind::Node,
            12,
            TemporalSample {
                entity_id: 9,
                property,
                event_time_nanos: time,
                sequence_index: revision,
                value: ScalarValue::Float(OrderedFloat(value)),
            },
            1_000,
        )?;
    }
    let incremental = store.rollup_buckets("ten_nanos", 9, 0, 40)?;
    let mut rebuilt = store.clone();
    rebuilt.rebuild_rollup("ten_nanos")?;
    assert_eq!(incremental, rebuilt.rollup_buckets("ten_nanos", 9, 0, 40)?);
    assert_eq!(incremental[0].count, 2);
    assert_eq!(incremental[0].sum, Some(6.0));
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(&store, &mut encoded)
        .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
    let restored: TemporalStore = ciborium::de::from_reader(encoded.as_slice())
        .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
    assert_eq!(restored.rollup_buckets("ten_nanos", 9, 0, 40)?, incremental);
    Ok(())
}
