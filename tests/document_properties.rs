// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc};

use irongraph::{
    Bookmark, DocumentItem, DocumentMap, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::{
        EdgeInput, GraphStore, NodeInput, TemporalDeclaration, TemporalSample, TemporalStore,
        TemporalType, TypedColumn,
    },
    types::{EntityKind, PropertyId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

fn context<'a>(
    graph: &'a GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    revision: u64,
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
        parameters,
        bookmark: Bookmark {
            term: 1,
            index: revision.saturating_sub(1),
        },
        mutation_revision: revision,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_000,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

fn result_document() -> ResultValue {
    ResultValue::Map(BTreeMap::from([
        (
            "enabled".to_owned(),
            ResultValue::Scalar(ScalarValue::Boolean(true)),
        ),
        (
            "nested".to_owned(),
            ResultValue::Map(BTreeMap::from([(
                "name".to_owned(),
                ResultValue::Scalar(ScalarValue::String(Arc::from("sensor"))),
            )])),
        ),
        (
            "scores".to_owned(),
            ResultValue::List(vec![
                ResultValue::Scalar(ScalarValue::Integer(1)),
                ResultValue::Scalar(ScalarValue::Integer(2)),
            ]),
        ),
    ]))
}

fn canonical_map() -> Result<DocumentMap> {
    DocumentMap::new(BTreeMap::from([
        (
            Arc::from("enabled"),
            DocumentItem::Scalar(ScalarValue::Boolean(true)),
        ),
        (
            Arc::from("nested"),
            DocumentItem::Map(BTreeMap::from([(
                Arc::from("name"),
                DocumentItem::Scalar(ScalarValue::String(Arc::from("sensor"))),
            )])),
        ),
        (
            Arc::from("scores"),
            DocumentItem::List(vec![
                DocumentItem::Scalar(ScalarValue::Integer(1)),
                DocumentItem::Scalar(ScalarValue::Integer(2)),
            ]),
        ),
    ]))
}

#[test]
fn cypher_parameter_literal_equality_and_projection_round_trip_documents() -> Result<()> {
    let mut graph = GraphStore::default();
    let mut parameters = BTreeMap::new();
    parameters.insert("payload".to_owned(), result_document());
    let output = QueryEngine.execute(
        "CREATE (d:Doc {payload: $payload}) RETURN d.payload AS payload",
        &mut context(&graph, parameters, 1),
    )?;
    for mutation in output.graph_mutations {
        graph.apply(mutation)?;
    }

    let result = QueryEngine.execute(
        "MATCH (d:Doc {payload: {enabled: true, nested: {name: 'sensor'}, scores: [1.0, 2]}}) \
         RETURN d.payload AS payload, d.payload.scores[1] AS second",
        &mut context(&graph, BTreeMap::new(), 2),
    )?;
    let batch = &result.result.batches[0];
    assert_eq!(batch.row_count, 1);
    assert_eq!(batch.columns[0].values, vec![result_document()]);
    assert_eq!(
        batch.columns[1].values,
        vec![ResultValue::Scalar(ScalarValue::Integer(2))]
    );
    Ok(())
}

#[test]
fn node_and_relationship_document_columns_survive_restart_bytes_exactly() -> Result<()> {
    let mut graph = GraphStore::default();
    let entity = graph.catalog_mut().intern_label("Entity")?;
    let relation = graph.catalog_mut().intern_relationship_type("LINK")?;
    let payload = graph.catalog_mut().intern_property("payload")?;
    let document = canonical_map()?;
    for id in [1, 2] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![entity],
            properties: vec![(payload, ScalarValue::Map(document.clone()))],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: relation,
        layer: Layer::Observed,
        revision: 3,
        properties: vec![(payload, ScalarValue::Map(document.clone()))],
    })?;

    let snapshot = graph.snapshot()?;
    for columns in [&snapshot.node_properties, &snapshot.edge_properties] {
        if !matches!(columns.column(payload), Some(TypedColumn::Map { .. })) {
            return Err(irongraph::Error::internal(
                "document property is not a flat map column",
            ));
        }
    }
    assert!(snapshot.resident_bytes() >= document.as_bytes().len().saturating_mul(3));

    let encoded = postcard::to_stdvec(&graph)
        .map_err(|error| irongraph::Error::invalid_data(error.to_string()))?;
    let reopened: GraphStore = postcard::from_bytes(&encoded)
        .map_err(|error| irongraph::Error::invalid_data(error.to_string()))?;
    let node = reopened
        .node(NodeId(1))
        .ok_or_else(|| irongraph::Error::internal("node is absent after restart"))?;
    let edge = reopened
        .edge(EdgeId(1))
        .ok_or_else(|| irongraph::Error::internal("relationship is absent after restart"))?;
    assert_eq!(
        node.property(payload),
        Some(ScalarValue::Map(document.clone()))
    );
    assert_eq!(edge.property(payload), Some(ScalarValue::Map(document)));
    Ok(())
}

#[test]
fn temporal_samples_reject_documents_before_mutation() -> Result<()> {
    let mut temporal = TemporalStore::default();
    temporal.declare(
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: 7,
            property: PropertyId(3),
            value_type: TemporalType::Float,
            retention_nanos: 1_000,
        },
        1_000,
    )?;
    let error = temporal
        .append(
            EntityKind::Node,
            7,
            TemporalSample {
                entity_id: 1,
                property: PropertyId(3),
                event_time_nanos: 1_000,
                sequence_index: 1,
                value: ScalarValue::Map(DocumentMap::new(BTreeMap::from([(
                    Arc::from("value"),
                    DocumentItem::Scalar(ScalarValue::Float(OrderedFloat(1.0))),
                )]))?),
            },
            1_000,
        )
        .expect_err("documents must never enter temporal columns");
    assert!(error.message.contains("scalar values only"));
    assert!(
        temporal
            .current(EntityKind::Node, 7, 1, PropertyId(3))
            .is_none()
    );
    Ok(())
}
