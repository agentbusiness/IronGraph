use super::*;
use std::cell::RefCell;

struct ProbeState {
    payload: Arc<str>,
    observations: Vec<usize>,
}

thread_local! {
    static PAYLOAD_PROBE: RefCell<Option<ProbeState>> = const { RefCell::new(None) };
}

pub(super) fn observe_payload_owners() {
    PAYLOAD_PROBE.with(|probe| {
        if let Some(probe) = probe.borrow_mut().as_mut() {
            probe.observations.push(Arc::strong_count(&probe.payload));
        }
    });
}

struct PayloadProbe;

impl PayloadProbe {
    fn begin(payload: &Arc<str>) -> Self {
        PAYLOAD_PROBE.with(|probe| {
            assert!(probe.borrow().is_none());
            *probe.borrow_mut() = Some(ProbeState {
                payload: Arc::clone(payload),
                observations: Vec::new(),
            });
        });
        Self
    }

    fn baseline(&self) -> usize {
        PAYLOAD_PROBE.with(|probe| {
            Arc::strong_count(&probe.borrow().as_ref().expect("probe missing").payload)
        })
    }

    fn observations(&self) -> Vec<usize> {
        PAYLOAD_PROBE.with(|probe| {
            probe
                .borrow()
                .as_ref()
                .expect("probe missing")
                .observations
                .clone()
        })
    }
}

impl Drop for PayloadProbe {
    fn drop(&mut self) {
        PAYLOAD_PROBE.with(|probe| *probe.borrow_mut() = None);
    }
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
            term: 1,
            index: graph.revision(),
        },
        mutation_revision: graph.revision() + 1,
        resolved_time_nanos: 0,
        next_node_id: 10,
        next_edge_id: 10,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1000,
        max_batch_rows: 3,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

fn fixture() -> Result<(GraphStore, Arc<str>)> {
    let graph = GraphStore::default();
    let label = graph.catalog().intern_label("Owner")?;
    let value = graph.catalog().intern_property("value")?;
    let body = graph.catalog().intern_property("body")?;
    let kind = graph.catalog().intern_relationship_type("R")?;
    let payload: Arc<str> = "complete user content ".repeat(100_000).into();
    for id in 1..=2 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![
                (value, ScalarValue::Integer(42)),
                (body, ScalarValue::String(Arc::clone(&payload))),
            ],
        })?;
    }
    for id in 1..=2 {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: vec![(body, ScalarValue::String(Arc::clone(&payload)))],
        })?;
    }
    Ok((graph, payload))
}

fn integers(output: &ExecutionOutput) -> Vec<i64> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| {
            batch.columns.iter().map(|column| {
                let ResultValue::Scalar(ScalarValue::Integer(value)) = &column.values[0] else {
                    panic!("expected INTEGER")
                };
                *value
            })
        })
        .collect()
}

#[test]
fn direct_scalar_count_matches_generic_batches_and_stream_failures() -> Result<()> {
    let (graph, payload) = fixture()?;
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&payload);
    let mut query = context(&graph);
    let view = GraphReadView::new(&graph);
    for batch_rows in [0, 1, 3] {
        query.max_batch_rows = batch_rows;
        for count in [0, 2, i64::MAX as u64] {
            let name = "count alias";
            let mut row = Row::new();
            row.insert(
                name.to_owned(),
                BindingValue::Value(ResultValue::Scalar(ScalarValue::Integer(count as i64))),
            );
            let expected = rows_to_result(
                &[row],
                &[name.to_owned()],
                &view,
                query.bookmark,
                StatementStats::default(),
                batch_rows,
                None,
            )?;
            assert_eq!(
                emit_scalar_count(count, name, &query, None)?.result,
                expected
            );
            let mut items = Vec::new();
            let output = emit_scalar_count(
                count,
                name,
                &query,
                Some(&mut |item| {
                    items.push(item);
                    Ok(())
                }),
            )?;
            assert!(output.result.batches.is_empty());
            let [
                ExecutionStreamItem::Schema(schema),
                ExecutionStreamItem::Batch(batch),
            ] = items.as_slice()
            else {
                panic!("expected schema then one batch");
            };
            assert_eq!(*schema, expected.schema);
            assert_eq!(std::slice::from_ref(batch), expected.batches);
        }
    }
    assert_eq!(
        emit_scalar_count(i64::MAX as u64 + 1, "count", &query, None)
            .unwrap_err()
            .code,
        ErrorCode::ResultBudgetExceeded
    );
    for fail_at in [1, 2] {
        let mut calls = 0;
        let error = emit_scalar_count(
            2,
            "count",
            &query,
            Some(&mut |_| {
                calls += 1;
                if calls == fail_at {
                    Err(Error::new(ErrorCode::Cancelled, "cancelled stream"))
                } else {
                    Ok(())
                }
            }),
        )
        .unwrap_err();
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert_eq!(calls, fail_at);
    }
    for (fast, generic) in [
        (
            "MATCH (n:Owner) RETURN count(n) AS c",
            "MATCH (n:Owner) RETURN count(n) + 0 AS c",
        ),
        (
            "MATCH (n:Missing) RETURN count(n) AS c",
            "MATCH (n:Missing) RETURN count(n) + 0 AS c",
        ),
        (
            "MATCH ()-[r:R]->() RETURN count(r) AS c",
            "MATCH ()-[r:R]->() RETURN count(r) + 0 AS c",
        ),
    ] {
        assert_eq!(
            QueryEngine.execute(fast, &mut query)?.result,
            QueryEngine.execute(generic, &mut query)?.result
        );
    }
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&payload), owners);
    Ok(())
}

#[test]
fn entity_count_actual_queries_skip_large_payload_hydration_with_positive_control() -> Result<()> {
    let (graph, payload) = fixture()?;
    let probe = PayloadProbe::begin(&payload);
    let baseline = probe.baseline();
    let output = QueryEngine.execute(
        "MATCH (n:Owner)-[:R]->(m) RETURN count(m), count(DISTINCT m)",
        &mut context(&graph),
    )?;
    assert_eq!(integers(&output), vec![2, 1]);
    assert_eq!(probe.observations(), vec![baseline, baseline]);
    drop(probe);
    let probe = PayloadProbe::begin(&payload);
    let baseline = probe.baseline();
    let output=QueryEngine.execute("MATCH (n:Owner)-[:R]->(m) UNWIND [m,m,null,1] AS entity RETURN count(entity), count(DISTINCT entity)",&mut context(&graph))?;
    assert_eq!(integers(&output), vec![6, 2]);
    let observed = probe.observations();
    assert_eq!(observed.len(), 2);
    assert!(
        observed.iter().all(|owners| *owners > baseline),
        "positive control did not detect actual fallback hydration"
    );
    println!("CANONICAL_ENTITY_COUNT_PASS");
    Ok(())
}

#[test]
fn entity_count_identity_kind_collisions_nulls_aliases_and_groups() -> Result<()> {
    let (graph, payload) = fixture()?;
    let probe = PayloadProbe::begin(&payload);
    let baseline = probe.baseline();
    let output=QueryEngine.execute("MATCH (n:Owner)-[r:R]->(m) UNWIND [n,n,r,null] AS entity RETURN count(entity) AS all, count(DISTINCT entity) AS unique",&mut context(&graph))?;
    assert_eq!(integers(&output), vec![6, 3]);
    assert_eq!(probe.observations(), vec![baseline, baseline]);
    drop(probe);
    let output=QueryEngine.execute("MATCH (n:Owner)-[:R]->(m) RETURN n.value AS category, count(m) AS all, count(DISTINCT m) AS unique",&mut context(&graph))?;
    assert_eq!(integers(&output), vec![42, 2, 1]);
    for (source, expected) in [
        (
            "UNWIND [null,null] AS entity RETURN count(entity), count(DISTINCT entity)",
            vec![0, 0],
        ),
        (
            "UNWIND [1,1,null] AS entity RETURN count(entity), count(DISTINCT entity)",
            vec![2, 1],
        ),
        (
            "UNWIND [] AS entity RETURN count(entity), count(DISTINCT entity)",
            vec![0, 0],
        ),
    ] {
        assert_eq!(
            integers(&QueryEngine.execute(source, &mut context(&graph))?),
            expected
        );
    }
    Ok(())
}

#[test]
fn entity_count_deleted_bindings_missing_identity_and_cancellation() -> Result<()> {
    let (graph, _payload) = fixture()?;
    for source in [
        "MATCH (n:Owner) DETACH DELETE n RETURN count(n)",
        "MATCH (n:Owner) DETACH DELETE n WITH n UNWIND [n,1] AS entity RETURN count(entity)",
        "MATCH ()-[r:R]->() DELETE r RETURN count(DISTINCT r)",
    ] {
        let mut query = context(&graph);
        query.capabilities.write = true;
        let error = QueryEngine
            .execute(source, &mut query)
            .expect_err("deleted entity was counted");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert!(error.message.contains("DeletedEntityAccess"));
    }
    let view = GraphReadView::new(&graph);
    let mut row = Row::new();
    row.insert("entity".into(), BindingValue::Node(u32::MAX));
    let error = count_entity_bindings(&[row], "entity", false, &view, &context(&graph))
        .expect_err("missing identity was counted");
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(error.message, "node binding does not exist");
    let mut query = context(&graph);
    query.cancellation.cancel();
    let error = QueryEngine
        .execute("MATCH (n:Owner)-[:R]->(m) RETURN count(m)", &mut query)
        .expect_err("cancelled entity aggregate executed");
    assert_eq!(error.code, ErrorCode::Cancelled);
    Ok(())
}

#[test]
fn entity_count_temporal_context_keeps_identity_only_semantics() -> Result<()> {
    let (graph, payload) = fixture()?;
    let label = graph
        .catalog()
        .label("Owner")
        .ok_or_else(|| Error::internal("label"))?;
    let value = graph
        .catalog()
        .property("value")
        .ok_or_else(|| Error::internal("property"))?;
    let temporal = TemporalStore::default();
    temporal.declare(
        crate::graph::TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: label.0,
            property: value,
            value_type: crate::graph::TemporalType::Integer,
            retention_nanos: 1000,
        },
        20,
    )?;
    temporal.append(
        EntityKind::Node,
        label.0,
        TemporalSample {
            entity_id: 1,
            property: value,
            event_time_nanos: 10,
            sequence_index: 1,
            value: ScalarValue::Integer(43),
        },
        20,
    )?;
    for source in [
        "MATCH (n:Owner)-[:R]->(m) RETURN count(m),count(DISTINCT m)",
        "AT TIME 10 MATCH (n:Owner)-[:R]->(m) RETURN count(m),count(DISTINCT m)",
    ] {
        let probe = PayloadProbe::begin(&payload);
        let baseline = probe.baseline();
        let mut query = context(&graph);
        query.temporal = Some(&temporal);
        assert_eq!(
            integers(&QueryEngine.execute(source, &mut query)?),
            vec![2, 1]
        );
        assert_eq!(probe.observations(), vec![baseline, baseline]);
    }
    Ok(())
}
