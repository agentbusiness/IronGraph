use super::*;
use std::{sync::mpsc, thread, time::Duration};

#[test]
fn inline_label_cells_preserve_ids_reclaim_borrowed_sets_and_bound_dirty_churn() -> Result<()> {
    let accounting = Arc::new(AtomicUsize::new(0));
    let cell = LabelsCell::default();
    for id in [0, u64::from(u32::MAX), u64::MAX - 2, u64::MAX - 1, u64::MAX] {
        cell.store(vec![LabelId(id)], &accounting);
        assert_eq!(&*cell.handle(), &[LabelId(id)]);
        assert!(cell.contains_all(&[LabelId(id)]));
        assert!(!cell.contains_all(&[LabelId(id ^ 1)]));
        if id < u64::MAX - 1 {
            assert!(cell.multiple.load().is_none());
            assert_eq!(accounting.load(Ordering::Acquire), 0);
        }
    }
    cell.store(vec![LabelId(1), LabelId(2)], &accounting);
    let borrowed = cell.handle();
    cell.store(vec![LabelId(3)], &accounting);
    assert!(cell.multiple.load().is_none());
    assert_eq!(&*borrowed, &[LabelId(1), LabelId(2)]);
    assert!(accounting.load(Ordering::Acquire) > 0);
    drop(borrowed);
    assert_eq!(accounting.load(Ordering::Acquire), 0);

    let graph = GraphStore::default();
    let first = graph.catalog().intern_label("First")?;
    let second = graph.catalog().intern_label("Second")?;
    let body = graph.catalog().intern_property("body")?;
    let payload: Arc<str> = "complete owning domain content ".repeat(100_000).into();
    for id in 1..=1025 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![first],
            properties: vec![(body, ScalarValue::String(Arc::clone(&payload)))],
        })?;
    }
    graph.add_node_labels(NodeId(1), vec![second], 2)?;
    graph.remove_node_labels(NodeId(1), vec![first], 3)?;
    graph.remove_node_labels(NodeId(1), vec![second], 4)?;
    graph.add_node_labels(NodeId(1), vec![first], 5)?;
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&payload);
    for cycle in 0..1000 {
        let revision = 6 + cycle * 4;
        graph.add_node_labels(NodeId(1), vec![second], revision)?;
        let node = graph
            .node(NodeId(1))
            .ok_or_else(|| missing("label owner"))?;
        assert!(node.has_labels(&[first, second]));
        graph.remove_node_labels(NodeId(1), vec![first], revision + 1)?;
        assert_eq!(&*node.labels(), &[second]);
        assert!(node.has_labels(&[second]));
        assert!(!node.has_labels(&[first]));
        graph.remove_node_labels(NodeId(1), vec![second], revision + 2)?;
        assert!(node.labels().is_empty());
        assert!(!node.has_labels(&[second]));
        graph.add_node_labels(NodeId(1), vec![first], revision + 3)?;
        assert_eq!(&*node.labels(), &[first]);
        assert!(node.has_labels(&[first]));
        assert_eq!(graph.0.label_bytes.load(Ordering::Acquire), 0);
        assert_eq!(graph.resident_bytes(), bytes);
        assert_eq!(Arc::strong_count(&payload), owners);
    }
    println!("CANONICAL_INLINE_LABELS_PASS");
    Ok(())
}

#[test]
fn borrowed_adjacency_views_stop_on_error_and_check_excluded_layers() -> Result<()> {
    let (graph, _) = fixture()?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Knowledge,
        revision: 1,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    let kind = graph.catalog().intern_relationship_type("R")?;
    for id in 1..=2048 {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    let node = graph
        .node(NodeId(1))
        .ok_or_else(|| Error::internal("source missing"))?;
    let bytes = graph.resident_bytes();
    let mut checkpoints = 0;
    let error = graph
        .try_visit_neighbors(
            node,
            true,
            LayerMask::KNOWLEDGE,
            || {
                checkpoints += 1;
                Err(Error::new(
                    ErrorCode::Cancelled,
                    "cancelled excluded adjacency",
                ))
            },
            |_, _| panic!("excluded edge reached visitor"),
        )
        .expect_err("filtered edges postponed cancellation");
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert_eq!(checkpoints, 1);
    let mut visited = 0;
    let error = graph
        .try_visit_neighbors(
            node,
            true,
            LayerMask::ALL,
            || Ok(()),
            |edge, target| {
                visited += 1;
                assert_eq!(edge.source(), NodeId(1));
                assert_eq!(edge.target(), target.id());
                assert_eq!(edge.relationship_type(), kind);
                Err(Error::new(ErrorCode::Cancelled, "stop first neighbor"))
            },
        )
        .expect_err("visitor failure did not stop adjacency");
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert_eq!(visited, 1);
    assert_eq!(graph.resident_bytes(), bytes);
    println!("CANONICAL_BORROWED_ADJACENCY_PASS");
    Ok(())
}

#[test]
fn borrowed_adjacency_pages_reload_recycled_cells_without_retaining_values() -> Result<()> {
    let graph = GraphStore::default();
    let body = graph.catalog().intern_property("body")?;
    let kind = graph.catalog().intern_relationship_type("R")?;
    let text: Arc<str> = "complete owning domain content ".repeat(100_000).into();
    for id in 1..=4 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(body, ScalarValue::String(Arc::clone(&text)))],
        })?;
    }
    for id in 1..=600 {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&text);
    let node = graph.node(NodeId(1)).unwrap();
    let mut count = 0;
    assert!(graph.visit_neighbor_denses(node, true, LayerMask::ALL, |_, _| count += 1));
    assert_eq!(count, 600);
    let barrier = std::sync::Barrier::new(2);
    thread::scope(|scope| -> Result<()> {
        let reader = scope.spawn(|| {
            let mut visited = 0;
            assert!(
                graph.visit_neighbor_denses(node, true, LayerMask::ALL, |dense, _| {
                    assert_eq!(dense, 1);
                    visited += 1;
                    if visited == 1 {
                        barrier.wait();
                        barrier.wait();
                    }
                })
            );
            assert_eq!(
                visited, 1,
                "recycled adjacency must stop at its allocation fence"
            );
        });
        barrier.wait();
        let writes = (|| -> Result<()> {
            for id in 1..=600 {
                graph.delete_edge(EdgeId(id), 2)?;
            }
            for id in 1001..=1600 {
                graph.insert_edge(EdgeInput {
                    id: EdgeId(id),
                    source: NodeId(3),
                    target: NodeId(4),
                    relationship_type: kind,
                    layer: Layer::Observed,
                    revision: 3,
                    properties: Vec::new(),
                })?;
            }
            Ok(())
        })();
        barrier.wait();
        reader
            .join()
            .map_err(|_| Error::internal("adjacency reader panicked"))?;
        writes
    })?;
    assert_eq!(graph.edge_slot_count(), 600);
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&text), owners);
    graph.validate_structure()
}

fn fixture() -> Result<(GraphStore, PropertyId)> {
    let graph = GraphStore::default();
    let property = graph.catalog().intern_property("value")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Knowledge,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(property, ScalarValue::Integer(41))],
    })?;
    Ok((graph, property))
}

#[test]
fn integer_candidate_pages_are_borrowed_bounded_and_conservative() -> Result<()> {
    let graph = GraphStore::default();
    let value = graph.catalog().intern_property("value")?;
    let body = graph.catalog().intern_property("body")?;
    let payload: Arc<str> = "real user content ".repeat(100_000).into();
    for id in 1..=1025 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (
                    value,
                    ScalarValue::Integer(if id % 257 == 0 { 42 } else { 43 }),
                ),
                (body, ScalarValue::String(Arc::clone(&payload))),
            ],
        })?;
    }
    let reader = graph
        .node_property_reader(value)
        .ok_or_else(|| missing("column"))?;
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&payload);
    let mut candidates = Vec::new();
    let mut partitioned = Vec::new();
    for start in (17..1009).step_by(101) {
        reader.visit_integer_candidates_range(
            start,
            (start + 101).min(1009),
            42,
            || Ok(()),
            |dense| {
                partitioned.push(dense);
                Ok(())
            },
        )?;
    }
    assert_eq!(partitioned, vec![256, 513, 770]);
    for _ in 0..1000 {
        candidates.clear();
        reader.visit_integer_candidates(
            graph.node_slot_count(),
            42,
            || Ok(()),
            |dense| {
                candidates.push(dense);
                Ok(())
            },
        )?;
        assert_eq!(candidates, vec![256, 513, 770]);
        assert_eq!(graph.resident_bytes(), bytes);
        assert_eq!(Arc::strong_count(&payload), owners);
    }
    graph.set_node_property(NodeId(1), value, ScalarValue::Float(42.0.into()), 2)?;
    graph.set_node_property(NodeId(1025), value, ScalarValue::Null, 3)?;
    candidates.clear();
    reader.visit_integer_candidates(
        graph.node_slot_count(),
        42,
        || Ok(()),
        |dense| {
            candidates.push(dense);
            Ok(())
        },
    )?;
    assert_eq!(candidates, vec![0, 256, 513, 770, 1024]);
    let error = reader
        .visit_integer_candidates(
            graph.node_slot_count(),
            42,
            || Err(Error::new(ErrorCode::Cancelled, "cancelled")),
            |_| panic!("visited after cancellation"),
        )
        .expect_err("ignored cancellation");
    assert_eq!(error.code, ErrorCode::Cancelled);
    println!("CANONICAL_INTEGER_PAGE_SCAN_PASS");
    Ok(())
}

#[test]
fn primitive_bits_null_and_monotonic_type_promotion() -> Result<()> {
    let (graph, property) = fixture()?;
    for (offset, integer) in [i64::MIN, -1, 0, i64::MAX].into_iter().enumerate() {
        graph.set_node_property(
            NodeId(1),
            property,
            ScalarValue::Integer(integer),
            2 + offset as u64,
        )?;
        let node = graph.node(NodeId(1)).ok_or_else(|| missing("node"))?;
        assert_eq!(node.property(property), Some(ScalarValue::Integer(integer)));
        assert_eq!(
            graph
                .node_property_reader(property)
                .and_then(|reader| reader.get_integer(node)),
            Some(integer)
        );
    }
    let float = f64::from_bits(0x7ff8_0000_0000_0123);
    graph.set_node_property(NodeId(1), property, ScalarValue::Float(float.into()), 6)?;
    let node = graph.node(NodeId(1)).ok_or_else(|| missing("node"))?;
    let Some(ScalarValue::Float(actual)) = node.property(property) else {
        return Err(Error::internal("float type changed"));
    };
    assert_eq!(actual.0.to_bits(), float.to_bits());
    assert_eq!(
        graph
            .node_property_reader(property)
            .and_then(|reader| reader.get_integer(node)),
        None
    );
    for (revision, value) in [
        (7, ScalarValue::Boolean(false)),
        (8, ScalarValue::Boolean(true)),
        (9, ScalarValue::Null),
        (10, ScalarValue::Integer(-7)),
    ] {
        graph.set_node_property(NodeId(1), property, value.clone(), revision)?;
        assert_eq!(
            node.property(property),
            (!matches!(value, ScalarValue::Null)).then_some(value)
        );
    }
    let column = graph
        .node_property_column(property)
        .ok_or_else(|| missing("column"))?;
    assert_eq!(
        column
            .cells
            .get(node.dense() as usize)
            .ok_or_else(|| missing("cell"))?
            .mode
            .load(Ordering::Acquire),
        VARIABLE_CELL
    );
    assert_eq!(column.value_count(), 1);
    assert_eq!(column.kind_mask(), 1 << 2);
    Ok(())
}

#[test]
fn promotion_stages_remain_readable_without_waiting_for_owner_writer() -> Result<()> {
    let (graph, property) = fixture()?;
    let node = graph.node(NodeId(1)).ok_or_else(|| missing("node"))?;
    let column = graph
        .node_property_column(property)
        .ok_or_else(|| missing("column"))?;
    let cell = column
        .cells
        .get(node.dense() as usize)
        .ok_or_else(|| missing("cell"))?;
    // Hold the actual production owner permit, then expose the exact production promotion stages.
    let mut owner = Some(graph.claim_node(node)?);
    cell.payload.store(Some(ValueAllocation::new(
        ScalarValue::String("promoted".into()),
        &column.accounting,
    )));
    thread::scope(|scope| -> Result<()> {
        let (send, receive) = mpsc::channel();
        let reader = scope.spawn(move || {
            let result = node.property(property);
            let _ = send.send(result);
        });
        let actual = receive.recv_timeout(Duration::from_secs(2));
        if actual.is_err() {
            drop(owner.take());
        }
        reader
            .join()
            .map_err(|_| Error::internal("reader panicked"))?;
        assert_eq!(
            actual.map_err(|error| Error::internal(error.to_string()))?,
            Some(ScalarValue::Integer(41))
        );
        Ok(())
    })?;
    cell.mode.store(VARIABLE_CELL, Ordering::Release);
    assert_eq!(
        node.property(property),
        Some(ScalarValue::String("promoted".into()))
    );
    cell.bits.store(0, Ordering::Release);
    assert_eq!(
        node.property(property),
        Some(ScalarValue::String("promoted".into()))
    );
    assert_eq!(column.get_integer(node.dense()), None);
    drop(owner);
    Ok(())
}

#[test]
fn held_owner_writer_delays_second_writer_but_never_reader() -> Result<()> {
    let (graph, property) = fixture()?;
    let node = graph.node(NodeId(1)).ok_or_else(|| missing("node"))?;
    let owner = graph.claim_node(node)?;
    thread::scope(|scope| -> Result<()> {
        let (started_send, started_receive) = mpsc::channel();
        let (done_send, done_receive) = mpsc::channel();
        let writer_graph = &graph;
        let writer = scope.spawn(move || {
            let _ = started_send.send(());
            let result =
                writer_graph.set_node_property_view(node, property, ScalarValue::Integer(99), 2);
            let _ = done_send.send(result);
        });
        started_receive
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| Error::internal(error.to_string()))?;
        let (send, receive) = mpsc::channel();
        let reader = scope.spawn(move || {
            let _ = send.send(node.property(property));
        });
        let read = receive.recv_timeout(Duration::from_secs(2));
        let pending = done_receive.try_recv();
        drop(owner);
        writer
            .join()
            .map_err(|_| Error::internal("writer panicked"))?;
        reader
            .join()
            .map_err(|_| Error::internal("reader panicked"))?;
        assert_eq!(
            read.map_err(|error| Error::internal(error.to_string()))?,
            Some(ScalarValue::Integer(41))
        );
        assert!(matches!(pending, Err(mpsc::TryRecvError::Empty)));
        done_receive
            .recv_timeout(Duration::from_secs(2))
            .map_err(|error| Error::internal(error.to_string()))??;
        assert_eq!(node.property(property), Some(ScalarValue::Integer(99)));
        Ok(())
    })
}

#[test]
fn captured_stale_setter_cannot_create_column_on_recycled_owner() -> Result<()> {
    let (graph, property) = fixture()?;
    let fresh = graph.catalog().intern_property("new_column")?;
    let old = graph.node(NodeId(1)).ok_or_else(|| missing("node"))?;
    graph.delete_node(NodeId(1), false, 2)?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Knowledge,
        revision: 3,
        labels: Vec::new(),
        properties: vec![(property, ScalarValue::Integer(77))],
    })?;
    let replacement = graph
        .node(NodeId(2))
        .ok_or_else(|| missing("replacement"))?;
    assert_eq!(old.dense(), replacement.dense());
    for target in [property, fresh] {
        assert!(
            graph
                .set_node_property_view(old, target, ScalarValue::String("stale".into()), 4)
                .is_err()
        );
    }
    assert!(graph.node_property_column(fresh).is_none());
    assert_eq!(old.property(property), None);
    assert_eq!(
        replacement.property(property),
        Some(ScalarValue::Integer(77))
    );
    assert_eq!(replacement.property(fresh), None);
    assert!(
        graph
            .set_node_property(NodeId(2), property, ScalarValue::Integer(0), 2)
            .is_err()
    );
    assert_eq!(replacement.revision(), 3);
    Ok(())
}

#[test]
fn resolved_endpoint_ordinals_cannot_attach_to_recycled_identity() -> Result<()> {
    let (graph, property) = fixture()?;
    let kind = graph.catalog().intern_relationship_type("LINK")?;
    let old_dense = graph
        .node(NodeId(1))
        .ok_or_else(|| missing("node"))?
        .dense();
    graph.delete_node(NodeId(1), false, 2)?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Knowledge,
        revision: 3,
        labels: Vec::new(),
        properties: vec![(property, ScalarValue::Integer(77))],
    })?;
    assert_eq!(
        graph
            .node(NodeId(2))
            .ok_or_else(|| missing("replacement"))?
            .dense(),
        old_dense
    );
    let slots = graph.edge_slot_count();
    let error = graph
        .append_edge(
            EdgeInput {
                id: EdgeId(1),
                source: NodeId(1),
                target: NodeId(1),
                relationship_type: kind,
                layer: Layer::Knowledge,
                revision: 4,
                properties: Vec::new(),
            },
            old_dense,
            old_dense,
            true,
        )
        .expect_err("a captured endpoint ordinal attached to its replacement owner");
    assert_eq!(error.message, "relationship endpoint identity changed");
    assert_eq!(graph.edge_count(), 0);
    assert_eq!(graph.edge_slot_count(), slots);
    assert!(
        graph
            .expand_out(NodeId(2), None, LayerMask::ALL)?
            .is_empty()
    );
    Ok(())
}

#[test]
fn sparse_numeric_pages_and_borrowed_payload_have_bounded_accounting() -> Result<()> {
    let bytes = Arc::new(AtomicUsize::new(0));
    let properties = Properties {
        columns: HashMap::default(),
        bytes: Arc::clone(&bytes),
    };
    let property = PropertyId(0);
    properties.set(2_000_000, property, ScalarValue::Integer(i64::MIN))?;
    assert_eq!(
        properties.get(2_000_000, property),
        Some(ScalarValue::Integer(i64::MIN))
    );
    assert_eq!(properties.get(1_000_000, property), None);
    assert!(
        bytes.load(Ordering::Acquire) < 1_000_000,
        "sparse row allocated its entire ordinal prefix"
    );
    properties.set(
        2_000_000,
        property,
        ScalarValue::String("a".repeat(65536).into()),
    )?;
    let column = properties
        .column(property, false)
        .ok_or_else(|| missing("column"))?;
    let cell = column.cells.get(2_000_000).ok_or_else(|| missing("cell"))?;
    let borrowed = cell.payload.load_full().ok_or_else(|| missing("payload"))?;
    let borrowed_bytes = borrowed.bytes;
    for _ in 0..1000 {
        properties.set(
            2_000_000,
            property,
            ScalarValue::String("b".repeat(65536).into()),
        )?;
    }
    let retained = bytes.load(Ordering::Acquire);
    drop(borrowed);
    assert_eq!(retained - bytes.load(Ordering::Acquire), borrowed_bytes);
    properties.clear(2_000_000);
    assert_eq!(column.value_count(), 0);
    assert_eq!(properties.get(2_000_000, property), None);
    assert!(bytes.load(Ordering::Acquire) < 1_000_000);
    Ok(())
}
