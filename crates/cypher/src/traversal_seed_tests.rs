use super::*;

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
        next_node_id: 100,
        next_edge_id: 100,
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

fn fixture() -> Result<GraphStore> {
    let graph = GraphStore::default();
    let label = graph.catalog().intern_label("Seed")?;
    let value = graph.catalog().intern_property("value")?;
    let kind = graph.catalog().intern_relationship_type("R")?;
    for (id, property) in [
        (1, Some(ScalarValue::Integer(42))),
        (2, Some(ScalarValue::Integer(1))),
        (3, Some(ScalarValue::Integer(2))),
        (4, Some(ScalarValue::Null)),
        (5, None),
        (6, Some(ScalarValue::Integer(3))),
        (7, Some(ScalarValue::Integer(42))),
        (8, Some(ScalarValue::Integer(4))),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: property
                .map(|property| vec![(value, property)])
                .unwrap_or_default(),
        })?;
    }
    for (id, source, target) in [
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (4, 2, 3),
        (5, 2, 1),
        (6, 3, 4),
        (7, 8, 2),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn count(output: &ExecutionOutput) -> i64 {
    let ResultValue::Scalar(ScalarValue::Integer(count)) =
        &output.result.batches[0].columns[0].values[0]
    else {
        panic!("traversal fixture did not return INTEGER count");
    };
    *count
}

#[test]
fn fused_canonical_trails_match_generic_counts_and_bounded_dirty_memory() -> Result<()> {
    let graph = fixture()?;
    let temporal = TemporalStore::default();
    let mut native_context = context(&graph);
    native_context.temporal = Some(&temporal);
    let before = canonical_trail_count::reductions();
    let native = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->()-[:R]->(m) WHERE n.value=42 RETURN count(m)",
        &mut native_context,
    )?;
    assert_eq!(count(&native), 6);
    assert_eq!(canonical_trail_count::reductions(), before + 1);
    let body = graph.catalog().intern_property("body")?;
    let payload: Arc<str> = "complete source text ".repeat(100_000).into();
    for node in graph.nodes() {
        graph.set_node_property(
            node.id(),
            body,
            ScalarValue::String(Arc::clone(&payload)),
            2,
        )?;
    }
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&payload);
    for pattern in [
        "(n:Seed)-[:R]->(m)",
        "(n:Seed)-[:R]->()-[:R]->(m)",
        "(n:Seed)-[:R*0..3]->(m)",
        "(n:Seed)-[:R*2..3]->(m)",
        "(n:Seed)<-[:R*1..3]-(m)",
        "(n:Seed)-[:R*0..2]->()-[:R*1..2]->(m:Seed)",
        "(n:Seed)-[:MISSING]->(m)",
    ] {
        for argument in ["m", "DISTINCT m"] {
            let query = format!("MATCH {pattern} WHERE n.value=42 RETURN count({argument})");
            let before = canonical_trail_count::reductions();
            let fused = QueryEngine.execute(&query, &mut context(&graph))?;
            assert_eq!(canonical_trail_count::reductions(), before + 1, "{query}");
            let generic = QueryEngine.execute(&format!("{query} + 0"), &mut context(&graph))?;
            assert_eq!(count(&fused), count(&generic), "{query}");
            // The fused reduction must still acquire every dependency required by real trails.
            for entity in generic.dependencies.entities.keys() {
                assert!(
                    fused.dependencies.entities.contains_key(entity),
                    "{query}: {entity:?}"
                );
            }
        }
    }
    for _ in 0..1000 {
        let result = QueryEngine.execute(
            "MATCH (n:Seed)-[:R]->()-[:R]->(m) WHERE n.value=42 RETURN count(m)",
            &mut context(&graph),
        )?;
        assert_eq!(count(&result), 6);
    }
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&payload), owners);
    println!("CANONICAL_FUSED_TRAILS_PASS");
    Ok(())
}

#[test]
fn traversal_seed_actual_trails_nulls_parameters_and_dependencies() -> Result<()> {
    let graph = fixture()?;
    for (source, expected) in [
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
            3,
        ),
        (
            "MATCH (n:Seed)-[:R]->()-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
            6,
        ),
        (
            "MATCH (n:Seed)-[:R*1..3]->(m) WHERE n.value = 42 RETURN count(DISTINCT m)",
            4,
        ),
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE 42 = n.value RETURN count(m)",
            3,
        ),
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value = $seed RETURN count(m)",
            3,
        ),
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE n.missing = 42 RETURN count(m)",
            0,
        ),
    ] {
        let mut context = context(&graph);
        context
            .parameters
            .insert("seed".into(), ResultValue::Scalar(ScalarValue::Integer(42)));
        let output = QueryEngine.execute(source, &mut context)?;
        assert_eq!(count(&output), expected, "{source}");
        assert!(!output.dependencies.predicates.is_empty());
        if expected != 0 {
            // This isolated rejected seed formerly acquired an entity dependency before filtering.
            assert!(
                !output
                    .dependencies
                    .entities
                    .contains_key(&EntityDependency::Node(NodeId(6)))
            );
        }
    }
    println!("CANONICAL_TRAVERSAL_SEED_PASS");
    Ok(())
}

#[test]
fn parallel_trail_reduction_matches_busy_serial_and_global_budget() -> Result<()> {
    let _control = CPU_POOL_TEST_CONTROL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let graph = fixture()?;
    let label = graph
        .catalog()
        .label("Seed")
        .ok_or_else(|| Error::internal("label"))?;
    let value = graph
        .catalog()
        .property("value")
        .ok_or_else(|| Error::internal("value"))?;
    let body = graph.catalog().intern_property("body")?;
    let kind = graph
        .catalog()
        .relationship_type("R")
        .ok_or_else(|| Error::internal("type"))?;
    let payload: Arc<str> = "complete original content ".repeat(100_000).into();
    for id in 9..=72 {
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
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(id),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }
    let query = "MATCH (n:Seed)-[:R]->()-[:R]->(m) WHERE n.value=42 RETURN count(m)";
    for id in 73..=32768 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(value, ScalarValue::Integer(1))],
        })?;
    }
    let expected = QueryEngine.execute(&format!("{query} + 0"), &mut context(&graph))?;
    assert_eq!(count(&expected), 134);
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&payload);
    let pool = integer_scan_pool().ok_or_else(|| Error::internal("CPU pool unavailable"))?;
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let permit = loop {
        if let Some(permit) = pool.try_claim() {
            break permit;
        }
        assert!(Instant::now() < deadline, "pool remained occupied");
        std::thread::yield_now();
    };
    let before = canonical_trail_count::parallel_reductions();
    let serial = QueryEngine.execute(query, &mut context(&graph))?;
    assert_eq!(count(&serial), 134);
    assert_eq!(canonical_trail_count::parallel_reductions(), before);
    drop(permit);
    loop {
        let output = QueryEngine.execute(query, &mut context(&graph))?;
        assert_eq!(count(&output), 134);
        assert_eq!(output.dependencies.entities, serial.dependencies.entities);
        if canonical_trail_count::parallel_reductions() > before {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "parallel path never acquired its pool"
        );
        std::thread::yield_now();
    }
    let mut limited = context(&graph);
    limited.max_result_rows = 100;
    assert_eq!(
        QueryEngine
            .execute(query, &mut limited)
            .expect_err("workers bypassed total budget")
            .code,
        ErrorCode::ResultBudgetExceeded
    );
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&payload), owners);
    println!("CANONICAL_PARALLEL_TRAILS_PASS");
    Ok(())
}

#[test]
fn traversal_seed_mixed_numeric_fallback_and_recycled_rows() -> Result<()> {
    let graph = fixture()?;
    let label = graph
        .catalog()
        .label("Seed")
        .ok_or_else(|| Error::internal("label"))?;
    let value = graph
        .catalog()
        .property("value")
        .ok_or_else(|| Error::internal("property"))?;
    let kind = graph
        .catalog()
        .relationship_type("R")
        .ok_or_else(|| Error::internal("type"))?;
    graph.insert_node(NodeInput {
        id: NodeId(9),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![label],
        properties: vec![(value, ScalarValue::Float(42.0.into()))],
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(8),
        source: NodeId(9),
        target: NodeId(8),
        relationship_type: kind,
        layer: Layer::Observed,
        revision: 1,
        properties: Vec::new(),
    })?;
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
        &mut context(&graph),
    )?;
    assert_eq!(count(&output), 4);
    assert!(
        output
            .dependencies
            .entities
            .contains_key(&EntityDependency::Node(NodeId(6)))
    );
    graph.delete_node(NodeId(9), true, 2)?;
    graph.insert_node(NodeInput {
        id: NodeId(10),
        layer: Layer::Observed,
        revision: 3,
        labels: vec![label],
        properties: vec![(value, ScalarValue::Integer(1))],
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(9),
        source: NodeId(10),
        target: NodeId(8),
        relationship_type: kind,
        layer: Layer::Observed,
        revision: 3,
        properties: Vec::new(),
    })?;
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
        &mut context(&graph),
    )?;
    assert_eq!(count(&output), 3);
    assert!(
        !output
            .dependencies
            .entities
            .contains_key(&EntityDependency::Node(NodeId(10)))
    );
    Ok(())
}

#[test]
fn traversal_seed_overlay_journal_errors_budget_and_cancellation() -> Result<()> {
    let graph = fixture()?;
    let mut first = context(&graph);
    first.capabilities.write = true;
    first.capabilities.schema = true;
    let pending = QueryEngine.execute(
        "CREATE (:Seed {value:42})-[:R]->(:Seed {value:1}) RETURN 1",
        &mut first,
    )?;
    let mut second = context(&graph);
    second.prior_graph_mutations = &pending.graph_mutations;
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
        &mut second,
    )?;
    assert_eq!(count(&output), 4);
    assert_eq!(graph.node_count(), 8);
    let mut inline = context(&graph);
    inline.capabilities.write = true;
    inline.capabilities.schema = true;
    let output=QueryEngine.execute("CREATE (:Seed {value:42})-[:R]->(:Seed {value:1}) WITH 1 AS created MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",&mut inline)?;
    assert_eq!(count(&output), 4);
    let mut limited = context(&graph);
    limited.max_result_rows = 2;
    let error = QueryEngine
        .execute(
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
            &mut limited,
        )
        .expect_err("actual expansion bypassed budget");
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    let error = QueryEngine
        .execute(
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN 1 / 0",
            &mut context(&graph),
        )
        .expect_err("projection error was skipped");
    assert_eq!(error.code, ErrorCode::QueryType);
    let mut cancelled = context(&graph);
    cancelled.cancellation.cancel();
    let error = QueryEngine
        .execute(
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
            &mut cancelled,
        )
        .expect_err("cancelled traversal executed");
    assert_eq!(error.code, ErrorCode::Cancelled);
    Ok(())
}

#[test]
fn traversal_seed_captured_integer_column_promotes_without_losing_equal_float() -> Result<()> {
    let graph = fixture()?;
    let property = graph
        .catalog()
        .property("value")
        .ok_or_else(|| Error::internal("property"))?;
    assert!(graph.node_property_is_integer(property));
    let reader = graph
        .node_property_reader(property)
        .ok_or_else(|| Error::internal("column"))?;
    let node = graph
        .node(NodeId(1))
        .ok_or_else(|| Error::internal("node"))?;
    assert_eq!(reader.get_integer(node), Some(42));
    // This is the exact interleaving after integer eligibility/column capture, before its read.
    let shared = graph.clone();
    std::thread::scope(|scope| {
        scope
            .spawn(move || {
                shared.set_node_property(NodeId(1), property, ScalarValue::Float(42.0.into()), 2)
            })
            .join()
            .expect("writer panicked")
    })?;
    assert_eq!(reader.get_integer(node), None);
    assert!(canonical_seed_integer_may_match(
        reader.get_integer(node),
        42
    ));
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
        &mut context(&graph),
    )?;
    assert_eq!(count(&output), 3);
    assert!(!canonical_seed_integer_may_match(Some(41), 42));
    Ok(())
}

#[test]
fn traversal_seed_start_pattern_errors_and_native_guard_retain_fallback() -> Result<()> {
    let graph = fixture()?;
    graph.catalog().intern_property("x")?;
    let error = QueryEngine
        .execute(
            "MATCH (n:Seed {x:1 / 0})-[:R]->(m) WHERE n.value=99 RETURN count(m)",
            &mut context(&graph),
        )
        .expect_err("early selection suppressed start-pattern error");
    assert_eq!(error.code, ErrorCode::QueryType);
    let mut native = context(&graph);
    native.capabilities.require_native_execution = true;
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
        &mut native,
    )?;
    assert_eq!(count(&output), 3);
    assert!(
        output
            .dependencies
            .entities
            .contains_key(&EntityDependency::Node(NodeId(6)))
    );
    Ok(())
}

#[test]
fn traversal_seed_temporal_current_and_history_keep_scalar_semantics() -> Result<()> {
    let graph = fixture()?;
    let value = graph
        .catalog()
        .property("value")
        .ok_or_else(|| Error::internal("property"))?;
    let label = graph
        .catalog()
        .label("Seed")
        .ok_or_else(|| Error::internal("label"))?;
    let temporal = TemporalStore::default();
    let mut empty = context(&graph);
    empty.temporal = Some(&temporal);
    let output = QueryEngine.execute(
        "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
        &mut empty,
    )?;
    assert_eq!(count(&output), 3);
    assert!(
        !output
            .dependencies
            .entities
            .contains_key(&EntityDependency::Node(NodeId(6)))
    );
    graph.set_node_property(NodeId(1), value, ScalarValue::Integer(1), 2)?;
    temporal.declare(
        crate::graph::TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: label.0,
            property: value,
            value_type: crate::graph::TemporalType::Integer,
            retention_nanos: 1_000_000_000,
        },
        20,
    )?;
    for (event_time_nanos, sequence_index, value) in [(10, 1, 42), (20, 2, 43)] {
        temporal.append(
            EntityKind::Node,
            label.0,
            TemporalSample {
                entity_id: 1,
                property: graph
                    .catalog()
                    .property("value")
                    .ok_or_else(|| Error::internal("property"))?,
                event_time_nanos,
                sequence_index,
                value: ScalarValue::Integer(value),
            },
            20,
        )?;
    }
    for (source, expected) in [
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value=43 RETURN count(m)",
            3,
        ),
        (
            "MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
            0,
        ),
        (
            "AT TIME 10 MATCH (n:Seed)-[:R]->(m) WHERE n.value=42 RETURN count(m)",
            3,
        ),
        (
            "AT TIME 10 MATCH (n:Seed)-[:R]->(m) WHERE n.value=43 RETURN count(m)",
            0,
        ),
    ] {
        let mut query = context(&graph);
        query.temporal = Some(&temporal);
        let output = QueryEngine.execute(source, &mut query)?;
        assert_eq!(count(&output), expected, "{source}");
    }
    Ok(())
}
