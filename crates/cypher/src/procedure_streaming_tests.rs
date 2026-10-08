use super::*;

#[test]
fn canonical_relationship_stream_matches_general_rows_and_keeps_batches_bounded() -> Result<()> {
    let graph = fixture()?;
    let bytes = graph.resident_bytes();
    for query in [
        "MATCH (a)-[r]->(b) RETURN a, r, b",
        "MATCH (a)<-[r:LINK]-(b) RETURN id(a) AS source, id(r) AS relationship, id(b) AS target",
        "MATCH (a)-[r:MISSING]->(b) RETURN a, r, b",
        "MATCH (a:Missing)-[r]->(b) RETURN a, r, b",
        "MATCH (a)-[r]->(a) RETURN a, r",
        "MATCH (a)-[r]->(b) RETURN id(a), id(r), id(b) LIMIT 0",
        "MATCH (a)-[r]->(b) RETURN id(a), id(r), id(b) LIMIT 2",
        "MATCH (a)-[r]->(b) WHERE id(a) = 3 RETURN a, r, b",
        "MATCH (a)-[r]-(b) RETURN id(a), id(r), id(b)",
    ] {
        let expected = QueryEngine.execute_unoptimized(query, &mut context(&graph))?;
        let mut batches = Vec::new();
        let actual = QueryEngine.execute_streaming(query, &mut context(&graph), &mut |item| {
            if let ExecutionStreamItem::Batch(batch) = item {
                assert!(batch.row_count <= 3);
                batches.push(batch);
            }
            Ok(())
        })?;
        assert!(
            actual.result.batches.is_empty(),
            "yielded results must not be retained"
        );
        let records = |batches: &[ResultBatch]| {
            let mut records = batches
                .iter()
                .flat_map(|batch| {
                    (0..batch.row_count).map(|row| {
                        batch
                            .columns
                            .iter()
                            .map(|column| format!("{:?}", column.values[row]))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            records.sort();
            records
        };
        assert_eq!(
            records(&batches),
            records(&expected.result.batches),
            "{query}"
        );
        assert_eq!(
            graph.resident_bytes(),
            bytes,
            "no graph image or cache retained"
        );
    }
    let mut context = context(&graph);
    let error = QueryEngine
        .execute_streaming(
            "MATCH (a)-[r]->(b) RETURN a, r, b",
            &mut context,
            &mut |_| Err(Error::new(ErrorCode::Cancelled, "consumer stopped")),
        )
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Cancelled);
    Ok(())
}

#[test]
fn parallel_triangle_ranges_share_dirty_canonical_storage_and_match_busy_serial() -> Result<()> {
    let _control = CPU_POOL_TEST_CONTROL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let graph = fixture()?;
    let body = graph
        .catalog()
        .property("body")
        .ok_or_else(|| Error::internal("body missing"))?;
    let text: Arc<str> = "complete canonical owner content ".repeat(32768).into();
    for id in 9..=32770 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(body, ScalarValue::String(Arc::clone(&text)))],
        })?;
    }
    let bytes = graph.resident_bytes();
    let owners = Arc::strong_count(&text);
    let query =
        "USE LAYER OBSERVED CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount";
    let pool = integer_scan_pool().ok_or_else(|| Error::internal("CPU pool unavailable"))?;
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let permit = loop {
        if let Some(permit) = pool.try_claim() {
            break permit;
        }
        if Instant::now() >= deadline {
            return Err(Error::internal("CPU pool stayed busy"));
        }
        std::thread::yield_now();
    };
    let before = PARALLEL_TRIANGLE_REDUCTIONS.with(std::cell::Cell::get);
    let serial = execute(&graph, query)?;
    assert_eq!(column(&serial, "triangleCount"), vec![integer(1)]);
    assert_eq!(
        PARALLEL_TRIANGLE_REDUCTIONS.with(std::cell::Cell::get),
        before
    );
    drop(permit);
    loop {
        let parallel = execute(&graph, query)?;
        assert_eq!(
            column(&parallel, "triangleCount"),
            column(&serial, "triangleCount")
        );
        if PARALLEL_TRIANGLE_REDUCTIONS.with(std::cell::Cell::get) > before {
            break;
        }
        if Instant::now() >= deadline {
            return Err(Error::internal("parallel triangle path unavailable"));
        }
        std::thread::yield_now();
    }
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&text), owners);
    let query = "USE LAYER OBSERVED CALL graph.clusteringcoefficient() YIELD node, coefficient RETURN count(*) AS nodes, sum(coefficient) AS coefficients";
    let run = || {
        let mut context = context(&graph);
        context.max_result_rows = graph.node_count();
        QueryEngine.execute(query, &mut context)
    };
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    let permit = loop {
        if let Some(permit) = pool.try_claim() {
            break permit;
        }
        if Instant::now() >= deadline {
            return Err(Error::internal("CPU pool stayed busy before clustering"));
        }
        std::thread::yield_now();
    };
    let before = PARALLEL_CLUSTERING_REDUCTIONS.with(std::cell::Cell::get);
    let serial = run()?;
    assert_eq!(
        column(&serial, "nodes"),
        vec![integer(
            graph.node_count_in_layers(crate::graph::LayerMask::OBSERVED) as i64
        )]
    );
    assert_eq!(
        PARALLEL_CLUSTERING_REDUCTIONS.with(std::cell::Cell::get),
        before
    );
    drop(permit);
    loop {
        let parallel = run()?;
        assert_eq!(column(&parallel, "nodes"), column(&serial, "nodes"));
        assert_eq!(
            column(&parallel, "coefficients"),
            column(&serial, "coefficients")
        );
        if PARALLEL_CLUSTERING_REDUCTIONS.with(std::cell::Cell::get) > before {
            break;
        }
        if Instant::now() >= deadline {
            return Err(Error::internal("parallel clustering path unavailable"));
        }
        std::thread::yield_now();
    }
    assert_eq!(graph.resident_bytes(), bytes);
    assert_eq!(Arc::strong_count(&text), owners);
    println!("CANONICAL_PARALLEL_TRIANGLES_PASS");
    println!("CANONICAL_PARALLEL_CLUSTERING_PASS");
    Ok(())
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
    let body = graph.catalog().intern_property("body")?;
    let weight = graph.catalog().intern_property("weight")?;
    let kind = graph.catalog().intern_relationship_type("LINK")?;
    let text: Arc<str> = "complete owner text ".repeat(32768).into();
    for id in 1..=8 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: if id == 8 {
                Layer::Workspace
            } else {
                Layer::Observed
            },
            revision: 1,
            labels: Vec::new(),
            properties: vec![(body, ScalarValue::String(Arc::clone(&text)))],
        })?;
    }
    for (id, source, target) in [
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 1),
        (4, 3, 4),
        (5, 5, 6),
        (6, 4, 8),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: kind,
            layer: if id == 6 {
                Layer::Workspace
            } else {
                Layer::Observed
            },
            revision: 1,
            properties: vec![(weight, ScalarValue::Integer(1))],
        })?;
    }
    Ok(graph)
}

fn execute(graph: &GraphStore, source: &str) -> Result<ExecutionOutput> {
    QueryEngine.execute(source, &mut context(graph))
}

fn rows(output: &ExecutionOutput) -> usize {
    output
        .result
        .batches
        .iter()
        .map(|batch| batch.row_count)
        .sum()
}

fn column(output: &ExecutionOutput, name: &str) -> Vec<ResultValue> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| {
            batch
                .columns
                .iter()
                .filter(move |column| column.name == name)
                .flat_map(|column| column.values.iter().cloned())
        })
        .collect()
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn procedure_state(graph: &GraphStore) -> ExecutionState<'_> {
    ExecutionState {
        graph: GraphReadView::new(graph),
        temporal: TemporalReadView::new(None),
        mutations: Vec::new(),
        temporal_mutations: Vec::new(),
        dependencies: TransactionDependencies::default(),
        next_node_id: 100,
        next_edge_id: 100,
        revision: 2,
        created_at: None,
        read_layers: crate::graph::LayerMask::OBSERVED,
        write_layer: Layer::Observed,
        stats: StatementStats::default(),
        final_columns: Vec::new(),
        scope_columns: BTreeSet::new(),
        administrative: None,
        vector_searches: Vec::new(),
        resident_backend: None,
        allow_context_backend: false,
        existential_subquery_depth: 0,
        initial_scan_cap: None,
    }
}

fn assert_count(graph: &GraphStore, source: &str, expected: i64) -> Result<()> {
    let output = execute(graph, source)?;
    assert_eq!(rows(&output), 1, "{source}");
    assert_eq!(
        output.result.schema,
        vec![("counted".to_owned(), ColumnType::Integer)],
        "{source}"
    );
    assert_eq!(
        column(&output, "counted"),
        vec![integer(expected)],
        "{source}"
    );
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn shortest_path_reuses_neighbors_and_preserves_parallel_relationship_ties() -> Result<()> {
    let graph = fixture()?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(100),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: graph.catalog().relationship_type("LINK").unwrap(),
        layer: Layer::Observed,
        revision: 2,
        properties: Vec::new(),
    })?;
    let bytes = graph.resident_bytes();
    let output = execute(
        &graph,
        "USE LAYER OBSERVED CALL graph.shortestpath(1, 4) YIELD path, cost RETURN path, cost",
    )?;
    let paths = column(&output, "path");
    let [
        ResultValue::Path {
            nodes,
            relationships,
        },
    ] = paths.as_slice()
    else {
        panic!("expected one path");
    };
    assert_eq!(
        nodes.iter().map(|node| node.id).collect::<Vec<_>>(),
        vec![NodeId(1), NodeId(2), NodeId(3), NodeId(4)]
    );
    assert_eq!(
        relationships.iter().map(|edge| edge.id).collect::<Vec<_>>(),
        vec![EdgeId(1), EdgeId(2), EdgeId(4)]
    );
    assert_eq!(column(&output, "cost"), vec![integer(3)]);
    assert_eq!(graph.resident_bytes(), bytes);
    Ok(())
}

#[test]
fn procedure_streaming_parity_all_actual_algorithms_and_disconnected_layers() -> Result<()> {
    let graph = fixture()?;
    let before = graph.resident_bytes();
    for (call, yielded, projection, expected) in [
        (
            "graph.degree()",
            "node, outDegree, inDegree, degree",
            "id(node) AS id, outDegree, inDegree, degree ORDER BY id",
            7,
        ),
        (
            "graph.bfs(1)",
            "node, distance",
            "id(node) AS id, distance ORDER BY id",
            4,
        ),
        (
            "graph.dfs(1)",
            "node, order",
            "id(node) AS id, order ORDER BY order",
            4,
        ),
        ("graph.shortestpath(1, 4)", "path, cost", "path, cost", 1),
        (
            "graph.dijkstra(1, 'weight')",
            "node, cost, predecessor",
            "id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
            4,
        ),
        (
            "graph.wcc()",
            "node, component",
            "id(node) AS id, component ORDER BY id",
            7,
        ),
        (
            "graph.scc()",
            "node, component",
            "id(node) AS id, component ORDER BY id",
            7,
        ),
        (
            "graph.pagerank()",
            "node, score",
            "id(node) AS id, score ORDER BY id",
            7,
        ),
        ("graph.trianglecount()", "triangleCount", "triangleCount", 1),
        (
            "graph.clusteringcoefficient()",
            "node, coefficient",
            "id(node) AS id, coefficient ORDER BY id",
            7,
        ),
        (
            "graph.kcore()",
            "node, core",
            "id(node) AS id, core ORDER BY id",
            7,
        ),
        (
            "graph.louvain()",
            "node, community",
            "id(node) AS id, community ORDER BY id",
            7,
        ),
    ] {
        let ordinary =
            format!("USE LAYER OBSERVED CALL {call} YIELD {yielded} RETURN {projection}");
        let output = execute(&graph, &ordinary)?;
        assert_eq!(rows(&output), expected, "{ordinary}");
        let counted =
            format!("USE LAYER OBSERVED CALL {call} YIELD {yielded} RETURN count(*) AS counted");
        let (plan, _) = prepared_plan(&counted, &context(&graph), None)?;
        assert!(
            matches!(plan.operators.as_slice(), [PhysicalOperator::BuiltinProcedure(_), PhysicalOperator::Project { keep_scope: false, projection }] if projection_is_only_count_star(projection)),
            "parity fixture did not exercise the terminal procedure count sink: {counted}"
        );
        assert_count(
            &graph,
            &counted,
            i64::try_from(expected).map_err(|_| Error::internal("fixture cardinality"))?,
        )?;
        assert!(output.graph_mutations.is_empty());
        if call == "graph.trianglecount()" {
            assert_eq!(column(&output, "triangleCount"), vec![integer(1)]);
        }
        if call == "graph.bfs(1)" {
            assert_eq!(
                column(&output, "id"),
                vec![integer(1), integer(2), integer(3), integer(4)]
            );
            assert_eq!(
                column(&output, "distance"),
                vec![integer(0), integer(1), integer(2), integer(3)]
            );
        }
    }
    assert_count(
        &graph,
        "USE LAYER OBSERVED CALL graph.shortestpath(1, 6) YIELD path RETURN count(*) AS counted",
        0,
    )?;
    assert_eq!(graph.resident_bytes(), before);
    assert_eq!(graph.node_count(), 8);
    assert_eq!(graph.edge_count(), 6);
    println!("PROCEDURE_STREAMING_PARITY_PASS");
    Ok(())
}

#[test]
fn procedure_streaming_contract_aliases_empty_filters_and_fallbacks() -> Result<()> {
    let graph = fixture()?;
    let alias = execute(
        &graph,
        "USE LAYER OBSERVED CALL graph.degree() YIELD node AS owner RETURN count(*) AS first, count(*) AS second",
    )?;
    assert_eq!(column(&alias, "first"), vec![integer(7)]);
    assert_eq!(column(&alias, "second"), vec![integer(7)]);
    for (query, expected) in [
        (
            "CALL graph.degree() YIELD node RETURN count(node) AS counted",
            7,
        ),
        (
            "CALL graph.degree() YIELD node RETURN count(DISTINCT node) AS counted",
            7,
        ),
        (
            "CALL graph.degree() YIELD degree WHERE degree > 0 RETURN count(*) AS counted",
            6,
        ),
        (
            "CALL graph.degree() YIELD node RETURN count(*) + 1 AS counted",
            8,
        ),
        (
            "UNWIND [1, 2] AS input CALL graph.degree() YIELD node RETURN count(*) AS counted",
            14,
        ),
    ] {
        assert_count(&graph, &format!("USE LAYER OBSERVED {query}"), expected)?;
    }
    let grouped = execute(
        &graph,
        "USE LAYER OBSERVED CALL graph.degree() YIELD degree RETURN degree, count(*) AS counted ORDER BY degree",
    )?;
    assert_eq!(
        column(&grouped, "degree"),
        vec![integer(0), integer(1), integer(2), integer(3)]
    );
    assert_eq!(
        column(&grouped, "counted"),
        vec![integer(1), integer(3), integer(2), integer(1)]
    );
    let empty = GraphStore::default();
    for call in [
        "graph.degree()",
        "graph.wcc()",
        "graph.scc()",
        "graph.pagerank()",
        "graph.clusteringcoefficient()",
        "graph.kcore()",
        "graph.louvain()",
    ] {
        assert_count(
            &empty,
            &format!("CALL {call} YIELD node RETURN count(*) AS counted"),
            0,
        )?;
    }
    assert_count(
        &empty,
        "CALL graph.trianglecount() YIELD triangleCount RETURN count(*) AS counted",
        1,
    )?;
    println!("PROCEDURE_STREAMING_CONTRACT_PASS");
    Ok(())
}

#[test]
fn procedure_streaming_contract_count_executes_errors_and_exact_row_budget() -> Result<()> {
    let graph = fixture()?;
    let weight = graph
        .catalog()
        .property("weight")
        .ok_or_else(|| Error::internal("weight"))?;
    for value in [
        ScalarValue::Integer(-1),
        ScalarValue::String("invalid weight".into()),
    ] {
        graph.set_edge_property(EdgeId(1), weight, value, 2)?;
        for terminal in ["id(node) AS id", "count(*) AS counted"] {
            let query = format!(
                "USE LAYER OBSERVED CALL graph.dijkstra(1, 'weight') YIELD node RETURN {terminal}"
            );
            assert!(
                execute(&graph, &query).is_err(),
                "COUNT bypassed Dijkstra: {query}"
            );
        }
    }
    for terminal in ["score", "count(*) AS counted"] {
        let query = format!(
            "USE LAYER OBSERVED CALL graph.pagerank(0.85, 0.0001, 0) YIELD score RETURN {terminal}"
        );
        assert!(
            execute(&graph, &query).is_err(),
            "COUNT bypassed PageRank: {query}"
        );
    }
    for budget in [6, 7] {
        for terminal in ["id(node) AS id", "count(*) AS counted"] {
            let mut execution = context(&graph);
            execution.max_result_rows = budget;
            let result = QueryEngine.execute(
                &format!("USE LAYER OBSERVED CALL graph.degree() YIELD node RETURN {terminal}"),
                &mut execution,
            );
            if budget == 6 {
                assert_eq!(
                    result
                        .expect_err("intermediate procedure budget was bypassed")
                        .code,
                    ErrorCode::ResultBudgetExceeded
                );
            } else {
                let output = result?;
                assert_eq!(
                    rows(&output),
                    if terminal.starts_with("count") { 1 } else { 7 }
                );
            }
        }
    }
    println!("PROCEDURE_STREAMING_CONTRACT_PASS");
    Ok(())
}

#[test]
fn procedure_streaming_contract_mutation_journal_and_owner_text_stay_canonical() -> Result<()> {
    let graph = fixture()?;
    let seen = graph.catalog().intern_property("seen")?;
    let mut execution = context(&graph);
    execution.capabilities.write = true;
    let output = QueryEngine.execute("USE LAYER OBSERVED WRITE LAYER OBSERVED CALL graph.degree() YIELD node SET node.seen = true RETURN count(*) AS counted", &mut execution)?;
    assert_eq!(column(&output, "counted"), vec![integer(7)]);
    assert_eq!(output.graph_mutations.iter().filter(|mutation| matches!(mutation, GraphMutation::SetNodeProperty { property, .. } if *property == seen)).count(), 7);
    assert_eq!(
        graph.node(NodeId(1)).and_then(|node| node.property(seen)),
        None
    );
    let mut read = context(&graph);
    read.prior_graph_mutations = &output.graph_mutations;
    let journal = QueryEngine.execute(
        "USE LAYER OBSERVED MATCH (node) WHERE node.seen = true RETURN count(node) AS counted",
        &mut read,
    )?;
    assert_eq!(column(&journal, "counted"), vec![integer(7)]);
    let body = graph
        .catalog()
        .property("body")
        .ok_or_else(|| Error::internal("body"))?;
    let Some(ScalarValue::String(original)) =
        graph.node(NodeId(1)).and_then(|node| node.property(body))
    else {
        return Err(Error::internal("canonical body absent"));
    };
    let materialized = execute(
        &graph,
        "USE LAYER OBSERVED CALL graph.degree() YIELD node RETURN node",
    )?;
    for value in column(&materialized, "node") {
        let ResultValue::Node(node) = value else {
            return Err(Error::internal("procedure node changed type"));
        };
        let Some(ScalarValue::String(text)) = node.properties.get("body") else {
            return Err(Error::internal("materialized body absent"));
        };
        assert!(
            Arc::ptr_eq(text, &original),
            "procedure copied the complete source text"
        );
    }
    println!("PROCEDURE_STREAMING_CONTRACT_PASS");
    Ok(())
}

#[test]
fn procedure_streaming_contract_empty_window_marker_keeps_actual_emit_budget() -> Result<()> {
    let graph = fixture()?;
    let mut execution = context(&graph);
    let (plan, _) = prepared_plan(
        "USE LAYER OBSERVED CALL graph.degree() YIELD node RETURN count(*) AS counted",
        &execution,
        None,
    )?;
    let [
        PhysicalOperator::BuiltinProcedure(call),
        PhysicalOperator::Project { projection, .. },
    ] = plan.operators.as_slice()
    else {
        return Err(Error::internal("terminal procedure fixture plan changed"));
    };
    let marker = Row::from([(INTERNAL_WINDOW_EMPTY.to_owned(), BindingValue::Null)]);
    let mut generic_state = procedure_state(&graph);
    let ResolvedProcedure::Builtin(signature) =
        validate_call(call, None, Some(&execution.parameters))?
    else {
        return Err(Error::internal("degree is not builtin"));
    };
    let generic = builtin_graph_procedure(
        vec![marker.clone()],
        call,
        signature,
        &mut generic_state,
        &execution,
    )?;
    assert_eq!(generic.len(), 7);
    assert_eq!(count_star_rows(&generic, &execution)?, 0);
    for budget in [6, 7] {
        execution.max_result_rows = budget;
        let mut state = procedure_state(&graph);
        let result = builtin_graph_procedure_count(
            vec![marker.clone()],
            call,
            projection,
            &mut state,
            &execution,
        );
        if budget == 6 {
            assert_eq!(
                result
                    .expect_err("empty-window markers bypassed actual procedure emission budget")
                    .code,
                ErrorCode::ResultBudgetExceeded
            );
        } else {
            let output = result?;
            assert_eq!(output.len(), 1);
            assert!(
                matches!(output[0].get("counted"), Some(BindingValue::Value(value)) if *value == integer(0))
            );
        }
    }
    let mut marker_alias = call.clone();
    marker_alias.yields[0].alias = Some(INTERNAL_WINDOW_EMPTY.to_owned());
    let aliased = builtin_graph_procedure(
        vec![Row::new()],
        &marker_alias,
        signature,
        &mut procedure_state(&graph),
        &execution,
    )?;
    assert_eq!(aliased.len(), 7);
    assert_eq!(count_star_rows(&aliased, &execution)?, 0);
    for budget in [6, 7] {
        execution.max_result_rows = budget;
        let result = builtin_graph_procedure_count(
            vec![Row::new()],
            &marker_alias,
            projection,
            &mut procedure_state(&graph),
            &execution,
        );
        if budget == 6 {
            assert_eq!(
                result
                    .expect_err("marker alias bypassed actual emitted row budget")
                    .code,
                ErrorCode::ResultBudgetExceeded
            );
        } else {
            let output = result?;
            assert!(
                matches!(output[0].get("counted"), Some(BindingValue::Value(value)) if *value == integer(0))
            );
        }
    }
    execution.cancellation.cancel();
    assert_eq!(
        builtin_graph_procedure_count(
            Vec::new(),
            call,
            projection,
            &mut procedure_state(&graph),
            &execution
        )
        .expect_err("zero-emission count ignored caller cancellation")
        .code,
        ErrorCode::Cancelled
    );
    println!("PROCEDURE_STREAMING_CONTRACT_PASS");
    Ok(())
}

#[test]
fn procedure_streaming_contract_cancel_between_actual_emitted_fields() -> Result<()> {
    let graph = fixture()?;
    let execution = context(&graph);
    let (plan, _) = prepared_plan(
        "USE LAYER OBSERVED CALL graph.degree() YIELD node RETURN id(node)",
        &execution,
        None,
    )?;
    let call = plan
        .operators
        .iter()
        .find_map(|operator| match operator {
            PhysicalOperator::BuiltinProcedure(call) => Some(call),
            _ => None,
        })
        .ok_or_else(|| Error::internal("procedure call absent"))?;
    let state = ExecutionState {
        graph: GraphReadView::new(&graph),
        temporal: TemporalReadView::new(None),
        mutations: Vec::new(),
        temporal_mutations: Vec::new(),
        dependencies: TransactionDependencies::default(),
        next_node_id: 100,
        next_edge_id: 100,
        revision: 2,
        created_at: None,
        read_layers: crate::graph::LayerMask::OBSERVED,
        write_layer: Layer::Observed,
        stats: StatementStats::default(),
        final_columns: Vec::new(),
        scope_columns: BTreeSet::new(),
        administrative: None,
        vector_searches: Vec::new(),
        resident_backend: None,
        allow_context_backend: false,
        existential_subquery_depth: 0,
        initial_scan_cap: None,
    };
    let (outgoing, incoming, nodes, ordinals) =
        state.graph.algorithm_adjacency(state.read_layers)?;
    let mut emitted = 0;
    let error = execute_graph_procedure(
        "graph.degree",
        call,
        &Row::new(),
        &state,
        &execution,
        &outgoing,
        &incoming,
        &nodes,
        &ordinals,
        &mut |fields: &[(&'static str, BindingValue)]| {
            assert_eq!(fields.len(), 4);
            assert!(matches!(fields[0], ("node", BindingValue::Node(_))));
            emitted += 1;
            if emitted == 3 {
                execution.cancellation.cancel();
            }
            Ok(())
        },
    )
    .expect_err("procedure ignored cancellation between field emissions");
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert_eq!(emitted, 3);
    println!("PROCEDURE_STREAMING_CONTRACT_PASS");
    Ok(())
}
