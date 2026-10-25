use super::*;
use crate::graph::{GraphIndexDefinition, GraphIndexKind};

fn context<'a>(graph: &'a GraphStore, indexes: &'a IndexCatalog) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: Some(indexes),
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 1,
            index: graph.revision(),
        },
        resolved_time_nanos: 0,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: usize::MAX,
        max_batch_rows: 1024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
        next_node_id: graph.node_slot_count() as u64 + 100,
        next_edge_id: graph.edge_slot_count() as u64 + 100,
        mutation_revision: graph.revision() + 1,
    }
}

fn fixture(size: u64) -> Result<(GraphStore, IndexCatalog)> {
    let graph = GraphStore::default();
    let label = graph.catalog().intern_label("Code")?;
    let sid = graph.catalog().intern_property("sid")?;
    let body = graph.catalog().intern_property("body")?;
    let text: Arc<str> = "canonical payload ".repeat(4096).into();
    for id in 1..=size {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![
                (sid, ScalarValue::Integer(id as i64)),
                (body, ScalarValue::String(Arc::clone(&text))),
            ],
        })?;
    }
    let indexes = IndexCatalog::default();
    indexes.create(
        &graph,
        GraphIndexDefinition {
            name: "code_sid".into(),
            kind: GraphIndexKind::Equality,
            label,
            properties: vec![sid],
            unique: false,
        },
    )?;
    Ok((graph, indexes))
}

fn rows(count: usize, size: u64, payload: usize) -> ResultValue {
    ResultValue::List(
        (0..count)
            .map(|i| {
                ResultValue::Map(BTreeMap::from([
                    (
                        "source".into(),
                        ResultValue::Scalar(ScalarValue::Integer((i as u64 % size + 1) as i64)),
                    ),
                    (
                        "target".into(),
                        ResultValue::Scalar(ScalarValue::Integer(
                            ((i as u64 + 1) % size + 1) as i64,
                        )),
                    ),
                    (
                        "payload".into(),
                        ResultValue::List(
                            (0..payload)
                                .map(|j| ResultValue::Scalar(ScalarValue::Integer(j as i64)))
                                .collect(),
                        ),
                    ),
                ]))
            })
            .collect(),
    )
}

fn count(output: &ExecutionOutput) -> i64 {
    let ResultValue::Scalar(ScalarValue::Integer(value)) =
        output.result.batches[0].columns[0].values[0]
    else {
        panic!("expected count");
    };
    value
}

fn state<'a>(context: &ExecutionContext<'a>) -> Result<ExecutionState<'a>> {
    Ok(ExecutionState {
        graph: transaction_graph_view(context)?,
        temporal: transaction_temporal_view(context)?,
        mutations: Vec::new(),
        temporal_mutations: Vec::new(),
        dependencies: TransactionDependencies::default(),
        next_node_id: context.next_node_id,
        next_edge_id: context.next_edge_id,
        revision: context.mutation_revision,
        created_at: None,
        read_layers: crate::graph::LayerMask::all(),
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
    })
}

#[test]
fn indexed_endpoint_matching_preserves_prior_transaction_changes() -> Result<()> {
    let (graph, indexes) = fixture(8)?;
    let sid = graph.catalog().property("sid").unwrap();
    let label = graph.catalog().label("Code").unwrap();
    let mutations = vec![
        GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property: sid,
            value: ScalarValue::Integer(99),
            revision: 2,
        },
        GraphMutation::DeleteNode {
            node: NodeId(2),
            detach: false,
            revision: 2,
        },
        GraphMutation::InsertNode(NodeInput {
            id: NodeId(9),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![(sid, ScalarValue::Integer(99))],
        }),
    ];
    let mut ctx = context(&graph, &indexes);
    ctx.prior_graph_mutations = &mutations;
    let view = state(&ctx)?;
    let pattern = NodePattern {
        variable: Some("n".into()),
        labels: vec!["Code".into()],
        properties: vec![("sid".into(), Expression::Literal(ScalarValue::Integer(99)))],
        property_predicate_present: true,
    };
    let candidates = host_index_candidates(
        &Row::new(),
        &pattern,
        &[label],
        &ScanAccessPath::Unspecified,
        &view,
        &ctx,
    )?;
    assert!(candidates.is_some(), "prior writes disabled indexed lookup");
    let output = QueryEngine.execute(
        "UNWIND [1,2,99] AS sid MATCH (n:Code {sid: sid}) RETURN count(*)",
        &mut ctx,
    )?;
    assert_eq!(count(&output), 2);
    assert_eq!(graph.node_count(), 8);
    assert_eq!(
        graph.node(NodeId(1)).unwrap().property(sid),
        Some(ScalarValue::Integer(1))
    );
    Ok(())
}

#[test]
fn single_endpoint_lookup_preserves_payload_allocation() -> Result<()> {
    let (graph, indexes) = fixture(4096)?;
    let ctx = context(&graph, &indexes);
    let mut state = state(&ctx)?;
    let payload = vec![BindingValue::Value(ResultValue::Scalar(ScalarValue::Integer(7))); 65536];
    let pointer = payload.as_ptr();
    let row = Row::from([("payload".into(), BindingValue::List(payload))]);
    let plan = plan(bind_with_parameters(
        parse("MATCH (n:Code {sid: 7}) RETURN n")?,
        graph.catalog(),
        ctx.capabilities,
        &ctx.parameters,
    )?)?;
    let PhysicalOperator::ScanPattern {
        pattern, access, ..
    } = &plan.operators[0]
    else {
        panic!("expected node scan")
    };
    let output = match_pattern(vec![row], pattern, access, false, true, &mut state, &ctx)?;
    assert_eq!(output.len(), 1);
    let Some(BindingValue::List(payload)) = output[0].get("payload") else {
        panic!("missing payload")
    };
    assert_eq!(
        pointer,
        payload.as_ptr(),
        "endpoint match copied the payload"
    );
    Ok(())
}

#[test]
fn unwind_edge_write_keeps_complete_payloads() -> Result<()> {
    let (graph, indexes) = fixture(8)?;
    let mut ctx = context(&graph, &indexes);
    ctx.capabilities.write = true;
    ctx.capabilities.schema = true;
    ctx.parameters.insert("rows".into(), rows(17, 8, 1024));
    let output = QueryEngine.execute("UNWIND $rows AS row MATCH (a:Code {sid: row.source}), (b:Code {sid: row.target}) CREATE (a)-[:R {payload: row.payload}]->(b) RETURN count(*)", &mut ctx)?;
    assert_eq!(count(&output), 17);
    assert_eq!(output.result.statistics.relationships_created, 17);
    assert_eq!(graph.edge_count(), 0);
    let payload_id = output
        .graph_mutations
        .iter()
        .find_map(|mutation| match mutation {
            GraphMutation::DeclareProperty { name, id } if name == "payload" => Some(*id),
            _ => None,
        })
        .unwrap();
    for mutation in &output.graph_mutations {
        if let GraphMutation::InsertEdge(edge) = mutation {
            let value = edge
                .properties
                .iter()
                .find(|(property, _)| *property == payload_id)
                .unwrap()
                .1
                .clone();
            assert_eq!(
                ResultValue::from_property(value)?,
                ResultValue::List(
                    (0..1024)
                        .map(|value| ResultValue::Scalar(ScalarValue::Integer(value)))
                        .collect()
                )
            );
        }
    }
    assert_eq!(
        output
            .graph_mutations
            .iter()
            .filter(|m| matches!(m, GraphMutation::InsertEdge(_)))
            .count(),
        17
    );
    Ok(())
}

#[test]
fn unwind_endpoint_fanout_missing_rows_and_empty_schema() -> Result<()> {
    let (graph, indexes) = fixture(8)?;
    let prior = [GraphMutation::InsertNode(NodeInput {
        id: NodeId(9),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![graph.catalog().label("Code").unwrap()],
        properties: vec![(
            graph.catalog().property("sid").unwrap(),
            ScalarValue::Integer(1),
        )],
    })];
    let mut ctx = context(&graph, &indexes);
    ctx.capabilities.write = true;
    ctx.capabilities.schema = true;
    ctx.prior_graph_mutations = &prior;
    let row = |source| {
        ResultValue::Map(BTreeMap::from([
            (
                "source".into(),
                ResultValue::Scalar(ScalarValue::Integer(source)),
            ),
            (
                "target".into(),
                ResultValue::Scalar(ScalarValue::Integer(3)),
            ),
        ]))
    };
    ctx.parameters.insert(
        "rows".into(),
        ResultValue::List(vec![row(1), row(1), row(99)]),
    );
    let output = QueryEngine.execute("UNWIND $rows AS row MATCH (a:Code {sid: row.source}), (b:Code {sid: row.target}) CREATE (a)-[:R]->(b) RETURN count(*)", &mut ctx)?;
    assert_eq!(count(&output), 4);
    assert_eq!(output.result.statistics.relationships_created, 4);
    ctx.parameters
        .insert("rows".into(), ResultValue::List(Vec::new()));
    let output = QueryEngine.execute(
        "UNWIND $rows AS row MATCH (a:Code {sid: row.source}), (b:Code {sid: row.target}) RETURN *",
        &mut ctx,
    )?;
    assert_eq!(
        output
            .result
            .schema
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b", "row"]
    );
    assert!(
        output
            .result
            .batches
            .iter()
            .all(|batch| batch.row_count == 0)
    );
    Ok(())
}

#[test]
#[ignore = "release performance measurement"]
fn unwind_write_profile() -> Result<()> {
    let size = 4096;
    let (graph, indexes) = fixture(size)?;
    let kind = graph.catalog().intern_relationship_type("EXISTING")?;
    let prior = [GraphMutation::InsertEdge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: kind,
        layer: Layer::Observed,
        revision: 2,
        properties: Vec::new(),
    })];
    for payload in [0, 256] {
        for dirty in [false, true] {
            for (stage, query) in [
                ("unwind", "UNWIND $rows AS row RETURN count(*)"),
                (
                    "match",
                    "UNWIND $rows AS row MATCH (a:Code {sid: row.source}), (b:Code {sid: row.target}) RETURN count(*)",
                ),
                (
                    "create",
                    "UNWIND $rows AS row MATCH (a:Code {sid: row.source}), (b:Code {sid: row.target}) CREATE (a)-[:R]->(b) RETURN count(*)",
                ),
            ] {
                let mut timings = Vec::new();
                for _ in 0..5 {
                    let mut ctx = context(&graph, &indexes);
                    ctx.capabilities.write = true;
                    ctx.capabilities.schema = true;
                    ctx.prior_graph_mutations = if dirty { &prior } else { &[] };
                    ctx.parameters
                        .insert("rows".into(), rows(512, size, payload));
                    let start = Instant::now();
                    let output = QueryEngine.execute(query, &mut ctx)?;
                    timings.push(start.elapsed().as_micros());
                    assert_eq!(count(&output), 512);
                    assert_eq!(
                        output.result.statistics.relationships_created,
                        if stage == "create" { 512 } else { 0 }
                    );
                }
                timings.sort_unstable();
                eprintln!(
                    "UNWIND_PROFILE nodes={size} rows=512 payload_items={payload} prior_edge={dirty} stage={stage} median_us={} samples_us={timings:?}",
                    timings[2]
                );
            }
        }
    }
    eprintln!("UNWIND_PROFILE_COMPLETE");
    Ok(())
}
