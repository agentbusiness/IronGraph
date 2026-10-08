use super::*;

fn context(graph: &GraphStore, project_id: ProjectId) -> ExecutionContext<'_> {
    ExecutionContext {
        project_id,
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
        next_node_id: 40_000,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1,
        max_batch_rows: 1,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

struct CancelAfterReduction;

impl CancelAfterReduction {
    fn enable() -> Self {
        INTEGER_SCAN_CANCEL_AFTER_REDUCTION.store(true, AtomicOrdering::Release);
        Self
    }
}

impl Drop for CancelAfterReduction {
    fn drop(&mut self) {
        INTEGER_SCAN_CANCEL_AFTER_REDUCTION.store(false, AtomicOrdering::Release);
    }
}

#[test]
fn range_scheduling_cancellation_after_actual_reduction_prevents_publication() -> Result<()> {
    let graph = GraphStore::default();
    let label = graph.catalog().intern_label("RangeScheduling")?;
    let property = graph.catalog().intern_property("value")?;
    // This crosses parallel admission and leaves an uneven final partition.
    for row in 0..32_769_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(row + 1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(
                property,
                ScalarValue::Integer(
                    i64::try_from(row % 1000)
                        .map_err(|error| Error::internal(error.to_string()))?,
                ),
            )],
        })?;
    }
    for (source, expected) in [
        (
            "MATCH (n:RangeScheduling) WHERE n.value >= 400 AND n.value < 600 RETURN count(n) AS matched",
            6600,
        ),
        (
            "MATCH (n:RangeScheduling) WHERE n.value > 1000 RETURN count(n) AS matched",
            0,
        ),
    ] {
        let mut normal = context(&graph, ProjectId(uuid::Uuid::nil()));
        let (plan, _) = prepared_plan(source, &normal, None)?;
        let output = stream_integer_node_aggregates(&plan, &normal, &mut None)?
            .ok_or_else(|| Error::internal("range scheduling fixture missed integer reduction"))?;
        assert_eq!(
            output.result.schema,
            vec![("matched".to_owned(), ColumnType::Integer)]
        );
        assert_eq!(
            output.result.batches[0].columns[0].values,
            vec![ResultValue::Scalar(ScalarValue::Integer(expected))]
        );
        assert!(output.dependencies.entities.is_empty());
        assert!(!output.dependencies.predicates.is_empty());

        let hook = CancelAfterReduction::enable();
        // The test hook is scoped to one reserved project and cannot cancel ordinary queries.
        let ordinary = QueryEngine.execute(source, &mut normal)?;
        assert_eq!(ordinary.result.batches, output.result.batches);
        assert!(!normal.cancellation.is_cancelled());
        assert!(INTEGER_SCAN_CANCEL_AFTER_REDUCTION.load(AtomicOrdering::Acquire));

        let cancelled = context(&graph, ProjectId(uuid::Uuid::from_u128(u128::MAX)));
        assert!(!cancelled.cancellation.is_cancelled());
        let mut emitted = 0_usize;
        let mut emit = |_item: ExecutionStreamItem| -> Result<()> {
            emitted += 1;
            Ok(())
        };
        let mut stream: Option<&mut dyn FnMut(ExecutionStreamItem) -> Result<()>> = Some(&mut emit);
        let result = stream_integer_node_aggregates(&plan, &cancelled, &mut stream);
        assert!(matches!(result, Err(error) if error.code == ErrorCode::Cancelled));
        assert!(cancelled.cancellation.is_cancelled());
        assert_eq!(emitted, 0, "cancelled reduction published schema or rows");
        assert!(!INTEGER_SCAN_CANCEL_AFTER_REDUCTION.load(AtomicOrdering::Acquire));
        drop(hook);

        // A fresh context still returns its scalar result after the one-shot hook was consumed.
        let fresh = QueryEngine.execute(source, &mut context(&graph, cancelled.project_id))?;
        assert_eq!(fresh.result.batches, output.result.batches);
    }
    assert_eq!(graph.node_slot_count(), 32_769);
    eprintln!("CANONICAL_RANGE_SCHEDULING_PASS");
    Ok(())
}
