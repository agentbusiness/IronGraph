// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::BTreeMap;

use irongraph::{
    Bookmark, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

fn context(graph: &GraphStore, write: bool, revision: u64) -> ExecutionContext<'_> {
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
            index: revision.saturating_sub(1),
        },
        mutation_revision: revision,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

#[test]
fn cpu_create_read_and_checkpoint_support_mixed_property_types() -> Result<()> {
    let mut graph = GraphStore::default();
    let created = QueryEngine.execute(
        "CREATE (a:Mixed {slot: 1, var: 0}), \
                (b:Mixed {slot: 2, var: 'xx'}), \
                (c:Mixed {slot: 3}) \
         RETURN a.var AS integer_value, b.var AS string_value, c.var AS missing_value",
        &mut context(&graph, true, 1),
    )?;
    let created_batch = &created.result.batches[0];
    assert_eq!(created_batch.row_count, 1);
    assert_eq!(
        created_batch.columns[0].values,
        vec![ResultValue::Scalar(ScalarValue::Integer(0))]
    );
    assert_eq!(
        created_batch.columns[1].values,
        vec![ResultValue::Scalar(ScalarValue::String("xx".into()))]
    );
    assert_eq!(
        created_batch.columns[2].values,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );
    for mutation in created.graph_mutations {
        graph.apply(mutation)?;
    }

    let property = graph
        .catalog()
        .property("var")
        .ok_or_else(|| irongraph::Error::internal("var property was not declared"))?;
    let snapshot = graph.snapshot()?;
    assert!(snapshot.node_properties.is_mixed(property));
    assert!(snapshot.node_properties.column(property).is_none());
    assert_eq!(
        graph.node_property_accepts(property, &ScalarValue::Boolean(true)),
        Some(true)
    );

    let read = QueryEngine.execute(
        "MATCH (n:Mixed) RETURN n.var AS value ORDER BY n.slot",
        &mut context(&graph, false, 2),
    )?;
    assert_eq!(
        read.result.batches[0].columns[0].values,
        vec![
            ResultValue::Scalar(ScalarValue::Integer(0)),
            ResultValue::Scalar(ScalarValue::String("xx".into())),
            ResultValue::Scalar(ScalarValue::Null),
        ]
    );

    let postcard = postcard::to_stdvec(&graph)
        .map_err(|error| irongraph::Error::internal(format!("postcard failed: {error}")))?;
    let restored: GraphStore = postcard::from_bytes(&postcard)
        .map_err(|error| irongraph::Error::internal(format!("postcard failed: {error}")))?;
    let read = QueryEngine.execute(
        "MATCH (n:Mixed) RETURN n.var AS value ORDER BY n.slot",
        &mut context(&restored, false, 2),
    )?;
    assert_eq!(
        read.result.batches[0].columns[0].values,
        vec![
            ResultValue::Scalar(ScalarValue::Integer(0)),
            ResultValue::Scalar(ScalarValue::String("xx".into())),
            ResultValue::Scalar(ScalarValue::Null),
        ]
    );

    let mut cbor = Vec::new();
    ciborium::ser::into_writer(&graph, &mut cbor)
        .map_err(|error| irongraph::Error::internal(format!("CBOR failed: {error}")))?;
    let restored: GraphStore = ciborium::de::from_reader(cbor.as_slice())
        .map_err(|error| irongraph::Error::internal(format!("CBOR failed: {error}")))?;
    assert_eq!(
        restored.snapshot()?.node_properties.get(0, property),
        Some(ScalarValue::Integer(0))
    );
    assert_eq!(
        restored.snapshot()?.node_properties.get(1, property),
        Some(ScalarValue::String("xx".into()))
    );
    assert_eq!(restored.snapshot()?.node_properties.get(2, property), None);
    Ok(())
}
