// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, ProcedureCatalog, ProcedureDefinition,
        ProcedureField, ProcedureValueType, QueryEngine, ResultValue,
    },
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

fn context(graph: &GraphStore, parameters: BTreeMap<String, ResultValue>) -> ExecutionContext<'_> {
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
        bookmark: Bookmark { term: 1, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn field(name: &str, value_type: ProcedureValueType) -> Result<ProcedureField> {
    ProcedureField::new(name, value_type, true)
}

fn string(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn city_catalog() -> Result<ProcedureCatalog> {
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.my.proc",
        vec![
            field("name", ProcedureValueType::String)?,
            field("id", ProcedureValueType::Integer)?,
        ],
        vec![
            field("city", ProcedureValueType::String)?,
            field("country_code", ProcedureValueType::Integer)?,
        ],
        vec![
            vec![string("Stefan"), integer(1), string("Berlin"), integer(49)],
            vec![string("Stefan"), integer(2), string("München"), integer(49)],
            vec![string("Petra"), integer(1), string("London"), integer(44)],
        ],
    )?)?;
    Ok(catalog)
}

fn execute(
    query: &str,
    graph: &GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    catalog: &ProcedureCatalog,
) -> Result<ExecutionOutput> {
    let mut context = context(graph, parameters);
    QueryEngine.execute_with_procedures(query, context.with_procedures(catalog))
}

fn first_row(output: &ExecutionOutput) -> Result<Vec<ResultValue>> {
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("result has no batch"))?;
    Ok(batch
        .columns
        .iter()
        .map(|column| column.values[0].clone())
        .collect())
}

#[test]
fn explicit_and_implicit_calls_match_typed_rows() -> Result<()> {
    let graph = GraphStore::default();
    let catalog = city_catalog()?;
    let explicit = execute(
        "CALL test.my.proc('Stefan', 1) YIELD city, country_code AS code RETURN city, code",
        &graph,
        BTreeMap::new(),
        &catalog,
    )?;
    assert_eq!(explicit.result.schema[0].0, "city");
    assert_eq!(first_row(&explicit)?, vec![string("Berlin"), integer(49)]);

    let implicit = execute(
        "CALL test.my.proc",
        &graph,
        BTreeMap::from([
            ("name".to_owned(), string("Stefan")),
            ("id".to_owned(), integer(2)),
        ]),
        &catalog,
    )?;
    assert_eq!(first_row(&implicit)?, vec![string("München"), integer(49)]);
    Ok(())
}

#[test]
fn void_procedure_preserves_each_input_row() -> Result<()> {
    let graph = GraphStore::default();
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.doNothing",
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?)?;
    let output = execute(
        "UNWIND [1, 2, 3] AS n CALL test.doNothing() RETURN n",
        &graph,
        BTreeMap::new(),
        &catalog,
    )?;
    let values = &output.result.batches[0].columns[0].values;
    assert_eq!(values, &[integer(1), integer(2), integer(3)]);
    Ok(())
}

#[test]
fn call_binding_surfaces_standard_compile_errors() -> Result<()> {
    let graph = GraphStore::default();
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.one",
        vec![field("input", ProcedureValueType::Integer)?],
        vec![field("output", ProcedureValueType::String)?],
        vec![vec![integer(1), string("one")]],
    )?)?;

    for (query, code, detail) in [
        (
            "CALL test.one()",
            ErrorCode::QuerySyntax,
            "InvalidNumberOfArguments",
        ),
        (
            "CALL test.one(true)",
            ErrorCode::QuerySyntax,
            "InvalidArgumentType",
        ),
        ("CALL test.one", ErrorCode::QueryType, "MissingParameter"),
        (
            "WITH 'bound' AS output CALL test.one(1) YIELD output RETURN output",
            ErrorCode::QuerySyntax,
            "VariableAlreadyBound",
        ),
        (
            "UNWIND [1] AS n CALL test.one(count(n)) YIELD output RETURN output",
            ErrorCode::QuerySyntax,
            "InvalidAggregation",
        ),
    ] {
        let error = execute(query, &graph, BTreeMap::new(), &catalog)
            .err()
            .ok_or_else(|| {
                irongraph::Error::internal(format!(
                    "invalid procedure call unexpectedly succeeded: {query}"
                ))
            })?;
        assert_eq!(error.code, code, "query: {query}; error: {error}");
        assert!(
            error.message.contains(detail),
            "query: {query}; expected {detail}; error: {error}"
        );
    }
    Ok(())
}
