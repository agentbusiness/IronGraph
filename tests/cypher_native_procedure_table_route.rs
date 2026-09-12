// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, ProcedureCatalog, ProcedureDefinition,
        ProcedureField, ProcedureValueType, QueryEngine, QueryResult, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

fn field(name: &str, value_type: ProcedureValueType, nullable: bool) -> Result<ProcedureField> {
    ProcedureField::new(name, value_type, nullable)
}

fn null() -> ResultValue {
    ResultValue::Scalar(ScalarValue::Null)
}

fn boolean(value: bool) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Boolean(value))
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn float(value: f64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Float(OrderedFloat(value)))
}

fn string(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn parameters(values: &[(&str, ResultValue)]) -> BTreeMap<String, ResultValue> {
    values
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.clone()))
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum OfficialFixture {
    DoNothing,
    Labels,
    CityLookup,
    NumberInput,
    FloatInput,
    NullableInteger,
    PairOutput,
}

#[derive(Clone, Debug)]
struct RouteCase {
    id: &'static str,
    fixture: OfficialFixture,
    query: &'static str,
    parameters: BTreeMap<String, ResultValue>,
    expected_columns: Vec<&'static str>,
    expected_rows: Vec<Vec<ResultValue>>,
}

fn case(
    id: &'static str,
    fixture: OfficialFixture,
    query: &'static str,
    expected_columns: &[&'static str],
    expected_rows: Vec<Vec<ResultValue>>,
) -> RouteCase {
    RouteCase {
        id,
        fixture,
        query,
        parameters: BTreeMap::new(),
        expected_columns: expected_columns.to_vec(),
        expected_rows,
    }
}

fn official_cases() -> Vec<RouteCase> {
    let labels = vec![vec![string("A")], vec![string("B")], vec![string("C")]];
    let city = vec![vec![string("Berlin"), integer(49)]];
    let wisdom = vec![vec![string("wisdom")]];
    let about_right = vec![vec![string("about right")]];
    let close_enough = vec![vec![string("close enough")]];
    let nix = vec![vec![string("nix")]];
    let pair = vec![vec![integer(1), integer(2)]];

    vec![
        case(
            "Call1[1]",
            OfficialFixture::DoNothing,
            "CALL test.doNothing()",
            &[],
            vec![],
        ),
        case(
            "Call1[2]",
            OfficialFixture::DoNothing,
            "CALL test.doNothing",
            &[],
            vec![],
        ),
        case(
            "Call1[5]",
            OfficialFixture::Labels,
            "CALL test.labels()",
            &["label"],
            labels.clone(),
        ),
        case(
            "Call1[6]",
            OfficialFixture::Labels,
            "CALL test.labels() YIELD label RETURN label",
            &["label"],
            labels,
        ),
        case(
            "Call2[1]",
            OfficialFixture::CityLookup,
            "CALL test.my.proc('Stefan', 1) YIELD city, country_code RETURN city, country_code",
            &["city", "country_code"],
            city.clone(),
        ),
        case(
            "Call2[2]",
            OfficialFixture::CityLookup,
            "CALL test.my.proc('Stefan', 1)",
            &["city", "country_code"],
            city.clone(),
        ),
        RouteCase {
            id: "Call2[3]",
            fixture: OfficialFixture::CityLookup,
            query: "CALL test.my.proc",
            parameters: parameters(&[("name", string("Stefan")), ("id", integer(1))]),
            expected_columns: vec!["city", "country_code"],
            expected_rows: city,
        },
        case(
            "Call3[1]",
            OfficialFixture::NumberInput,
            "CALL test.my.proc(42)",
            &["out"],
            wisdom.clone(),
        ),
        case(
            "Call3[2]",
            OfficialFixture::NumberInput,
            "CALL test.my.proc(42) YIELD out RETURN out",
            &["out"],
            wisdom,
        ),
        case(
            "Call3[3]",
            OfficialFixture::NumberInput,
            "CALL test.my.proc(42.3)",
            &["out"],
            about_right.clone(),
        ),
        case(
            "Call3[4]",
            OfficialFixture::NumberInput,
            "CALL test.my.proc(42.3) YIELD out RETURN out",
            &["out"],
            about_right,
        ),
        case(
            "Call3[5]",
            OfficialFixture::FloatInput,
            "CALL test.my.proc(42)",
            &["out"],
            close_enough.clone(),
        ),
        case(
            "Call3[6]",
            OfficialFixture::FloatInput,
            "CALL test.my.proc(42) YIELD out RETURN out",
            &["out"],
            close_enough,
        ),
        case(
            "Call4[1]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null)",
            &["out"],
            nix.clone(),
        ),
        case(
            "Call4[2]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null) YIELD out RETURN out",
            &["out"],
            nix.clone(),
        ),
        case(
            "Call5[1]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null) YIELD out RETURN out",
            &["out"],
            nix.clone(),
        ),
        case(
            "Call5[2]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null) YIELD out RETURN *",
            &["out"],
            nix.clone(),
        ),
        case(
            "Call5[3][33]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a, b RETURN a, b",
            &["a", "b"],
            pair.clone(),
        ),
        case(
            "Call5[3][34]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD b, a RETURN a, b",
            &["a", "b"],
            pair.clone(),
        ),
        case(
            "Call5[4][35]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS c, b AS d RETURN c, d",
            &["c", "d"],
            pair.clone(),
        ),
        case(
            "Call5[4][36]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS b, b AS d RETURN b, d",
            &["b", "d"],
            pair.clone(),
        ),
        case(
            "Call5[4][37]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS c, b AS a RETURN c, a",
            &["c", "a"],
            pair.clone(),
        ),
        case(
            "Call5[4][38]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS b, b AS a RETURN b, a",
            &["b", "a"],
            pair.clone(),
        ),
        case(
            "Call5[4][39]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS c, b AS b RETURN c, b",
            &["c", "b"],
            pair.clone(),
        ),
        case(
            "Call5[4][40]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS c, b RETURN c, b",
            &["c", "b"],
            pair.clone(),
        ),
        case(
            "Call5[4][41]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS a, b AS d RETURN a, d",
            &["a", "d"],
            pair.clone(),
        ),
        case(
            "Call5[4][42]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a, b AS d RETURN a, d",
            &["a", "d"],
            pair.clone(),
        ),
        case(
            "Call5[4][43]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS a, b AS b RETURN a, b",
            &["a", "b"],
            pair.clone(),
        ),
        case(
            "Call5[4][44]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a AS a, b RETURN a, b",
            &["a", "b"],
            pair.clone(),
        ),
        case(
            "Call5[4][45]",
            OfficialFixture::PairOutput,
            "CALL test.my.proc(null) YIELD a, b AS b RETURN a, b",
            &["a", "b"],
            pair.clone(),
        ),
        case(
            "Call5[8]",
            OfficialFixture::CityLookup,
            "CALL test.my.proc('Stefan', 1) YIELD *",
            &["city", "country_code"],
            vec![vec![string("Berlin"), integer(49)]],
        ),
        case(
            "Call6[1]",
            OfficialFixture::Labels,
            "CALL test.labels() YIELD label \
             WITH count(*) AS c \
             CALL test.labels() YIELD label \
             RETURN *",
            &["c", "label"],
            vec![
                vec![integer(3), string("A")],
                vec![integer(3), string("B")],
                vec![integer(3), string("C")],
            ],
        ),
        case(
            "Call6[2]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null) YIELD out WITH out RETURN out",
            &["out"],
            nix.clone(),
        ),
        case(
            "Call6[3]",
            OfficialFixture::NullableInteger,
            "CALL test.my.proc(null) YIELD out WITH out AS a RETURN a",
            &["a"],
            nix,
        ),
    ]
}

fn official_catalog(fixture: OfficialFixture) -> Result<ProcedureCatalog> {
    let mut catalog = ProcedureCatalog::default();
    let definition = match fixture {
        OfficialFixture::DoNothing => {
            ProcedureDefinition::new("test.doNothing", Vec::new(), Vec::new(), Vec::new())?
        }
        OfficialFixture::Labels => ProcedureDefinition::new(
            "test.labels",
            Vec::new(),
            vec![field("label", ProcedureValueType::String, true)?],
            vec![vec![string("A")], vec![string("B")], vec![string("C")]],
        )?,
        OfficialFixture::CityLookup => ProcedureDefinition::new(
            "test.my.proc",
            vec![
                field("name", ProcedureValueType::String, true)?,
                field("id", ProcedureValueType::Integer, true)?,
            ],
            vec![
                field("city", ProcedureValueType::String, true)?,
                field("country_code", ProcedureValueType::Integer, true)?,
            ],
            vec![
                vec![string("Andres"), integer(1), string("Malmö"), integer(46)],
                vec![string("Tobias"), integer(1), string("Malmö"), integer(46)],
                vec![string("Mats"), integer(1), string("Malmö"), integer(46)],
                vec![string("Stefan"), integer(1), string("Berlin"), integer(49)],
                vec![string("Stefan"), integer(2), string("München"), integer(49)],
                vec![string("Petra"), integer(1), string("London"), integer(44)],
            ],
        )?,
        OfficialFixture::NumberInput => ProcedureDefinition::new(
            "test.my.proc",
            vec![field("in", ProcedureValueType::Number, true)?],
            vec![field("out", ProcedureValueType::String, true)?],
            vec![
                vec![integer(42), string("wisdom")],
                vec![float(42.3), string("about right")],
            ],
        )?,
        OfficialFixture::FloatInput => ProcedureDefinition::new(
            "test.my.proc",
            vec![field("in", ProcedureValueType::Float, true)?],
            vec![field("out", ProcedureValueType::String, true)?],
            vec![vec![float(42.0), string("close enough")]],
        )?,
        OfficialFixture::NullableInteger => ProcedureDefinition::new(
            "test.my.proc",
            vec![field("in", ProcedureValueType::Integer, true)?],
            vec![field("out", ProcedureValueType::String, true)?],
            vec![vec![null(), string("nix")]],
        )?,
        OfficialFixture::PairOutput => ProcedureDefinition::new(
            "test.my.proc",
            vec![field("in", ProcedureValueType::Integer, true)?],
            vec![
                field("a", ProcedureValueType::Integer, true)?,
                field("b", ProcedureValueType::Integer, true)?,
            ],
            vec![vec![null(), integer(1), integer(2)]],
        )?,
    };
    catalog.register(definition)?;
    Ok(catalog)
}

fn image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 0,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    parameters: BTreeMap<String, ResultValue>,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
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
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 256,
        max_batch_rows: 256,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn result_rows(result: &QueryResult) -> Vec<Vec<ResultValue>> {
    result
        .batches
        .iter()
        .flat_map(|batch| {
            (0..batch.row_count).map(|row| {
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect()
            })
        })
        .collect()
}

fn execute(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    catalog: &ProcedureCatalog,
    query: &str,
    parameters: BTreeMap<String, ResultValue>,
) -> Result<ExecutionOutput> {
    let mut execution = context(graph, backend, parameters);
    QueryEngine.execute_with_procedures(query, execution.with_procedures(catalog))
}

fn assert_output(
    label: &str,
    output: &ExecutionOutput,
    expected_columns: &[&str],
    expected_rows: &[Vec<ResultValue>],
) {
    let columns = output
        .result
        .schema
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(columns, expected_columns, "{label}: projected columns");
    assert_eq!(result_rows(&output.result), expected_rows, "{label}: rows");
    assert_eq!(
        output.result.statistics,
        StatementStats::default(),
        "{label}: query statistics"
    );
    assert!(!output.result.truncated, "{label}: truncated result");
    assert!(
        output.graph_mutations.is_empty(),
        "{label}: graph side effects"
    );
    assert!(
        output.temporal_mutations.is_empty(),
        "{label}: temporal side effects"
    );
}

fn execute_official_case(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    case: &RouteCase,
) -> Result<ExecutionOutput> {
    let catalog = official_catalog(case.fixture)?;
    execute(
        graph,
        backend,
        &catalog,
        case.query,
        case.parameters.clone(),
    )
}

fn cpu_backend(graph: &GraphStore) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(image(graph)?)?;
    Ok(backend)
}

#[test]
fn exact_34_graph_free_official_call_scenarios_pass_the_cpu_reference() -> Result<()> {
    let graph = GraphStore::default();
    let backend = cpu_backend(&graph)?;
    let cases = official_cases();
    assert_eq!(cases.len(), 34);
    assert_eq!(
        cases
            .iter()
            .map(|case| case.id)
            .collect::<BTreeSet<_>>()
            .len(),
        34,
        "every pinned TCK scenario must have a unique identity"
    );

    for case in &cases {
        let output = execute_official_case(&graph, &backend, case)?;
        assert_output(
            case.id,
            &output,
            &case.expected_columns,
            &case.expected_rows,
        );
    }
    Ok(())
}

fn call1_in_query_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let name = graph.catalog_mut().intern_property("name")?;
    for (offset, (label, value)) in [("A", "a"), ("B", "b"), ("C", "c")].into_iter().enumerate() {
        let label = graph.catalog_mut().intern_label(label)?;
        let revision = u64::try_from(offset + 1)
            .map_err(|_| Error::internal("Call1 fixture revision overflowed"))?;
        graph.insert_node(NodeInput {
            id: NodeId(revision),
            layer: Layer::Observed,
            revision,
            labels: vec![label],
            properties: vec![(name, ScalarValue::String(Arc::from(value)))],
        })?;
    }
    Ok(graph)
}

#[test]
fn call1_3_and_4_preserve_each_incoming_graph_row_on_the_cpu_reference() -> Result<()> {
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.doNothing",
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?)?;

    let empty_graph = GraphStore::default();
    let empty_backend = cpu_backend(&empty_graph)?;
    let output = execute(
        &empty_graph,
        &empty_backend,
        &catalog,
        "MATCH (n) CALL test.doNothing() RETURN n",
        BTreeMap::new(),
    )?;
    assert_output("Call1[3]", &output, &["n"], &[]);

    let graph = call1_in_query_fixture()?;
    let backend = cpu_backend(&graph)?;
    let output = execute(
        &graph,
        &backend,
        &catalog,
        "MATCH (n) CALL test.doNothing() RETURN n.name AS name",
        BTreeMap::new(),
    )?;
    assert_output(
        "Call1[4]",
        &output,
        &["name"],
        &[vec![string("a")], vec![string("b")], vec![string("c")]],
    );
    Ok(())
}

#[test]
fn cpu_reference_preserves_non_identity_in_query_procedure_cardinality() -> Result<()> {
    let graph = call1_in_query_fixture()?;
    let backend = cpu_backend(&graph)?;
    let cases = [
        (
            ProcedureDefinition::new_with_row_count(
                "test.notIdentity",
                Vec::new(),
                Vec::new(),
                0,
                Vec::new(),
            )?,
            "MATCH (n) CALL test.notIdentity() RETURN n",
            "zero-row relation",
            0,
        ),
        (
            ProcedureDefinition::new_with_row_count(
                "test.notIdentity",
                Vec::new(),
                Vec::new(),
                2,
                Vec::new(),
            )?,
            "MATCH (n) CALL test.notIdentity() RETURN n",
            "two-row relation",
            6,
        ),
        (
            ProcedureDefinition::new(
                "test.notIdentity",
                vec![field("input", ProcedureValueType::Integer, false)?],
                Vec::new(),
                vec![vec![integer(1)], vec![integer(1)]],
            )?,
            "MATCH (n) CALL test.notIdentity(1) RETURN n",
            "input-bearing relation",
            6,
        ),
        (
            ProcedureDefinition::new(
                "test.notIdentity",
                Vec::new(),
                vec![field("output", ProcedureValueType::Integer, false)?],
                vec![vec![integer(1)]],
            )?,
            "MATCH (n) CALL test.notIdentity() YIELD output RETURN output",
            "output-bearing relation",
            3,
        ),
    ];

    for (definition, query, label, expected_rows) in cases {
        let mut catalog = ProcedureCatalog::default();
        catalog.register(definition)?;
        let output = execute(&graph, &backend, &catalog, query, BTreeMap::new())?;
        assert_eq!(
            output
                .result
                .batches
                .iter()
                .map(|batch| batch.row_count)
                .sum::<usize>(),
            expected_rows,
            "{label}"
        );
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ProbeCase {
    name: &'static str,
    query: &'static str,
    parameters: BTreeMap<String, ResultValue>,
    expected_columns: Vec<&'static str>,
    expected_rows: Vec<Vec<ResultValue>>,
}

fn probe(
    name: &'static str,
    query: &'static str,
    parameter_values: &[(&str, ResultValue)],
    expected_rows: Vec<Vec<ResultValue>>,
) -> ProbeCase {
    ProbeCase {
        name,
        query,
        parameters: parameters(parameter_values),
        expected_columns: vec!["ordinal"],
        expected_rows,
    }
}

fn adversarial_catalog() -> Result<ProcedureCatalog> {
    let positive_nan = f64::from_bits(0x7ff0_0000_0000_0042);
    let negative_nan = f64::from_bits(0xfff8_0000_0000_beef);
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.booleanExact",
        vec![field("value", ProcedureValueType::Boolean, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        vec![
            vec![boolean(true), integer(0)],
            vec![boolean(true), integer(1)],
            vec![boolean(false), integer(2)],
            vec![null(), integer(3)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.integerExact",
        vec![field("value", ProcedureValueType::Integer, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        vec![
            vec![integer(7), integer(0)],
            vec![integer(7), integer(1)],
            vec![integer(8), integer(2)],
            vec![null(), integer(3)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.floatExact",
        vec![field("value", ProcedureValueType::Float, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        vec![
            vec![float(-0.0), integer(0)],
            vec![float(0.0), integer(1)],
            vec![float(7.0), integer(2)],
            vec![float(positive_nan), integer(3)],
            vec![float(negative_nan), integer(4)],
            vec![null(), integer(5)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.numberExact",
        vec![field("value", ProcedureValueType::Number, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        vec![
            vec![integer(7), integer(0)],
            vec![float(7.0), integer(1)],
            vec![float(7.5), integer(2)],
            vec![integer(0), integer(3)],
            vec![float(-0.0), integer(4)],
            vec![float(0.0), integer(5)],
            vec![float(positive_nan), integer(6)],
            vec![float(negative_nan), integer(7)],
            vec![null(), integer(8)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.stringExact",
        vec![field("value", ProcedureValueType::String, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        vec![
            vec![string("Malmö"), integer(0)],
            vec![string("Malmö"), integer(1)],
            vec![string("Malmo"), integer(2)],
            vec![string("München"), integer(3)],
            vec![string("💾"), integer(4)],
            vec![null(), integer(5)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new_with_row_count(
        "test.empty",
        vec![field("value", ProcedureValueType::Integer, true)?],
        vec![field("ordinal", ProcedureValueType::Integer, false)?],
        0,
        Vec::new(),
    )?)?;
    catalog.register(ProcedureDefinition::new_with_row_count(
        "test.emptyZeroColumn",
        Vec::new(),
        Vec::new(),
        0,
        Vec::new(),
    )?)?;
    catalog.register(ProcedureDefinition::new_with_row_count(
        "test.unitZeroColumn",
        Vec::new(),
        Vec::new(),
        1,
        Vec::new(),
    )?)?;
    Ok(catalog)
}

fn adversarial_cases() -> Vec<ProbeCase> {
    let positive_nan = f64::from_bits(0x7ff0_0000_0000_0042);
    vec![
        probe(
            "BOOLEAN true preserves duplicate source-row order",
            "CALL test.booleanExact($value) YIELD ordinal RETURN ordinal",
            &[("value", boolean(true))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "BOOLEAN false is exact",
            "CALL test.booleanExact($value) YIELD ordinal RETURN ordinal",
            &[("value", boolean(false))],
            vec![vec![integer(2)]],
        ),
        probe(
            "nullable BOOLEAN matches the null fixture row",
            "CALL test.booleanExact($value) YIELD ordinal RETURN ordinal",
            &[("value", null())],
            vec![vec![integer(3)]],
        ),
        probe(
            "INTEGER is exact and preserves duplicate rows",
            "CALL test.integerExact($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(7))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "INTEGER no-match produces an empty output relation",
            "CALL test.integerExact($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(999))],
            vec![],
        ),
        probe(
            "FLOAT canonical negative zero equals both zero rows",
            "CALL test.floatExact($value) YIELD ordinal RETURN ordinal",
            &[("value", float(-0.0))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "FLOAT accepts an INTEGER invocation exactly",
            "CALL test.floatExact($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(7))],
            vec![vec![integer(2)]],
        ),
        probe(
            "FLOAT NaN never equals canonical NaN rows",
            "CALL test.floatExact($value) YIELD ordinal RETURN ordinal",
            &[("value", float(positive_nan))],
            vec![],
        ),
        probe(
            "NUMBER INTEGER equals INTEGER and FLOAT seven",
            "CALL test.numberExact($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(7))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "NUMBER FLOAT equals INTEGER and FLOAT seven",
            "CALL test.numberExact($value) YIELD ordinal RETURN ordinal",
            &[("value", float(7.0))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "NUMBER zero crosses integer and canonical FLOAT zero",
            "CALL test.numberExact($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(0))],
            vec![vec![integer(3)], vec![integer(4)], vec![integer(5)]],
        ),
        probe(
            "NUMBER NaN never equals canonical NaN rows",
            "CALL test.numberExact($value) YIELD ordinal RETURN ordinal",
            &[("value", float(positive_nan))],
            vec![],
        ),
        probe(
            "nullable NUMBER matches only its null row",
            "CALL test.numberExact($value) YIELD ordinal RETURN ordinal",
            &[("value", null())],
            vec![vec![integer(8)]],
        ),
        probe(
            "STRING compares exact UTF-8 bytes and preserves duplicates",
            "CALL test.stringExact($value) YIELD ordinal RETURN ordinal",
            &[("value", string("Malmö"))],
            vec![vec![integer(0)], vec![integer(1)]],
        ),
        probe(
            "STRING does not fold diacritics",
            "CALL test.stringExact($value) YIELD ordinal RETURN ordinal",
            &[("value", string("Malmo"))],
            vec![vec![integer(2)]],
        ),
        probe(
            "STRING handles exact supplementary UTF-8",
            "CALL test.stringExact($value) YIELD ordinal RETURN ordinal",
            &[("value", string("💾"))],
            vec![vec![integer(4)]],
        ),
        probe(
            "nullable STRING matches only its null row",
            "CALL test.stringExact($value) YIELD ordinal RETURN ordinal",
            &[("value", null())],
            vec![vec![integer(5)]],
        ),
        probe(
            "typed zero-row relation remains empty",
            "CALL test.empty($value) YIELD ordinal RETURN ordinal",
            &[("value", integer(1))],
            vec![],
        ),
    ]
}

fn execute_adversarial_suite(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
) -> Result<Vec<QueryResult>> {
    let catalog = adversarial_catalog()?;
    adversarial_cases()
        .into_iter()
        .map(|case| {
            let output = execute(
                graph,
                backend,
                &catalog,
                case.query,
                case.parameters.clone(),
            )?;
            assert_output(
                case.name,
                &output,
                &case.expected_columns,
                &case.expected_rows,
            );
            Ok(output.result)
        })
        .collect()
}

#[test]
fn typed_selection_edge_cases_pass_the_cpu_reference() -> Result<()> {
    let graph = GraphStore::default();
    let backend = cpu_backend(&graph)?;
    assert_eq!(execute_adversarial_suite(&graph, &backend)?.len(), 18);

    let empty = ProcedureDefinition::new_with_row_count(
        "test.zeroColumn",
        Vec::new(),
        Vec::new(),
        0,
        Vec::new(),
    )?
    .resident_table()?;
    let unit = ProcedureDefinition::new_with_row_count(
        "test.zeroColumn",
        Vec::new(),
        Vec::new(),
        1,
        Vec::new(),
    )?
    .resident_table()?;
    assert_eq!(empty.row_count(), 0);
    assert_eq!(unit.row_count(), 1);
    assert!(empty.columns().is_empty());
    assert!(unit.columns().is_empty());
    assert_ne!(empty.fingerprint(), unit.fingerprint());
    Ok(())
}

fn versioned_definition(version: char) -> Result<ProcedureDefinition> {
    let rows = match version {
        'A' => vec![
            vec![integer(1), string("A-first")],
            vec![integer(1), string("A-second")],
        ],
        'B' => vec![
            vec![integer(1), string("B-second")],
            vec![integer(1), string("B-first")],
        ],
        _ => return Err(Error::invalid_data("unknown versioned procedure fixture")),
    };
    ProcedureDefinition::new(
        "test.versioned",
        vec![field("value", ProcedureValueType::Integer, false)?],
        vec![field("payload", ProcedureValueType::String, false)?],
        rows,
    )
}

fn execute_version(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    version: char,
) -> Result<QueryResult> {
    let definition = versioned_definition(version)?;
    let mut catalog = ProcedureCatalog::default();
    catalog.register(definition)?;
    let output = execute(
        graph,
        backend,
        &catalog,
        "CALL test.versioned(1) YIELD payload RETURN payload",
        BTreeMap::new(),
    )?;
    let expected = match version {
        'A' => vec![vec![string("A-first")], vec![string("A-second")]],
        'B' => vec![vec![string("B-second")], vec![string("B-first")]],
        _ => unreachable!(),
    };
    assert_output(
        &format!("versioned table {version}"),
        &output,
        &["payload"],
        &expected,
    );
    Ok(output.result)
}

fn execute_table_identity_sequence(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
) -> Result<Vec<QueryResult>> {
    let fingerprint_a = versioned_definition('A')?.resident_table()?.fingerprint();
    let fingerprint_b = versioned_definition('B')?.resident_table()?.fingerprint();
    assert_ne!(fingerprint_a, fingerprint_b);
    assert_eq!(
        fingerprint_a,
        versioned_definition('A')?.resident_table()?.fingerprint()
    );
    Ok(vec![
        execute_version(graph, backend, 'A')?,
        execute_version(graph, backend, 'B')?,
        execute_version(graph, backend, 'A')?,
    ])
}

#[test]
fn table_fingerprint_and_execution_sequence_do_not_reuse_stale_rows_on_cpu() -> Result<()> {
    let graph = GraphStore::default();
    let backend = cpu_backend(&graph)?;
    let sequence = execute_table_identity_sequence(&graph, &backend)?;
    assert_ne!(sequence[0], sequence[1]);
    assert_eq!(sequence[0], sequence[2]);
    Ok(())
}

/// A CPU implementation advertised as Metal is deliberate sabotage. Every CALL must enter the
/// backend through project pinning and then the typed procedure ABI; host-side matching would make
/// these queries succeed without that route. Older graph/vector APIs remain forbidden escapes.
struct CpuMasqueradingAsMetal {
    inner: CpuBackend,
    unexpected_query_calls: Arc<AtomicUsize>,
}

impl CpuMasqueradingAsMetal {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner,
            unexpected_query_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn reject<T>(&self, route: &str) -> Result<T> {
        self.unexpected_query_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("procedure-table test rejects sabotaged backend route {route}"),
        ))
    }
}

impl ExecutionBackend for CpuMasqueradingAsMetal {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, _project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        self.reject("pin_project")
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.inner.resident_graph_revision(project)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.inner.resident_bookmark(project)
    }

    fn scan_nodes(
        &self,
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject("execute_node_pipeline")
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject("exact_l2")
    }
}

#[test]
fn cpu_advertised_as_metal_cannot_execute_any_of_the_34_calls_or_host_match_rows() -> Result<()> {
    let graph = GraphStore::default();
    let backend = CpuMasqueradingAsMetal::new(cpu_backend(&graph)?);
    let observations = Arc::clone(&backend.unexpected_query_calls);
    assert_eq!(backend.kind(), BackendKind::Metal);

    for case in official_cases() {
        let error = execute_official_case(&graph, &backend, &case)
            .expect_err("a CPU implementation advertised as Metal must not execute CALL");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{}", case.id);
        assert!(
            error.message.contains("procedure-table") && error.message.contains("pin_project"),
            "{}: unexpected fail-closed error: {error}",
            case.id
        );
    }
    assert_eq!(
        observations.load(Ordering::SeqCst),
        official_cases().len(),
        "each CALL must stop exactly at pin_project without using an older unreceipted method"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_backend(graph: &GraphStore) -> Result<MetalBackend> {
    let mut backend = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    backend.admit_project(image(graph)?)?;
    assert_eq!(
        backend.resident_bookmark(PROJECT),
        Some(Bookmark::default())
    );
    Ok(backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_matches_cpu_for_all_procedure_table_calls_without_fallback() -> Result<()> {
    static METAL_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let graph = GraphStore::default();
    let cpu = cpu_backend(&graph)?;
    let metal = metal_backend(&graph)?;
    assert_eq!(cpu.kind(), BackendKind::Cpu);
    assert_eq!(metal.kind(), BackendKind::Metal);

    let mut failures = Vec::new();
    for case in official_cases() {
        let cpu_output = execute_official_case(&graph, &cpu, &case)?;
        match execute_official_case(&graph, &metal, &case) {
            Ok(metal_output) => {
                if metal_output.result != cpu_output.result {
                    failures.push(format!(
                        "{}: CPU/Metal result mismatch: CPU={:?}; Metal={:?}",
                        case.id, cpu_output.result, metal_output.result
                    ));
                }
                assert_output(
                    case.id,
                    &metal_output,
                    &case.expected_columns,
                    &case.expected_rows,
                );
            }
            Err(error) => failures.push(format!("{}: real Metal failed: {error}", case.id)),
        }
    }

    let cpu_adversarial = execute_adversarial_suite(&graph, &cpu)?;
    match execute_adversarial_suite(&graph, &metal) {
        Ok(metal_adversarial) => {
            if metal_adversarial != cpu_adversarial {
                failures.push("typed adversarial CPU/Metal results differ".to_owned());
            }
        }
        Err(error) => failures.push(format!("typed adversarial real Metal failed: {error}")),
    }

    let cpu_sequence = execute_table_identity_sequence(&graph, &cpu)?;
    match execute_table_identity_sequence(&graph, &metal) {
        Ok(metal_sequence) => {
            if metal_sequence != cpu_sequence {
                failures.push(
                    "table-fingerprint A/B/A sequence returned stale or reordered rows".to_owned(),
                );
            }
        }
        Err(error) => failures.push(format!(
            "table-fingerprint A/B/A real Metal sequence failed: {error}"
        )),
    }

    assert!(
        failures.is_empty(),
        "strict native procedure-table acceptance has {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
