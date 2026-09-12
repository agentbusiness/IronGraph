// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ProcedureCatalog, ProcedureDefinition, ProcedureField,
        ProcedureValueType, QueryEngine, ResultValue,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

const CANONICAL_NAN_BITS: u64 = 0x7ff8_0000_0000_0000;
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;

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

fn decoded_strings<'a>(validity: &[u8], offsets: &[u32], utf8: &'a [u8]) -> Vec<Option<&'a str>> {
    validity
        .iter()
        .enumerate()
        .map(|(row, valid)| {
            if *valid == 0 {
                return None;
            }
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            Some(std::str::from_utf8(&utf8[start..end]).expect("resident string must be UTF-8"))
        })
        .collect()
}

#[test]
fn row_count_distinguishes_empty_relation_from_one_zero_column_row() -> Result<()> {
    let empty = ProcedureDefinition::new_with_row_count(
        "test.zeroColumn",
        Vec::new(),
        Vec::new(),
        0,
        Vec::new(),
    )?
    .resident_table()?;
    assert_eq!(empty.row_count(), 0);
    assert!(empty.columns().is_empty());

    let unit = ProcedureDefinition::new_with_row_count(
        "test.zeroColumn",
        Vec::new(),
        Vec::new(),
        1,
        Vec::new(),
    )?
    .resident_table()?;
    assert_eq!(unit.row_count(), 1);
    assert!(unit.columns().is_empty());
    assert_ne!(empty.fingerprint(), unit.fingerprint());

    // This is the pinned CALL fixture spelling for test.doNothing(). Existing CPU semantics
    // interpret it as the unit relation. The default constructor preserves that behavior
    // deliberately, while the explicit constructor carries the cardinality itself.
    let legacy_unit =
        ProcedureDefinition::new("test.doNothing", Vec::new(), Vec::new(), Vec::new())?
            .resident_table()?;
    let explicit_unit = ProcedureDefinition::new_with_row_count(
        "test.doNothing",
        Vec::new(),
        Vec::new(),
        1,
        Vec::new(),
    )?
    .resident_table()?;
    let tuple_spelled_unit =
        ProcedureDefinition::new("test.doNothing", Vec::new(), Vec::new(), vec![Vec::new()])?
            .resident_table()?;
    assert_eq!(legacy_unit.row_count(), 1);
    assert_eq!(explicit_unit.row_count(), 1);
    assert_eq!(legacy_unit.fingerprint(), explicit_unit.fingerprint());
    assert_eq!(legacy_unit.fingerprint(), tuple_spelled_unit.fingerprint());
    legacy_unit.validate()?;
    explicit_unit.validate()?;
    empty.validate()?;
    Ok(())
}

#[test]
fn export_preserves_signatures_ordinals_nulls_duplicates_and_utf8_order() -> Result<()> {
    let table = ProcedureDefinition::new(
        "test.lookup",
        vec![
            field("id", ProcedureValueType::Integer, true)?,
            field("ratio", ProcedureValueType::Float, true)?,
            field("number", ProcedureValueType::Number, true)?,
        ],
        vec![
            field("active", ProcedureValueType::Boolean, true)?,
            field("city", ProcedureValueType::String, true)?,
        ],
        vec![
            vec![
                integer(1),
                integer(7),
                integer(7),
                boolean(true),
                string("Malmö"),
            ],
            vec![null(), null(), float(7.5), null(), string("München")],
            vec![
                integer(1),
                integer(7),
                integer(7),
                boolean(true),
                string("Malmö"),
            ],
            vec![integer(2), float(8.25), float(8.25), boolean(false), null()],
        ],
    )?
    .resident_table()?;

    assert_eq!(table.name(), "test.lookup");
    assert_eq!(table.row_count(), 4);
    assert_eq!(table.inputs().len(), 3);
    assert_eq!(table.outputs().len(), 2);
    assert_eq!(
        table
            .inputs()
            .iter()
            .map(|field| (
                field.name(),
                field.value_type(),
                field.nullable(),
                field.source_ordinal()
            ))
            .collect::<Vec<_>>(),
        vec![
            ("id", ProcedureValueType::Integer, true, 0),
            ("ratio", ProcedureValueType::Float, true, 1),
            ("number", ProcedureValueType::Number, true, 2),
        ]
    );
    assert_eq!(table.outputs()[0].source_ordinal(), 3);
    assert_eq!(table.outputs()[1].source_ordinal(), 4);

    let integers = table.column(0).unwrap().as_integer().unwrap();
    assert_eq!(integers.validity(), &[1, 0, 1, 1]);
    assert_eq!(integers.values(), &[1, 0, 1, 2]);

    let floats = table.column(1).unwrap().as_float().unwrap();
    assert_eq!(floats.validity(), &[1, 0, 1, 1]);
    assert_eq!(
        floats.value_bits(),
        &[7.0_f64.to_bits(), 0, 7.0_f64.to_bits(), 8.25_f64.to_bits()]
    );

    let numbers = table.column(2).unwrap().as_number().unwrap();
    assert!(numbers.kinds()[0].is_integer());
    assert!(numbers.kinds()[1].is_float());
    assert!(numbers.kinds()[2].is_integer());
    assert!(numbers.kinds()[3].is_float());
    assert_eq!(numbers.integer_values(), &[7, 0, 7, 0]);
    assert_eq!(
        numbers.float_value_bits(),
        &[0, 7.5_f64.to_bits(), 0, 8.25_f64.to_bits()]
    );

    let booleans = table.column(3).unwrap().as_boolean().unwrap();
    assert_eq!(booleans.validity(), &[1, 0, 1, 1]);
    assert_eq!(booleans.values(), &[1, 0, 1, 0]);

    let strings = table.column(4).unwrap().as_string().unwrap();
    assert_eq!(strings.validity(), &[1, 1, 1, 0]);
    assert_eq!(strings.offsets(), &[0, 6, 14, 20, 20]);
    assert_eq!(
        decoded_strings(strings.validity(), strings.offsets(), strings.utf8()),
        vec![Some("Malmö"), Some("München"), Some("Malmö"), None]
    );
    assert_eq!(strings.utf8(), "MalmöMünchenMalmö".as_bytes());
    table.validate()?;
    Ok(())
}

#[test]
fn float_integer_normalization_is_stable_while_number_remains_tagged() -> Result<()> {
    let float_from_integer = ProcedureDefinition::new(
        "test.float",
        vec![field("value", ProcedureValueType::Float, false)?],
        Vec::new(),
        vec![vec![integer(42)]],
    )?
    .resident_table()?;
    let float_from_float = ProcedureDefinition::new(
        "test.float",
        vec![field("value", ProcedureValueType::Float, false)?],
        Vec::new(),
        vec![vec![float(42.0)]],
    )?
    .resident_table()?;
    assert_eq!(float_from_integer, float_from_float);
    assert_eq!(
        float_from_integer.columns()[0]
            .as_float()
            .unwrap()
            .value_bits(),
        &[42.0_f64.to_bits()]
    );

    let number = ProcedureDefinition::new(
        "test.number",
        vec![field("value", ProcedureValueType::Number, true)?],
        Vec::new(),
        vec![vec![integer(42)], vec![float(42.0)], vec![null()]],
    )?
    .resident_table()?;
    let column = number.columns()[0].as_number().unwrap();
    assert!(column.kinds()[0].is_integer());
    assert!(column.kinds()[1].is_float());
    assert!(column.kinds()[2].is_null());
    assert_eq!(column.validity(), &[1, 1, 0]);
    assert_eq!(column.integer_values(), &[42, 0, 0]);
    assert_eq!(column.float_value_bits(), &[0, 42.0_f64.to_bits(), 0]);
    Ok(())
}

#[test]
fn float_and_number_canonicalize_nan_payload_sign_and_negative_zero_exactly() -> Result<()> {
    let positive_payload_nan = f64::from_bits(0x7ff0_0000_0000_0042);
    let negative_payload_nan = f64::from_bits(0xfff8_0000_0000_beef);

    let floats = ProcedureDefinition::new(
        "test.floatCanonical",
        vec![field("value", ProcedureValueType::Float, false)?],
        Vec::new(),
        vec![
            vec![float(positive_payload_nan)],
            vec![float(negative_payload_nan)],
            vec![float(-0.0)],
            vec![float(0.0)],
        ],
    )?
    .resident_table()?;
    let float_column = floats.columns()[0].as_float().unwrap();
    assert_eq!(float_column.validity(), &[1, 1, 1, 1]);
    assert_eq!(
        float_column.value_bits(),
        &[CANONICAL_NAN_BITS, CANONICAL_NAN_BITS, 0, 0]
    );

    let numbers = ProcedureDefinition::new(
        "test.numberCanonical",
        vec![field("value", ProcedureValueType::Number, false)?],
        Vec::new(),
        vec![
            vec![float(positive_payload_nan)],
            vec![float(negative_payload_nan)],
            vec![float(-0.0)],
            vec![float(0.0)],
            vec![integer(0)],
        ],
    )?
    .resident_table()?;
    let number_column = numbers.columns()[0].as_number().unwrap();
    assert_eq!(number_column.validity(), &[1, 1, 1, 1, 1]);
    assert!(
        number_column.kinds()[..4]
            .iter()
            .all(|kind| kind.is_float())
    );
    assert!(number_column.kinds()[4].is_integer());
    assert_eq!(number_column.integer_values(), &[0, 0, 0, 0, 0]);
    assert_eq!(
        number_column.float_value_bits(),
        &[CANONICAL_NAN_BITS, CANONICAL_NAN_BITS, 0, 0, 0]
    );
    floats.validate()?;
    numbers.validate()?;
    Ok(())
}

#[test]
fn malformed_rows_are_rejected_before_export() -> Result<()> {
    let cases = [
        ProcedureDefinition::new(
            "test.badArity",
            vec![field("value", ProcedureValueType::Integer, true)?],
            Vec::new(),
            vec![Vec::new()],
        ),
        ProcedureDefinition::new(
            "test.badType",
            vec![field("value", ProcedureValueType::Integer, true)?],
            Vec::new(),
            vec![vec![string("not-an-integer")]],
        ),
        ProcedureDefinition::new(
            "test.badNull",
            vec![field("value", ProcedureValueType::String, false)?],
            Vec::new(),
            vec![vec![null()]],
        ),
        ProcedureDefinition::new_with_row_count(
            "test.badCardinality",
            vec![field("value", ProcedureValueType::Integer, false)?],
            Vec::new(),
            2,
            vec![vec![integer(1)]],
        ),
        ProcedureDefinition::new_with_row_count(
            "test.badZeroColumnCardinality",
            Vec::new(),
            Vec::new(),
            0,
            vec![Vec::new()],
        ),
    ];

    for result in cases {
        let error = result.expect_err("malformed procedure row must be rejected");
        assert_eq!(error.code, ErrorCode::InvalidData);
    }
    Ok(())
}

#[test]
fn fingerprint_covers_complete_schema_rows_tags_and_canonical_float_payloads() -> Result<()> {
    fn definition(
        name: &str,
        nullable: bool,
        rows: Vec<Vec<ResultValue>>,
    ) -> Result<ProcedureDefinition> {
        ProcedureDefinition::new(
            name,
            Vec::new(),
            vec![field("city", ProcedureValueType::String, nullable)?],
            rows,
        )
    }

    let baseline = definition(
        "test.cities",
        true,
        vec![vec![string("Malmö")], vec![string("München")]],
    )?;
    let first = baseline.resident_table()?;
    let second = baseline.resident_table()?;
    assert_eq!(first.fingerprint(), second.fingerprint());
    assert_ne!(first.fingerprint().as_bytes(), &[0; 32]);

    let reversed = definition(
        "test.cities",
        true,
        vec![vec![string("München")], vec![string("Malmö")]],
    )?
    .resident_table()?;
    let duplicate = definition(
        "test.cities",
        true,
        vec![
            vec![string("Malmö")],
            vec![string("München")],
            vec![string("München")],
        ],
    )?
    .resident_table()?;
    let renamed = definition(
        "test.otherCities",
        true,
        vec![vec![string("Malmö")], vec![string("München")]],
    )?
    .resident_table()?;
    let non_nullable = definition(
        "test.cities",
        false,
        vec![vec![string("Malmö")], vec![string("München")]],
    )?
    .resident_table()?;
    let renamed_field = ProcedureDefinition::new(
        "test.cities",
        Vec::new(),
        vec![field("place", ProcedureValueType::String, true)?],
        vec![vec![string("Malmö")], vec![string("München")]],
    )?
    .resident_table()?;

    for changed in [
        &reversed,
        &duplicate,
        &renamed,
        &non_nullable,
        &renamed_field,
    ] {
        assert_ne!(first.fingerprint(), changed.fingerprint());
    }

    let empty_string_schema = ProcedureDefinition::new_with_row_count(
        "test.schemaType",
        Vec::new(),
        vec![field("value", ProcedureValueType::String, true)?],
        0,
        Vec::new(),
    )?
    .resident_table()?;
    let empty_integer_schema = ProcedureDefinition::new_with_row_count(
        "test.schemaType",
        Vec::new(),
        vec![field("value", ProcedureValueType::Integer, true)?],
        0,
        Vec::new(),
    )?
    .resident_table()?;
    assert_ne!(
        empty_string_schema.fingerprint(),
        empty_integer_schema.fingerprint()
    );

    // Both relations have one empty NUMBER column at source ordinal zero. Direction is the only
    // semantic distinction, so this directly protects input-vs-output fingerprint coverage.
    let input_schema = ProcedureDefinition::new_with_row_count(
        "test.schemaDirection",
        vec![field("value", ProcedureValueType::Number, true)?],
        Vec::new(),
        0,
        Vec::new(),
    )?
    .resident_table()?;
    let output_schema = ProcedureDefinition::new_with_row_count(
        "test.schemaDirection",
        Vec::new(),
        vec![field("value", ProcedureValueType::Number, true)?],
        0,
        Vec::new(),
    )?
    .resident_table()?;
    assert_ne!(input_schema.fingerprint(), output_schema.fingerprint());

    let ordinal_ab = ProcedureDefinition::new_with_row_count(
        "test.schemaOrdinal",
        vec![
            field("first", ProcedureValueType::String, true)?,
            field("second", ProcedureValueType::String, true)?,
        ],
        Vec::new(),
        0,
        Vec::new(),
    )?
    .resident_table()?;
    let ordinal_ba = ProcedureDefinition::new_with_row_count(
        "test.schemaOrdinal",
        vec![
            field("second", ProcedureValueType::String, true)?,
            field("first", ProcedureValueType::String, true)?,
        ],
        Vec::new(),
        0,
        Vec::new(),
    )?
    .resident_table()?;
    assert_ne!(ordinal_ab.fingerprint(), ordinal_ba.fingerprint());
    assert_eq!(ordinal_ab.inputs()[0].source_ordinal(), 0);
    assert_eq!(ordinal_ab.inputs()[1].source_ordinal(), 1);
    assert_eq!(ordinal_ba.inputs()[0].source_ordinal(), 0);
    assert_eq!(ordinal_ba.inputs()[1].source_ordinal(), 1);

    let integer_number = ProcedureDefinition::new(
        "test.numberKind",
        vec![field("value", ProcedureValueType::Number, false)?],
        Vec::new(),
        vec![vec![integer(42)]],
    )?
    .resident_table()?;
    let float_number = ProcedureDefinition::new(
        "test.numberKind",
        vec![field("value", ProcedureValueType::Number, false)?],
        Vec::new(),
        vec![vec![float(42.0)]],
    )?
    .resident_table()?;
    assert_ne!(integer_number.fingerprint(), float_number.fingerprint());
    assert!(integer_number.columns()[0].as_number().unwrap().kinds()[0].is_integer());
    assert!(float_number.columns()[0].as_number().unwrap().kinds()[0].is_float());

    let float_payload = |value| -> Result<_> {
        ProcedureDefinition::new(
            "test.floatPayload",
            vec![field("value", ProcedureValueType::Float, false)?],
            Vec::new(),
            vec![vec![float(value)]],
        )?
        .resident_table()
    };
    assert_ne!(
        float_payload(1.25)?.fingerprint(),
        float_payload(2.5)?.fingerprint()
    );
    assert_eq!(
        float_payload(f64::from_bits(0x7ff0_0000_0000_0042))?.fingerprint(),
        float_payload(f64::from_bits(0xfff8_0000_0000_beef))?.fingerprint()
    );
    assert_eq!(
        float_payload(-0.0)?.fingerprint(),
        float_payload(0.0)?.fingerprint()
    );
    Ok(())
}

fn procedure_catalog() -> Result<ProcedureCatalog> {
    let mut catalog = ProcedureCatalog::default();
    catalog.register(ProcedureDefinition::new(
        "test.lookup",
        vec![
            field("name", ProcedureValueType::String, true)?,
            field("id", ProcedureValueType::Integer, true)?,
        ],
        vec![
            field("city", ProcedureValueType::String, true)?,
            field("code", ProcedureValueType::Integer, true)?,
        ],
        vec![
            vec![string("Stefan"), integer(1), string("Berlin"), integer(49)],
            vec![string("Stefan"), integer(2), string("München"), integer(49)],
        ],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.nullable",
        vec![field("input", ProcedureValueType::Integer, true)?],
        vec![field("out", ProcedureValueType::String, true)?],
        vec![vec![null(), string("nix")]],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.floatInput",
        vec![field("input", ProcedureValueType::Float, true)?],
        vec![field("out", ProcedureValueType::String, true)?],
        vec![vec![float(42.0), string("close enough")]],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.labels",
        Vec::new(),
        vec![field("label", ProcedureValueType::String, true)?],
        vec![vec![string("A")], vec![string("B")], vec![string("C")]],
    )?)?;
    catalog.register(ProcedureDefinition::new(
        "test.doNothing",
        Vec::new(),
        Vec::new(),
        Vec::new(),
    )?)?;
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
        max_result_rows: 128,
        max_batch_rows: 128,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn result_rows(result: &irongraph::cypher::QueryResult) -> Vec<Vec<ResultValue>> {
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

#[test]
fn graph_free_call_uses_the_resident_table_plan_for_cpu_reference_semantics() -> Result<()> {
    let graph = GraphStore::default();
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image(&graph)?)?;
    let catalog = procedure_catalog()?;

    let cases = [
        (
            "CALL test.lookup('Stefan', 1) YIELD city, code AS country_code WITH city AS place, country_code RETURN place, country_code",
            vec![vec![string("Berlin"), integer(49)]],
        ),
        (
            "CALL test.nullable(null) YIELD out RETURN *",
            vec![vec![string("nix")]],
        ),
        (
            "CALL test.floatInput(42) YIELD out RETURN out",
            vec![vec![string("close enough")]],
        ),
        (
            "CALL test.labels() YIELD label RETURN label",
            vec![vec![string("A")], vec![string("B")], vec![string("C")]],
        ),
    ];
    for (query, expected) in cases {
        let mut execution = context(&graph, &cpu, BTreeMap::new());
        let output =
            QueryEngine.execute_with_procedures(query, execution.with_procedures(&catalog))?;
        assert_eq!(result_rows(&output.result), expected, "{query}");
    }

    let mut execution = context(&graph, &cpu, BTreeMap::new());
    let output = QueryEngine
        .execute_with_procedures("CALL test.doNothing()", execution.with_procedures(&catalog))?;
    assert!(output.result.schema.is_empty());
    assert!(output.result.batches.is_empty());
    Ok(())
}

struct NoFallbackMetalBackend {
    inner: Box<dyn ExecutionBackend>,
    unexpected_query_calls: Arc<AtomicUsize>,
}

impl NoFallbackMetalBackend {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            unexpected_query_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn reject<T>(&self, route: &str) -> Result<T> {
        self.unexpected_query_calls.fetch_add(1, Ordering::SeqCst);
        Err(irongraph::Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("procedure route rejected sabotaged backend method {route}"),
        ))
    }
}

impl ExecutionBackend for NoFallbackMetalBackend {
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
fn metal_table_plan_enters_only_the_pinned_typed_route_without_legacy_fallback() -> Result<()> {
    let graph = GraphStore::default();
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image(&graph)?)?;
    let backend = NoFallbackMetalBackend::new(cpu);
    let observations = Arc::clone(&backend.unexpected_query_calls);
    let catalog = procedure_catalog()?;
    let mut execution = context(&graph, &backend, BTreeMap::new());
    let error = QueryEngine
        .execute_with_procedures(
            "CALL test.lookup('Stefan', 1) YIELD city, code RETURN city, code",
            execution.with_procedures(&catalog),
        )
        .expect_err("a sabotaged Metal backend must stop at project pinning");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert!(
        error.message.contains("procedure") && error.message.contains("pin_project"),
        "unexpected admission error: {error}"
    );
    assert_eq!(
        observations.load(Ordering::SeqCst),
        1,
        "the plan must enter pin_project exactly once and no older backend route"
    );
    Ok(())
}
