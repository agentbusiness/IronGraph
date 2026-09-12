// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance for the five non-temporal scalar/list examples in
//! `WithOrderBy1.feature` scenario [45].
//!
//! The CPU semantic reference is deliberately advertised as Metal before project pinning. That
//! makes generic host execution ineligible while preserving honest CPU receipt provenance after
//! pinning. A passing case must cross one sealed Unit segmented-aggregation boundary containing
//! the complete projection, UNWIND, comparison-rank, ORDER BY, collect, range, and equality work.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentExecutionObligation, ResidentExecutionReceipt,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierExpression, ResidentQuantifierFunction, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentQuantifierSlot, ResidentQuantifierValue,
        ResidentRowProgramRequest, ResidentRowProgramResult, ResidentSegmentedAggregate,
        ResidentSegmentedAggregateKind, ResidentSegmentedAggregationOperation,
        ResidentSegmentedAggregationRequest, ResidentSegmentedAggregationResult,
        ResidentSegmentedAggregationSource, ResidentSegmentedUnitPrelude, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 45, index: 0 };
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 4_096;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-20260720-after-pattern2-null2.json";
const REPORT_SHA256: &str = "e1717bb8d877acbd2ebd8f75679e1e53bf751ffe5d0174fecf7cf293d9cd968d";
const FEATURE_SUFFIX: &str = "features/clauses/with-orderBy/WithOrderBy1.feature";
const SCENARIO_PREFIX: &str = "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarDomain {
    Boolean,
    Integer,
    Float,
    String,
    IntegerList,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ScalarConsistencyCase {
    /// Zero-based position in the pinned JSON report.
    report_index: usize,
    /// One-based identifier rendered at the end of the expanded report name.
    displayed_id: usize,
    example_name: &'static str,
    expanded_name: &'static str,
    values_literal: &'static str,
    value_count: usize,
    domain: ScalarDomain,
}

const CASES: [ScalarConsistencyCase; 5] = [
    ScalarConsistencyCase {
        report_index: 1_007,
        displayed_id: 1_008,
        example_name: "booleans",
        expanded_name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1008]",
        values_literal: "[true, false]",
        value_count: 2,
        domain: ScalarDomain::Boolean,
    },
    ScalarConsistencyCase {
        report_index: 1_008,
        displayed_id: 1_009,
        example_name: "integers",
        expanded_name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1009]",
        values_literal: "[351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10]",
        value_count: 13,
        domain: ScalarDomain::Integer,
    },
    ScalarConsistencyCase {
        report_index: 1_009,
        displayed_id: 1_010,
        example_name: "floats",
        expanded_name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1010]",
        values_literal: "[351.5, -3974856.01, -3.203957, 123.0002, 123.0001, 123.00013, 123.00011, 0.0100000, 0.0999999, 0.00000001, 3.0, 209345.87, -10.654]",
        value_count: 13,
        domain: ScalarDomain::Float,
    },
    ScalarConsistencyCase {
        report_index: 1_010,
        displayed_id: 1_011,
        example_name: "string",
        expanded_name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1011]",
        values_literal: "['Sort', 'order', ' ', 'should', 'be', '', 'consistent', 'with', 'comparisons', ', ', 'where', 'comparisons are', 'defined', '!']",
        value_count: 14,
        domain: ScalarDomain::String,
    },
    ScalarConsistencyCase {
        report_index: 1_011,
        displayed_id: 1_012,
        example_name: "lists",
        expanded_name: "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName> [1012]",
        values_literal: "[[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]]",
        value_count: 8,
        domain: ScalarDomain::IntegerList,
    },
];

fn query(case: ScalarConsistencyCase) -> String {
    format!(
        "WITH {} AS values\n\
         WITH values, size(values) AS numOfValues\n\
         UNWIND values AS value\n\
         WITH size([x IN values WHERE x < value]) AS x, value, numOfValues\n\
           ORDER BY value\n\
         WITH numOfValues, collect(x) AS orderedX\n\
         RETURN orderedX = range(0, numOfValues-1) AS equal",
        case.values_literal
    )
}

fn expected_values(case: ScalarConsistencyCase) -> ResidentQuantifierValue {
    let values = match case.domain {
        ScalarDomain::Boolean => vec![
            ResidentQuantifierValue::Boolean(true),
            ResidentQuantifierValue::Boolean(false),
        ],
        ScalarDomain::Integer => [
            351, -3_974_856, 93, -3, 123, 0, 3, -2, 20_934_587, 1, 20_934_585, 20_934_586, -10,
        ]
        .into_iter()
        .map(ResidentQuantifierValue::Integer)
        .collect(),
        ScalarDomain::Float => [
            351.5_f64,
            -3_974_856.01,
            -3.203_957,
            123.000_2,
            123.000_1,
            123.000_13,
            123.000_11,
            0.010_000_0,
            0.099_999_9,
            0.000_000_01,
            3.0,
            209_345.87,
            -10.654,
        ]
        .into_iter()
        .map(|value| ResidentQuantifierValue::Float(value.to_bits()))
        .collect(),
        ScalarDomain::String => [
            "Sort",
            "order",
            " ",
            "should",
            "be",
            "",
            "consistent",
            "with",
            "comparisons",
            ", ",
            "where",
            "comparisons are",
            "defined",
            "!",
        ]
        .into_iter()
        .map(|value| ResidentQuantifierValue::String(value.to_owned()))
        .collect(),
        ScalarDomain::IntegerList => [
            &[2, 2][..],
            &[2, -2],
            &[1, 2],
            &[],
            &[1],
            &[300, 0],
            &[1, -20],
            &[2, -2, 100],
        ]
        .into_iter()
        .map(|values| {
            ResidentQuantifierValue::List(
                values
                    .iter()
                    .copied()
                    .map(ResidentQuantifierValue::Integer)
                    .collect(),
            )
        })
        .collect(),
    };
    ResidentQuantifierValue::List(values)
}

fn expected_rank_range(case: ScalarConsistencyCase) -> ResidentQuantifierValue {
    ResidentQuantifierValue::List(
        (0..case.value_count)
            .map(|rank| ResidentQuantifierValue::Integer(rank as i64))
            .collect(),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CallSnapshot {
    pins: usize,
    segmented: usize,
    sealed: usize,
    unit_source: usize,
    host_relation: usize,
    quantifier: usize,
    row: usize,
    node_pipeline: usize,
    nullable: usize,
    sort_rows: usize,
    other: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReceiptCapture {
    receipts: Vec<ResidentExecutionReceipt>,
    graph_read_dependency_count: usize,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    segmented: AtomicUsize,
    sealed: AtomicUsize,
    unit_source: AtomicUsize,
    host_relation: AtomicUsize,
    quantifier: AtomicUsize,
    row: AtomicUsize,
    node_pipeline: AtomicUsize,
    nullable: AtomicUsize,
    sort_rows: AtomicUsize,
    other: AtomicUsize,
    requests: Mutex<Vec<ResidentSegmentedAggregationRequest>>,
    receipt_captures: Mutex<Vec<ReceiptCapture>>,
}

impl Observations {
    fn snapshot(&self) -> CallSnapshot {
        CallSnapshot {
            pins: self.pins.load(Ordering::SeqCst),
            segmented: self.segmented.load(Ordering::SeqCst),
            sealed: self.sealed.load(Ordering::SeqCst),
            unit_source: self.unit_source.load(Ordering::SeqCst),
            host_relation: self.host_relation.load(Ordering::SeqCst),
            quantifier: self.quantifier.load(Ordering::SeqCst),
            row: self.row.load(Ordering::SeqCst),
            node_pipeline: self.node_pipeline.load(Ordering::SeqCst),
            nullable: self.nullable.load(Ordering::SeqCst),
            sort_rows: self.sort_rows.load(Ordering::SeqCst),
            other: self.other.load(Ordering::SeqCst),
        }
    }
}

/// Before pinning this wrapper advertises Metal, so strict execution cannot fall through to the
/// generic CPU evaluator. Its pinned form exposes the honest CPU kind and delegates only the one
/// complete segmented command. All observable alternative routes fail closed.
struct ObservedSegmentedBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl ObservedSegmentedBackend {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(Observations::default()),
        }
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    fn rejected<T>(&self, route: &'static str) -> Result<T> {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict scalar consistency gate rejected `{route}`"),
        ))
    }

    fn reject_other<T>(&self, route: &'static str) -> Result<T> {
        self.observations.other.fetch_add(1, Ordering::SeqCst);
        self.rejected(route)
    }
}

impl ExecutionBackend for ObservedSegmentedBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            BackendKind::Cpu
        } else {
            BackendKind::Metal
        }
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

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        if self.pinned {
            return self.reject_other("pin_project_twice");
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            pinned: true,
            observations: Arc::clone(&self.observations),
        }))
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
        self.reject_other("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_other("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_other("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_other("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_other("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.observations.sort_rows.fetch_add(1, Ordering::SeqCst);
        self.rejected("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_other("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_other("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.observations
            .node_pipeline
            .fetch_add(1, Ordering::SeqCst);
        self.rejected("execute_node_pipeline")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.observations.row.fetch_add(1, Ordering::SeqCst);
        self.rejected("execute_row_program")
    }

    fn execute_quantifier_program(
        &self,
        _request: &ResidentQuantifierProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        self.observations.quantifier.fetch_add(1, Ordering::SeqCst);
        self.rejected("execute_quantifier_program")
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        false
    }

    fn execute_nullable_relation(
        &self,
        _request: &ResidentNullableRelationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        self.observations.nullable.fetch_add(1, Ordering::SeqCst);
        self.rejected("execute_nullable_relation")
    }

    fn supports_native_segmented_aggregation(&self) -> bool {
        self.inner.supports_native_segmented_aggregation()
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        if !self.pinned {
            return self.reject_other("execute_segmented_aggregation_without_pin");
        }
        self.observations.segmented.fetch_add(1, Ordering::SeqCst);
        let Some(program) = request.program.as_ref() else {
            self.observations
                .host_relation
                .fetch_add(1, Ordering::SeqCst);
            return self.rejected("legacy_host_relation_aggregation");
        };
        if request.input.row_count != 0
            || request.input.column_count != 0
            || !request.input.cells.is_empty()
            || !request.input.arena.is_empty()
        {
            self.observations
                .host_relation
                .fetch_add(1, Ordering::SeqCst);
            return self.rejected("sealed_program_with_host_rows");
        }
        if !matches!(
            &program.source,
            ResidentSegmentedAggregationSource::Unit { .. }
        ) {
            return self.reject_other("non_unit_segmented_source");
        }
        request.validate()?;
        self.observations.sealed.fetch_add(1, Ordering::SeqCst);
        self.observations.unit_source.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());

        let raw = self
            .inner
            .execute_segmented_aggregation(request, cancellation)?;
        let validated = raw
            .clone()
            .validate_for_publication(request, BackendKind::Cpu)?;
        self.observations
            .receipt_captures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(ReceiptCapture {
                receipts: validated.receipts().to_vec(),
                graph_read_dependency_count: validated.graph_read_dependencies().len(),
            });
        Ok(raw)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_other("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_other("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_other("exact_l2")
    }
}

struct Fixture {
    graph: GraphStore,
}

impl Fixture {
    fn new() -> Self {
        Self {
            graph: GraphStore::default(),
        }
    }

    fn backend(&self) -> Result<ObservedSegmentedBackend> {
        let image = ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Ok(ObservedSegmentedBackend::new(cpu))
    }
}

fn context<'a>(fixture: &'a Fixture, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: BOOKMARK,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 8,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute(
    fixture: &Fixture,
    backend: &dyn ExecutionBackend,
    query: &str,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(query, &mut context(fixture, backend))
}

fn output_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal(
                "scalar consistency output has an invalid columnar shape",
            ));
        }
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn assert_exact_public_output(case: ScalarConsistencyCase, output: &ExecutionOutput) -> Result<()> {
    assert_eq!(
        output.result.schema,
        vec![("equal".to_owned(), ColumnType::Boolean)],
        "{} lost its exact output schema",
        case.example_name
    );
    assert_eq!(
        output_rows(output)?,
        vec![vec![ResultValue::Scalar(ScalarValue::Boolean(true))]],
        "{} did not preserve ORDER BY/comparison consistency",
        case.example_name
    );
    assert_eq!(output.result.bookmark, BOOKMARK);
    assert_eq!(output.result.statistics, StatementStats::default());
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    Ok(())
}

fn expression_slot(expression: &ResidentQuantifierExpression) -> Result<ResidentQuantifierSlot> {
    let ResidentQuantifierExpression::Slot(slot) = expression else {
        return Err(Error::internal(format!(
            "expected a sealed slot expression, got {expression:?}"
        )));
    };
    Ok(*slot)
}

fn assert_size_expression(
    expression: &ResidentQuantifierExpression,
    source: ResidentQuantifierSlot,
    static_size: usize,
) -> Result<()> {
    match expression {
        ResidentQuantifierExpression::Function {
            function: ResidentQuantifierFunction::Size,
            arguments,
        } if arguments.as_slice() == [ResidentQuantifierExpression::Slot(source)] => Ok(()),
        ResidentQuantifierExpression::Literal(ResidentQuantifierValue::Integer(size))
            if *size == static_size as i64 =>
        {
            Ok(())
        }
        _ => Err(Error::internal(format!(
            "sealed size(values) expression changed: {expression:?}"
        ))),
    }
}

fn assert_rank_expression(
    expression: &ResidentQuantifierExpression,
    values: ResidentQuantifierSlot,
    value: ResidentQuantifierSlot,
) -> Result<()> {
    let ResidentQuantifierExpression::Function {
        function: ResidentQuantifierFunction::Size,
        arguments,
    } = expression
    else {
        return Err(Error::internal(format!(
            "comparison rank is not computed by size(list-comprehension): {expression:?}"
        )));
    };
    let [
        ResidentQuantifierExpression::ListComprehension {
            variable,
            list,
            predicate: Some(predicate),
            projection: None,
        },
    ] = arguments.as_slice()
    else {
        return Err(Error::internal(format!(
            "comparison rank lost its exact list comprehension: {arguments:?}"
        )));
    };
    assert_eq!(expression_slot(list)?, values);
    let ResidentQuantifierExpression::Binary {
        left,
        operation: ResidentQuantifierBinary::Less,
        right,
    } = predicate.as_ref()
    else {
        return Err(Error::internal(format!(
            "comparison rank lost x < value: {predicate:?}"
        )));
    };
    assert_eq!(expression_slot(left)?, *variable);
    assert_eq!(expression_slot(right)?, value);
    Ok(())
}

fn assert_static_rank_range(
    expression: &ResidentQuantifierExpression,
    case: ScalarConsistencyCase,
) -> Result<()> {
    let expected = expected_rank_range(case);
    match expression {
        ResidentQuantifierExpression::Literal(actual) if *actual == expected => Ok(()),
        ResidentQuantifierExpression::List(items) => {
            let actual = items
                .iter()
                .map(|item| match item {
                    ResidentQuantifierExpression::Literal(value) => Ok(value.clone()),
                    _ => Err(Error::internal(format!(
                        "sealed rank range retained a dynamic item: {item:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(ResidentQuantifierValue::List(actual), expected);
            Ok(())
        }
        _ => Err(Error::internal(format!(
            "range(0, numOfValues-1) was not exactly normalized: {expression:?}"
        ))),
    }
}

fn source_obligation(
    source: &ResidentSegmentedAggregationSource,
) -> Result<ResidentExecutionObligation> {
    let ResidentSegmentedAggregationSource::Unit {
        prelude: ResidentSegmentedUnitPrelude::Empty,
        obligation,
    } = source
    else {
        return Err(Error::internal("scalar consistency source is not Unit"));
    };
    Ok(*obligation)
}

fn assert_sealed_program_and_receipts(
    case: ScalarConsistencyCase,
    request: &ResidentSegmentedAggregationRequest,
    capture: &ReceiptCapture,
) -> Result<()> {
    assert_eq!(request.project, PROJECT);
    assert_eq!(request.expected_bookmark, BOOKMARK);
    assert_eq!(request.expected_graph_revision, 0);
    assert_eq!(request.grouping_columns.len(), 1);
    assert_eq!(
        request.aggregates,
        vec![ResidentSegmentedAggregate {
            kind: ResidentSegmentedAggregateKind::Collect,
            input_column: Some(0),
            distinct: false,
            percentile: None,
        }]
    );
    assert_eq!(request.maximum_output_groups, 1);
    assert_eq!(
        request.maximum_output_cells,
        case.value_count as u32 + 1,
        "the sealed collect arena must reserve every ordered rank plus the final Boolean cell"
    );
    assert_eq!(request.maximum_output_arena_bytes, 0);
    assert_eq!(capture.graph_read_dependency_count, 0);

    let program = request
        .program
        .as_ref()
        .ok_or_else(|| Error::internal("scalar consistency request omitted its sealed program"))?;
    assert!(matches!(
        &program.source,
        ResidentSegmentedAggregationSource::Unit { .. }
    ));
    assert_eq!(program.maximum_intermediate_rows, case.value_count as u32);
    assert_eq!(program.maximum_list_items, case.value_count as u32);
    assert_eq!(program.random_seed, 0);
    assert_eq!(program.outputs.len(), 1);
    assert_eq!(program.outputs[0].name, "equal");

    let [
        first_project,
        size_project,
        unwind,
        rank_project,
        order,
        aggregate,
        final_project,
    ] = program.stages.as_slice()
    else {
        return Err(Error::internal(format!(
            "{} lowered {} stages instead of the exact seven-stage command",
            case.example_name,
            program.stages.len()
        )));
    };

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: first_bindings,
    } = &first_project.operation
    else {
        return Err(Error::internal(
            "stage 0 is not the literal values projection",
        ));
    };
    let [values_binding] = first_bindings.as_slice() else {
        return Err(Error::internal("stage 0 did not bind exactly `values`"));
    };
    assert_eq!(
        values_binding.expression,
        ResidentQuantifierExpression::Literal(expected_values(case))
    );

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: size_bindings,
    } = &size_project.operation
    else {
        return Err(Error::internal(
            "stage 1 is not values + size(values) projection",
        ));
    };
    let [copied_values, number_of_values] = size_bindings.as_slice() else {
        return Err(Error::internal(
            "stage 1 did not bind values and numOfValues exactly",
        ));
    };
    assert_eq!(
        expression_slot(&copied_values.expression)?,
        values_binding.output
    );
    assert_size_expression(
        &number_of_values.expression,
        values_binding.output,
        case.value_count,
    )?;

    let ResidentSegmentedAggregationOperation::Unwind {
        expression,
        output: unwound_value,
    } = &unwind.operation
    else {
        return Err(Error::internal("stage 2 is not UNWIND values"));
    };
    assert_eq!(expression_slot(expression)?, copied_values.output);

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: rank_bindings,
    } = &rank_project.operation
    else {
        return Err(Error::internal(
            "stage 3 is not rank + value + numOfValues projection",
        ));
    };
    let [rank, projected_value, projected_count] = rank_bindings.as_slice() else {
        return Err(Error::internal(
            "stage 3 did not bind x, value, and numOfValues exactly",
        ));
    };
    assert_rank_expression(&rank.expression, copied_values.output, *unwound_value)?;
    assert_eq!(
        expression_slot(&projected_value.expression)?,
        *unwound_value
    );
    assert_eq!(
        expression_slot(&projected_count.expression)?,
        number_of_values.output
    );

    let ResidentSegmentedAggregationOperation::Order { keys } = &order.operation else {
        return Err(Error::internal("stage 4 is not ORDER BY value"));
    };
    let [key] = keys.as_slice() else {
        return Err(Error::internal(
            "ORDER BY value did not lower exactly one key",
        ));
    };
    assert_eq!(expression_slot(&key.expression)?, projected_value.output);
    assert!(!key.descending);
    assert!(!key.nulls_first);

    let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
        &aggregate.operation
    else {
        return Err(Error::internal("stage 5 is not numOfValues + collect(x)"));
    };
    let ([group], [reduction]) = (groups.as_slice(), reductions.as_slice()) else {
        return Err(Error::internal(
            "stage 5 did not preserve one group and one reduction",
        ));
    };
    assert_eq!(expression_slot(&group.expression)?, projected_count.output);
    assert_eq!(reduction.kind, ResidentSegmentedAggregateKind::Collect);
    assert_eq!(
        reduction.input.as_ref(),
        Some(&ResidentQuantifierExpression::Slot(rank.output))
    );
    assert!(!reduction.distinct);

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: final_bindings,
    } = &final_project.operation
    else {
        return Err(Error::internal(
            "stage 6 is not the final equality projection",
        ));
    };
    let [equal] = final_bindings.as_slice() else {
        return Err(Error::internal("stage 6 did not bind exactly `equal`"));
    };
    let ResidentQuantifierExpression::Binary {
        left,
        operation: ResidentQuantifierBinary::Equal,
        right,
    } = &equal.expression
    else {
        return Err(Error::internal(
            "final projection lost orderedX = range(...) equality",
        ));
    };
    assert_eq!(expression_slot(left)?, reduction.output);
    assert_static_rank_range(right, case)?;
    assert_eq!(program.outputs[0].source, equal.output);

    let mut expected_obligations = vec![source_obligation(&program.source)?];
    for stage in &program.stages {
        expected_obligations.push(stage.obligation);
        if let ResidentSegmentedAggregationOperation::Aggregate { reductions, .. } =
            &stage.operation
        {
            expected_obligations.extend(reductions.iter().map(|item| item.obligation));
        }
    }
    assert_eq!(capture.receipts.len(), 9);
    assert_eq!(
        capture
            .receipts
            .iter()
            .map(|receipt| receipt.obligation)
            .collect::<Vec<_>>(),
        expected_obligations
    );
    assert!(capture.receipts.iter().all(|receipt| {
        receipt.execution == request.execution
            && receipt.completion == ResidentDeviceCompletion::CpuReference
    }));
    let rows = case.value_count as u64;
    assert_eq!(
        capture
            .receipts
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>(),
        vec![
            (1, 1),
            (1, 1),
            (1, 1),
            (1, rows),
            (rows, rows),
            (rows, rows),
            (rows, 1),
            (rows, 1),
            (1, 1),
        ]
    );
    Ok(())
}

#[test]
fn scalar_consistency_manifest_is_exact_and_non_overlapping() {
    assert_eq!(
        CASES.map(|case| case.report_index),
        [1_007, 1_008, 1_009, 1_010, 1_011]
    );
    assert_eq!(
        CASES.map(|case| case.displayed_id),
        [1_008, 1_009, 1_010, 1_011, 1_012]
    );
    assert_eq!(
        CASES.map(|case| case.example_name),
        ["booleans", "integers", "floats", "string", "lists"]
    );
    assert!(CASES.iter().all(|case| {
        case.displayed_id == case.report_index + 1
            && case.expanded_name == format!("{SCENARIO_PREFIX} [{}]", case.displayed_id)
    }));
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.report_index)
            .collect::<BTreeSet<_>>()
            .len(),
        CASES.len()
    );
    for case in CASES {
        let ResidentQuantifierValue::List(values) = expected_values(case) else {
            unreachable!("case oracle always returns a list")
        };
        assert_eq!(values.len(), case.value_count, "{}", case.example_name);
    }
}

#[test]
#[ignore = "external assurance gate: requires the exact fresh 3,682-Metal TCK report"]
fn fresh_report_pins_exact_scalar_consistency_failures() {
    let bytes = fs::read(REPORT_PATH).expect("fresh TCK report is readable");
    assert_eq!(hex::encode(Sha256::digest(&bytes)), REPORT_SHA256);
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).expect("fresh TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_682));
    assert_eq!(report["matched"].as_u64(), Some(3_682));
    assert_eq!(report["fully_conformant"].as_u64(), Some(3_682));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("fresh report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);

    for case in CASES {
        let scenario = &scenarios[case.report_index];
        assert!(
            scenario["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(FEATURE_SUFFIX))
        );
        assert_eq!(scenario["name"].as_str(), Some(case.expanded_name));
        assert_eq!(scenario["operation_count"].as_u64(), Some(1));
        assert_eq!(scenario["cpu_passed"].as_bool(), Some(true));
        assert_eq!(scenario["metal_passed"].as_bool(), Some(false));
        assert_eq!(scenario["cpu_metal_matched"].as_bool(), Some(false));
        assert_eq!(scenario["fully_conformant"].as_bool(), Some(false));
        assert_eq!(
            scenario["shared_failures"].as_array().map(Vec::len),
            Some(0)
        );
        assert_eq!(scenario["cpu_failures"].as_array().map(Vec::len), Some(0));
        let metal_failures = scenario["metal_failures"]
            .as_array()
            .expect("selected scenario has Metal failures");
        assert_eq!(metal_failures.len(), 1);
        assert!(
            metal_failures[0]
                .as_str()
                .is_some_and(|failure| failure.contains("GpuAdmissionFailure"))
        );
        assert_eq!(scenario["divergences"].as_array().map(Vec::len), Some(1));

        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let path = candidate["path"].as_str()?;
                let name = candidate["name"].as_str()?;
                (path.ends_with(FEATURE_SUFFIX) && name == case.expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(matches, vec![case.report_index]);
    }
}

#[test]
fn strict_cpu_reference_executes_exact_five_in_one_sealed_command_each() -> Result<()> {
    for case in CASES {
        let fixture = Fixture::new();
        let backend = fixture.backend()?;
        let observations = backend.observations();
        let output = execute(&fixture, &backend, &query(case))?;
        assert_exact_public_output(case, &output)?;
        assert_eq!(
            observations.snapshot(),
            CallSnapshot {
                pins: 1,
                segmented: 1,
                sealed: 1,
                unit_source: 1,
                ..CallSnapshot::default()
            },
            "{} crossed an alternative or partial backend route",
            case.example_name
        );

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let captures = observations
            .receipt_captures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ([request], [capture]) = (requests.as_slice(), captures.as_slice()) else {
            return Err(Error::internal(format!(
                "{} did not produce exactly one request and receipt capture",
                case.example_name
            )));
        };
        assert_sealed_program_and_receipts(case, request, capture)?;
    }
    Ok(())
}

#[test]
fn adversarial_near_misses_never_dispatch_the_scalar_segmented_route() -> Result<()> {
    let near_misses = [
        (
            "nested lists deeper than the exact integer-list domain",
            "WITH [[[1]], [[2]]] AS values\n\
             WITH values, size(values) AS numOfValues\n\
             UNWIND values AS value\n\
             WITH size([x IN values WHERE x < value]) AS x, value, numOfValues ORDER BY value\n\
             WITH numOfValues, collect(x) AS orderedX\n\
             RETURN orderedX = range(0, numOfValues-1) AS equal",
        ),
        (
            "mixed inner list values are outside the exact integer-list domain",
            "WITH [[1], ['2']] AS values\n\
             WITH values, size(values) AS numOfValues\n\
             UNWIND values AS value\n\
             WITH size([x IN values WHERE x < value]) AS x, value, numOfValues ORDER BY value\n\
             WITH numOfValues, collect(x) AS orderedX\n\
             RETURN orderedX = range(0, numOfValues-1) AS equal",
        ),
        (
            "MAP ordering has no scalar consistency semantics",
            "WITH [{k: 1}, {k: 2}] AS values\n\
             WITH values, size(values) AS numOfValues\n\
             UNWIND values AS value\n\
             WITH size([x IN values WHERE x < value]) AS x, value, numOfValues ORDER BY value\n\
             WITH numOfValues, collect(x) AS orderedX\n\
             RETURN orderedX = range(0, numOfValues-1) AS equal",
        ),
        (
            "DISTINCT collect is not the ordered non-distinct proof",
            "WITH [true, false] AS values\n\
             WITH values, size(values) AS numOfValues\n\
             UNWIND values AS value\n\
             WITH size([x IN values WHERE x < value]) AS x, value, numOfValues ORDER BY value\n\
             WITH numOfValues, collect(DISTINCT x) AS orderedX\n\
             RETURN orderedX = range(0, numOfValues-1) AS equal",
        ),
        (
            "volatile secondary ordering is outside the exact proof",
            "WITH [true, false] AS values\n\
             WITH values, size(values) AS numOfValues\n\
             UNWIND values AS value\n\
             WITH size([x IN values WHERE x < value]) AS x, value, numOfValues\n\
               ORDER BY value, rand()\n\
             WITH numOfValues, collect(x) AS orderedX\n\
             RETURN orderedX = range(0, numOfValues-1) AS equal",
        ),
    ];

    for (name, query) in near_misses {
        let fixture = Fixture::new();
        let backend = fixture.backend()?;
        let observations = backend.observations();
        execute(&fixture, &backend, query)
            .expect_err("an unproved scalar-consistency near miss must fail closed");
        assert_eq!(
            observations.snapshot(),
            CallSnapshot::default(),
            "near miss reached a backend command: {name}"
        );
        assert!(
            observations
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty(),
            "near miss captured a segmented request: {name}"
        );
        assert!(
            observations
                .receipt_captures
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty(),
            "near miss captured segmented receipts: {name}"
        );
    }
    Ok(())
}
