// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance for the five temporal examples in `WithOrderBy1.feature`
//! scenario [45].
//!
//! The two remaining mixed graph/document total-order examples are intentionally not part of
//! this gate: they require graph handles, paths, heterogeneous documents, NaN, and NULL result
//! publication, while these five cases can extend the existing seven-stage Unit consistency
//! command with one homogeneous raw temporal-constructor list.

use std::{
    collections::BTreeMap,
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
        BindCapabilities, Clause, ColumnType, ExecutionContext, ExecutionOutput, Expression,
        QueryEngine, ResultValue, Statement, StatementStats, parse,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentExecutionObligation, ResidentExecutionReceipt,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectImage,
        ResidentQuantifierBinary, ResidentQuantifierExpression, ResidentQuantifierFunction,
        ResidentSegmentedAggregate, ResidentSegmentedAggregateKind,
        ResidentSegmentedAggregationOperation, ResidentSegmentedAggregationRequest,
        ResidentSegmentedAggregationResult, ResidentSegmentedAggregationSource,
        ResidentSegmentedUnitPrelude, ResidentSortRequest, ResidentSortResult,
        ResidentTemporalValueFunction, ResidentTemporalValueInput, ResidentTemporalValueInvocation,
        ResidentTemporalValueProgramRequest, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5749_5448_4f52_4445_525f_5445_4d50_0001,
));
const BOOKMARK: Bookmark = Bookmark { term: 45, index: 0 };
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const REPORT: &str = "/tmp/irongraph-tck-full-20260721-after-create3-list12-delete5.json";
const REPORT_SHA256: &str = "66383b9f0075863b17778ad4348af6c729e8ccc204c8fe67b8d5b67712ffa288";
const FEATURE_SUFFIX: &str = "features/clauses/with-orderBy/WithOrderBy1.feature";
const SCENARIO_PREFIX: &str = "[45] Sort order should be consistent with comparisons where comparisons are defined #Example: <exampleName>";
const STALE_FAILURE_INDICES: [usize; 7] = [951, 952, 1_012, 1_013, 1_014, 1_015, 1_016];
const MIXED_FAILURE_NAMES: [&str; 2] = [
    "[21] Sort distinct types in ascending order",
    "[22] Sort distinct types in descending order",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TemporalCase {
    report_index: usize,
    displayed_id: usize,
    example_name: &'static str,
    values_literal: &'static str,
    value_count: usize,
    function: ResidentTemporalValueFunction,
}

const CASES: [TemporalCase; 5] = [
    TemporalCase {
        report_index: 1_012,
        displayed_id: 1_013,
        example_name: "dates",
        values_literal: "[date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 12, day: 24}), date({year: 1984, month: 10, day: 12}), date({year: 1985, month: 5, day: 6}), date({year: 1980, month: 10, day: 24}), date({year: 1984, month: 10, day: 11})]",
        value_count: 6,
        function: ResidentTemporalValueFunction::Date,
    },
    TemporalCase {
        report_index: 1_013,
        displayed_id: 1_014,
        example_name: "localtimes",
        values_literal: "[localtime({hour: 10, minute: 35}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}), localtime({hour: 12, minute: 35, second: 13}), localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 15})]",
        value_count: 6,
        function: ResidentTemporalValueFunction::LocalTime,
    },
    TemporalCase {
        report_index: 1_014,
        displayed_id: 1_015,
        example_name: "times",
        values_literal: "[time({hour: 10, minute: 35, timezone: '-08:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})]",
        value_count: 6,
        function: ResidentTemporalValueFunction::Time,
    },
    TemporalCase {
        report_index: 1_015,
        displayed_id: 1_016,
        example_name: "localdatetimes",
        values_literal: "[localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}), localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}), localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}), localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})]",
        value_count: 5,
        function: ResidentTemporalValueFunction::LocalDateTime,
    },
    TemporalCase {
        report_index: 1_016,
        displayed_id: 1_017,
        example_name: "datetimes",
        values_literal: "[datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})]",
        value_count: 5,
        function: ResidentTemporalValueFunction::DateTime,
    },
];

fn expanded_name(case: TemporalCase) -> String {
    format!("{SCENARIO_PREFIX} [{}]", case.displayed_id)
}

fn query(case: TemporalCase) -> String {
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

fn expected_temporal_map(entries: &[(String, Expression)]) -> Result<Vec<u8>> {
    let mut packet = vec![
        1,
        u8::try_from(entries.len())
            .map_err(|_| Error::internal("expected temporal map exceeds u8 field count"))?,
    ];
    for (key, value) in entries {
        packet.push(match key.as_str() {
            "year" => 1,
            "month" => 2,
            "day" => 3,
            "hour" => 9,
            "minute" => 10,
            "second" => 11,
            "nanosecond" => 14,
            "timezone" => 15,
            _ => {
                return Err(Error::internal(format!(
                    "unexpected temporal map key `{key}`"
                )));
            }
        });
        match value {
            Expression::Literal(ScalarValue::Null) => packet.push(0),
            Expression::Literal(ScalarValue::Integer(value)) => {
                packet.push(1);
                packet.extend_from_slice(&value.to_le_bytes());
            }
            Expression::Literal(ScalarValue::String(value)) => {
                packet.push(3);
                let bytes = value.as_bytes();
                packet.extend_from_slice(
                    &u32::try_from(bytes.len())
                        .map_err(|_| Error::internal("temporal map string exceeds u32"))?
                        .to_le_bytes(),
                );
                packet.extend_from_slice(bytes);
            }
            _ => {
                return Err(Error::internal(format!(
                    "unexpected temporal map value `{value:?}`"
                )));
            }
        }
    }
    Ok(packet)
}

fn expected_temporal_program(case: TemporalCase) -> Result<ResidentTemporalValueProgramRequest> {
    let parsed = parse(&format!("RETURN {} AS values", case.values_literal))?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("expected temporal source is not a query"));
    };
    let [Clause::Return(projection)] = body.clauses.as_slice() else {
        return Err(Error::internal("expected temporal source lost RETURN"));
    };
    let [item] = projection.items.as_slice() else {
        return Err(Error::internal("expected temporal source changed width"));
    };
    let Expression::List(items) = &item.expression else {
        return Err(Error::internal("expected temporal source is not a list"));
    };
    if items.len() != case.value_count {
        return Err(Error::internal("expected temporal source changed length"));
    }
    let mut invocations = Vec::with_capacity(items.len());
    for item in items {
        let Expression::Function {
            name,
            distinct: false,
            arguments,
        } = item
        else {
            return Err(Error::internal(
                "expected temporal item is not a constructor",
            ));
        };
        let [Expression::Map(entries)] = arguments.as_slice() else {
            return Err(Error::internal(
                "expected temporal constructor lost its raw map",
            ));
        };
        let expected_name = match case.function {
            ResidentTemporalValueFunction::Date => "date",
            ResidentTemporalValueFunction::LocalTime => "localtime",
            ResidentTemporalValueFunction::Time => "time",
            ResidentTemporalValueFunction::LocalDateTime => "localdatetime",
            ResidentTemporalValueFunction::DateTime => "datetime",
            ResidentTemporalValueFunction::Duration => {
                return Err(Error::internal(
                    "Duration entered the temporal consistency gate",
                ));
            }
        };
        if name.len() != 1 || name[0] != expected_name {
            return Err(Error::internal(format!(
                "expected temporal constructor changed family: {name:?}"
            )));
        }
        invocations.push(ResidentTemporalValueInvocation {
            function: case.function,
            input: ResidentTemporalValueInput::Map(expected_temporal_map(entries)?),
        });
    }
    Ok(ResidentTemporalValueProgramRequest {
        output_registers: (0..items.len())
            .map(|register| {
                u16::try_from(register)
                    .map_err(|_| Error::internal("expected temporal register exceeds u16"))
            })
            .collect::<Result<Vec<_>>>()?,
        invocations,
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CallSnapshot {
    pins: usize,
    segmented: usize,
    forbidden: usize,
}

#[derive(Clone, Debug)]
struct ReceiptCapture {
    receipts: Vec<ResidentExecutionReceipt>,
    graph_read_dependency_count: usize,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    segmented: AtomicUsize,
    forbidden: AtomicUsize,
    requests: Mutex<Vec<ResidentSegmentedAggregationRequest>>,
    captures: Mutex<Vec<ReceiptCapture>>,
}

impl Observations {
    fn snapshot(&self) -> CallSnapshot {
        CallSnapshot {
            pins: self.pins.load(Ordering::SeqCst),
            segmented: self.segmented.load(Ordering::SeqCst),
            forbidden: self.forbidden.load(Ordering::SeqCst),
        }
    }
}

/// Before pinning this wrapper advertises Metal, making host execution ineligible. Its pinned
/// form exposes honest CPU provenance and delegates only the complete segmented command.
struct StrictTemporalConsistencyBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictTemporalConsistencyBackend {
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

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations.forbidden.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict temporal consistency gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictTemporalConsistencyBackend {
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
        if self.pinned || project != PROJECT {
            return self.reject("pin_project");
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

    fn supports_native_segmented_aggregation(&self) -> bool {
        self.inner.supports_native_segmented_aggregation()
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        if !self.pinned {
            return self.reject("execute_segmented_aggregation_without_pin");
        }
        let Some(program) = request.program.as_ref() else {
            return self.reject("legacy_host_relation_aggregation");
        };
        if request.input.row_count != 0
            || request.input.column_count != 0
            || !request.input.cells.is_empty()
            || !request.input.arena.is_empty()
            || !matches!(
                &program.source,
                ResidentSegmentedAggregationSource::Unit { .. }
            )
        {
            return self.reject("non_unit_or_host_materialized_aggregation");
        }
        request.validate()?;
        self.observations.segmented.fetch_add(1, Ordering::SeqCst);
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
            .captures
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

struct Fixture {
    graph: GraphStore,
}

impl Fixture {
    fn new() -> Self {
        Self {
            graph: GraphStore::default(),
        }
    }

    fn backend(&self) -> Result<StrictTemporalConsistencyBackend> {
        let image = ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Ok(StrictTemporalConsistencyBackend::new(cpu))
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
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 4_096,
        max_batch_rows: 8,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn output_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal(
                "temporal consistency output has an invalid columnar shape",
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

fn assert_public_output(case: TemporalCase, output: &ExecutionOutput) -> Result<()> {
    assert_eq!(
        output.result.schema,
        [("equal".to_owned(), ColumnType::Boolean)],
        "{} changed its output schema",
        case.example_name
    );
    assert_eq!(
        output_rows(output)?,
        [vec![ResultValue::Scalar(ScalarValue::Boolean(true))]],
        "{} lost ORDER/comparison consistency",
        case.example_name
    );
    assert_eq!(output.result.bookmark, BOOKMARK);
    assert_eq!(output.result.statistics, StatementStats::default());
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    Ok(())
}

fn expression_slot(
    expression: &ResidentQuantifierExpression,
) -> Result<irongraph::gpu::ResidentQuantifierSlot> {
    let ResidentQuantifierExpression::Slot(slot) = expression else {
        return Err(Error::internal(format!(
            "expected sealed slot expression, got {expression:?}"
        )));
    };
    Ok(*slot)
}

fn assert_rank_expression(
    expression: &ResidentQuantifierExpression,
    values: irongraph::gpu::ResidentQuantifierSlot,
    value: irongraph::gpu::ResidentQuantifierSlot,
) -> Result<()> {
    let ResidentQuantifierExpression::Function {
        function: ResidentQuantifierFunction::Size,
        arguments,
    } = expression
    else {
        return Err(Error::internal("rank is not size(list-comprehension)"));
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
        return Err(Error::internal("rank lost its exact list comprehension"));
    };
    assert_eq!(expression_slot(list)?, values);
    let ResidentQuantifierExpression::Binary {
        left,
        operation: ResidentQuantifierBinary::Less,
        right,
    } = predicate.as_ref()
    else {
        return Err(Error::internal("rank lost iterator < value"));
    };
    assert_eq!(expression_slot(left)?, *variable);
    assert_eq!(expression_slot(right)?, value);
    Ok(())
}

fn assert_size_expression(
    expression: &ResidentQuantifierExpression,
    values: irongraph::gpu::ResidentQuantifierSlot,
    count: usize,
) -> Result<()> {
    match expression {
        ResidentQuantifierExpression::Function {
            function: ResidentQuantifierFunction::Size,
            arguments,
        } if arguments.as_slice() == [ResidentQuantifierExpression::Slot(values)] => Ok(()),
        ResidentQuantifierExpression::Literal(
            irongraph::gpu::ResidentQuantifierValue::Integer(actual),
        ) if *actual == count as i64 => Ok(()),
        _ => Err(Error::internal(format!(
            "size(values) changed its sealed expression: {expression:?}"
        ))),
    }
}

fn assert_rank_range(expression: &ResidentQuantifierExpression, count: usize) -> Result<()> {
    let ResidentQuantifierExpression::Literal(irongraph::gpu::ResidentQuantifierValue::List(
        values,
    )) = expression
    else {
        return Err(Error::internal(
            "final rank range is not one immutable list",
        ));
    };
    assert_eq!(values.len(), count);
    assert!(values.iter().enumerate().all(|(index, value)| {
        value == &irongraph::gpu::ResidentQuantifierValue::Integer(index as i64)
    }));
    Ok(())
}

fn assert_sealed_request(
    case: TemporalCase,
    request: &ResidentSegmentedAggregationRequest,
    capture: &ReceiptCapture,
) -> Result<()> {
    assert_eq!(request.project, PROJECT);
    assert_eq!(request.expected_bookmark, BOOKMARK);
    assert_eq!(request.expected_graph_revision, 0);
    assert_eq!(request.grouping_columns.len(), 1);
    assert_eq!(
        request.aggregates,
        [ResidentSegmentedAggregate {
            kind: ResidentSegmentedAggregateKind::Collect,
            input_column: Some(0),
            distinct: false,
            percentile: None,
        }]
    );
    assert_eq!(request.maximum_output_groups, 1);
    assert_eq!(request.maximum_output_cells, case.value_count as u32 + 1);
    assert_eq!(request.maximum_output_arena_bytes, 0);
    assert_eq!(capture.graph_read_dependency_count, 0);

    let program = request
        .program
        .as_ref()
        .ok_or_else(|| Error::internal("temporal consistency request has no sealed program"))?;
    let ResidentSegmentedAggregationSource::Unit {
        prelude:
            ResidentSegmentedUnitPrelude::RawTemporalList {
                program: temporal_program,
                output: temporal_output,
            },
        obligation: source_obligation,
    } = &program.source
    else {
        return Err(Error::internal(
            "temporal consistency source lost its raw temporal Unit prelude",
        ));
    };
    assert_eq!(temporal_program, &expected_temporal_program(case)?);
    assert_eq!(program.slot_count, 11);
    assert_eq!(program.maximum_intermediate_rows, case.value_count as u32);
    assert_eq!(program.maximum_list_items, case.value_count as u32);
    assert_eq!(program.random_seed, 0);
    assert_eq!(program.outputs.len(), 1);
    assert_eq!(program.outputs[0].name, "equal");

    let [first, size, unwind, rank, order, aggregate, final_project] = program.stages.as_slice()
    else {
        return Err(Error::internal(format!(
            "{} did not lower exactly seven stages",
            case.example_name
        )));
    };
    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: first_bindings,
    } = &first.operation
    else {
        return Err(Error::internal(
            "stage 0 is not the temporal-list projection",
        ));
    };
    let [values_binding] = first_bindings.as_slice() else {
        return Err(Error::internal("stage 0 did not bind exactly `values`"));
    };
    assert_eq!(*temporal_output, values_binding.output);
    assert_eq!(
        expression_slot(&values_binding.expression)?,
        values_binding.output
    );

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: size_bindings,
    } = &size.operation
    else {
        return Err(Error::internal("stage 1 is not values + size projection"));
    };
    let [copied_values, number_of_values] = size_bindings.as_slice() else {
        return Err(Error::internal("stage 1 changed projection width"));
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
    } = &rank.operation
    else {
        return Err(Error::internal("stage 3 is not rank projection"));
    };
    let [rank_binding, projected_value, projected_count] = rank_bindings.as_slice() else {
        return Err(Error::internal("stage 3 changed projection width"));
    };
    assert_rank_expression(
        &rank_binding.expression,
        copied_values.output,
        *unwound_value,
    )?;
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
        return Err(Error::internal("stage 4 changed ORDER key count"));
    };
    assert_eq!(expression_slot(&key.expression)?, projected_value.output);
    assert!(!key.descending && !key.nulls_first);

    let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
        &aggregate.operation
    else {
        return Err(Error::internal("stage 5 is not collect aggregation"));
    };
    let ([group], [reduction]) = (groups.as_slice(), reductions.as_slice()) else {
        return Err(Error::internal("stage 5 changed group/reduction count"));
    };
    assert_eq!(expression_slot(&group.expression)?, projected_count.output);
    assert_eq!(reduction.kind, ResidentSegmentedAggregateKind::Collect);
    assert_eq!(
        reduction.input.as_ref(),
        Some(&ResidentQuantifierExpression::Slot(rank_binding.output))
    );
    assert!(!reduction.distinct);

    let ResidentSegmentedAggregationOperation::Project {
        keep_scope: false,
        bindings: final_bindings,
    } = &final_project.operation
    else {
        return Err(Error::internal("stage 6 is not equality projection"));
    };
    let [equal] = final_bindings.as_slice() else {
        return Err(Error::internal("stage 6 changed projection width"));
    };
    let ResidentQuantifierExpression::Binary {
        left,
        operation: ResidentQuantifierBinary::Equal,
        right,
    } = &equal.expression
    else {
        return Err(Error::internal("stage 6 lost rank-range equality"));
    };
    assert_eq!(expression_slot(left)?, reduction.output);
    assert_rank_range(right, case.value_count)?;
    assert_eq!(program.outputs[0].source, equal.output);

    let mut expected_obligations = Vec::<ResidentExecutionObligation>::with_capacity(9);
    expected_obligations.push(*source_obligation);
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
        [
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
fn stale_seven_split_into_two_non_overlapping_native_contracts() {
    assert_eq!(
        CASES.map(|case| case.report_index),
        [1_012, 1_013, 1_014, 1_015, 1_016]
    );
    assert_eq!(
        CASES.map(|case| case.displayed_id),
        [1_013, 1_014, 1_015, 1_016, 1_017]
    );
    assert_eq!(
        CASES.map(|case| case.example_name),
        [
            "dates",
            "localtimes",
            "times",
            "localdatetimes",
            "datetimes"
        ]
    );
    let mut split = vec![951, 952];
    split.extend(CASES.map(|case| case.report_index));
    assert_eq!(split, STALE_FAILURE_INDICES);
    assert_eq!(MIXED_FAILURE_NAMES.len(), 2);
    assert!(CASES.iter().all(|case| {
        case.displayed_id == case.report_index + 1
            && expanded_name(*case) == format!("{SCENARIO_PREFIX} [{}]", case.displayed_id)
    }));
}

#[test]
#[ignore = "external assurance gate: requires the exact stale 3,703-Metal full report"]
fn stale_report_pins_exact_seven_with_order_by_failures() {
    let bytes = fs::read(REPORT).expect("stale report is readable");
    assert_eq!(hex::encode(Sha256::digest(&bytes)), REPORT_SHA256);
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).expect("stale report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_703));
    assert_eq!(report["matched"].as_u64(), Some(3_703));
    assert_eq!(report["fully_conformant"].as_u64(), Some(3_703));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("stale report has scenarios");
    let failures = scenarios
        .iter()
        .enumerate()
        .filter_map(|(index, scenario)| {
            (scenario["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(FEATURE_SUFFIX))
                && scenario["cpu_passed"].as_bool() == Some(true)
                && scenario["metal_passed"].as_bool() == Some(false))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(failures, STALE_FAILURE_INDICES);
    for (index, expected) in [951, 952].into_iter().zip(MIXED_FAILURE_NAMES) {
        assert_eq!(scenarios[index]["name"].as_str(), Some(expected));
    }
    for case in CASES {
        let scenario = &scenarios[case.report_index];
        let expected_name = expanded_name(case);
        assert_eq!(scenario["name"].as_str(), Some(expected_name.as_str()));
        assert_eq!(scenario["operation_count"].as_u64(), Some(1));
        assert_eq!(scenario["cpu_failures"].as_array().map(Vec::len), Some(0));
        assert_eq!(scenario["metal_failures"].as_array().map(Vec::len), Some(1));
        assert!(
            scenario["metal_failures"][0]
                .as_str()
                .is_some_and(|failure| failure.contains("GpuAdmissionFailure"))
        );
    }
}

#[test]
fn strict_cpu_reference_executes_exact_five_temporal_cases_as_one_command_each() -> Result<()> {
    for case in CASES {
        let fixture = Fixture::new();
        let backend = fixture.backend()?;
        let observations = backend.observations();
        let output = QueryEngine.execute(&query(case), &mut context(&fixture, &backend))?;
        assert_public_output(case, &output)?;
        assert_eq!(
            observations.snapshot(),
            CallSnapshot {
                pins: 1,
                segmented: 1,
                forbidden: 0,
            },
            "{} used a fallback or partial route",
            case.example_name
        );
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let captures = observations
            .captures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let ([request], [capture]) = (requests.as_slice(), captures.as_slice()) else {
            return Err(Error::internal(format!(
                "{} did not produce one request and receipt capture",
                case.example_name
            )));
        };
        assert_sealed_request(case, request, capture)?;
    }
    Ok(())
}
