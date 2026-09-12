// Test-only module. Clippy's `allow-expect-in-tests` covers `#[test]` bodies but not the helper
// functions those tests call, and a failed expectation in a fixture is the intended way for a test
// to fail. The production denial of `expect` is unaffected.
#![allow(clippy::expect_used)]

//! Strict native-route proof for TypeConversion4 [5]/[8]/[9].
//!
//! The outer backend reports Metal until one immutable project pin, then permits exactly one
//! complete quantifier-program call into the CPU semantic reference. Every decomposed primitive
//! is poisoned, so generic evaluation or host-side UNWIND/projection cannot satisfy this test.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentQuantifierExpression,
        ResidentQuantifierFunction, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentQuantifierSource, ResidentQuantifierStage,
        ResidentQuantifierValue, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5459_5045_434f_4e56_345f_544f_5354_5238,
));
const MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy)]
enum ExactShape {
    MixedAnyListComprehension,
    CoalesceAfterToString,
    ToStringAfterCoalesce,
}

#[derive(Clone, Copy)]
struct Case {
    report_index: usize,
    query: &'static str,
    shape: ExactShape,
}

const CASES: [Case; 3] = [
    Case {
        report_index: 3857,
        query: "RETURN [x IN [1, 2.3, true, 'apa'] | toString(x)] AS list",
        shape: ExactShape::MixedAnyListComprehension,
    },
    Case {
        report_index: 3860,
        query: "UNWIND ['male', 'female', null] AS gen RETURN coalesce(toString(gen), 'x') AS result",
        shape: ExactShape::CoalesceAfterToString,
    },
    Case {
        report_index: 3861,
        query: "UNWIND ['male', 'female', null] AS gen RETURN toString(coalesce(gen, 'x')) AS result",
        shape: ExactShape::ToStringAfterCoalesce,
    },
];

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    quantifier_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentQuantifierProgramRequest>>,
}

struct StrictToStringBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictToStringBackend {
    fn new(graph: &GraphStore) -> Result<Self> {
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(ResidentProjectImage::build(
            PROJECT,
            bookmark(graph),
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?)?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(Observations::default()),
        })
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict TypeConversion4 backend rejected route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictToStringBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.inner.kind()
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
        if self.pinned {
            return self.reject("admit_project");
        }
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject("replace_all_projects");
        }
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        if self.pinned {
            return self.reject("evict_project");
        }
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        if self.pinned {
            self.observations
                .forbidden_calls
                .fetch_add(1, Ordering::SeqCst);
        } else {
            self.inner.advance_bookmark(bookmark);
        }
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

    fn supports_native_quantifier_program(&self) -> bool {
        true
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        if !self.pinned {
            return self.reject("execute_quantifier_program_unpinned");
        }
        request.validate()?;
        self.observations
            .quantifier_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_quantifier_program(request, cancellation)
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

fn bookmark(graph: &GraphStore) -> Bookmark {
    Bookmark {
        term: 84,
        index: graph.revision(),
    }
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
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
        parameters: BTreeMap::new(),
        bookmark: bookmark(graph),
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: false,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn result_rows(output: &irongraph::cypher::ExecutionOutput) -> Vec<Vec<ResultValue>> {
    output
        .result
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

fn literal_x(expression: &ResidentQuantifierExpression) -> bool {
    matches!(
        expression,
        ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(value))
            if value == "x"
    )
}

fn assert_exact_request(
    request: &ResidentQuantifierProgramRequest,
    shape: ExactShape,
) -> Result<()> {
    request.validate()?;
    if !matches!(&request.source, ResidentQuantifierSource::Unit) {
        return Err(Error::internal(
            "TypeConversion4 changed its graph-free Unit source",
        ));
    }
    let output_slot = match shape {
        ExactShape::MixedAnyListComprehension => {
            let [
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings,
                },
            ] = request.program.stages.as_slice()
            else {
                return Err(Error::internal(
                    "TypeConversion4 [5] changed its exact Unit/project stage shape",
                ));
            };
            let [binding] = bindings.as_slice() else {
                return Err(Error::internal(
                    "TypeConversion4 [5] changed its final projection width",
                ));
            };
            let ResidentQuantifierExpression::ListComprehension {
                variable,
                list,
                predicate: None,
                projection: Some(projection),
            } = &binding.expression
            else {
                return Err(Error::internal(
                    "TypeConversion4 [5] changed its list-comprehension expression",
                ));
            };
            if !matches!(
                list.as_ref(),
                ResidentQuantifierExpression::List(values)
                    if matches!(
                        values.as_slice(),
                        [
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Integer(1)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Float(bits)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Boolean(true)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::String(text)
                            ),
                        ] if *bits == 2.3_f64.to_bits() && text == "apa"
                    )
            ) || !matches!(
                projection.as_ref(),
                ResidentQuantifierExpression::Function {
                    function: ResidentQuantifierFunction::ToString,
                    arguments,
                } if matches!(
                    arguments.as_slice(),
                    [ResidentQuantifierExpression::Slot(slot)] if slot == variable
                )
            ) {
                return Err(Error::internal(
                    "TypeConversion4 [5] changed its mixed scalar toString projection",
                ));
            }
            binding.output
        }
        ExactShape::CoalesceAfterToString | ExactShape::ToStringAfterCoalesce => {
            let [
                ResidentQuantifierStage::Unwind {
                    expression: ResidentQuantifierExpression::List(values),
                    output: unwind_output,
                },
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings,
                },
            ] = request.program.stages.as_slice()
            else {
                return Err(Error::internal(
                    "TypeConversion4 changed its exact UNWIND/project stage shape",
                ));
            };
            if !matches!(
                values.as_slice(),
                [
                    ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(male)),
                    ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(female)),
                    ResidentQuantifierExpression::Literal(ResidentQuantifierValue::Null),
                ] if male == "male" && female == "female"
            ) {
                return Err(Error::internal(
                    "TypeConversion4 changed its immutable UNWIND values",
                ));
            }
            let [binding] = bindings.as_slice() else {
                return Err(Error::internal(
                    "TypeConversion4 changed its final projection width",
                ));
            };
            let exact = match shape {
                ExactShape::CoalesceAfterToString => matches!(
                    &binding.expression,
                    ResidentQuantifierExpression::Function {
                        function: ResidentQuantifierFunction::Coalesce,
                        arguments,
                    } if matches!(
                        arguments.as_slice(),
                        [
                            ResidentQuantifierExpression::Function {
                                function: ResidentQuantifierFunction::ToString,
                                arguments: inner,
                            },
                            suffix,
                        ] if matches!(inner.as_slice(), [ResidentQuantifierExpression::Slot(slot)] if slot == unwind_output)
                            && literal_x(suffix)
                    )
                ),
                ExactShape::ToStringAfterCoalesce => matches!(
                    &binding.expression,
                    ResidentQuantifierExpression::Function {
                        function: ResidentQuantifierFunction::ToString,
                        arguments,
                    } if matches!(
                        arguments.as_slice(),
                        [ResidentQuantifierExpression::Function {
                            function: ResidentQuantifierFunction::Coalesce,
                            arguments: inner,
                        }] if matches!(
                            inner.as_slice(),
                            [ResidentQuantifierExpression::Slot(slot), suffix]
                                if slot == unwind_output && literal_x(suffix)
                        )
                    )
                ),
                ExactShape::MixedAnyListComprehension => unreachable!("outer match excluded it"),
            };
            if !exact {
                return Err(Error::internal(
                    "TypeConversion4 changed its sealed toString/coalesce expression tree",
                ));
            }
            binding.output
        }
    };
    if !matches!(
        request.program.outputs.as_slice(),
        [output] if output.source == output_slot
            && match shape {
                ExactShape::MixedAnyListComprehension => output.name == "list",
                ExactShape::CoalesceAfterToString | ExactShape::ToStringAfterCoalesce => {
                    output.name == "result"
                }
            }
    ) {
        return Err(Error::internal(
            "TypeConversion4 changed its final output binding",
        ));
    }

    // The ABI rejects adjacent broader function shapes while constructing the immutable request;
    // neither malformed command is sent to a backend or granted a fresh execution boundary.
    for arity in [0_usize, 2] {
        let mut program = request.program.clone();
        let Some(ResidentQuantifierStage::Project { bindings, .. }) = program.stages.last_mut()
        else {
            unreachable!("exact stage shape checked above")
        };
        bindings[0].expression = ResidentQuantifierExpression::Function {
            function: ResidentQuantifierFunction::ToString,
            arguments: vec![
                ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(
                    "x".to_owned(),
                ));
                arity
            ],
        };
        let error = ResidentQuantifierProgramRequest::build_with_source(
            request.generation,
            request.execution,
            request.source.clone(),
            program,
            request.max_rows,
            request.max_list_items,
            request.random_seed,
        )
        .expect_err("malformed toString arity unexpectedly entered the sealed ABI");
        if error.code != ErrorCode::QueryType {
            return Err(Error::internal(format!(
                "malformed toString arity {arity} returned {error}"
            )));
        }
    }
    Ok(())
}

#[test]
fn type_conversion4_string_null_unwind_uses_one_complete_quantifier_command() -> Result<()> {
    let graph = GraphStore::default();

    for case in CASES {
        let (expected_schema, expected_rows) = match case.shape {
            ExactShape::MixedAnyListComprehension => (
                vec![("list".to_owned(), ColumnType::List)],
                vec![vec![ResultValue::List(vec![
                    ResultValue::Scalar(ScalarValue::String("1".into())),
                    ResultValue::Scalar(ScalarValue::String("2.3".into())),
                    ResultValue::Scalar(ScalarValue::String("true".into())),
                    ResultValue::Scalar(ScalarValue::String("apa".into())),
                ])]],
            ),
            ExactShape::CoalesceAfterToString | ExactShape::ToStringAfterCoalesce => (
                vec![("result".to_owned(), ColumnType::String)],
                vec![
                    vec![ResultValue::Scalar(ScalarValue::String("male".into()))],
                    vec![ResultValue::Scalar(ScalarValue::String("female".into()))],
                    vec![ResultValue::Scalar(ScalarValue::String("x".into()))],
                ],
            ),
        };
        let reference = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictToStringBackend::new(&graph)?;
        let observations = backend.observations();
        let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;

        assert_eq!(
            native.result, reference.result,
            "report {}",
            case.report_index
        );
        assert_eq!(
            native.result.schema, expected_schema,
            "report {}",
            case.report_index
        );
        assert_eq!(
            result_rows(&native),
            expected_rows,
            "report {}",
            case.report_index
        );
        assert_eq!(native.result.statistics, StatementStats::default());
        assert!(native.graph_mutations.is_empty());
        assert!(native.temporal_mutations.is_empty());
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.quantifier_calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let [request] = requests.as_slice() else {
            return Err(Error::internal(format!(
                "report {} changed native request cardinality",
                case.report_index
            )));
        };
        assert_exact_request(request, case.shape)?;
    }
    Ok(())
}
