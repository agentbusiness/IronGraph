// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Fresh-report manifest for the five remaining native `List12.feature` gaps.
//!
//! This pins identities, exact primary queries, and a collision-safe implementation partition.
//! It deliberately supplies no substitute backend semantics: the certified generic CPU run is the
//! oracle, while native CPU/Metal execution gates belong beside each production tranche.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use serde::Deserialize;
use sha2::{Digest as _, Sha256};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BinaryOperator, BindCapabilities, ExecutionContext, ExecutionOutput, Expression,
        PhysicalOperator, QueryEngine, ResultValue, StatementStats, bind, parse, plan,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentMutationProjectionSource, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentNullableRelationRequest, ResidentNullableRelationResult, ResidentProjectImage,
        ResidentQuantifierExpression, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSegmentedAggregateKind, ResidentSegmentedAggregationOperation,
        ResidentSegmentedAggregationRequest, ResidentSegmentedAggregationResult,
        ResidentSegmentedAggregationSource, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NameCatalog, NodeInput,
        TemporalStore,
    },
    types::{EdgeId, LabelId, Layer, NodeId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const REPORT: &str =
    "/tmp/irongraph-tck-full-20260721-after-string-predicate-precedence-merge9-3.json";
const REPORT_SHA256: &str = "77943059a3208e2adf43a771e757e1d800f135fd4a055aca076efb2aa041d213";
const FEATURE: &str = "expressions/list/List12.feature";
const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(0x4c49_5354_3132_0000_0000_0000_0003));
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    /// Reuses the resident segmented graph aggregation ABI after an exact fixed-path rewrite.
    CollectedPathHeadNormalization,
    /// Reuses nullable `collect()` after proving `x <> null` can never retain an element.
    NullableFilteredCollectSize,
    /// Needs aggregation, entity-list provenance, pre-write values, UNWIND, SET, and publication.
    CollectedEntityMutation,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    report_index: usize,
    name: &'static str,
    query: &'static str,
    tranche: Tranche,
}

const CASES: [Case; 5] = [
    Case {
        report_index: 1682,
        name: "[1] Collect and extract using a list comprehension",
        query: concat!(
            "MATCH (a:Label1) WITH collect(a) AS nodes ",
            "WITH nodes, [x IN nodes | x.name] AS oldNames ",
            "UNWIND nodes AS n SET n.name = 'newName' RETURN n.name, oldNames"
        ),
        tranche: Tranche::CollectedEntityMutation,
    },
    Case {
        report_index: 1683,
        name: "[2] Collect and filter using a list comprehension",
        query: concat!(
            "MATCH (a:Label1) WITH collect(a) AS nodes ",
            "WITH nodes, [x IN nodes WHERE x.name = 'original'] AS noopFiltered ",
            "UNWIND nodes AS n SET n.name = 'newName' ",
            "RETURN n.name, size(noopFiltered)"
        ),
        tranche: Tranche::CollectedEntityMutation,
    },
    Case {
        report_index: 1684,
        name: "[3] Size of list comprehension",
        query: concat!(
            "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) ",
            "RETURN size([x IN collect(r) WHERE x <> null]) AS cn"
        ),
        tranche: Tranche::NullableFilteredCollectSize,
    },
    Case {
        report_index: 1685,
        name: "[4] Returning a list comprehension",
        query: concat!(
            "MATCH p = (n)-->() ",
            "RETURN [x IN collect(p) | head(nodes(x))] AS p"
        ),
        tranche: Tranche::CollectedPathHeadNormalization,
    },
    Case {
        report_index: 1686,
        name: "[5] Using a list comprehension in a WITH",
        query: concat!(
            "MATCH p = (n:A)-->() ",
            "WITH [x IN collect(p) | head(nodes(x))] AS p, count(n) AS c ",
            "RETURN p, c"
        ),
        tranche: Tranche::CollectedPathHeadNormalization,
    },
];

#[derive(Debug, Deserialize)]
struct Report {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    matched: usize,
    fully_conformant: usize,
    scenarios: Vec<ReportScenario>,
}

#[derive(Debug, Deserialize)]
struct ReportScenario {
    path: String,
    name: String,
    operation_count: usize,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
}

#[derive(Default)]
struct NativeObservations {
    pins: AtomicUsize,
    segmented_calls: AtomicUsize,
    mutation_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentSegmentedAggregationRequest>>,
    mutation_requests: Mutex<Vec<ResidentNodePipelineRequest>>,
}

/// Advertises Metal before pinning so `require_native_execution` cannot fall through to the
/// generic CPU evaluator. The pinned wrapper is honestly CPU-backed and delegates only one
/// complete segmented graph command; every observable competing route is poisoned.
struct StrictSegmentedBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<NativeObservations>,
}

impl StrictSegmentedBackend {
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
            observations: Arc::new(NativeObservations::default()),
        })
    }

    fn observations(&self) -> Arc<NativeObservations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict List12 backend rejected route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictSegmentedBackend {
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
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        let continuation = request
            .mutation
            .as_ref()
            .and_then(|mutation| mutation.continuation.as_ref());
        let complete_list12_mutation = continuation.is_some_and(|continuation| {
            continuation.value_program.as_ref().is_some_and(|program| {
                matches!(
                    program.source,
                    irongraph::gpu::ResidentQuantifierSource::Unit
                ) && program.program.outputs.len() == 1
            }) && continuation
                .segmented_value_program
                .as_ref()
                .is_some_and(|nested| {
                    matches!(
                        nested.program.as_ref().map(|program| &program.source),
                        Some(ResidentSegmentedAggregationSource::GraphRelation { .. })
                    )
                })
                && matches!(
                    continuation.outputs.as_slice(),
                    [first, second]
                        if first.source == ResidentMutationProjectionSource::ComputedValue(0)
                            && second.source == ResidentMutationProjectionSource::ComputedValue(1)
                )
        });
        if !self.pinned || !complete_list12_mutation {
            return self.reject("execute_node_pipeline");
        }
        self.observations
            .mutation_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .mutation_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject("execute_row_program")
    }

    fn execute_quantifier_program(
        &self,
        _request: &ResidentQuantifierProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        self.reject("execute_quantifier_program")
    }

    fn execute_nullable_relation(
        &self,
        _request: &ResidentNullableRelationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        self.reject("execute_nullable_relation")
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
        request.validate()?;
        if request.input.row_count != 0
            || request.input.column_count != 0
            || !request.input.cells.is_empty()
            || !request.input.arena.is_empty()
            || !matches!(
                request.program.as_ref().map(|program| &program.source),
                Some(ResidentSegmentedAggregationSource::GraphRelation { .. })
            )
        {
            return self.reject("non_graph_or_host_shaped_segmented_command");
        }
        self.observations
            .segmented_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner
            .execute_segmented_aggregation(request, cancellation)
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
        term: 12,
        index: graph.revision(),
    }
}

fn populated_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let c = graph.catalog_mut().intern_label("C")?;
    let t = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, labels) in [(1, vec![a]), (2, vec![b]), (3, vec![c])] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties: Vec::new(),
        })?;
    }
    for (id, target) in [(4, 2), (5, 3)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(1),
            target: NodeId(target),
            relationship_type: t,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn collected_entity_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("Label1")?;
    let name = graph.catalog_mut().intern_property("name")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![label],
        properties: vec![(name, ScalarValue::String("original".into()))],
    })?;
    Ok(graph)
}

fn execution_context<'a>(
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
        next_node_id: 100,
        next_edge_id: 100,
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

fn execute(
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(
        query,
        &mut execution_context(graph, backend, require_native_execution),
    )
}

fn execute_write(
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<ExecutionOutput> {
    let mut context = execution_context(graph, backend, require_native_execution);
    context.capabilities.write = true;
    QueryEngine.execute(query, &mut context)
}

fn result_rows(output: &ExecutionOutput) -> Vec<Vec<ResultValue>> {
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

#[test]
fn exact_five_case_manifest_has_a_collision_safe_two_one_two_partition() {
    let expected = [
        (Tranche::CollectedPathHeadNormalization, 2),
        (Tranche::NullableFilteredCollectSize, 1),
        (Tranche::CollectedEntityMutation, 2),
    ];
    assert_eq!(
        expected.iter().map(|(_, count)| count).sum::<usize>(),
        CASES.len()
    );

    let mut identities = BTreeSet::new();
    for case in CASES {
        assert!(
            identities.insert((case.report_index, case.name)),
            "duplicate List12 manifest identity at {}",
            case.report_index
        );
        assert!(!case.query.is_empty());
    }
    for (tranche, count) in expected {
        assert_eq!(
            CASES.iter().filter(|case| case.tranche == tranche).count(),
            count,
            "wrong count for {tranche:?}"
        );
    }
}

#[test]
fn exact_list12_three_to_five_physical_plans_retain_the_rewrite_proofs() -> Result<()> {
    for case in &CASES[2..] {
        let planned = plan(bind(
            parse(case.query)?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?)?;
        assert!(
            planned.read_only,
            "{} unexpectedly planned a write",
            case.name
        );
        assert!(
            planned.unions.is_empty(),
            "{} unexpectedly planned UNION",
            case.name
        );
        let aggregate = planned
            .operators
            .iter()
            .find_map(|operator| match operator {
                PhysicalOperator::Project { projection, .. } => Some(projection),
                _ => None,
            })
            .unwrap_or_else(|| panic!("{} omitted its aggregate boundary", case.name));
        assert!(!aggregate.distinct);
        match case.report_index {
            1684 => {
                assert_eq!(aggregate.items.len(), 1);
                assert!(matches!(
                    &aggregate.items[0].expression,
                    Expression::Function { name, distinct: false, arguments }
                        if name.len() == 1
                            && name[0].eq_ignore_ascii_case("size")
                            && matches!(
                                arguments.as_slice(),
                                [Expression::ListComprehension {
                                    predicate: Some(predicate),
                                    projection: None,
                                    ..
                                }] if matches!(
                                    predicate.as_ref(),
                                    Expression::Binary {
                                        operation: BinaryOperator::NotEqual,
                                        right,
                                        ..
                                    } if matches!(right.as_ref(), Expression::Literal(ScalarValue::Null))
                                )
                            )
                ));
            }
            1685 | 1686 => {
                assert!(matches!(
                    &aggregate.items[0].expression,
                    Expression::ListComprehension {
                        predicate: None,
                        projection: Some(projected),
                        ..
                    } if matches!(
                        projected.as_ref(),
                        Expression::Function { name, distinct: false, arguments }
                            if name.len() == 1
                                && name[0].eq_ignore_ascii_case("head")
                                && matches!(
                                    arguments.as_slice(),
                                    [Expression::Function {
                                        name,
                                        distinct: false,
                                        ..
                                    }] if name.len() == 1 && name[0].eq_ignore_ascii_case("nodes")
                                )
                    )
                ));
                assert_eq!(
                    aggregate.items.len(),
                    usize::from(case.report_index == 1686) + 1
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn assert_public_result_matches_oracle(
    oracle: &ExecutionOutput,
    native: &ExecutionOutput,
) -> Result<()> {
    assert_eq!(native.result.schema, oracle.result.schema);
    assert_eq!(result_rows(native), result_rows(oracle));
    assert_eq!(native.result.bookmark, oracle.result.bookmark);
    assert_eq!(native.result.statistics, StatementStats::default());
    assert_eq!(native.result.statistics, oracle.result.statistics);
    assert_eq!(native.result.truncated, oracle.result.truncated);
    assert!(native.graph_mutations.is_empty());
    assert!(native.temporal_mutations.is_empty());
    Ok(())
}

fn assert_write_result_matches_oracle(
    oracle: &ExecutionOutput,
    native: &ExecutionOutput,
) -> Result<()> {
    assert_eq!(native.result.schema, oracle.result.schema);
    assert_eq!(result_rows(native), result_rows(oracle));
    assert_eq!(native.result.bookmark, oracle.result.bookmark);
    assert_eq!(native.result.statistics, oracle.result.statistics);
    assert_eq!(native.result.truncated, oracle.result.truncated);
    assert_eq!(
        serde_json::to_value(&native.graph_mutations)
            .map_err(|error| Error::internal(error.to_string()))?,
        serde_json::to_value(&oracle.graph_mutations)
            .map_err(|error| Error::internal(error.to_string()))?
    );
    assert!(native.temporal_mutations.is_empty());
    assert!(oracle.temporal_mutations.is_empty());
    Ok(())
}

fn assert_null_filtered_collect_request(request: &ResidentSegmentedAggregationRequest) {
    let program = request.program.as_ref().expect("sealed segmented program");
    assert!(matches!(
        &program.source,
        ResidentSegmentedAggregationSource::GraphRelation { .. }
    ));
    let [aggregate_stage, project_stage] = program.stages.as_slice() else {
        panic!("List12 [3] changed its aggregate/project stage shape");
    };
    let ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } =
        &aggregate_stage.operation
    else {
        panic!("List12 [3] changed its aggregate stage shape");
    };
    let ResidentSegmentedAggregationOperation::Project { bindings, .. } = &project_stage.operation
    else {
        panic!("List12 [3] changed its project stage shape");
    };
    assert!(groups.is_empty());
    let [reduction] = reductions.as_slice() else {
        panic!("List12 [3] changed its single reduction");
    };
    assert_eq!(reduction.kind, ResidentSegmentedAggregateKind::Collect);
    assert!(!reduction.distinct);
    assert!(matches!(
        &reduction.input,
        Some(ResidentQuantifierExpression::Literal(
            irongraph::gpu::ResidentQuantifierValue::Null
        ))
    ));
    let [binding] = bindings.as_slice() else {
        panic!("List12 [3] changed its final projection width");
    };
    assert!(matches!(
        &binding.expression,
        ResidentQuantifierExpression::Function { function, arguments }
            if *function == irongraph::gpu::ResidentQuantifierFunction::Size
                && matches!(
                    arguments.as_slice(),
                    [ResidentQuantifierExpression::Slot(slot)] if *slot == reduction.output
                )
    ));
}

fn assert_collected_path_head_request(
    request: &ResidentSegmentedAggregationRequest,
    with_count: bool,
) {
    let program = request.program.as_ref().expect("sealed segmented program");
    let ResidentSegmentedAggregationSource::GraphRelation {
        request: source, ..
    } = &program.source
    else {
        panic!("List12 path-head route lost its graph relation");
    };
    assert_eq!(
        source
            .program
            .stages
            .iter()
            .filter(|stage| {
                matches!(
                    stage,
                    irongraph::gpu::ResidentNullableRelationStage::NodeScan { .. }
                        | irongraph::gpu::ResidentNullableRelationStage::Expand { .. }
                )
            })
            .count(),
        2
    );
    let aggregate = program
        .stages
        .iter()
        .find_map(|stage| match &stage.operation {
            ResidentSegmentedAggregationOperation::Aggregate { groups, reductions } => {
                Some((groups, reductions))
            }
            _ => None,
        })
        .expect("List12 path-head route omitted aggregate stage");
    assert!(aggregate.0.is_empty());
    assert_eq!(aggregate.1.len(), usize::from(with_count) + 1);
    assert_eq!(aggregate.1[0].kind, ResidentSegmentedAggregateKind::Collect);
    assert!(!aggregate.1[0].distinct);
    assert!(matches!(
        &aggregate.1[0].input,
        Some(ResidentQuantifierExpression::Slot(_))
    ));
    if with_count {
        assert_eq!(
            aggregate.1[1].kind,
            ResidentSegmentedAggregateKind::CountValue
        );
        assert!(matches!(
            &aggregate.1[1].input,
            Some(ResidentQuantifierExpression::Slot(_))
        ));
    }
}

#[test]
fn strict_cpu_executes_list12_one_and_two_as_one_prewrite_mutation_command() -> Result<()> {
    let graph = collected_entity_graph()?;
    let backend = StrictSegmentedBackend::new(&graph)?;
    let observations = backend.observations();

    for (offset, case) in CASES[..2].iter().enumerate() {
        let oracle = execute_write(&graph, None, false, case.query)?;
        let native = execute_write(&graph, Some(&backend), true, case.query)?;
        assert_write_result_matches_oracle(&oracle, &native)?;
        let expected_second = if offset == 0 {
            ResultValue::List(vec![ResultValue::Scalar(ScalarValue::String(
                "original".into(),
            ))])
        } else {
            ResultValue::Scalar(ScalarValue::Integer(1))
        };
        assert_eq!(
            result_rows(&native),
            vec![vec![
                ResultValue::Scalar(ScalarValue::String("newName".into())),
                expected_second,
            ]],
            "{}",
            case.name
        );
        assert_eq!(
            native.result.statistics,
            StatementStats {
                properties_set: 1,
                ..StatementStats::default()
            }
        );
        let [GraphMutation::SetNodeProperty { node, value, .. }] =
            native.graph_mutations.as_slice()
        else {
            panic!(
                "{} changed its exact mutation publication: {:?}",
                case.name, native.graph_mutations
            );
        };
        assert_eq!(*node, NodeId(1));
        assert_eq!(*value, ScalarValue::String("newName".into()));
    }

    assert_eq!(observations.pins.load(Ordering::SeqCst), 2);
    assert_eq!(observations.mutation_calls.load(Ordering::SeqCst), 2);
    assert_eq!(observations.segmented_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    let requests = observations
        .mutation_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        let continuation = request
            .mutation
            .as_ref()
            .and_then(|mutation| mutation.continuation.as_ref())
            .expect("complete mutation continuation");
        assert!(continuation.value_program.as_ref().is_some_and(|program| {
            matches!(
                program.source,
                irongraph::gpu::ResidentQuantifierSource::Unit
            ) && program.program.outputs.len() == 1
        }));
        let nested = continuation
            .segmented_value_program
            .as_ref()
            .expect("pre-write graph program");
        nested.validate()?;
        assert_eq!(nested.maximum_output_groups, 1);
        assert_eq!(continuation.outputs.len(), 2);
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_list12_three_to_five_as_one_complete_segmented_command() -> Result<()> {
    let empty = GraphStore::default();
    let empty_backend = StrictSegmentedBackend::new(&empty)?;
    let empty_observations = empty_backend.observations();
    let oracle = execute(&empty, None, false, CASES[2].query)?;
    let native = execute(&empty, Some(&empty_backend), true, CASES[2].query)?;
    assert_public_result_matches_oracle(&oracle, &native)?;
    assert_eq!(
        result_rows(&native),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(0))]]
    );
    assert_eq!(empty_observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(empty_observations.segmented_calls.load(Ordering::SeqCst), 1);
    assert_eq!(empty_observations.forbidden_calls.load(Ordering::SeqCst), 0);
    let empty_requests = empty_observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(empty_requests.len(), 1);
    assert_null_filtered_collect_request(&empty_requests[0]);
    drop(empty_requests);

    let graph = populated_graph()?;
    let backend = StrictSegmentedBackend::new(&graph)?;
    let observations = backend.observations();
    for case in &CASES[2..] {
        let oracle = execute(&graph, None, false, case.query)?;
        let native = execute(&graph, Some(&backend), true, case.query)?;
        assert_public_result_matches_oracle(&oracle, &native)?;
    }
    assert_eq!(observations.pins.load(Ordering::SeqCst), 3);
    assert_eq!(observations.segmented_calls.load(Ordering::SeqCst), 3);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(requests.len(), 3);
    assert_null_filtered_collect_request(&requests[0]);
    assert_collected_path_head_request(&requests[1], false);
    assert_collected_path_head_request(&requests[2], true);
    Ok(())
}

#[test]
fn list12_rewrites_reject_structural_near_misses() -> Result<()> {
    let graph = populated_graph()?;
    for query in [
        "MATCH p = (n)-->() RETURN [x IN collect(p) | last(nodes(x))] AS p",
        "MATCH p = (n)-->() RETURN [x IN collect(p) WHERE true | head(nodes(x))] AS p",
        concat!(
            "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) ",
            "RETURN size([x IN collect(r) WHERE x IS NOT NULL]) AS cn"
        ),
    ] {
        let backend = StrictSegmentedBackend::new(&graph)?;
        let observations = backend.observations();
        let error = execute(&graph, Some(&backend), true, query)
            .expect_err("List12 structural near miss unexpectedly entered native execution");
        assert_eq!(
            error.code,
            ErrorCode::GpuAdmissionFailure,
            "{query}: {error:?}"
        );
        assert_eq!(observations.segmented_calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[test]
#[ignore = "reads the pinned local full-run artifact"]
fn fresh_full_report_pins_exactly_the_five_remaining_list12_failures() {
    let bytes = fs::read(REPORT).unwrap_or_else(|error| panic!("failed to read {REPORT}: {error}"));
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), REPORT_SHA256);
    let report: Report = serde_json::from_slice(&bytes)
        .unwrap_or_else(|error| panic!("failed to decode {REPORT}: {error}"));
    assert_eq!(
        (
            report.total,
            report.cpu_passed,
            report.metal_passed,
            report.matched,
            report.fully_conformant,
            report.scenarios.len(),
        ),
        (3_897, 3_897, 3_749, 3_749, 3_749, 3_897)
    );

    let exact_gap = report
        .scenarios
        .iter()
        .enumerate()
        .filter(|(_, scenario)| {
            scenario.path.ends_with(FEATURE) && scenario.cpu_passed && !scenario.metal_passed
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert_eq!(
        exact_gap,
        CASES
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>()
    );

    for case in CASES {
        let actual = &report.scenarios[case.report_index];
        assert!(actual.path.ends_with(FEATURE));
        assert_eq!(actual.name, case.name);
        assert_eq!(actual.operation_count, 1);
        assert!(actual.cpu_passed);
        assert!(!actual.metal_passed);
        assert!(!actual.fully_conformant);
        assert!(actual.shared_failures.is_empty());
        assert!(actual.cpu_failures.is_empty());
        assert!(actual.divergences.is_empty());
        assert_eq!(actual.metal_failures.len(), 1);

        let query_prefix = case.query.chars().take(120).collect::<String>();
        let failure = &actual.metal_failures[0];
        assert!(
            failure.contains(&format!("`{query_prefix}")),
            "fresh report lost the exact query prefix for {}",
            case.name
        );
        assert!(failure.contains("GpuAdmissionFailure"));
        assert!(failure.contains("no complete resident implementation"));
    }
}
