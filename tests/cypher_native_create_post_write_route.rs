// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict public-route acceptance for all 14 openCypher `Create6.feature` scenarios.
//!
//! The manifest pins the exact seven post-write shapes for node CREATE and the same seven for
//! relationship CREATE. The generic CPU oracle proves rows and side effects. The strict observer
//! advertises Metal and permits only one pinned generalized row-mutation command; an incomplete
//! CREATE primitive, a decomposed row operator, or any legacy graph primitive is rejected. Until
//! the complete row-CREATE continuation exists, the active routing test requires admission to fail
//! before dispatch. Ignored strict-CPU and real-Metal tests are the success gates.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentGroup, ResidentGroupRequest,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-next.json";
const FEATURE: &str = "clauses/create/Create6.feature";
const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4352_4541_5445_365f_504f_5354_5752_0001,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CreateDomain {
    Node,
    Relationship,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TailShape {
    LimitZero,
    SkipAll,
    PageTwo,
    PageAll,
    Filter,
    ReturnAggregate,
    WithAggregate,
}

impl TailShape {
    const fn selected_rows(self) -> usize {
        match self {
            Self::LimitZero | Self::SkipAll => 1,
            Self::PageTwo
            | Self::PageAll
            | Self::Filter
            | Self::ReturnAggregate
            | Self::WithAggregate => 5,
        }
    }

    fn source_values(self) -> Vec<i64> {
        match self {
            Self::LimitZero | Self::SkipAll | Self::PageTwo | Self::PageAll => {
                vec![42; self.selected_rows()]
            }
            Self::Filter | Self::ReturnAggregate | Self::WithAggregate => vec![1, 2, 3, 4, 5],
        }
    }

    fn result_values(self) -> Vec<i64> {
        match self {
            Self::LimitZero | Self::SkipAll => Vec::new(),
            Self::PageTwo => vec![42, 42],
            Self::PageAll => vec![42; 5],
            Self::Filter => vec![2, 4],
            Self::ReturnAggregate | Self::WithAggregate => vec![15],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Create6Case {
    /// Zero-based index in `/tmp/irongraph-tck-full-next.json`.
    report_index: usize,
    scenario: u8,
    name: &'static str,
    domain: CreateDomain,
    tail: TailShape,
    query: &'static str,
}

impl Create6Case {
    fn label(self) -> String {
        format!(
            "TCK report index {} Create6 [{}] {}",
            self.report_index, self.scenario, self.name
        )
    }

    const fn output_name(self) -> &'static str {
        match self.tail {
            TailShape::LimitZero | TailShape::SkipAll => match self.domain {
                CreateDomain::Node => "n",
                CreateDomain::Relationship => "r",
            },
            TailShape::PageTwo | TailShape::PageAll | TailShape::Filter => "num",
            TailShape::ReturnAggregate | TailShape::WithAggregate => "sum",
        }
    }
}

const CASES: [Create6Case; 14] = [
    Create6Case {
        report_index: 116,
        scenario: 1,
        name: "Limiting to zero results after creating nodes affects the result set but not the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::LimitZero,
        query: "CREATE (n:N {num: 42}) RETURN n LIMIT 0",
    },
    Create6Case {
        report_index: 117,
        scenario: 2,
        name: "Skipping all results after creating nodes affects the result set but not the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::SkipAll,
        query: "CREATE (n:N {num: 42}) RETURN n SKIP 1",
    },
    Create6Case {
        report_index: 118,
        scenario: 3,
        name: "Skipping and limiting to a few results after creating nodes does not affect the result set nor the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::PageTwo,
        query: "UNWIND [42, 42, 42, 42, 42] AS x CREATE (n:N {num: x}) RETURN n.num AS num SKIP 2 LIMIT 2",
    },
    Create6Case {
        report_index: 119,
        scenario: 4,
        name: "Skipping zero result and limiting to all results after creating nodes does not affect the result set nor the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::PageAll,
        query: "UNWIND [42, 42, 42, 42, 42] AS x CREATE (n:N {num: x}) RETURN n.num AS num SKIP 0 LIMIT 5",
    },
    Create6Case {
        report_index: 120,
        scenario: 5,
        name: "Filtering after creating nodes affects the result set but not the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::Filter,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE (n:N {num: x}) WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    },
    Create6Case {
        report_index: 121,
        scenario: 6,
        name: "Aggregating in `RETURN` after creating nodes affects the result set but not the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::ReturnAggregate,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE (n:N {num: x}) RETURN sum(n.num) AS sum",
    },
    Create6Case {
        report_index: 122,
        scenario: 7,
        name: "Aggregating in `WITH` after creating nodes affects the result set but not the side effects",
        domain: CreateDomain::Node,
        tail: TailShape::WithAggregate,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE (n:N {num: x}) WITH sum(n.num) AS sum RETURN sum",
    },
    Create6Case {
        report_index: 123,
        scenario: 8,
        name: "Limiting to zero results after creating relationships affects the result set but not the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::LimitZero,
        query: "CREATE ()-[r:R {num: 42}]->() RETURN r LIMIT 0",
    },
    Create6Case {
        report_index: 124,
        scenario: 9,
        name: "Skipping all results after creating relationships affects the result set but not the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::SkipAll,
        query: "CREATE ()-[r:R {num: 42}]->() RETURN r SKIP 1",
    },
    Create6Case {
        report_index: 125,
        scenario: 10,
        name: "Skipping and limiting to a few results after creating relationships does not affect the result set nor the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::PageTwo,
        query: "UNWIND [42, 42, 42, 42, 42] AS x CREATE ()-[r:R {num: x}]->() RETURN r.num AS num SKIP 2 LIMIT 2",
    },
    Create6Case {
        report_index: 126,
        scenario: 11,
        name: "Skipping zero result and limiting to all results after creating relationships does not affect the result set nor the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::PageAll,
        query: "UNWIND [42, 42, 42, 42, 42] AS x CREATE ()-[r:R {num: x}]->() RETURN r.num AS num SKIP 0 LIMIT 5",
    },
    Create6Case {
        report_index: 127,
        scenario: 12,
        name: "Filtering after creating relationships affects the result set but not the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::Filter,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE ()-[r:R {num: x}]->() WITH r WHERE r.num % 2 = 0 RETURN r.num AS num",
    },
    Create6Case {
        report_index: 128,
        scenario: 13,
        name: "Aggregating in `RETURN` after creating relationships affects the result set but not the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::ReturnAggregate,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE ()-[r:R {num: x}]->() RETURN sum(r.num) AS sum",
    },
    Create6Case {
        report_index: 129,
        scenario: 14,
        name: "Aggregating in `WITH` after creating relationships affects the result set but not the side effects",
        domain: CreateDomain::Relationship,
        tail: TailShape::WithAggregate,
        query: "UNWIND [1, 2, 3, 4, 5] AS x CREATE ()-[r:R {num: x}]->() WITH sum(r.num) AS sum RETURN sum",
    },
];

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn empty() -> Self {
        let graph = GraphStore::default();
        Self {
            bookmark: Bookmark {
                term: 67,
                index: graph.revision(),
            },
            graph,
        }
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
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
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 2,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn expected_statistics(case: Create6Case) -> StatementStats {
    let selected = case.tail.selected_rows() as u64;
    match case.domain {
        CreateDomain::Node => StatementStats {
            nodes_created: selected,
            labels_added: selected,
            ..StatementStats::default()
        },
        CreateDomain::Relationship => StatementStats {
            nodes_created: selected.saturating_mul(2),
            relationships_created: selected,
            ..StatementStats::default()
        },
    }
}

fn integer_values(output: &ExecutionOutput, column_name: &str) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == column_name)
            .ok_or_else(|| {
                Error::internal(format!(
                    "Create6 result batch omitted expected column `{column_name}`"
                ))
            })?;
        for value in &column.values {
            let ResultValue::Scalar(ScalarValue::Integer(value)) = value else {
                return Err(Error::internal(format!(
                    "Create6 `{column_name}` result is not an integer: {value:?}"
                )));
            };
            values.push(*value);
        }
    }
    Ok(values)
}

fn assert_case_output(
    fixture: &Fixture,
    case: Create6Case,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    let expected_columns = vec![case.output_name().to_owned()];
    let actual_columns = output
        .result
        .schema
        .iter()
        .map(|(name, _)| name.clone())
        .collect::<Vec<_>>();
    if actual_columns != expected_columns {
        return Err(format!(
            "wrong result schema: expected {expected_columns:?}, got {actual_columns:?}"
        ));
    }
    let actual_rows = output
        .result
        .batches
        .iter()
        .map(|batch| batch.row_count)
        .sum::<usize>();
    let expected_values = case.tail.result_values();
    if actual_rows != expected_values.len() {
        return Err(format!(
            "wrong result cardinality: expected {}, got {actual_rows}",
            expected_values.len()
        ));
    }
    let actual_values =
        integer_values(output, case.output_name()).map_err(|error| error.message)?;
    if actual_values != expected_values {
        return Err(format!(
            "wrong result values: expected {expected_values:?}, got {actual_values:?}"
        ));
    }
    let expected_stats = expected_statistics(case);
    if output.result.statistics != expected_stats
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
    {
        return Err(format!(
            "wrong result metadata: expected {expected_stats:?} at {:?}, got {:?}",
            fixture.bookmark, output.result
        ));
    }
    if !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
    {
        return Err("CREATE emitted unrelated temporal, administrative, or vector work".to_owned());
    }

    let nodes = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::InsertNode(node) => Some(node),
            _ => None,
        })
        .collect::<Vec<_>>();
    let relationships = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::InsertEdge(edge) => Some(edge),
            _ => None,
        })
        .collect::<Vec<_>>();
    let source_values = case.tail.source_values();
    match case.domain {
        CreateDomain::Node => {
            if nodes.len() != case.tail.selected_rows() || !relationships.is_empty() {
                return Err(format!(
                    "wrong node CREATE effects: {} nodes and {} relationships",
                    nodes.len(),
                    relationships.len()
                ));
            }
            if nodes.iter().any(|node| {
                node.labels.len() != 1
                    || node.properties.len() != 1
                    || !matches!(&node.properties[0].1, ScalarValue::Integer(_))
            }) {
                return Err("created node lost its exact label/property shape".to_owned());
            }
            let mut actual = nodes
                .iter()
                .filter_map(|node| match &node.properties[0].1 {
                    ScalarValue::Integer(value) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>();
            actual.sort_unstable();
            let mut expected = source_values;
            expected.sort_unstable();
            if actual != expected {
                return Err(format!(
                    "created node properties differ: expected {expected:?}, got {actual:?}"
                ));
            }
        }
        CreateDomain::Relationship => {
            if nodes.len() != case.tail.selected_rows().saturating_mul(2)
                || relationships.len() != case.tail.selected_rows()
            {
                return Err(format!(
                    "wrong relationship CREATE effects: {} nodes and {} relationships",
                    nodes.len(),
                    relationships.len()
                ));
            }
            if nodes
                .iter()
                .any(|node| !node.labels.is_empty() || !node.properties.is_empty())
            {
                return Err("anonymous relationship endpoints acquired data".to_owned());
            }
            let node_ids = nodes.iter().map(|node| node.id).collect::<BTreeSet<_>>();
            if relationships.iter().any(|edge| {
                !node_ids.contains(&edge.source)
                    || !node_ids.contains(&edge.target)
                    || edge.properties.len() != 1
                    || !matches!(&edge.properties[0].1, ScalarValue::Integer(_))
            }) {
                return Err(
                    "created relationship lost an endpoint or exact property shape".to_owned(),
                );
            }
            let mut actual = relationships
                .iter()
                .filter_map(|edge| match &edge.properties[0].1 {
                    ScalarValue::Integer(value) => Some(*value),
                    _ => None,
                })
                .collect::<Vec<_>>();
            actual.sort_unstable();
            let mut expected = source_values;
            expected.sort_unstable();
            if actual != expected {
                return Err(format!(
                    "created relationship properties differ: expected {expected:?}, got {actual:?}"
                ));
            }
        }
    }
    if fixture.graph.nodes().next().is_some() || fixture.graph.edges().next().is_some() {
        return Err("query execution mutated the immutable fixture graph".to_owned());
    }
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_graph_write_calls: AtomicUsize,
    rejected_incomplete_routes: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// Advertises Metal while allowing only one pinned, complete row-CREATE mutation request.
struct StrictCreatePostWriteBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictCreatePostWriteBackend {
    fn strict_cpu(image: ResidentProjectImage) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Self::new(
            Box::new(cpu),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(image: ResidentProjectImage) -> Result<Self> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image)?;
        Self::new(
            Box::new(metal),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    fn new(
        inner: Box<dyn ExecutionBackend>,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("strict Create6 backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("strict Create6 backend has no admitted graph generation")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self
            .inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("Create6 publication lost resident bookmark"))?;
        self.expected_graph_revision = self
            .inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("Create6 publication lost graph generation"))?;
        Ok(())
    }

    fn reject_incomplete<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .rejected_incomplete_routes
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict Create6 observer rejected incomplete `{route}`; one row-CREATE command must own source, effects, post-write stages, and final relation"
            ),
        ))
    }
}

impl ExecutionBackend for StrictCreatePostWriteBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
            self.advertised_kind
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
            return self.reject_incomplete("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_incomplete("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Create6 pin changed backend provenance or immutable generation fence",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
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
        self.reject_incomplete("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_incomplete("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_incomplete("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_incomplete("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_incomplete("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_incomplete("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_incomplete("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_incomplete("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_incomplete("execute_node_pipeline")
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        if !self.pinned {
            return self.reject_incomplete("execute_row_mutation_on_unpinned_generation");
        }
        if request.create_body().is_none() {
            return self.reject_incomplete("execute_row_mutation_without_create_body");
        }
        request.validate()?;
        if self.inner.kind() != self.actual_kind
            || request.generation.project != PROJECT
            || request.generation.bookmark != self.expected_bookmark
            || request.generation.graph_revision != self.expected_graph_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Create6 row mutation escaped its pinned backend or immutable generation",
            ));
        }
        if self
            .observations
            .complete_graph_write_calls
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return self.reject_incomplete("execute_row_mutation_more_than_once");
        }
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_row_mutation(request, cancellation)
    }

    fn supports_native_create_node(&self) -> bool {
        false
    }

    fn execute_create_node(
        &self,
        _request: &ResidentCreateNodeRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        self.reject_incomplete("execute_create_node")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_incomplete("execute_row_program")
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_incomplete("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_incomplete("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_incomplete("exact_l2")
    }
}

fn run_generic_case(case: Create6Case) -> Result<()> {
    let fixture = Fixture::empty();
    let output = QueryEngine.execute(case.query, &mut context(&fixture, None, false))?;
    assert_case_output(&fixture, case, &output).map_err(Error::internal)
}

fn run_strict_case(
    case: Create6Case,
    backend: &StrictCreatePostWriteBackend,
    fixture: &Fixture,
) -> Result<()> {
    let observations = backend.observations();
    let output = QueryEngine.execute(case.query, &mut context(fixture, Some(backend), true))?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let complete = observations
        .complete_graph_write_calls
        .load(Ordering::SeqCst);
    let rejected = observations
        .rejected_incomplete_routes
        .load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || complete != 1 || rejected != 0 || requests.len() != 1 {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{}: strict route requires one pin, one complete row mutation, zero incomplete routes, and one retained request; got {pins}/{complete}/{rejected}/{}",
                case.label(),
                requests.len()
            ),
        ));
    }
    requests[0].validate()?;
    if requests[0].create_body().is_none() {
        return Err(Error::internal(format!(
            "{}: complete request lost its row-CREATE body",
            case.label()
        )));
    }
    drop(requests);
    assert_case_output(fixture, case, &output).map_err(Error::internal)
}

fn run_strict_cpu_suite() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        let result = (|| {
            let fixture = Fixture::empty();
            let backend = StrictCreatePostWriteBackend::strict_cpu(fixture.image()?)?;
            run_strict_case(case, &backend, &fixture)
        })();
        if let Err(error) = result {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict CPU Create6 acceptance failed for {} case(s):\n{}",
                failures.len(),
                failures.join("\n")
            ),
        ))
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn run_real_metal_suite() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        let result = (|| {
            let fixture = Fixture::empty();
            let backend = StrictCreatePostWriteBackend::real_metal(fixture.image()?)?;
            run_strict_case(case, &backend, &fixture)
        })();
        if let Err(error) = result {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict Metal Create6 acceptance failed for {} case(s):\n{}",
                failures.len(),
                failures.join("\n")
            ),
        ))
    }
}

#[derive(Deserialize)]
struct CertifiedReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    operation_count: usize,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    shared_failures: Vec<String>,
}

#[test]
fn literal_manifest_is_exactly_create6_report_indices_116_through_129() {
    assert_eq!(CASES.len(), 14);
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>(),
        (116_usize..=129).collect::<Vec<_>>()
    );
    assert_eq!(
        CASES.iter().map(|case| case.scenario).collect::<Vec<_>>(),
        (1_u8..=14).collect::<Vec<_>>()
    );
    assert!(
        CASES[..7]
            .iter()
            .all(|case| case.domain == CreateDomain::Node)
    );
    assert!(
        CASES[7..]
            .iter()
            .all(|case| case.domain == CreateDomain::Relationship)
    );
    for index in 0..7 {
        assert_eq!(CASES[index].tail, CASES[index + 7].tail);
    }
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.query)
            .collect::<BTreeSet<_>>()
            .len(),
        14
    );
    for case in CASES {
        assert!(!case.name.is_empty());
        assert!(!case.query.is_empty());
    }
}

#[test]
#[ignore = "external identity gate: requires /tmp/irongraph-tck-full-next.json"]
fn literal_manifest_matches_the_fresh_3897_scenario_report() -> Result<()> {
    let report: CertifiedReport = serde_json::from_slice(
        &std::fs::read(REPORT_PATH)
            .map_err(|error| Error::internal(format!("cannot read {REPORT_PATH}: {error}")))?,
    )
    .map_err(|error| Error::internal(format!("cannot decode {REPORT_PATH}: {error}")))?;
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);
    assert_eq!(report.metal_passed, 3_506);
    for case in CASES {
        let scenario = report.scenarios.get(case.report_index).ok_or_else(|| {
            Error::internal(format!("{} is outside the certified report", case.label()))
        })?;
        assert!(scenario.path.ends_with(FEATURE), "{}", case.label());
        assert_eq!(
            scenario.name,
            format!("[{}] {}", case.scenario, case.name),
            "{}",
            case.label()
        );
        assert_eq!(scenario.operation_count, 1, "{}", case.label());
        assert!(scenario.cpu_passed, "{}", case.label());
        assert!(!scenario.metal_passed, "{}", case.label());
        assert!(!scenario.fully_conformant, "{}", case.label());
        assert!(scenario.cpu_failures.is_empty(), "{}", case.label());
        assert!(scenario.shared_failures.is_empty(), "{}", case.label());
        assert_eq!(scenario.metal_failures.len(), 1, "{}", case.label());
        let failure = &scenario.metal_failures[0];
        assert!(failure.contains(case.query), "{}: {failure}", case.label());
        assert!(
            failure.contains("GpuAdmissionFailure")
                && failure.contains(
                    "active GPU execution class has no complete resident implementation for this query plan"
                ),
            "{}: {failure}",
            case.label()
        );
    }
    Ok(())
}

#[test]
fn generic_cpu_oracle_proves_exact_rows_and_create_side_effects_for_all_14() {
    let mut failures = Vec::new();
    for case in CASES {
        if let Err(error) = run_generic_case(case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert!(
        failures.is_empty(),
        "generic CPU Create6 oracle had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn strict_metal_route_is_complete_or_fails_closed_before_dispatch_for_all_14() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        let fixture = Fixture::empty();
        let backend = StrictCreatePostWriteBackend::strict_cpu(fixture.image()?)?;
        let observations = backend.observations();
        match QueryEngine.execute(
            case.query,
            &mut context(&fixture, Some(&backend), true),
        ) {
            Ok(output) => {
                let pins = observations.pins.load(Ordering::SeqCst);
                let complete = observations
                    .complete_graph_write_calls
                    .load(Ordering::SeqCst);
                let rejected = observations
                    .rejected_incomplete_routes
                    .load(Ordering::SeqCst);
                let requests = observations
                    .requests
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len();
                if pins != 1 || complete != 1 || rejected != 0 || requests != 1 {
                    failures.push(format!(
                        "{}: success did not use exactly one pinned complete command ({pins}/{complete}/{rejected}/{requests})",
                        case.label()
                    ));
                } else if let Err(error) = assert_case_output(&fixture, case, &output) {
                    failures.push(format!("{}: partial/incorrect success: {error}", case.label()));
                }
            }
            Err(error) if error.code == ErrorCode::GpuAdmissionFailure => {
                let pins = observations.pins.load(Ordering::SeqCst);
                let complete = observations
                    .complete_graph_write_calls
                    .load(Ordering::SeqCst);
                let rejected = observations
                    .rejected_incomplete_routes
                    .load(Ordering::SeqCst);
                if pins != 0 || complete != 0 || rejected != 0 {
                    failures.push(format!(
                        "{}: admission failure dispatched an incomplete prefix ({pins}/{complete}/{rejected}): {error}",
                        case.label()
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "{}: expected complete success or pre-dispatch admission failure, got {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "Create6 strict routing boundary had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn strict_cpu_reference_executes_all_14_through_one_pinned_complete_command() -> Result<()> {
    run_strict_cpu_suite()
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red acceptance gate: all 14 Create6 scenarios must match CPU on real Metal"]
fn real_metal_executes_all_14_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    run_real_metal_suite()
}
