// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance for openCypher TCK Remove3 scenarios [15]-[21].
//!
//! These seven scenarios remove a relationship property and then continue with LIMIT, SKIP,
//! filtering, or aggregation. The generic CPU tests pin the exact public Cypher semantics. The
//! strict tests advertise an accelerator and close every generic execution entrypoint: success is
//! valid only through one immutable generation pin and one complete mutation-continuation command.
//! Until that native relationship REMOVE program exists, the non-ignored routing test requires an
//! admission failure before any prefix dispatch. The ignored CPU-reference and real-Metal tests
//! remain deliberately red acceptance gates.

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
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, EntityDependency, ExecutionContext, ExecutionOutput,
        QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentEntityBinding,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentMutationOperation, ResidentMutationPostStage, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentObligationKind, ResidentObligationScope,
        ResidentProjectImage, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5245_4c5f_5245_4d4f_5645_5f33_5f3135,
));
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 32 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;
const FEATURE: &str = "features/clauses/remove/Remove3.feature";
const FEATURE_FILTER: &str = "clauses/remove/Remove3.feature";

const SCENARIO_NAMES: [&str; 7] = [
    "[15] Limiting to zero results after removing a property from relationships affects the result set but not the side effects",
    "[16] Skipping all results after removing a property from relationships affects the result set but not the side effects",
    "[17] Skipping and limiting to a few results after removing a property from relationships affects the result set but not the side effects",
    "[18] Skipping zero result and limiting to all results after removing a property from relationships does not affect the result set nor the side effects",
    "[19] Filtering after removing a property from relationships affects the result set but not the side effects",
    "[20] Aggregating in `RETURN` after removing a property from relationships affects the result set but not the side effects",
    "[21] Aggregating in `WITH` after removing a property from relationships affects the result set but not the side effects",
];

const QUERIES: [&str; 7] = [
    "MATCH ()-[r:R]->() REMOVE r.num RETURN r LIMIT 0",
    "MATCH ()-[r:R]->() REMOVE r.num RETURN r SKIP 1",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN r.num AS num SKIP 2 LIMIT 2",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN r.num AS num SKIP 0 LIMIT 5",
    "MATCH ()-[r:R]->() REMOVE r.name WITH r WHERE r.num % 2 = 0 RETURN r.num AS num",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN sum(r.num) AS sum",
    "MATCH ()-[r:R]->() REMOVE r.name WITH sum(r.num) AS sum RETURN sum",
];

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
    const ALL: [Self; 7] = [
        Self::LimitZero,
        Self::SkipAll,
        Self::PageTwo,
        Self::PageAll,
        Self::Filter,
        Self::ReturnAggregate,
        Self::WithAggregate,
    ];

    const fn index(self) -> usize {
        match self {
            Self::LimitZero => 0,
            Self::SkipAll => 1,
            Self::PageTwo => 2,
            Self::PageAll => 3,
            Self::Filter => 4,
            Self::ReturnAggregate => 5,
            Self::WithAggregate => 6,
        }
    }

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

    const fn expected_column(self) -> &'static str {
        match self {
            Self::LimitZero | Self::SkipAll => "r",
            Self::PageTwo | Self::PageAll | Self::Filter => "num",
            Self::ReturnAggregate | Self::WithAggregate => "sum",
        }
    }

    fn expected_values(self) -> Vec<i64> {
        match self {
            Self::LimitZero | Self::SkipAll => Vec::new(),
            Self::PageTwo => vec![42, 42],
            Self::PageAll => vec![42; 5],
            Self::Filter => vec![2, 4],
            Self::ReturnAggregate | Self::WithAggregate => vec![15],
        }
    }

    const fn removed_property(self) -> &'static str {
        match self {
            Self::LimitZero | Self::SkipAll => "num",
            Self::PageTwo
            | Self::PageAll
            | Self::Filter
            | Self::ReturnAggregate
            | Self::WithAggregate => "name",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TckCase {
    /// Zero-based index in the certified 3,897-scenario report.
    report_index: usize,
    /// Human-facing one-based report ID.
    displayed_report_id: usize,
    feature_scenario: u8,
    feature: &'static str,
    feature_filter: &'static str,
    name: &'static str,
    tail: TailShape,
    query: &'static str,
}

impl TckCase {
    fn label(self) -> String {
        format!(
            "report index {} / displayed ID {} / Remove3 {}",
            self.report_index, self.displayed_report_id, self.name
        )
    }
}

fn all_cases() -> impl Iterator<Item = TckCase> {
    TailShape::ALL.into_iter().map(|tail| {
        let offset = tail.index();
        TckCase {
            report_index: 687 + offset,
            displayed_report_id: 688 + offset,
            feature_scenario: 15 + u8::try_from(offset).expect("seven-case offset fits u8"),
            feature: FEATURE,
            feature_filter: FEATURE_FILTER,
            name: SCENARIO_NAMES[offset],
            tail,
            query: QUERIES[offset],
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PropertyTokenExpectation {
    GloballyUnknown,
    ExistsElsewhere,
}

#[derive(Clone, Copy, Debug)]
struct NoOpCase {
    name: &'static str,
    query: &'static str,
    property_name: &'static str,
    token: PropertyTokenExpectation,
}

const NO_OP_CASES: [NoOpCase; 2] = [
    NoOpCase {
        name: "REMOVE globally unknown relationship property",
        query: "MATCH ()-[r:R]->() REMOVE r.globallyUnknown RETURN r.num AS num",
        property_name: "globallyUnknown",
        token: PropertyTokenExpectation::GloballyUnknown,
    },
    NoOpCase {
        name: "REMOVE relationship property token present only on another relationship",
        query: "MATCH ()-[r:R]->() REMOVE r.onlyOnOther RETURN r.num AS num",
        property_name: "onlyOnOther",
        token: PropertyTokenExpectation::ExistsElsewhere,
    },
];

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    num: PropertyId,
    name: PropertyId,
}

impl Fixture {
    fn official(case: TckCase) -> Result<Self> {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("name")?;

        for offset in 0..case.tail.selected_rows() {
            let source = NodeId(u64::try_from(offset * 2 + 1).expect("fixture node ID fits u64"));
            let target = NodeId(u64::try_from(offset * 2 + 2).expect("fixture node ID fits u64"));
            let revision = u64::try_from(offset * 3).expect("fixture revision fits u64");
            graph.insert_node(NodeInput {
                id: source,
                layer: Layer::Observed,
                revision: revision + 1,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
            graph.insert_node(NodeInput {
                id: target,
                layer: Layer::Observed,
                revision: revision + 2,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;

            let numeric = match case.tail {
                TailShape::LimitZero
                | TailShape::SkipAll
                | TailShape::PageTwo
                | TailShape::PageAll => 42,
                TailShape::Filter | TailShape::ReturnAggregate | TailShape::WithAggregate => {
                    i64::try_from(offset + 1).expect("five fixture rows fit i64")
                }
            };
            let mut properties = vec![(num, ScalarValue::Integer(numeric))];
            if !matches!(case.tail, TailShape::LimitZero | TailShape::SkipAll) {
                properties.push((name, ScalarValue::String(Arc::<str>::from("a"))));
            }
            graph.insert_edge(EdgeInput {
                id: EdgeId(100 + u64::try_from(offset).expect("fixture edge ID fits u64")),
                source,
                target,
                relationship_type,
                layer: Layer::Observed,
                revision: revision + 3,
                properties,
            })?;
        }

        let bookmark = Bookmark {
            term: 53,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
        })
    }

    fn no_op_relationship_properties() -> Result<Self> {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let other_relationship_type = graph.catalog_mut().intern_relationship_type("OTHER_R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("onlyOnOther")?;

        for id in 1_u64..=4 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(100),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 5,
            properties: vec![(num, ScalarValue::Integer(7))],
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(200),
            source: NodeId(3),
            target: NodeId(4),
            relationship_type: other_relationship_type,
            layer: Layer::Observed,
            revision: 6,
            properties: vec![(name, ScalarValue::Integer(99))],
        })?;

        let bookmark = Bookmark {
            term: 59,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
        })
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

    fn cpu(&self) -> Result<CpuBackend> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(cpu)
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
        next_node_id: 1_000,
        next_edge_id: 1_000,
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

fn integer_values(
    output: &ExecutionOutput,
    column_name: &str,
) -> std::result::Result<Vec<i64>, String> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() {
            return Err("result batch has misaligned columns".to_owned());
        }
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == column_name)
            .ok_or_else(|| format!("result batch omitted expected `{column_name}` column"))?;
        for value in &column.values {
            match value {
                ResultValue::Scalar(ScalarValue::Integer(value)) => values.push(*value),
                _ => return Err(format!("expected integer result, got {value:?}")),
            }
        }
    }
    values.sort_unstable();
    Ok(values)
}

fn mutation_debug(output: &ExecutionOutput) -> Vec<String> {
    output
        .graph_mutations
        .iter()
        .map(|mutation| format!("{mutation:?}"))
        .collect()
}

fn assert_official_output(
    fixture: &Fixture,
    case: TckCase,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    let expected_type = if matches!(case.tail, TailShape::LimitZero | TailShape::SkipAll) {
        ColumnType::Null
    } else {
        ColumnType::Integer
    };
    let expected_schema = vec![(case.tail.expected_column().to_owned(), expected_type)];
    if output.result.schema != expected_schema {
        return Err(format!(
            "schema mismatch: expected {expected_schema:?}, got {:?}",
            output.result.schema
        ));
    }
    if integer_values(output, case.tail.expected_column())? != case.tail.expected_values() {
        return Err(format!(
            "row mismatch: expected {:?}, got {:?}",
            case.tail.expected_values(),
            integer_values(output, case.tail.expected_column())?
        ));
    }
    let expected_stats = StatementStats {
        properties_set: case.tail.selected_rows() as u64,
        ..StatementStats::default()
    };
    if output.result.statistics != expected_stats {
        return Err(format!(
            "statistics mismatch: expected {expected_stats:?}, got {:?}",
            output.result.statistics
        ));
    }
    if output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
    {
        return Err("result changed its bookmark, truncated, or emitted temporal work".to_owned());
    }

    let expected_property = if case.tail.removed_property() == "num" {
        fixture.num
    } else {
        fixture.name
    };
    let mut actual_edges = BTreeSet::new();
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::SetEdgeProperty {
                edge,
                property,
                value: ScalarValue::Null,
                ..
            } if *property == expected_property => {
                actual_edges.insert(*edge);
            }
            _ => {
                return Err(format!(
                    "unexpected relationship REMOVE mutation: {mutation:?}"
                ));
            }
        }
    }
    let expected_edges = (0..case.tail.selected_rows())
        .map(|offset| EdgeId(100 + u64::try_from(offset).expect("fixture edge offset fits u64")))
        .collect::<BTreeSet<_>>();
    if actual_edges != expected_edges || output.graph_mutations.len() != expected_edges.len() {
        return Err(format!(
            "mutation target mismatch: expected {expected_edges:?}, got {:?}",
            mutation_debug(output)
        ));
    }
    let expected_write_targets = expected_edges
        .iter()
        .copied()
        .map(EntityDependency::Relationship)
        .collect::<BTreeSet<_>>();
    if output.dependencies.write_targets != expected_write_targets {
        return Err(format!(
            "write targets mismatch: expected {expected_write_targets:?}, got {:?}",
            output.dependencies.write_targets
        ));
    }
    Ok(())
}

fn assert_no_op_fixture(fixture: &Fixture) -> std::result::Result<(), String> {
    let catalog = fixture.graph.catalog();
    if catalog.property("globallyUnknown").is_some()
        || catalog.property("onlyOnOther") != Some(fixture.name)
    {
        return Err(
            "no-op fixture does not distinguish unknown and elsewhere-only tokens".to_owned(),
        );
    }
    let matched = fixture
        .graph
        .edge(EdgeId(100))
        .ok_or_else(|| "no-op fixture omitted matched relationship 100".to_owned())?;
    let other = fixture
        .graph
        .edge(EdgeId(200))
        .ok_or_else(|| "no-op fixture omitted other relationship 200".to_owned())?;
    if matched.property(fixture.num) != Some(ScalarValue::Integer(7))
        || matched.property(fixture.name).is_some()
        || other.property(fixture.name) != Some(ScalarValue::Integer(99))
    {
        return Err("elsewhere-only property was placed on the matched relationship".to_owned());
    }
    Ok(())
}

fn assert_no_op_output(
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    if output.result.schema != [("num".to_owned(), ColumnType::Integer)]
        || integer_values(output, "num")? != [7]
    {
        return Err(format!(
            "no-op changed its result: schema={:?}, rows={:?}",
            output.result.schema,
            integer_values(output, "num")?
        ));
    }
    if output.result.statistics != StatementStats::default()
        || !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || !output.dependencies.write_targets.is_empty()
    {
        return Err(format!(
            "no-op published an effect: statistics={:?}, mutations={:?}, temporal={}, targets={:?}",
            output.result.statistics,
            mutation_debug(output),
            output.temporal_mutations.len(),
            output.dependencies.write_targets
        ));
    }
    if output.result.bookmark != fixture.bookmark || output.result.truncated {
        return Err("no-op changed its bookmark or truncation state".to_owned());
    }
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_command_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNodePipelineRequest>>,
}

/// Allows only one pinned complete mutation-continuation command and closes generic query routes.
struct ObservedRelationshipRemoveBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl ObservedRelationshipRemoveBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "relationship REMOVE acceptance did not construct real Metal",
            ));
        }
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    fn new<B: ExecutionBackend + 'static>(
        inner: B,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("relationship REMOVE backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("relationship REMOVE backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner: Box::new(inner),
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

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict relationship REMOVE test rejected `{route}` execution"),
        ))
    }

    fn refresh_expected_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement relationship REMOVE project has no bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement relationship REMOVE project has no revision")
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedRelationshipRemoveBackend {
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
            return self.reject_query_route("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_query_route("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "pinned relationship REMOVE generation changed before dispatch",
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
        self.refresh_expected_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)?;
        self.refresh_expected_fence()
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
        self.reject_query_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_query_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_query_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_query_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_query_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        if !self.pinned {
            return self.reject_query_route("execute_node_pipeline_on_unpinned_generation");
        }
        if request.project != PROJECT
            || request
                .mutation
                .as_ref()
                .and_then(|program| program.continuation.as_ref())
                .is_none()
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "relationship REMOVE request escaped its complete pinned generation",
            ));
        }
        request.validate_mutation()?;
        self.observations
            .complete_command_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_query_route("exact_l2")
    }
}

fn assert_effect_obligation(
    command_index: u16,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
    expected_kind: ResidentObligationKind,
) -> std::result::Result<(), String> {
    if kind != expected_kind || scope != ResidentObligationScope::MutationCommand(command_index) {
        return Err(format!(
            "command {command_index} obligation has wrong kind/scope: {kind:?}/{scope:?}"
        ));
    }
    Ok(())
}

fn assert_complete_request(
    fixture: &Fixture,
    request: &ResidentNodePipelineRequest,
    property_name: &str,
    property_token: Option<PropertyId>,
    expected_column: &str,
) -> std::result::Result<(), String> {
    request
        .validate_mutation()
        .map_err(|error| format!("complete request is invalid: {error}"))?;
    if request.project != PROJECT || request.offset != 0 || request.limit != usize::MAX {
        return Err(format!(
            "request escaped project or retained legacy pagination: {request:#?}"
        ));
    }
    let program = request
        .mutation
        .as_ref()
        .ok_or_else(|| "complete request omitted mutation program".to_owned())?;
    if program.commands.len() != 1 || program.max_intents == 0 {
        return Err(format!(
            "relationship REMOVE requires one bounded command, got {program:#?}"
        ));
    }
    let command = &program.commands[0];
    let property_index = match command.operation {
        ResidentMutationOperation::RemoveProperty {
            target: ResidentEntityBinding::Relationship(_),
            property_name,
        } => property_name,
        ref operation => {
            return Err(format!(
                "native operation is not relationship RemoveProperty: {operation:?}"
            ));
        }
    };
    if program
        .property_names
        .get(property_index as usize)
        .map(String::as_str)
        != Some(property_name)
        || program
            .property_tokens
            .get(property_index as usize)
            .copied()
            != Some(property_token)
    {
        return Err(format!(
            "relationship REMOVE property token/name mismatch: names={:?}, tokens={:?}",
            program.property_names, program.property_tokens
        ));
    }
    assert_effect_obligation(
        0,
        command.rhs_obligation.kind,
        command.rhs_obligation.scope,
        ResidentObligationKind::MutationRhs,
    )?;
    assert_effect_obligation(
        0,
        command.effect_obligation.kind,
        command.effect_obligation.scope,
        ResidentObligationKind::MutationEffect,
    )?;

    let continuation = program
        .continuation
        .as_ref()
        .ok_or_else(|| "relationship REMOVE omitted complete continuation".to_owned())?;
    if continuation.expected_bookmark != fixture.bookmark
        || continuation.expected_graph_revision != fixture.graph.revision()
        || continuation.expected_layout_version != fixture.graph.layout_version()
        || continuation.stage_obligations.len() != continuation.stages.len()
        || continuation.outputs.len() != 1
        || continuation.outputs[0].name != expected_column
        || continuation.maximum_output_cells
            != continuation
                .maximum_output_rows
                .saturating_mul(continuation.outputs.len())
        || continuation.maximum_output_arena_bytes != 0
    {
        return Err(format!(
            "relationship REMOVE continuation fence/shape is incomplete: {continuation:#?}"
        ));
    }
    let fingerprint = request
        .mutation_pipeline_fingerprint()
        .map_err(|error| format!("cannot recompute complete-command fingerprint: {error}"))?;
    if continuation.fingerprint != fingerprint {
        return Err("continuation fingerprint does not seal the complete request".to_owned());
    }
    Ok(())
}

fn assert_official_request(
    fixture: &Fixture,
    case: TckCase,
    request: &ResidentNodePipelineRequest,
) -> std::result::Result<(), String> {
    let property_token = if case.tail.removed_property() == "num" {
        fixture.num
    } else {
        fixture.name
    };
    assert_complete_request(
        fixture,
        request,
        case.tail.removed_property(),
        Some(property_token),
        case.tail.expected_column(),
    )?;
    let program = request.mutation.as_ref().expect("checked above");
    if program.max_intents < case.tail.selected_rows() {
        return Err(format!(
            "mutation capacity {} is smaller than {} selected rows",
            program.max_intents,
            case.tail.selected_rows()
        ));
    }
    let stages = &program.continuation.as_ref().expect("checked above").stages;
    let exact_tail = match (case.tail, stages.as_slice()) {
        (
            TailShape::LimitZero,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Limit { rows: 0 },
            ],
        )
        | (
            TailShape::SkipAll,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 1 },
            ],
        )
        | (
            TailShape::PageTwo,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 2 },
                ResidentMutationPostStage::Limit { rows: 2 },
            ],
        )
        | (
            TailShape::PageAll,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::Skip { rows: 0 },
                ResidentMutationPostStage::Limit { rows: 5 },
            ],
        )
        | (
            TailShape::Filter,
            [
                ResidentMutationPostStage::Project { .. },
                ResidentMutationPostStage::FilterIntegerModuloEquals {
                    divisor: 2,
                    operand: 0,
                    ..
                },
                ResidentMutationPostStage::Project { .. },
            ],
        )
        | (TailShape::ReturnAggregate, [ResidentMutationPostStage::SumInteger { .. }])
        | (TailShape::WithAggregate, [ResidentMutationPostStage::SumInteger { .. }]) => true,
        _ => false,
    };
    if !exact_tail {
        return Err(format!(
            "wrong ordered {:?} continuation: {stages:?}",
            case.tail
        ));
    }
    Ok(())
}

fn assert_no_op_request(
    fixture: &Fixture,
    case: NoOpCase,
    request: &ResidentNodePipelineRequest,
) -> std::result::Result<(), String> {
    let expected_token = match case.token {
        PropertyTokenExpectation::GloballyUnknown => None,
        PropertyTokenExpectation::ExistsElsewhere => Some(fixture.name),
    };
    assert_complete_request(fixture, request, case.property_name, expected_token, "num")?;
    let program = request.mutation.as_ref().expect("checked above");
    if program.max_intents < 1
        || !matches!(
            program
                .continuation
                .as_ref()
                .expect("checked above")
                .stages
                .as_slice(),
            [ResidentMutationPostStage::Project { .. }]
        )
    {
        return Err(format!(
            "no-op escaped its one-command projection continuation: {request:#?}"
        ));
    }
    Ok(())
}

fn assert_same_publication(
    reference: &ExecutionOutput,
    native: &ExecutionOutput,
) -> std::result::Result<(), String> {
    if native.result != reference.result
        || mutation_debug(native) != mutation_debug(reference)
        || native.dependencies.entities != reference.dependencies.entities
        || native.dependencies.write_targets != reference.dependencies.write_targets
        || native.temporal_mutations.len() != reference.temporal_mutations.len()
    {
        return Err(format!(
            "native publication differs from generic CPU:\nCPU result: {:#?}\nnative result: {:#?}\nCPU mutations: {:?}\nnative mutations: {:?}",
            reference.result,
            native.result,
            mutation_debug(reference),
            mutation_debug(native)
        ));
    }
    Ok(())
}

fn execute_native_official(
    fixture: &Fixture,
    backend: &ObservedRelationshipRemoveBackend,
    case: TckCase,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(case.query, &mut context(fixture, None, false))
        .map_err(|error| format!("generic CPU failed with {:?}: {error}", error.code))?;
    assert_official_output(fixture, case, &reference)?;

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = QueryEngine
        .execute(case.query, &mut context(fixture, Some(backend), true))
        .map_err(|error| {
            format!(
                "native failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1
        || observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "route was not exactly one pin, one complete command, and zero fallback calls"
                .to_owned(),
        );
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one complete request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "complete request disappeared".to_owned())?
    };
    assert_official_request(fixture, case, &request)?;
    assert_official_output(fixture, case, &native)?;
    assert_same_publication(&reference, &native)
}

fn execute_native_no_op(
    fixture: &Fixture,
    backend: &ObservedRelationshipRemoveBackend,
    case: NoOpCase,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(case.query, &mut context(fixture, None, false))
        .map_err(|error| format!("generic CPU failed with {:?}: {error}", error.code))?;
    assert_no_op_output(fixture, &reference)?;

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.complete_command_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let native = QueryEngine
        .execute(case.query, &mut context(fixture, Some(backend), true))
        .map_err(|error| {
            format!(
                "native no-op failed with {:?}: {error}; pins={}, complete calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.complete_command_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1
        || observations.complete_command_calls.load(Ordering::SeqCst) != calls_before + 1
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err("no-op route was not one pin/complete command with zero fallback".to_owned());
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain one no-op command".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "no-op command disappeared".to_owned())?
    };
    assert_no_op_request(fixture, case, &request)?;
    assert_no_op_output(fixture, &native)?;
    assert_same_publication(&reference, &native)
}

fn run_all_native_official(
    backend: &mut ObservedRelationshipRemoveBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = match Fixture::official(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: fixture failed: {error}", case.label()));
                continue;
            }
        };
        if let Err(error) = fixture
            .image()
            .and_then(|image| backend.replace_all_projects(vec![image]))
        {
            failures.push(format!(
                "{}: resident replacement failed: {error}",
                case.label()
            ));
            continue;
        }
        if let Err(error) = execute_native_official(&fixture, backend, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn run_native_no_ops(
    fixture: &Fixture,
    backend: &ObservedRelationshipRemoveBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for case in NO_OP_CASES {
        if let Err(error) = execute_native_no_op(fixture, backend, case) {
            failures.push(format!("{}: {error}", case.name));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn assert_no_failures(boundary: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{boundary} had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_is_exactly_remove3_relationship_property_scenarios_15_through_21() {
    let cases = all_cases().collect::<Vec<_>>();
    assert_eq!(cases.len(), 7);
    assert_eq!(
        cases
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>(),
        (687_usize..=693).collect::<Vec<_>>()
    );
    assert_eq!(
        cases
            .iter()
            .map(|case| case.displayed_report_id)
            .collect::<Vec<_>>(),
        (688_usize..=694).collect::<Vec<_>>()
    );
    assert_eq!(
        cases
            .iter()
            .map(|case| case.feature_scenario)
            .collect::<Vec<_>>(),
        (15_u8..=21).collect::<Vec<_>>()
    );
    assert!(cases.iter().all(|case| {
        case.feature == FEATURE
            && case.feature_filter == FEATURE_FILTER
            && case
                .name
                .starts_with(&format!("[{}]", case.feature_scenario))
            && case.query.contains("REMOVE r.")
    }));
    assert_eq!(
        cases.iter().map(|case| case.name).collect::<Vec<_>>(),
        SCENARIO_NAMES
    );
    assert_eq!(
        cases.iter().map(|case| case.query).collect::<Vec<_>>(),
        QUERIES
    );
}

#[test]
fn generic_cpu_oracle_proves_exact_seven_rows_statistics_and_relationship_mutations() -> Result<()>
{
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::official(case)?;
        match QueryEngine.execute(case.query, &mut context(&fixture, None, false)) {
            Ok(output) => {
                if let Err(error) = assert_official_output(&fixture, case, &output) {
                    failures.push(format!("{}: {error}", case.label()));
                }
            }
            Err(error) => failures.push(format!(
                "{}: generic CPU failed with {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert_no_failures("generic CPU Remove3 [15]-[21] oracle", failures);
    Ok(())
}

#[test]
fn generic_cpu_oracle_proves_target_local_relationship_remove_no_ops() -> Result<()> {
    let fixture = Fixture::no_op_relationship_properties()?;
    assert_no_op_fixture(&fixture).map_err(Error::internal)?;
    let mut failures = Vec::new();
    for case in NO_OP_CASES {
        match QueryEngine.execute(case.query, &mut context(&fixture, None, false)) {
            Ok(output) => {
                if let Err(error) = assert_no_op_output(&fixture, &output) {
                    failures.push(format!("{}: {error}", case.name));
                }
            }
            Err(error) => failures.push(format!(
                "{}: generic CPU failed with {:?}: {error}",
                case.name, error.code
            )),
        }
    }
    assert_no_op_fixture(&fixture).map_err(Error::internal)?;
    assert_no_failures("generic CPU relationship no-op oracle", failures);
    Ok(())
}

#[test]
fn native_relationship_remove_is_complete_or_fails_closed_before_dispatch() -> Result<()> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::official(case)?;
        let backend = ObservedRelationshipRemoveBackend::strict_cpu_reference(fixture.cpu()?)?;
        let observations = backend.observations();
        match QueryEngine.execute(case.query, &mut context(&fixture, Some(&backend), true)) {
            Ok(output) => {
                if observations.pins.load(Ordering::SeqCst) != 1
                    || observations.complete_command_calls.load(Ordering::SeqCst) != 1
                    || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
                {
                    failures.push(format!(
                        "{}: successful route was not one pin/complete command with zero fallback",
                        case.label()
                    ));
                } else {
                    let request = observations
                        .requests
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .last()
                        .cloned();
                    if let Some(request) = request {
                        if let Err(error) = assert_official_request(&fixture, case, &request) {
                            failures.push(format!("{}: {error}", case.label()));
                        }
                    } else {
                        failures.push(format!(
                            "{}: successful route retained no request",
                            case.label()
                        ));
                    }
                    if let Err(error) = assert_official_output(&fixture, case, &output) {
                        failures.push(format!("{}: partial publication: {error}", case.label()));
                    }
                }
            }
            Err(error) if error.code == ErrorCode::GpuAdmissionFailure => {
                if observations.pins.load(Ordering::SeqCst) != 0
                    || observations.complete_command_calls.load(Ordering::SeqCst) != 0
                    || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
                {
                    failures.push(format!(
                        "{}: fail-closed route dispatched a prefix or generic call",
                        case.label()
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "{}: expected complete success or fail-closed admission, got {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert_no_failures("relationship REMOVE monotonic routing boundary", failures);
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: Remove3 [15]-[21] require one complete native relationship REMOVE command"]
fn strict_cpu_reference_runs_exact_seven_through_one_pinned_complete_command() -> Result<()> {
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("relationship REMOVE manifest is empty"))?;
    let fixture = Fixture::official(first)?;
    let mut backend = ObservedRelationshipRemoveBackend::strict_cpu_reference(fixture.cpu()?)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    if let Err(failures) = run_all_native_official(&mut backend) {
        assert_no_failures("strict CPU relationship REMOVE", failures);
    }
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: target-local relationship REMOVE no-ops require backend effect receipts"]
fn strict_cpu_reference_proves_relationship_remove_no_ops_have_zero_effects() -> Result<()> {
    let fixture = Fixture::no_op_relationship_properties()?;
    assert_no_op_fixture(&fixture).map_err(Error::internal)?;
    let backend = ObservedRelationshipRemoveBackend::strict_cpu_reference(fixture.cpu()?)?;
    if let Err(failures) = run_native_no_ops(&fixture, &backend) {
        assert_no_failures("strict CPU relationship REMOVE no-ops", failures);
    }
    Ok(())
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
#[ignore = "red acceptance gate: exact seven plus no-op adversaries must match CPU on real Metal"]
fn real_metal_matches_cpu_for_exact_seven_and_no_ops_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("relationship REMOVE manifest is empty"))?;
    let fixture = Fixture::official(first)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    let mut backend = ObservedRelationshipRemoveBackend::real_metal(metal)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Metal);

    let mut failures = run_all_native_official(&mut backend)
        .err()
        .unwrap_or_default();
    let no_op = Fixture::no_op_relationship_properties()?;
    match no_op
        .image()
        .and_then(|image| backend.replace_all_projects(vec![image]))
    {
        Ok(()) => {
            if let Err(mut no_op_failures) = run_native_no_ops(&no_op, &backend) {
                failures.append(&mut no_op_failures);
            }
        }
        Err(error) => failures.push(format!("no-op resident replacement failed: {error}")),
    }
    assert_no_failures("real Metal relationship REMOVE acceptance", failures);
    Ok(())
}
