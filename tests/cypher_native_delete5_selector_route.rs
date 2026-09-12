// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact strict-native route gate for openCypher Delete5 scenarios [1]-[6].
//!
//! These six source queries pin the only admitted compiler shapes: a direct entity in a
//! single-key map, or a direct `collect(entity)` selected by one non-negative literal/parameter
//! ordinal (possibly beneath exact single-key maps). The observing backend closes every legacy
//! graph primitive, captures the sealed command, and delegates only that command to the CPU
//! semantic reference. Near misses must fail before project pinning or backend dispatch.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, EntityDependency, ExecutionContext, ExecutionOutput, QueryEngine,
        ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentDeleteRequest,
        ResidentDeleteResult, ResidentDeleteTargetSelector, ResidentEntityBinding, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodeBinding,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4445_4c45_5445_355f_5345_4c45_4354_4f52,
));
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 100_000;

const FRIEND_FIXTURE: &str = r#"
CREATE (u:User)
CREATE (u)-[:FRIEND]->()
CREATE (u)-[:FRIEND]->()
CREATE (u)-[:FRIEND]->()
CREATE (u)-[:FRIEND]->()
"#;

const TWO_USERS: &str = "CREATE (:User), (:User)";

const TWO_USER_CYCLE: &str = r#"
CREATE (a:User), (b:User)
CREATE (a)-[:R]->(b)
CREATE (b)-[:R]->(a)
"#;

#[derive(Clone, Copy)]
struct Delete5Case {
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    friend_index: Option<i64>,
    target: ResidentEntityBinding,
    selector: ResidentDeleteTargetSelector,
    detach: bool,
    selected_rows: usize,
    command_attempts: usize,
    read_dependencies: usize,
    nodes_deleted: usize,
    relationships_deleted: usize,
    relationships_removed: usize,
}

const CASES: &[Delete5Case] = &[
    Delete5Case {
        name: "Delete5 [1] node from collect",
        setup: FRIEND_FIXTURE,
        query: r#"
MATCH (:User)-[:FRIEND]->(n)
WITH collect(n) AS friends
DETACH DELETE friends[$friendIndex]
"#,
        friend_index: Some(1),
        target: ResidentEntityBinding::Node(ResidentNodeBinding::End),
        selector: ResidentDeleteTargetSelector::CollectedOrdinal { index: 1 },
        detach: true,
        selected_rows: 4,
        command_attempts: 1,
        read_dependencies: 9,
        nodes_deleted: 1,
        relationships_deleted: 0,
        relationships_removed: 1,
    },
    Delete5Case {
        name: "Delete5 [2] relationship from collect",
        setup: FRIEND_FIXTURE,
        query: r#"
MATCH (:User)-[r:FRIEND]->()
WITH collect(r) AS friendships
DETACH DELETE friendships[$friendIndex]
"#,
        friend_index: Some(1),
        target: ResidentEntityBinding::Relationship(0),
        selector: ResidentDeleteTargetSelector::CollectedOrdinal { index: 1 },
        detach: true,
        selected_rows: 4,
        command_attempts: 1,
        read_dependencies: 9,
        nodes_deleted: 0,
        relationships_deleted: 1,
        relationships_removed: 1,
    },
    Delete5Case {
        name: "Delete5 [3] nodes from map",
        setup: TWO_USERS,
        query: r#"
MATCH (u:User)
WITH {key: u} AS nodes
DELETE nodes.key
"#,
        friend_index: None,
        target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
        selector: ResidentDeleteTargetSelector::EachSelectedRow,
        detach: false,
        selected_rows: 2,
        command_attempts: 2,
        read_dependencies: 2,
        nodes_deleted: 2,
        relationships_deleted: 0,
        relationships_removed: 0,
    },
    Delete5Case {
        name: "Delete5 [4] relationships from map",
        setup: TWO_USER_CYCLE,
        query: r#"
MATCH (:User)-[r]->(:User)
WITH {key: r} AS rels
DELETE rels.key
"#,
        friend_index: None,
        target: ResidentEntityBinding::Relationship(0),
        selector: ResidentDeleteTargetSelector::EachSelectedRow,
        detach: false,
        selected_rows: 2,
        command_attempts: 2,
        read_dependencies: 4,
        nodes_deleted: 0,
        relationships_deleted: 2,
        relationships_removed: 2,
    },
    Delete5Case {
        name: "Delete5 [5] node from nested map and collect",
        setup: TWO_USER_CYCLE,
        query: r#"
MATCH (u:User)
WITH {key: collect(u)} AS nodeMap
DETACH DELETE nodeMap.key[0]
"#,
        friend_index: None,
        target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
        selector: ResidentDeleteTargetSelector::CollectedOrdinal { index: 0 },
        detach: true,
        selected_rows: 2,
        command_attempts: 1,
        read_dependencies: 2,
        nodes_deleted: 1,
        relationships_deleted: 0,
        relationships_removed: 2,
    },
    Delete5Case {
        name: "Delete5 [6] relationship from nested maps and collect",
        setup: TWO_USER_CYCLE,
        query: r#"
MATCH (:User)-[r]->(:User)
WITH {key: {key: collect(r)}} AS rels
DELETE rels.key.key[0]
"#,
        friend_index: None,
        target: ResidentEntityBinding::Relationship(0),
        selector: ResidentDeleteTargetSelector::CollectedOrdinal { index: 0 },
        detach: false,
        selected_rows: 2,
        command_attempts: 1,
        read_dependencies: 4,
        nodes_deleted: 0,
        relationships_deleted: 1,
        relationships_removed: 1,
    },
];

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    delete_calls: AtomicUsize,
    rejected_routes: AtomicUsize,
    requests: Mutex<Vec<ResidentDeleteRequest>>,
    raw_results: Mutex<Vec<String>>,
}

struct StrictDelete5Backend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictDelete5Backend {
    fn new(inner: CpuBackend) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("Delete5 backend omitted its resident bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("Delete5 backend omitted its resident revision"))?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .rejected_routes
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict Delete5 gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictDelete5Backend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            BackendKind::Cpu
        } else {
            // Strict planning must treat this observer as an accelerator. Only its pinned inner
            // command reports CPU-reference completion provenance.
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
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != BackendKind::Cpu
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Delete5 pin changed the immutable resident generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
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

    fn execute_delete_pipeline(
        &self,
        request: &ResidentDeleteRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentDeleteResult> {
        if !self.pinned {
            return self.reject("execute_delete_pipeline_on_unpinned_generation");
        }
        if request.project != PROJECT
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Delete5 request escaped its pinned resident generation",
            ));
        }
        self.observations
            .delete_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let result = self.inner.execute_delete_pipeline(request, cancellation);
        self.observations
            .raw_results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(format!("{result:#?}"));
        result
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

fn parameters(friend_index: Option<i64>) -> BTreeMap<String, ResultValue> {
    friend_index.map_or_else(BTreeMap::new, |index| {
        BTreeMap::from([(
            "friendIndex".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(index)),
        )])
    })
}

fn context<'a>(
    graph: &'a GraphStore,
    parameters: BTreeMap<String, ResultValue>,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let next_node_id = graph
        .nodes()
        .map(|node| node.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
    let next_edge_id = graph
        .edges()
        .map(|edge| edge.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
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
            term: 53,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id,
        next_edge_id,
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
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn fixture_graph(setup: &str) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let output = QueryEngine.execute(setup, &mut context(&graph, BTreeMap::new(), None, false))?;
    apply_mutations(&mut graph, &output.graph_mutations)?;
    Ok(graph)
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 53,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn admitted_backend(graph: &GraphStore) -> Result<StrictDelete5Backend> {
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(resident_image(graph)?)?;
    StrictDelete5Backend::new(cpu)
}

fn mutation_targets(output: &ExecutionOutput) -> BTreeSet<EntityDependency> {
    output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::DeleteNode { node, .. } => Some(EntityDependency::Node(*node)),
            GraphMutation::DeleteEdge { edge, .. } => Some(EntityDependency::Relationship(*edge)),
            _ => None,
        })
        .collect()
}

fn first_debug_list_value(raw: &str, field: &str) -> Option<usize> {
    raw.split_once(&format!("{field}: ["))?
        .1
        .trim_start()
        .split(',')
        .next()?
        .trim()
        .parse()
        .ok()
}

fn assert_exact_case(case: Delete5Case) -> Result<()> {
    let graph = fixture_graph(case.setup)?;
    let oracle = QueryEngine.execute(
        case.query,
        &mut context(&graph, parameters(case.friend_index), None, false),
    )?;
    let backend = admitted_backend(&graph)?;
    assert_eq!(backend.kind(), BackendKind::Metal, "{}", case.name);
    let observations = backend.observations();
    let native = QueryEngine.execute(
        case.query,
        &mut context(&graph, parameters(case.friend_index), Some(&backend), true),
    )?;

    assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{}", case.name);
    assert_eq!(
        observations.delete_calls.load(Ordering::SeqCst),
        1,
        "{}",
        case.name
    );
    assert_eq!(
        observations.rejected_routes.load(Ordering::SeqCst),
        0,
        "{}",
        case.name
    );
    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| Error::internal(format!("{} omitted its resident request", case.name)))?;
    assert_eq!(request.commands.len(), 1, "{}", case.name);
    assert_eq!(request.commands[0].target, case.target, "{}", case.name);
    assert_eq!(request.commands[0].selector, case.selector, "{}", case.name);
    assert_eq!(request.commands[0].detach, case.detach, "{}", case.name);
    assert!(request.continuation.is_none(), "{}", case.name);
    assert_eq!(request.selection.offset, 0, "{}", case.name);
    assert_eq!(request.selection.limit, usize::MAX, "{}", case.name);
    assert!(
        request.selection.max_output_rows >= case.selected_rows,
        "{} truncated its stable prewrite relation",
        case.name
    );

    let raw = observations
        .raw_results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| Error::internal(format!("{} omitted its raw result", case.name)))?;
    assert!(
        raw.contains(&format!("selected_rows: {},", case.selected_rows)),
        "{} changed prewrite cardinality: {raw}",
        case.name
    );
    assert_eq!(
        first_debug_list_value(&raw, "command_attempts"),
        Some(case.command_attempts),
        "{} used the wrong selector attempt cardinality: {raw}",
        case.name
    );

    let expected_statistics = StatementStats {
        nodes_deleted: case.nodes_deleted as u64,
        relationships_deleted: case.relationships_deleted as u64,
        ..StatementStats::default()
    };
    assert!(native.result.schema.is_empty(), "{}", case.name);
    assert!(native.result.batches.is_empty(), "{}", case.name);
    assert_eq!(
        native.result.statistics, expected_statistics,
        "{}",
        case.name
    );
    assert_eq!(
        native.result.statistics, oracle.result.statistics,
        "{}",
        case.name
    );
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations),
        "{} selected a different stable entity ordinal",
        case.name
    );
    assert_eq!(
        native.dependencies.entities.len(),
        case.read_dependencies,
        "{} did not retain every selected prewrite entity dependency",
        case.name
    );
    let targets = mutation_targets(&native);
    assert_eq!(native.dependencies.write_targets, targets, "{}", case.name);
    assert_eq!(
        targets.len(),
        case.nodes_deleted + case.relationships_deleted
    );

    let mut committed = graph.clone();
    apply_mutations(&mut committed, &native.graph_mutations)?;
    assert_eq!(
        graph.node_count() - committed.node_count(),
        case.nodes_deleted,
        "{}",
        case.name
    );
    assert_eq!(
        graph.edge_count() - committed.edge_count(),
        case.relationships_removed,
        "{}",
        case.name
    );
    Ok(())
}

#[test]
fn delete5_one_through_six_use_exact_selector_and_full_prewrite_relation() -> Result<()> {
    for case in CASES {
        assert_exact_case(*case)?;
    }
    Ok(())
}

#[test]
fn broader_delete_expressions_fail_before_pin_or_dispatch() -> Result<()> {
    let graph = fixture_graph(FRIEND_FIXTURE)?;
    let backend = admitted_backend(&graph)?;
    let observations = backend.observations();
    let near_misses = [
        (
            "arithmetic ordinal",
            "MATCH (:User)-[:FRIEND]->(n) WITH collect(n) AS friends DETACH DELETE friends[1 + 0]",
            BTreeMap::new(),
        ),
        (
            "negative ordinal",
            "MATCH (:User)-[:FRIEND]->(n) WITH collect(n) AS friends DETACH DELETE friends[-1]",
            BTreeMap::new(),
        ),
        (
            "non-integer parameter ordinal",
            "MATCH (:User)-[:FRIEND]->(n) WITH collect(n) AS friends DETACH DELETE friends[$friendIndex]",
            BTreeMap::from([(
                "friendIndex".to_owned(),
                ResultValue::Scalar(ScalarValue::String("1".into())),
            )]),
        ),
        (
            "distinct collect",
            "MATCH (:User)-[:FRIEND]->(n) WITH collect(DISTINCT n) AS friends DETACH DELETE friends[0]",
            BTreeMap::new(),
        ),
        (
            "collect expression",
            "MATCH (:User)-[:FRIEND]->(n) WITH collect({key:n}) AS friends DETACH DELETE friends[0]",
            BTreeMap::new(),
        ),
        (
            "dynamic map index",
            "MATCH (u:User) WITH {key:u} AS nodes DELETE nodes['key']",
            BTreeMap::new(),
        ),
        (
            "multi-key map",
            "MATCH (u:User) WITH {key:u, extra:u} AS nodes DELETE nodes.key",
            BTreeMap::new(),
        ),
        (
            "second wrapper projection",
            "MATCH (u:User) WITH {key:u} AS nodes WITH nodes AS aliases DELETE aliases.key",
            BTreeMap::new(),
        ),
    ];

    for (name, query, parameters) in near_misses {
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, parameters, Some(&backend), true),
            )
            .err()
            .ok_or_else(|| Error::internal(format!("{name} unexpectedly executed natively")))?;
        assert!(
            matches!(
                error.code,
                ErrorCode::GpuAdmissionFailure | ErrorCode::QuerySyntax | ErrorCode::QueryType
            ),
            "{name} returned the wrong failure: {error}"
        );
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            0,
            "{name} pinned a project"
        );
        assert_eq!(
            observations.delete_calls.load(Ordering::SeqCst),
            0,
            "{name} dispatched DELETE"
        );
        assert_eq!(
            observations.rejected_routes.load(Ordering::SeqCst),
            0,
            "{name} reached a legacy backend route"
        );
    }
    Ok(())
}
