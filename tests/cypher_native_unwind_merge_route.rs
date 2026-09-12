// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;
const FIRST_CREATED_NODE_ID: u64 = 100;

const UNWIND1_FEATURE: &str = "features/clauses/unwind/Unwind1.feature";
const UNWIND1_SCENARIO: &str = "[14] Unwind with merge";
const UNWIND_MERGE_QUERY: &str = "UNWIND $props AS prop \
     MERGE (p:Person {login: prop.login}) \
     SET p.name = prop.name \
     RETURN p.name, p.login";

#[derive(Clone, Debug)]
struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn empty() -> Self {
        Self::from_graph(GraphStore::default())
    }

    fn with_existing_person(login_value: &str, name_value: &str) -> Result<Self> {
        let mut graph = GraphStore::default();
        let person = graph.catalog_mut().intern_label("Person")?;
        let login = graph.catalog_mut().intern_property("login")?;
        let name = graph.catalog_mut().intern_property("name")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![person],
            properties: vec![
                (login, ScalarValue::String(Arc::from(login_value))),
                (name, ScalarValue::String(Arc::from(name_value))),
            ],
        })?;
        Ok(Self::from_graph(graph))
    }

    fn from_graph(graph: GraphStore) -> Self {
        Self {
            bookmark: Bookmark {
                term: 73,
                index: graph.revision(),
            },
            graph,
        }
    }

    fn resident_image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn cpu_backend(&self) -> Result<CpuBackend> {
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(self.resident_image()?)?;
        Ok(backend)
    }
}

fn string(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn property_map(login: Option<&str>, name: &str) -> ResultValue {
    ResultValue::Map(BTreeMap::from([
        (
            "login".to_owned(),
            login.map_or(ResultValue::Scalar(ScalarValue::Null), |value| {
                string(value)
            }),
        ),
        ("name".to_owned(), string(name)),
    ]))
}

fn parameters(rows: &[(Option<&str>, &str)]) -> BTreeMap<String, ResultValue> {
    BTreeMap::from([(
        "props".to_owned(),
        ResultValue::List(
            rows.iter()
                .map(|(login, name)| property_map(*login, name))
                .collect(),
        ),
    )])
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    parameters: BTreeMap<String, ResultValue>,
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
        parameters,
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: FIRST_CREATED_NODE_ID,
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

fn execute_cpu(fixture: &Fixture, rows: &[(Option<&str>, &str)]) -> Result<ExecutionOutput> {
    QueryEngine.execute(
        UNWIND_MERGE_QUERY,
        &mut context(fixture, None, parameters(rows), false),
    )
}

fn result_string(value: &ResultValue, column: &str) -> Result<String> {
    let ResultValue::Scalar(ScalarValue::String(value)) = value else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("`{column}` was not a non-null string: {value:?}"),
        ));
    };
    Ok(value.to_string())
}

fn output_rows(output: &ExecutionOutput) -> Result<Vec<(String, String)>> {
    let expected_schema = vec![
        ("p.name".to_owned(), ColumnType::String),
        ("p.login".to_owned(), ColumnType::String),
    ];
    if output.result.schema != expected_schema {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{UNWIND1_SCENARIO} returned the wrong schema: {:?}",
                output.result.schema
            ),
        ));
    }

    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if batch.columns.len() != 2
            || batch.columns[0].name != "p.name"
            || batch.columns[0].value_type != ColumnType::String
            || batch.columns[1].name != "p.login"
            || batch.columns[1].value_type != ColumnType::String
            || batch.columns[0].values.len() != batch.row_count
            || batch.columns[1].values.len() != batch.row_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("malformed UNWIND/MERGE result batch: {batch:?}"),
            ));
        }
        for row in 0..batch.row_count {
            rows.push((
                result_string(&batch.columns[0].values[row], "p.name")?,
                result_string(&batch.columns[1].values[row], "p.login")?,
            ));
        }
    }
    Ok(rows)
}

fn assert_output(
    fixture: &Fixture,
    output: &ExecutionOutput,
    expected_rows: &[(&str, &str)],
    expected_statistics: StatementStats,
) -> Result<()> {
    let mut actual = output_rows(output)?;
    let mut expected = expected_rows
        .iter()
        .map(|(name, login)| ((*name).to_owned(), (*login).to_owned()))
        .collect::<Vec<_>>();
    // Unwind1 [14] explicitly specifies "in any order". Multiplicity is retained by sorting the
    // complete vectors rather than collecting them into sets.
    actual.sort();
    expected.sort();
    if actual != expected {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{UNWIND1_FEATURE} / {UNWIND1_SCENARIO} rows differ; expected {expected:?}, got {actual:?}"
            ),
        ));
    }
    if output.result.statistics != expected_statistics
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "UNWIND/MERGE statistics, bookmark, or truncation changed: {:?}",
                output.result
            ),
        ));
    }
    if !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "UNWIND/MERGE emitted unrelated execution state",
        ));
    }
    Ok(())
}

fn published_graph(fixture: &Fixture, output: &ExecutionOutput) -> Result<GraphStore> {
    let mut graph = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(graph)
}

fn people_by_login(graph: &GraphStore) -> Result<BTreeMap<String, String>> {
    let person = graph
        .catalog()
        .label("Person")
        .ok_or_else(|| Error::internal("published graph has no Person label"))?;
    let login = graph
        .catalog()
        .property("login")
        .ok_or_else(|| Error::internal("published graph has no login property"))?;
    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| Error::internal("published graph has no name property"))?;

    let mut people = BTreeMap::new();
    for node in graph.nodes().filter(|node| node.labels().contains(&person)) {
        let Some(ScalarValue::String(login_value)) = node.property(login) else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("Person {:?} has no string login", node.id()),
            ));
        };
        let Some(ScalarValue::String(name_value)) = node.property(name) else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("Person {:?} has no string name", node.id()),
            ));
        };
        if people
            .insert(login_value.to_string(), name_value.to_string())
            .is_some()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("MERGE published duplicate Person login `{login_value}`"),
            ));
        }
    }
    Ok(people)
}

fn assert_published_people(
    fixture: &Fixture,
    output: &ExecutionOutput,
    expected: &[(&str, &str)],
) -> Result<()> {
    let graph = published_graph(fixture, output)?;
    let expected = expected
        .iter()
        .map(|(login, name)| ((*login).to_owned(), (*name).to_owned()))
        .collect::<BTreeMap<_, _>>();
    let actual = people_by_login(&graph)?;
    if actual != expected || graph.node_count() != expected.len() || graph.edge_count() != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("published UNWIND/MERGE graph differs; expected {expected:?}, got {actual:?}"),
        ));
    }
    Ok(())
}

fn mutation_counts(output: &ExecutionOutput) -> (usize, usize) {
    let inserted = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
        .count();
    let set = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::SetNodeProperty { .. }))
        .count();
    (inserted, set)
}

fn assert_only_node_merge_mutations(output: &ExecutionOutput) -> Result<()> {
    if let Some(unexpected) = output.graph_mutations.iter().find(|mutation| {
        !matches!(
            mutation,
            GraphMutation::DeclareLabel { .. }
                | GraphMutation::DeclareProperty { .. }
                | GraphMutation::InsertNode(_)
                | GraphMutation::SetNodeProperty { .. }
        )
    }) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("UNWIND/MERGE emitted an unrelated mutation: {unexpected:?}"),
        ));
    }
    Ok(())
}

#[test]
fn cpu_oracle_matches_exact_opencypher_unwind1_14_rows_and_side_effects() -> Result<()> {
    let fixture = Fixture::empty();
    let output = execute_cpu(
        &fixture,
        &[(Some("login1"), "name1"), (Some("login2"), "name2")],
    )?;

    assert_output(
        &fixture,
        &output,
        &[("name1", "login1"), ("name2", "login2")],
        StatementStats {
            nodes_created: 2,
            properties_set: 2,
            labels_added: 2,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (2, 2));
    assert_published_people(
        &fixture,
        &output,
        &[("login1", "name1"), ("login2", "name2")],
    )?;

    let declared_labels = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::DeclareLabel { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let declared_properties = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::DeclareProperty { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(declared_labels, BTreeSet::from(["Person"]));
    assert_eq!(declared_properties, BTreeSet::from(["login", "name"]));
    // The TCK's +labels side effect counts the one newly introduced label token, while the public
    // statement statistics count the two node-label assignments asserted above.
    assert_eq!(fixture.graph.node_count(), 0);
    assert!(fixture.graph.catalog().label("Person").is_none());
    Ok(())
}

#[test]
fn cpu_oracle_merge_updates_an_existing_person_and_creates_only_the_missing_person() -> Result<()> {
    let fixture = Fixture::with_existing_person("kept", "old")?;
    let output = execute_cpu(
        &fixture,
        &[(Some("kept"), "updated"), (Some("new"), "fresh")],
    )?;

    assert_output(
        &fixture,
        &output,
        &[("updated", "kept"), ("fresh", "new")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 2));
    assert!(!output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::DeclareLabel { .. } | GraphMutation::DeclareProperty { .. }
    )));
    assert_published_people(&fixture, &output, &[("kept", "updated"), ("new", "fresh")])?;

    let name = fixture
        .graph
        .catalog()
        .property("name")
        .ok_or_else(|| Error::internal("existing fixture lost name property"))?;
    assert_eq!(
        fixture
            .graph
            .node(NodeId(1))
            .and_then(|node| node.property(name)),
        Some(ScalarValue::String(Arc::from("old"))),
        "staged MERGE/SET mutations leaked into the canonical input graph"
    );
    Ok(())
}

#[test]
fn cpu_oracle_duplicate_keys_reuse_one_created_node_and_observe_prior_rows() -> Result<()> {
    let fixture = Fixture::empty();
    let output = execute_cpu(
        &fixture,
        &[(Some("same"), "first"), (Some("same"), "second")],
    )?;

    // Both projected rows bind the same statement-local node. By RETURN time the second ordered
    // SET is visible through both bindings.
    assert_output(
        &fixture,
        &output,
        &[("second", "same"), ("second", "same")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 2));

    let inserted_node = output
        .graph_mutations
        .iter()
        .find_map(|mutation| match mutation {
            GraphMutation::InsertNode(node) => Some(node.id),
            _ => None,
        })
        .ok_or_else(|| Error::internal("duplicate-key MERGE emitted no node intent"))?;
    let ordered_sets = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetNodeProperty { node, value, .. } => Some((*node, value.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        ordered_sets,
        vec![
            (inserted_node, ScalarValue::String(Arc::from("first"))),
            (inserted_node, ScalarValue::String(Arc::from("second"))),
        ],
        "duplicate-key updates lost deterministic input-row order or read-own-writes identity"
    );
    assert_published_people(&fixture, &output, &[("same", "second")])?;
    Ok(())
}

#[test]
fn cpu_oracle_null_merge_key_aborts_the_whole_statement_without_publication() -> Result<()> {
    let fixture = Fixture::empty();
    let error = execute_cpu(
        &fixture,
        &[
            (Some("would-have-been-created"), "first"),
            (None, "invalid"),
        ],
    )
    .expect_err("MERGE accepted a null key after staging an earlier input row");

    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(
        error.message.contains("MergeReadOwnWrites"),
        "null MERGE key lost its exact runtime category: {error:?}"
    );
    assert_eq!(fixture.graph.node_count(), 0);
    assert_eq!(fixture.graph.edge_count(), 0);
    assert_eq!(fixture.graph.revision(), 0);
    assert!(fixture.graph.catalog().label("Person").is_none());
    assert!(fixture.graph.catalog().property("login").is_none());
    assert!(fixture.graph.catalog().property("name").is_none());
    Ok(())
}

#[test]
fn cpu_oracle_empty_login_and_name_are_values_not_null_or_missing() -> Result<()> {
    let fixture = Fixture::empty();
    let output = execute_cpu(&fixture, &[(Some(""), "")])?;

    assert_output(
        &fixture,
        &output,
        &[("", "")],
        StatementStats {
            nodes_created: 1,
            properties_set: 1,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 1));
    assert_published_people(&fixture, &output, &[("", "")])?;
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_command_calls: AtomicUsize,
    legacy_or_fallback_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// Strict red acceptance boundary for the row-driven native MERGE implementation.
///
/// The unpinned wrapper advertises Metal so `require_native_execution` cannot enter the generic
/// CPU executor. Every currently exposed primitive or mutation-prefix route is poisoned. A valid
/// implementation must dispatch the dedicated complete row-mutation command on this pinned
/// wrapper. That one command must own UNWIND, map-field extraction, MERGE read-own-writes, SET, two
/// string result columns, and all mutation/result receipts before publication.
struct StrictUnwindMergeBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictUnwindMergeBackend {
    fn cpu_reference(inner: CpuBackend) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("strict UNWIND/MERGE backend has no bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("strict UNWIND/MERGE backend has no graph revision"))?;
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

    fn reject_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .legacy_or_fallback_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict UNWIND/MERGE test rejected legacy or partial `{route}` route"),
        ))
    }
}

impl ExecutionBackend for StrictUnwindMergeBackend {
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
            return self.reject_route("pin_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != BackendKind::Cpu
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "strict UNWIND/MERGE pin changed backend provenance or graph fence",
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
        self.reject_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_route("execute_node_pipeline")
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        if !self.pinned {
            return self.reject_route("execute_row_mutation_on_unpinned_generation");
        }
        request.validate()?;
        if request.generation.project != PROJECT
            || request.generation.bookmark != self.expected_bookmark
            || request.generation.graph_revision != self.expected_graph_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "native row-mutation request escaped its pinned immutable generation",
            ));
        }
        self.observations
            .complete_command_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_row_mutation(request, cancellation)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_route("exact_l2")
    }
}

fn execute_strict(
    fixture: &Fixture,
    rows: &[(Option<&str>, &str)],
) -> Result<(
    std::result::Result<ExecutionOutput, Error>,
    Arc<RouteObservations>,
)> {
    let backend = StrictUnwindMergeBackend::cpu_reference(fixture.cpu_backend()?)?;
    let observations = backend.observations();
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "strict UNWIND/MERGE backend did not advertise accelerator execution",
        ));
    }
    let outcome = QueryEngine.execute(
        UNWIND_MERGE_QUERY,
        &mut context(fixture, Some(&backend), parameters(rows), true),
    );
    Ok((outcome, observations))
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn execute_real_metal(fixture: &Fixture, rows: &[(Option<&str>, &str)]) -> Result<ExecutionOutput> {
    let mut backend = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    backend.admit_project(fixture.resident_image()?)?;
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "real row-mutation regression did not select Metal",
        ));
    }
    QueryEngine.execute(
        UNWIND_MERGE_QUERY,
        &mut context(fixture, Some(&backend), parameters(rows), true),
    )
}

fn assert_one_complete_native_command(
    observations: &RouteObservations,
) -> Result<ResidentRowMutationRequest> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let complete = observations.complete_command_calls.load(Ordering::SeqCst);
    let forbidden = observations.legacy_or_fallback_calls.load(Ordering::SeqCst);
    if pins != 1 || complete != 1 || forbidden != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "strict UNWIND/MERGE route was not one complete native command: pins={pins}, complete={complete}, forbidden={forbidden}"
            ),
        ));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "strict UNWIND/MERGE captured {} complete requests instead of one",
                requests.len()
            ),
        ));
    }
    Ok(requests[0].clone())
}

fn assert_complete_native_request(
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    if request.generation.project != PROJECT
        || request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
        || request.generation.layout_version != fixture.graph.layout_version()
        || request.first_node_id != FIRST_CREATED_NODE_ID
        || request.write_layer != Layer::Observed
        || request.maximum_output_rows != 2
        || request.input.len() != 2
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("row-driven MERGE request has the wrong generation or capacity: {request:?}"),
        ));
    }

    let expected_input = [
        BTreeMap::from([
            ("login".to_owned(), ScalarValue::String(Arc::from("login1"))),
            ("name".to_owned(), ScalarValue::String(Arc::from("name1"))),
        ]),
        BTreeMap::from([
            ("login".to_owned(), ScalarValue::String(Arc::from("login2"))),
            ("name".to_owned(), ScalarValue::String(Arc::from("name2"))),
        ]),
    ];
    for (actual, expected) in request.input.iter().zip(expected_input) {
        if actual.entries.iter().cloned().collect::<BTreeMap<_, _>>() != expected {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("native UNWIND source map was host-rewritten or lost fields: {actual:?}"),
            ));
        }
    }

    fn resolve<'a>(values: &'a [String], index: u16, kind: &str) -> Result<&'a str> {
        values
            .get(usize::from(index))
            .map(String::as_str)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    format!("native row-mutation {kind} index {index} is out of range"),
                )
            })
    }

    let program = &request.program;
    let merge_labels = program
        .merge
        .labels
        .iter()
        .map(|index| resolve(&program.label_names, *index, "label"))
        .collect::<Result<Vec<_>>>()?;
    let merge_keys = program
        .merge
        .keys
        .iter()
        .map(|key| {
            Ok((
                resolve(&program.property_names, key.property_name, "property")?,
                resolve(&program.input_keys, key.input_key, "input key")?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    if merge_labels != ["Person"]
        || merge_keys != [("login", "login")]
        || program.sets.len() != 1
        || resolve(
            &program.property_names,
            program.sets[0].property_name,
            "SET property",
        )? != "name"
        || resolve(
            &program.input_keys,
            program.sets[0].input_key,
            "SET input key",
        )? != "name"
        || program.sets[0].target_entity != program.merge.output_entity
        || program.outputs.len() != 2
        || program.outputs[0].name != "p.name"
        || program.outputs[1].name != "p.login"
        || program
            .outputs
            .iter()
            .any(|output| output.entity != program.merge.output_entity)
        || resolve(
            &program.property_names,
            program.outputs[0].property_name,
            "first output property",
        )? != "name"
        || resolve(
            &program.property_names,
            program.outputs[1].property_name,
            "second output property",
        )? != "login"
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("native command does not own exact Unwind1 [14]: {program:?}"),
        ));
    }
    if program.label_tokens.iter().any(Option::is_some)
        || program.property_tokens.iter().any(Option::is_some)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "empty-graph Unwind1 [14] unexpectedly used pre-existing schema tokens",
        ));
    }
    Ok(())
}

#[test]
fn strict_accelerator_admission_requires_one_complete_native_unwind_merge_command() -> Result<()> {
    let fixture = Fixture::empty();

    // This is intentionally a red acceptance test until the public row-driven mutation ABI and
    // its CPU reference are wired through the strict wrapper above. Success cannot come from the
    // generic executor: the unpinned backend advertises Metal and native execution is mandatory.
    let (outcome, observations) = execute_strict(
        &fixture,
        &[(Some("login1"), "name1"), (Some("login2"), "name2")],
    )?;
    let output = outcome.map_err(|error| {
            Error::new(
                error.code,
                format!(
                    "{UNWIND1_FEATURE} / {UNWIND1_SCENARIO} lacks one complete native row-mutation owner: {error}"
                ),
            )
        })?;
    let request = assert_one_complete_native_command(&observations)?;
    assert_complete_native_request(&fixture, &request)?;
    assert_output(
        &fixture,
        &output,
        &[("name1", "login1"), ("name2", "login2")],
        StatementStats {
            nodes_created: 2,
            properties_set: 2,
            labels_added: 2,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (2, 2));
    assert_published_people(
        &fixture,
        &output,
        &[("login1", "name1"), ("login2", "name2")],
    )?;
    assert_eq!(fixture.graph.node_count(), 0);
    Ok(())
}

#[test]
fn strict_native_merge_uses_the_resident_graph_for_existing_and_new_rows() -> Result<()> {
    let fixture = Fixture::with_existing_person("kept", "old")?;
    let (outcome, observations) = execute_strict(
        &fixture,
        &[(Some("kept"), "updated"), (Some("new"), "fresh")],
    )?;
    let output = outcome?;
    assert_one_complete_native_command(&observations)?.validate()?;
    assert_output(
        &fixture,
        &output,
        &[("updated", "kept"), ("fresh", "new")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 2));
    assert_published_people(&fixture, &output, &[("kept", "updated"), ("new", "fresh")])?;
    Ok(())
}

#[test]
fn strict_native_duplicate_keys_share_one_overlay_identity_and_ordered_writes() -> Result<()> {
    let fixture = Fixture::empty();
    let (outcome, observations) = execute_strict(
        &fixture,
        &[(Some("same"), "first"), (Some("same"), "second")],
    )?;
    let output = outcome?;
    assert_one_complete_native_command(&observations)?.validate()?;
    assert_output(
        &fixture,
        &output,
        &[("second", "same"), ("second", "same")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 2));
    let ordered_values = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetNodeProperty { value, .. } => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        ordered_values,
        vec![
            ScalarValue::String(Arc::from("first")),
            ScalarValue::String(Arc::from("second")),
        ]
    );
    assert_published_people(&fixture, &output, &[("same", "second")])?;
    Ok(())
}

#[test]
fn strict_native_null_key_fails_atomically_after_an_earlier_valid_row() -> Result<()> {
    let fixture = Fixture::empty();
    let (outcome, observations) = execute_strict(
        &fixture,
        &[
            (Some("would-have-been-created"), "first"),
            (None, "invalid"),
        ],
    )?;
    let error = outcome.expect_err("native row MERGE accepted a null key");
    assert_one_complete_native_command(&observations)?.validate()?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(
        error.message.contains("MergeReadOwnWrites"),
        "native null-key error diverged from the CPU semantic category: {error:?}"
    );
    assert_eq!(fixture.graph.node_count(), 0);
    assert_eq!(fixture.graph.revision(), 0);
    assert!(fixture.graph.catalog().label("Person").is_none());
    assert!(fixture.graph.catalog().property("login").is_none());
    assert!(fixture.graph.catalog().property("name").is_none());
    Ok(())
}

#[test]
fn strict_native_empty_strings_survive_merge_set_and_typed_projection() -> Result<()> {
    let fixture = Fixture::empty();
    let (outcome, observations) = execute_strict(&fixture, &[(Some(""), "")])?;
    let output = outcome?;
    assert_one_complete_native_command(&observations)?.validate()?;
    assert_output(
        &fixture,
        &output,
        &[("", "")],
        StatementStats {
            nodes_created: 1,
            properties_set: 1,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;
    assert_only_node_merge_mutations(&output)?;
    assert_eq!(mutation_counts(&output), (1, 1));
    assert_published_people(&fixture, &output, &[("", "")])?;
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a real Metal device and serialized hardware execution"]
fn real_metal_row_mutation_matches_cpu_for_new_existing_duplicate_and_empty_values() -> Result<()> {
    let empty = Fixture::empty();
    let output = execute_real_metal(
        &empty,
        &[(Some("login1"), "name1"), (Some("login2"), "name2")],
    )?;
    assert_output(
        &empty,
        &output,
        &[("name1", "login1"), ("name2", "login2")],
        StatementStats {
            nodes_created: 2,
            properties_set: 2,
            labels_added: 2,
            ..StatementStats::default()
        },
    )?;

    let duplicate =
        execute_real_metal(&empty, &[(Some("same"), "first"), (Some("same"), "second")])?;
    assert_output(
        &empty,
        &duplicate,
        &[("second", "same"), ("second", "same")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;

    let existing = Fixture::with_existing_person("kept", "old")?;
    let mixed = execute_real_metal(
        &existing,
        &[(Some("kept"), "updated"), (Some("new"), "fresh")],
    )?;
    assert_output(
        &existing,
        &mixed,
        &[("updated", "kept"), ("fresh", "new")],
        StatementStats {
            nodes_created: 1,
            properties_set: 2,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;

    let zero_rows = execute_real_metal(&existing, &[])?;
    assert_output(&existing, &zero_rows, &[], StatementStats::default())?;

    let empty_strings = execute_real_metal(&empty, &[(Some(""), "")])?;
    assert_output(
        &empty,
        &empty_strings,
        &[("", "")],
        StatementStats {
            nodes_created: 1,
            properties_set: 1,
            labels_added: 1,
            ..StatementStats::default()
        },
    )?;

    let existing_empty_fixture = Fixture::with_existing_person("", "old")?;
    let existing_empty = execute_real_metal(&existing_empty_fixture, &[(Some(""), "")])?;
    assert_output(
        &existing_empty_fixture,
        &existing_empty,
        &[("", "")],
        StatementStats {
            properties_set: 1,
            ..StatementStats::default()
        },
    )
}
