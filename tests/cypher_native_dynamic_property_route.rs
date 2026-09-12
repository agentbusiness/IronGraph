// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentGroup, ResidentGroupRequest,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableRelationBindingKind,
        ResidentNullableRelationOutputSource, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentNullableRelationStage, ResidentProjectImage,
        ResidentRowMutationRequest, ResidentRowMutationResult, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentSortRequest, ResidentSortResult,
        ResidentVariablePathRequest, ResidentVariablePathResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4752_4150_4837_5f44_594e_414d_4943_0001,
));
const BOOKMARK: Bookmark = Bookmark { term: 37, index: 7 };
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    nullable_calls: AtomicUsize,
    create_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    nullable_requests: Mutex<Vec<ResidentNullableRelationRequest>>,
    create_requests: Mutex<Vec<ResidentCreateNodeRequest>>,
}

struct StrictDynamicPropertyBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictDynamicPropertyBackend {
    fn new(graph: &GraphStore) -> Result<Self> {
        let image = ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(image)?;
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
            format!("strict Graph7 backend rejected unsealed route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictDynamicPropertyBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.inner.kind()
        } else {
            // The unpinned observer represents the active accelerator class so strict-native
            // execution rejects an unadmitted complete plan before the generic row executor can
            // issue even a scan. Once pinned, publication validates the real CPU reference
            // receipt against its actual completion kind.
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
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

    fn execute_row_mutation(
        &self,
        _request: &ResidentRowMutationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        self.reject("execute_row_mutation")
    }

    fn supports_native_create_node(&self) -> bool {
        true
    }

    fn execute_create_node(
        &self,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        if !self.pinned {
            return self.reject("execute_create_node_unpinned");
        }
        request.validate()?;
        self.observations
            .create_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .create_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_create_node(request, cancellation)
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject("execute_row_program")
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        true
    }

    fn supports_nullable_relation_string_property_equality(&self) -> bool {
        true
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        if !self.pinned {
            return self.reject("execute_nullable_relation_unpinned");
        }
        request.validate()?;
        self.observations
            .nullable_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .nullable_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_nullable_relation(request, cancellation)
    }

    fn execute_variable_path(
        &self,
        _request: &ResidentVariablePathRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        self.reject("execute_variable_path")
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

fn read_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let name = graph.catalog_mut().intern_property("name")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(name, ScalarValue::String("Apa".into()))],
    })?;
    Ok(graph)
}

fn graph6_node_expression_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let existing = graph.catalog_mut().intern_property("existing")?;
    graph.catalog_mut().intern_property("missing")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(existing, ScalarValue::Integer(42))],
    })?;
    Ok(graph)
}

fn graph6_relationship_expression_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
    let existing = graph.catalog_mut().intern_property("existing")?;
    graph.catalog_mut().intern_property("missing")?;
    for id in [NodeId(1), NodeId(2)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 1,
        properties: vec![(existing, ScalarValue::Integer(42))],
    })?;
    Ok(graph)
}

fn generic_read_context(graph: &GraphStore) -> ExecutionContext<'_> {
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
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 32,
        max_batch_rows: 32,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    write: bool,
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
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write,
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 32,
        max_batch_rows: 32,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn assert_single_apa(output: &ExecutionOutput) -> Result<()> {
    assert_eq!(
        output.result.schema,
        [("value".to_owned(), ColumnType::String)]
    );
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal("Graph7 result changed batch count"));
    };
    assert_eq!(batch.row_count, 1);
    assert!(matches!(
        batch.columns.as_slice(),
        [column]
            if column.name == "value"
                && column.value_type == ColumnType::String
                && column.values
                    == [ResultValue::Scalar(ScalarValue::String("Apa".into()))]
    ));
    assert!(!output.result.truncated);
    Ok(())
}

#[test]
fn graph7_read_uses_one_sealed_nullable_string_property_command() -> Result<()> {
    let graph = read_fixture()?;
    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| Error::internal("Graph7 fixture omitted `name`"))?;
    let backend = StrictDynamicPropertyBackend::new(&graph)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        "MATCH (n {name: 'Apa'}) RETURN n['nam' + 'e'] AS value",
        &mut context(&graph, &backend, false, BTreeMap::new()),
    )?;
    assert_single_apa(&output)?;
    assert_eq!(output.result.statistics, StatementStats::default());
    assert!(output.graph_mutations.is_empty());
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.create_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

    let requests = observations
        .nullable_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal("Graph7 read changed native request count"));
    };
    request.validate()?;
    let [
        ResidentNullableRelationStage::NodeScan { .. },
        ResidentNullableRelationStage::FinalProject { bindings },
    ] = request.program.stages.as_slice()
    else {
        return Err(Error::internal("Graph7 read changed sealed stage shape"));
    };
    assert!(matches!(
        bindings.as_slice(),
        [binding]
            if binding.name == "value"
                && matches!(
                    binding.source,
                    ResidentNullableRelationOutputSource::StringProperty {
                        kind: ResidentNullableRelationBindingKind::Node,
                        property,
                        ..
                    } if property == name
                )
    ));
    Ok(())
}

#[test]
fn graph6_entity_list_expression_properties_use_one_sealed_nullable_command() -> Result<()> {
    let cases = [
        (
            graph6_node_expression_fixture()?,
            "MATCH (n) WITH [123, n] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing",
            ResidentNullableRelationBindingKind::Node,
            1563_usize,
        ),
        (
            graph6_relationship_expression_fixture()?,
            "MATCH ()-[r]->() WITH [123, r] AS list RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing",
            ResidentNullableRelationBindingKind::Relationship,
            1567_usize,
        ),
    ];

    for (graph, query, expected_kind, report_index) in cases {
        let reference = QueryEngine.execute(query, &mut generic_read_context(&graph))?;
        let existing = graph
            .catalog()
            .property("existing")
            .ok_or_else(|| Error::internal("Graph6 fixture omitted `existing`"))?;
        let backend = StrictDynamicPropertyBackend::new(&graph)?;
        let observations = backend.observations();
        let native = QueryEngine.execute(
            query,
            &mut context(&graph, &backend, false, BTreeMap::new()),
        )?;

        assert_eq!(native.result, reference.result, "report {report_index}");
        assert_eq!(native.result.statistics, StatementStats::default());
        assert!(native.graph_mutations.is_empty());
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.create_calls.load(Ordering::SeqCst), 0);
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

        let requests = observations
            .nullable_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let [request] = requests.as_slice() else {
            return Err(Error::internal(format!(
                "Graph6 report {report_index} changed native request count"
            )));
        };
        request.validate()?;
        assert!(
            !request
                .program
                .stages
                .iter()
                .any(|stage| matches!(stage, ResidentNullableRelationStage::ScopeProject { .. })),
            "report {report_index} retained the compiler-erased list wrapper"
        );
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            request.program.stages.last()
        else {
            return Err(Error::internal(format!(
                "Graph6 report {report_index} omitted its final projection"
            )));
        };
        assert!(matches!(
            bindings.as_slice(),
            [missing, missing_too, present]
                if missing.name == "(list[1]).missing"
                    && missing_too.name == "(list[1]).missingToo"
                    && present.name == "(list[1]).existing"
                    && matches!(
                        missing.source,
                        ResidentNullableRelationOutputSource::NullProperty { kind, .. }
                            if kind == expected_kind
                    )
                    && matches!(
                        missing_too.source,
                        ResidentNullableRelationOutputSource::NullProperty { kind, .. }
                            if kind == expected_kind
                    )
                    && matches!(
                        present.source,
                        ResidentNullableRelationOutputSource::IntegerProperty {
                            kind,
                            property,
                            ..
                        } if kind == expected_kind && property == existing
                    )
        ));
    }

    let graph = graph6_node_expression_fixture()?;
    for query in [
        "MATCH (n) WITH [123, n] AS list RETURN (list[1]).missing, list AS retained",
        "MATCH (n) WITH [124, n] AS list RETURN (list[1]).missing",
    ] {
        let backend = StrictDynamicPropertyBackend::new(&graph)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, &backend, false, BTreeMap::new()),
            )
            .expect_err("a Graph6 list-wrapper near miss unexpectedly entered native dispatch");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 0, "{query}");
        assert_eq!(
            observations.nullable_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert_eq!(
            observations.create_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert_eq!(
            observations.forbidden_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
    }
    Ok(())
}

fn assert_graph7_create(query: &str, parameters: BTreeMap<String, ResultValue>) -> Result<()> {
    let graph = GraphStore::default();
    let backend = StrictDynamicPropertyBackend::new(&graph)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(query, &mut context(&graph, &backend, true, parameters))?;
    assert_single_apa(&output)?;
    assert_eq!(
        output.result.statistics,
        StatementStats {
            nodes_created: 1,
            ..StatementStats::default()
        }
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.create_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

    let requests = observations
        .create_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(
            "Graph7 CREATE changed native request count",
        ));
    };
    request.validate()?;
    assert_eq!(request.property_names, ["name".to_owned()]);

    let mut declaration = None;
    let mut inserted = None;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::DeclareProperty { name, id } if name == "name" => {
                declaration = Some(*id);
            }
            GraphMutation::InsertNode(node) => inserted = Some(node),
            other => {
                return Err(Error::internal(format!(
                    "Graph7 CREATE emitted unrelated mutation {other:?}"
                )));
            }
        }
    }
    let property =
        declaration.ok_or_else(|| Error::internal("Graph7 omitted property declaration"))?;
    let node = inserted.ok_or_else(|| Error::internal("Graph7 omitted node insertion"))?;
    assert_eq!(node.id, NodeId(100));
    assert_eq!(
        node.properties,
        vec![(property, ScalarValue::String("Apa".into()))]
    );
    Ok(())
}

#[test]
fn graph7_create_literal_and_parameter_keys_use_the_existing_overlay_property() -> Result<()> {
    assert_graph7_create(
        "CREATE (n {name: 'Apa'}) RETURN n['nam' + 'e'] AS value",
        BTreeMap::new(),
    )?;
    assert_graph7_create(
        "CREATE (n {name: 'Apa'}) RETURN n[$idx] AS value",
        BTreeMap::from([(
            "idx".to_owned(),
            ResultValue::Scalar(ScalarValue::String("name".into())),
        )]),
    )?;
    assert_graph7_create(
        "CREATE (n {name: 'Apa'}) RETURN n['na' + ('m' + 'e')] AS value",
        BTreeMap::new(),
    )
}

#[test]
fn graph7_dynamic_non_string_and_wider_shapes_dispatch_nothing() -> Result<()> {
    let string_parameter = BTreeMap::from([(
        "idx".to_owned(),
        ResultValue::Scalar(ScalarValue::String("name".into())),
    )]);
    let null_parameter =
        BTreeMap::from([("idx".to_owned(), ResultValue::Scalar(ScalarValue::Null))]);
    let integer_parameter = BTreeMap::from([(
        "idx".to_owned(),
        ResultValue::Scalar(ScalarValue::Integer(1)),
    )]);
    for (query, write, parameters) in [
        (
            "MATCH (n {name: 'Apa'}) RETURN n[null] AS value",
            false,
            BTreeMap::new(),
        ),
        (
            "MATCH (n {name: 'Apa'}) RETURN n[1] AS value",
            false,
            BTreeMap::new(),
        ),
        (
            "MATCH (n {name: 'Apa'}) RETURN n[$idx] AS value",
            false,
            null_parameter,
        ),
        (
            "MATCH (n {name: 'Apa'}) RETURN n[$idx] AS value",
            false,
            integer_parameter.clone(),
        ),
        (
            "MATCH (n {name: 'Apa'}) WITH n AS m RETURN m[$idx] AS value",
            false,
            string_parameter.clone(),
        ),
        (
            "MATCH (n {name: 'Apa'}) RETURN n[$idx] AS value, n.name AS direct",
            false,
            string_parameter.clone(),
        ),
        (
            "CREATE (n {name: 'Apa'}) RETURN n[null] AS value",
            true,
            BTreeMap::new(),
        ),
        (
            "CREATE (n {name: 'Apa'}) RETURN n[$idx] AS value",
            true,
            integer_parameter,
        ),
        (
            "CREATE (n {name: 'Apa'}) RETURN n[$idx] AS value, n.name AS direct",
            true,
            string_parameter,
        ),
    ] {
        let graph = if write {
            GraphStore::default()
        } else {
            read_fixture()?
        };
        let backend = StrictDynamicPropertyBackend::new(&graph)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(&graph, &backend, write, parameters))
            .expect_err("unsupported Graph7 neighbor must fail closed");
        assert!(
            matches!(
                error.code,
                ErrorCode::GpuAdmissionFailure | ErrorCode::QueryType
            ),
            "unexpected fail-closed error for {query}: {error:?}"
        );
        assert_eq!(observations.pins.load(Ordering::SeqCst), 0, "{query}");
        assert_eq!(
            observations.nullable_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert_eq!(
            observations.create_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert_eq!(
            observations.forbidden_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
    }
    Ok(())
}
