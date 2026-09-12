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
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES, ResidentDirection, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableNodeDomain,
        ResidentNullableRelationBindingKind, ResidentNullableRelationFilterPlacement,
        ResidentNullableRelationOutputSource, ResidentNullableRelationPredicate,
        ResidentNullableRelationPredicateValue, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentNullableRelationStage,
        ResidentNullableRelationTarget, ResidentNullableRelationshipDomain, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4c49_5354_3132_5f4c_4142_454c_5f49_4e36,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const QUERY: &str = "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b";

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    nullable_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNullableRelationRequest>>,
}

struct StrictLabelMembershipBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictLabelMembershipBackend {
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
            format!("strict List12 [6] backend rejected route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictLabelMembershipBackend {
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
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_nullable_relation(request, cancellation)
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
        term: 61,
        index: graph.revision(),
    }
}

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let c = graph.catalog_mut().intern_label("C")?;
    let relationship = graph.catalog_mut().intern_relationship_type("T")?;
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, labels, properties) in [
        (1, vec![a], vec![(name, ScalarValue::String("c".into()))]),
        (2, vec![b], Vec::new()),
        (3, vec![c], Vec::new()),
        (4, vec![a], Vec::new()),
        (5, Vec::new(), Vec::new()),
        (
            6,
            vec![a],
            vec![(name, ScalarValue::String("anything".into()))],
        ),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties,
        })?;
    }
    for (id, source, target) in [(1, 1, 2), (2, 1, 3), (3, 4, 3), (4, 6, 5)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: relationship,
            layer: Layer::Observed,
            revision: 10 + id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn result_node_ids(output: &ExecutionOutput) -> Result<Vec<NodeId>> {
    if output.result.schema != [("b".to_owned(), ColumnType::Node)]
        || output.result.statistics != StatementStats::default()
        || !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
    {
        return Err(Error::internal(
            "List12 [6] produced an unexpected result contract",
        ));
    }
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns[0].values)
        .map(|value| match value {
            ResultValue::Node(node) => Ok(node.id),
            other => Err(Error::internal(format!(
                "List12 [6] returned non-node value {other:?}"
            ))),
        })
        .collect()
}

fn collect_membership_leaves<'a>(
    predicate: &'a ResidentNullableRelationPredicate,
    leaves: &mut Vec<&'a ResidentNullableRelationPredicate>,
) -> bool {
    match predicate {
        ResidentNullableRelationPredicate::Or(left, right) => {
            collect_membership_leaves(left, leaves) && collect_membership_leaves(right, leaves)
        }
        ResidentNullableRelationPredicate::And(_, _) => {
            leaves.push(predicate);
            true
        }
        _ => false,
    }
}

#[test]
fn list12_6_executes_once_as_the_balanced_catalog_label_predicate() -> Result<()> {
    let graph = fixture()?;
    let backend = StrictLabelMembershipBackend::new(&graph)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(QUERY, &mut context(&graph, &backend))?;

    // The missing source property reaches a labelled target and must remain unknown; the
    // present source property reaching an unlabelled target must be false. Only (:C) survives.
    assert_eq!(result_node_ids(&output)?, [NodeId(3)]);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(
            "List12 [6] changed nullable request cardinality",
        ));
    };
    request.validate()?;
    let [
        ResidentNullableRelationStage::NodeScan {
            output: source_slot,
            labels: ResidentNullableNodeDomain::Any,
            ..
        },
        ResidentNullableRelationStage::Expand {
            source,
            relationship: None,
            different_from,
            target: ResidentNullableRelationTarget::Introduce(target_slot),
            direction: ResidentDirection::Outgoing,
            relationship_types: ResidentNullableRelationshipDomain::Any,
            target_labels: ResidentNullableNodeDomain::Any,
            ..
        },
        ResidentNullableRelationStage::FinalProject { bindings },
    ] = request.program.stages.as_slice()
    else {
        return Err(Error::internal(
            "List12 [6] changed its sealed relation shape",
        ));
    };
    assert_eq!(source, source_slot);
    assert!(different_from.is_empty());
    assert!(matches!(
        bindings.as_slice(),
        [binding]
            if binding.name == "b"
                && binding.source
                    == (ResidentNullableRelationOutputSource::Entity {
                        slot: *target_slot,
                        kind: ResidentNullableRelationBindingKind::Node,
                    })
    ));

    let [filter] = request.predicate_program.filters.as_slice() else {
        return Err(Error::internal(
            "List12 [6] did not emit exactly one native filter",
        ));
    };
    assert_eq!(
        filter.placement,
        ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 }
    );
    let mut leaves = Vec::new();
    assert!(collect_membership_leaves(&filter.predicate, &mut leaves));
    assert_eq!(leaves.len(), graph.catalog().labels().count());

    let name = graph
        .catalog()
        .property("name")
        .ok_or_else(|| Error::internal("List12 [6] fixture omitted name property"))?;
    let mut actual = BTreeSet::new();
    for leaf in leaves {
        let ResidentNullableRelationPredicate::And(label, comparison) = leaf else {
            unreachable!("leaf collector admits only conjunctions")
        };
        let ResidentNullableRelationPredicate::HasLabels {
            node,
            labels: ResidentNullableNodeDomain::Known(labels),
        } = label.as_ref()
        else {
            return Err(Error::internal("List12 [6] leaf omitted HasLabels"));
        };
        let [label] = labels.as_slice() else {
            return Err(Error::internal(
                "List12 [6] leaf did not name exactly one label",
            ));
        };
        assert_eq!(node, target_slot);
        let ResidentNullableRelationPredicate::CompareString {
            left:
                ResidentNullableRelationPredicateValue::StringProperty {
                    slot,
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                },
            operation: CompareOp::Eq,
            right: ResidentNullableRelationPredicateValue::String(value),
        } = comparison.as_ref()
        else {
            return Err(Error::internal(
                "List12 [6] leaf omitted its string equality",
            ));
        };
        assert_eq!(slot, source_slot);
        assert_eq!(*property, name);
        actual.insert((*label, value.to_string()));
    }
    let expected = graph
        .catalog()
        .labels()
        .map(|(label, name)| (label, name.to_lowercase()))
        .collect::<BTreeSet<_>>();
    assert_eq!(actual, expected);
    Ok(())
}

fn assert_no_native_dispatch(graph: &GraphStore, query: &str) -> Result<()> {
    let backend = StrictLabelMembershipBackend::new(graph)?;
    let observations = backend.observations();
    let error = QueryEngine
        .execute(query, &mut context(graph, &backend))
        .expect_err("List12 [6] near miss unexpectedly executed");
    assert_eq!(
        error.code,
        ErrorCode::GpuAdmissionFailure,
        "{query}: {error:?}"
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), 0, "{query}");
    assert_eq!(
        observations.nullable_calls.load(Ordering::SeqCst),
        0,
        "{query}"
    );
    assert_eq!(
        observations.forbidden_calls.load(Ordering::SeqCst),
        0,
        "{query}"
    );
    Ok(())
}

#[test]
fn list12_6_variable_function_projection_direction_and_property_near_misses_fail_closed()
-> Result<()> {
    let graph = fixture()?;
    for query in [
        "MATCH (n)-->(b) WHERE n.other IN [x IN labels(b) | toLower(x)] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(n) | toLower(x)] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | upper(x)] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | x] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b)] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) WHERE true | toLower(x)] RETURN b",
        "MATCH (n)<--(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b",
        "MATCH (n)--(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b",
        "MATCH (n)-[:T]->(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN n",
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b AS b",
    ] {
        assert_no_native_dispatch(&graph, query)?;
    }
    Ok(())
}

#[test]
fn list12_6_catalog_instruction_and_string_images_are_bounded_before_dispatch() -> Result<()> {
    let mut too_many_labels = GraphStore::default();
    too_many_labels.catalog_mut().intern_property("name")?;
    // One label leaf needs five instructions and the balanced OR needs one more per join. Forty-
    // three labels therefore need 257 instructions and exceed the compact 256-instruction proof.
    for index in 0..43 {
        too_many_labels
            .catalog_mut()
            .intern_label(&format!("L{index}"))?;
    }
    assert_no_native_dispatch(&too_many_labels, QUERY)?;

    let mut oversized_literal = GraphStore::default();
    oversized_literal.catalog_mut().intern_property("name")?;
    oversized_literal
        .catalog_mut()
        .intern_label(&"X".repeat(RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES + 1))?;
    assert_no_native_dispatch(&oversized_literal, QUERY)
}
