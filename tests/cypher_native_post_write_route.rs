// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    fs,
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
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;
const REPORT_PATH: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const REMOVE_FEATURE: &str = "features/clauses/remove/Remove3.feature";
const SET_FEATURE: &str = "features/clauses/set/Set6.feature";

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    pipeline_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNodePipelineRequest>>,
}

/// Test-only guard around the only acceptable native whole-statement boundary.
///
/// The root CPU reference deliberately advertises Metal so a missing resident implementation
/// cannot fall through to the generic executor. The object returned by `pin_project` reports the
/// honest backend kind so receipt validation still distinguishes CPU-reference completion from
/// real Metal completion. Every semantic entrypoint except one mutation-bearing node pipeline is
/// closed. The pipeline itself is closed on the root object: dispatch must use the immutable
/// generation returned by `pin_project`.
struct ObservedWholeStatementBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl ObservedWholeStatementBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    fn wrong_receipt_provenance(inner: CpuBackend) -> Result<Self> {
        // The real implementation emits CpuReference receipts, but the pinned wrapper claims
        // Metal. Publication must reject those receipts instead of accepting the intent values.
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "post-write acceptance test did not construct a real Metal backend",
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
            Error::internal("post-write test backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("post-write test backend has no admitted graph revision")
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
            format!("strict post-write test rejected `{route}` execution"),
        ))
    }

    fn refresh_expected_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement post-write project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement post-write project has no graph revision")
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedWholeStatementBackend {
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
                "pinned post-write generation does not match the admitted immutable fence",
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
            || request.mutation.is_none()
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "post-write pipeline is not tied to the pinned mutation generation",
            ));
        }
        self.observations
            .pipeline_calls
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostWriteShape {
    LimitZero,
    SkipAll,
    PageTwo,
    PageAll,
    Filter,
    ReturnAggregate,
    WithAggregate,
}

impl PostWriteShape {
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

    const fn input_rows(self) -> usize {
        match self {
            Self::LimitZero | Self::SkipAll => 1,
            _ => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PostWriteGroup {
    RemoveNodeProperty,
    RemoveNodeLabel,
    RemoveRelationshipProperty,
    SetNodeProperty,
    SetNodeLabel,
    SetRelationshipProperty,
}

impl PostWriteGroup {
    const ALL: [Self; 6] = [
        Self::RemoveNodeProperty,
        Self::RemoveNodeLabel,
        Self::RemoveRelationshipProperty,
        Self::SetNodeProperty,
        Self::SetNodeLabel,
        Self::SetRelationshipProperty,
    ];

    const fn first_id(self) -> u16 {
        match self {
            Self::RemoveNodeProperty => 673,
            Self::RemoveNodeLabel => 680,
            Self::RemoveRelationshipProperty => 687,
            Self::SetNodeProperty => 855,
            Self::SetNodeLabel => 862,
            Self::SetRelationshipProperty => 869,
        }
    }

    const fn feature(self) -> &'static str {
        match self {
            Self::RemoveNodeProperty | Self::RemoveNodeLabel | Self::RemoveRelationshipProperty => {
                REMOVE_FEATURE
            }
            Self::SetNodeProperty | Self::SetNodeLabel | Self::SetRelationshipProperty => {
                SET_FEATURE
            }
        }
    }

    const fn is_relationship(self) -> bool {
        matches!(
            self,
            Self::RemoveRelationshipProperty | Self::SetRelationshipProperty
        )
    }

    const fn changes_numeric_property(self) -> bool {
        matches!(self, Self::SetNodeProperty | Self::SetRelationshipProperty)
    }

    const fn names(self) -> &'static [&'static str; 7] {
        match self {
            Self::RemoveNodeProperty => &REMOVE_NODE_PROPERTY_NAMES,
            Self::RemoveNodeLabel => &REMOVE_NODE_LABEL_NAMES,
            Self::RemoveRelationshipProperty => &REMOVE_RELATIONSHIP_PROPERTY_NAMES,
            Self::SetNodeProperty => &SET_NODE_PROPERTY_NAMES,
            Self::SetNodeLabel => &SET_NODE_LABEL_NAMES,
            Self::SetRelationshipProperty => &SET_RELATIONSHIP_PROPERTY_NAMES,
        }
    }

    const fn queries(self) -> &'static [&'static str; 7] {
        match self {
            Self::RemoveNodeProperty => &REMOVE_NODE_PROPERTY_QUERIES,
            Self::RemoveNodeLabel => &REMOVE_NODE_LABEL_QUERIES,
            Self::RemoveRelationshipProperty => &REMOVE_RELATIONSHIP_PROPERTY_QUERIES,
            Self::SetNodeProperty => &SET_NODE_PROPERTY_QUERIES,
            Self::SetNodeLabel => &SET_NODE_LABEL_QUERIES,
            Self::SetRelationshipProperty => &SET_RELATIONSHIP_PROPERTY_QUERIES,
        }
    }
}

const REMOVE_NODE_PROPERTY_NAMES: [&str; 7] = [
    "[1] Limiting to zero results after removing a property from nodes affects the result set but not the side effects",
    "[2] Skipping all results after removing a property from nodes affects the result set but not the side effects",
    "[3] Skipping and limiting to a few results after removing a property from nodes affects the result set but not the side effects",
    "[4] Skipping zero results and limiting to all results after removing a property from nodes does not affect the result set nor the side effects",
    "[5] Filtering after removing a property from nodes affects the result set but not the side effects",
    "[6] Aggregating in `RETURN` after removing a property from nodes affects the result set but not the side effects",
    "[7] Aggregating in `WITH` after removing a property from nodes affects the result set but not the side effects",
];

const REMOVE_NODE_LABEL_NAMES: [&str; 7] = [
    "[8] Limiting to zero results after removing a label from nodes affects the result set but not the side effects",
    "[9] Skipping all results after removing a label from nodes affects the result set but not the side effects",
    "[10] Skipping and limiting to a few results after removing a label from nodes affects the result set but not the side effects",
    "[11] Skipping zero result and limiting to all results after removing a label from nodes does not affect the result set nor the side effects",
    "[12] Filtering after removing a label from nodes affects the result set but not the side effects",
    "[13] Aggregating in `RETURN` after removing a label from nodes affects the result set but not the side effects",
    "[14] Aggregating in `WITH` after removing a label from nodes affects the result set but not the side effects",
];

const REMOVE_RELATIONSHIP_PROPERTY_NAMES: [&str; 7] = [
    "[15] Limiting to zero results after removing a property from relationships affects the result set but not the side effects",
    "[16] Skipping all results after removing a property from relationships affects the result set but not the side effects",
    "[17] Skipping and limiting to a few results after removing a property from relationships affects the result set but not the side effects",
    "[18] Skipping zero result and limiting to all results after removing a property from relationships does not affect the result set nor the side effects",
    "[19] Filtering after removing a property from relationships affects the result set but not the side effects",
    "[20] Aggregating in `RETURN` after removing a property from relationships affects the result set but not the side effects",
    "[21] Aggregating in `WITH` after removing a property from relationships affects the result set but not the side effects",
];

const SET_NODE_PROPERTY_NAMES: [&str; 7] = [
    "[1] Limiting to zero results after setting a property on nodes affects the result set but not the side effects",
    "[2] Skipping all results after setting a property on nodes affects the result set but not the side effects",
    "[3] Skipping and limiting to a few results after setting a property on nodes affects the result set but not the side effects",
    "[4] Skipping zero results and limiting to all results after setting a property on nodes does not affect the result set nor the side effects",
    "[5] Filtering after setting a property on nodes affects the result set but not the side effects",
    "[6] Aggregating in `RETURN` after setting a property on nodes affects the result set but not the side effects",
    "[7] Aggregating in `WITH` after setting a property on nodes affects the result set but not the side effects",
];

const SET_NODE_LABEL_NAMES: [&str; 7] = [
    "[8] Limiting to zero results after adding a label on nodes affects the result set but not the side effects",
    "[9] Skipping all results after adding a label on nodes affects the result set but not the side effects",
    "[10] Skipping and limiting to a few results after adding a label on nodes affects the result set but not the side effects",
    "[11] Skipping zero result and limiting to all results after adding a label on nodes does not affect the result set nor the side effects",
    "[12] Filtering after adding a label on nodes affects the result set but not the side effects",
    "[13] Aggregating in `RETURN` after adding a label on nodes affects the result set but not the side effects",
    "[14] Aggregating in `WITH` after adding a label on nodes affects the result set but not the side effects",
];

const SET_RELATIONSHIP_PROPERTY_NAMES: [&str; 7] = [
    "[15] Limiting to zero results after setting a property on relationships affects the result set but not the side effects",
    "[16] Skipping all results after setting a property on relationships affects the result set but not the side effects",
    "[17] Skipping and limiting to a few results after setting a property on relationships affects the result set but not the side effects",
    "[18] Skipping zero result and limiting to all results after setting a property on relationships does not affect the result set nor the side effects",
    "[19] Filtering after setting a property on relationships affects the result set but not the side effects",
    "[20] Aggregating in `RETURN` after setting a property on relationships affects the result set but not the side effects",
    "[21] Aggregating in `WITH` after setting a property on relationships affects the result set but not the side effects",
];

const REMOVE_NODE_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH (n:N) REMOVE n.num RETURN n LIMIT 0",
    "MATCH (n:N) REMOVE n.num RETURN n SKIP 1",
    "MATCH (n:N) REMOVE n.name RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) REMOVE n.name RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) REMOVE n.name WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) REMOVE n.name RETURN sum(n.num) AS sum",
    "MATCH (n:N) REMOVE n.name WITH sum(n.num) AS sum RETURN sum",
];

const REMOVE_NODE_LABEL_QUERIES: [&str; 7] = [
    "MATCH (n:N) REMOVE n:N RETURN n LIMIT 0",
    "MATCH (n:N) REMOVE n:N RETURN n SKIP 1",
    "MATCH (n:N) REMOVE n:N RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) REMOVE n:N RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) REMOVE n:N WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) REMOVE n:N RETURN sum(n.num) AS sum",
    "MATCH (n:N) REMOVE n:N WITH sum(n.num) AS sum RETURN sum",
];

const REMOVE_RELATIONSHIP_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH ()-[r:R]->() REMOVE r.num RETURN r LIMIT 0",
    "MATCH ()-[r:R]->() REMOVE r.num RETURN r SKIP 1",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN r.num AS num SKIP 2 LIMIT 2",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN r.num AS num SKIP 0 LIMIT 5",
    "MATCH ()-[r:R]->() REMOVE r.name WITH r WHERE r.num % 2 = 0 RETURN r.num AS num",
    "MATCH ()-[r:R]->() REMOVE r.name RETURN sum(r.num) AS sum",
    "MATCH ()-[r:R]->() REMOVE r.name WITH sum(r.num) AS sum RETURN sum",
];

const SET_NODE_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH (n:N) SET n.num = 43 RETURN n LIMIT 0",
    "MATCH (n:N) SET n.num = 43 RETURN n SKIP 1",
    "MATCH (n:N) SET n.num = 42 RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) SET n.num = 42 RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) SET n.num = n.num + 1 WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) SET n.num = n.num + 1 RETURN sum(n.num) AS sum",
    "MATCH (n:N) SET n.num = n.num + 1 WITH sum(n.num) AS sum RETURN sum",
];

const SET_NODE_LABEL_QUERIES: [&str; 7] = [
    "MATCH (n:N) SET n:Foo RETURN n LIMIT 0",
    "MATCH (n:N) SET n:Foo RETURN n SKIP 1",
    "MATCH (n:N) SET n:Foo RETURN n.num AS num SKIP 2 LIMIT 2",
    "MATCH (n:N) SET n:Foo RETURN n.num AS num SKIP 0 LIMIT 5",
    "MATCH (n:N) SET n:Foo WITH n WHERE n.num % 2 = 0 RETURN n.num AS num",
    "MATCH (n:N) SET n:Foo RETURN sum(n.num) AS sum",
    "MATCH (n:N) SET n:Foo WITH sum(n.num) AS sum RETURN sum",
];

const SET_RELATIONSHIP_PROPERTY_QUERIES: [&str; 7] = [
    "MATCH ()-[r:R]->() SET r.num = 43 RETURN r LIMIT 0",
    "MATCH ()-[r:R]->() SET r.num = 43 RETURN r SKIP 1",
    "MATCH ()-[r:R]->() SET r.num = 42 RETURN r.num AS num SKIP 2 LIMIT 2",
    "MATCH ()-[r:R]->() SET r.num = 42 RETURN r.num AS num SKIP 0 LIMIT 5",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 WITH r WHERE r.num % 2 = 0 RETURN r.num AS num",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 RETURN sum(r.num) AS sum",
    "MATCH ()-[r:R]->() SET r.num = r.num + 1 WITH sum(r.num) AS sum RETURN sum",
];

#[derive(Clone, Copy, Debug)]
struct TckCase {
    id: u16,
    feature: &'static str,
    name: &'static str,
    group: PostWriteGroup,
    shape: PostWriteShape,
    query: &'static str,
}

#[derive(Debug, Deserialize)]
struct CertifiedReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Debug, Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    cpu_metal_matched: bool,
    fully_conformant: bool,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
}

fn certified_report() -> Result<CertifiedReport> {
    let bytes = fs::read(REPORT_PATH).map_err(|error| {
        Error::internal(format!(
            "cannot read certified post-write report {REPORT_PATH}: {error}"
        ))
    })?;
    serde_json::from_slice(&bytes).map_err(|error| {
        Error::internal(format!(
            "cannot decode certified post-write report {REPORT_PATH}: {error}"
        ))
    })
}

impl TckCase {
    fn label(self) -> String {
        format!("TCK {} {}: {}", self.id, self.feature, self.name)
    }

    const fn expected_column(self) -> &'static str {
        match self.shape {
            PostWriteShape::LimitZero | PostWriteShape::SkipAll => {
                if self.group.is_relationship() {
                    "r"
                } else {
                    "n"
                }
            }
            PostWriteShape::PageTwo | PostWriteShape::PageAll | PostWriteShape::Filter => "num",
            PostWriteShape::ReturnAggregate | PostWriteShape::WithAggregate => "sum",
        }
    }

    fn expected_values(self) -> Vec<i64> {
        match self.shape {
            PostWriteShape::LimitZero | PostWriteShape::SkipAll => Vec::new(),
            PostWriteShape::PageTwo => vec![42, 42],
            PostWriteShape::PageAll => vec![42, 42, 42, 42, 42],
            PostWriteShape::Filter if self.group.changes_numeric_property() => vec![2, 4, 6],
            PostWriteShape::Filter => vec![2, 4],
            PostWriteShape::ReturnAggregate | PostWriteShape::WithAggregate
                if self.group.changes_numeric_property() =>
            {
                vec![20]
            }
            PostWriteShape::ReturnAggregate | PostWriteShape::WithAggregate => vec![15],
        }
    }

    fn expected_stats(self) -> StatementStats {
        let changed = self.shape.input_rows() as u64;
        match self.group {
            PostWriteGroup::RemoveNodeProperty
            | PostWriteGroup::RemoveRelationshipProperty
            | PostWriteGroup::SetNodeProperty
            | PostWriteGroup::SetRelationshipProperty => StatementStats {
                properties_set: changed,
                ..StatementStats::default()
            },
            PostWriteGroup::RemoveNodeLabel => StatementStats {
                labels_removed: changed,
                ..StatementStats::default()
            },
            PostWriteGroup::SetNodeLabel => StatementStats {
                labels_added: changed,
                ..StatementStats::default()
            },
        }
    }
}

fn all_cases() -> impl Iterator<Item = TckCase> {
    PostWriteGroup::ALL.into_iter().flat_map(|group| {
        PostWriteShape::ALL.into_iter().map(move |shape| {
            let index = shape.index();
            TckCase {
                id: group.first_id() + u16::try_from(index).expect("seven-case index fits u16"),
                feature: group.feature(),
                name: group.names()[index],
                group,
                shape,
                query: group.queries()[index],
            }
        })
    })
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    num: PropertyId,
    name: PropertyId,
    n_label: LabelId,
}

impl Fixture {
    fn new(case: TckCase) -> Result<Self> {
        let mut graph = GraphStore::default();
        let n_label = graph.catalog_mut().intern_label("N")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let rows = case.shape.input_rows();

        if case.group.is_relationship() {
            for offset in 0..rows {
                let source = NodeId(u64::try_from(offset * 2 + 1).expect("fixture ID fits u64"));
                let target = NodeId(u64::try_from(offset * 2 + 2).expect("fixture ID fits u64"));
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
                let edge = EdgeId(100 + u64::try_from(offset).expect("fixture ID fits u64"));
                graph.insert_edge(EdgeInput {
                    id: edge,
                    source,
                    target,
                    relationship_type,
                    layer: Layer::Observed,
                    revision: revision + 3,
                    properties: Self::properties(case, offset, num, name),
                })?;
            }
        } else {
            for offset in 0..rows {
                let node = NodeId(u64::try_from(offset + 1).expect("fixture ID fits u64"));
                graph.insert_node(NodeInput {
                    id: node,
                    layer: Layer::Observed,
                    revision: node.0,
                    labels: vec![n_label],
                    properties: Self::properties(case, offset, num, name),
                })?;
            }
        }

        let bookmark = Bookmark {
            term: 37,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            num,
            name,
            n_label,
        })
    }

    fn properties(
        case: TckCase,
        offset: usize,
        num: PropertyId,
        name: PropertyId,
    ) -> Vec<(PropertyId, ScalarValue)> {
        let numeric = match case.shape {
            PostWriteShape::LimitZero | PostWriteShape::SkipAll => 42,
            PostWriteShape::PageTwo | PostWriteShape::PageAll
                if !case.group.changes_numeric_property() =>
            {
                42
            }
            _ => i64::try_from(offset + 1).expect("five fixture rows fit i64"),
        };
        let mut properties = vec![(num, ScalarValue::Integer(numeric))];
        if matches!(
            case.group,
            PostWriteGroup::RemoveNodeProperty | PostWriteGroup::RemoveRelationshipProperty
        ) && !matches!(
            case.shape,
            PostWriteShape::LimitZero | PostWriteShape::SkipAll
        ) {
            properties.push((name, ScalarValue::String(Arc::<str>::from("a"))));
        }
        properties
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

    fn strict_cpu_backend(&self) -> Result<ObservedWholeStatementBackend> {
        ObservedWholeStatementBackend::strict_cpu_reference(self.cpu()?)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedWholeStatementBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedWholeStatementBackend::real_metal(metal)
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
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
            require_native_execution: true,
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
    case: TckCase,
) -> std::result::Result<Vec<i64>, String> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() {
            return Err("result batch has misaligned columns".to_owned());
        }
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == case.expected_column())
            .ok_or_else(|| {
                format!(
                    "result batch omitted expected `{}` column",
                    case.expected_column()
                )
            })?;
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

fn entity_mutations(output: &ExecutionOutput) -> Vec<&GraphMutation> {
    output
        .graph_mutations
        .iter()
        .filter(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty { .. }
                    | GraphMutation::SetEdgeProperty { .. }
                    | GraphMutation::AddNodeLabels { .. }
                    | GraphMutation::RemoveNodeLabels { .. }
            )
        })
        .collect()
}

fn mutation_debug(output: &ExecutionOutput) -> Vec<String> {
    output
        .graph_mutations
        .iter()
        .map(|mutation| format!("{mutation:?}"))
        .collect()
}

fn assert_expected_output(
    fixture: &Fixture,
    case: TckCase,
    output: &ExecutionOutput,
) -> std::result::Result<(), String> {
    let expected_type = if matches!(
        case.shape,
        PostWriteShape::LimitZero | PostWriteShape::SkipAll
    ) {
        ColumnType::Null
    } else {
        ColumnType::Integer
    };
    let expected_schema = vec![(case.expected_column().to_owned(), expected_type)];
    if output.result.schema != expected_schema {
        return Err(format!(
            "schema mismatch: expected {expected_schema:?}, got {:?}",
            output.result.schema
        ));
    }
    let actual_values = integer_values(output, case)?;
    let mut expected_values = case.expected_values();
    expected_values.sort_unstable();
    if actual_values != expected_values {
        return Err(format!(
            "row mismatch: expected {expected_values:?}, got {actual_values:?}"
        ));
    }
    if output.result.statistics != case.expected_stats() {
        return Err(format!(
            "statement-statistics mismatch: expected {:?}, got {:?}",
            case.expected_stats(),
            output.result.statistics
        ));
    }
    if output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
    {
        return Err(
            "post-write result changed its bookmark, truncated, or emitted temporal work"
                .to_owned(),
        );
    }

    let mutations = entity_mutations(output);
    if mutations.len() != case.shape.input_rows() {
        return Err(format!(
            "expected {} entity mutations, got {}: {:?}",
            case.shape.input_rows(),
            mutations.len(),
            mutation_debug(output)
        ));
    }
    if output.dependencies.write_targets.len() != case.shape.input_rows() {
        return Err(format!(
            "expected {} distinct write targets, got {:?}",
            case.shape.input_rows(),
            output.dependencies.write_targets
        ));
    }

    let actions_are_exact = match case.group {
        PostWriteGroup::RemoveNodeProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty {
                    property,
                    value: ScalarValue::Null,
                    ..
                } if *property == if matches!(case.shape, PostWriteShape::LimitZero | PostWriteShape::SkipAll) {
                    fixture.num
                } else {
                    fixture.name
                }
            )
        }),
        PostWriteGroup::RemoveRelationshipProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetEdgeProperty {
                    property,
                    value: ScalarValue::Null,
                    ..
                } if *property == if matches!(case.shape, PostWriteShape::LimitZero | PostWriteShape::SkipAll) {
                    fixture.num
                } else {
                    fixture.name
                }
            )
        }),
        PostWriteGroup::SetNodeProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetNodeProperty { property, value: ScalarValue::Integer(_), .. }
                    if *property == fixture.num
            )
        }),
        PostWriteGroup::SetRelationshipProperty => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::SetEdgeProperty { property, value: ScalarValue::Integer(_), .. }
                    if *property == fixture.num
            )
        }),
        PostWriteGroup::RemoveNodeLabel => mutations.iter().all(|mutation| {
            matches!(
                mutation,
                GraphMutation::RemoveNodeLabels { labels, .. }
                    if labels.as_slice() == [fixture.n_label]
            )
        }),
        PostWriteGroup::SetNodeLabel => {
            let declared = output.graph_mutations.iter().find_map(|mutation| match mutation {
                GraphMutation::DeclareLabel { name, id } if name == "Foo" => Some(*id),
                _ => None,
            });
            declared.is_some_and(|foo| {
                mutations.iter().all(|mutation| {
                    matches!(
                        mutation,
                        GraphMutation::AddNodeLabels { labels, .. } if labels.as_slice() == [foo]
                    )
                })
            })
        }
    };
    if !actions_are_exact {
        return Err(format!(
            "mutation action mismatch: {:?}",
            mutation_debug(output)
        ));
    }
    Ok(())
}

fn assert_native_request(
    case: TckCase,
    request: &ResidentNodePipelineRequest,
) -> std::result::Result<(), String> {
    if request.project != PROJECT || request.mutation.is_none() {
        return Err(format!(
            "whole-statement request omitted the project or mutation program: {request:#?}"
        ));
    }
    let (offset, limit) = match case.shape {
        PostWriteShape::LimitZero => (0, 0),
        PostWriteShape::SkipAll => (1, MAX_RESULT_ROWS + 1),
        PostWriteShape::PageTwo => (2, 2),
        PostWriteShape::PageAll => (0, 5),
        _ => return Ok(()),
    };
    if request.offset != offset || request.limit != limit {
        return Err(format!(
            "post-write pagination escaped the native request: expected offset/limit {offset}/{limit}, got {}/{}",
            request.offset, request.limit
        ));
    }
    Ok(())
}

fn assert_matches_reference(
    reference: &ExecutionOutput,
    native: &ExecutionOutput,
) -> std::result::Result<(), String> {
    if native.result != reference.result {
        return Err(format!(
            "native result differs from exact CPU semantics:\nCPU: {:#?}\nnative: {:#?}",
            reference.result, native.result
        ));
    }
    if mutation_debug(native) != mutation_debug(reference)
        || native.dependencies.entities != reference.dependencies.entities
        || native.dependencies.write_targets != reference.dependencies.write_targets
    {
        return Err(format!(
            "native mutation/dependency publication differs from CPU:\nCPU mutations: {:?}\nnative mutations: {:?}",
            mutation_debug(reference),
            mutation_debug(native)
        ));
    }
    Ok(())
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedWholeStatementBackend,
    case: TckCase,
) -> std::result::Result<(), String> {
    let reference = QueryEngine
        .execute(case.query, &mut context(fixture, None))
        .map_err(|error| format!("CPU oracle failed with {:?}: {error}", error.code))?;
    assert_expected_output(fixture, case, &reference)?;

    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.pipeline_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();

    let native = QueryEngine
        .execute(case.query, &mut context(fixture, Some(backend)))
        .map_err(|error| {
            format!(
                "native query failed with {:?}: {error}; pins={}, pipeline calls={}, rejected routes={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.pipeline_calls.load(Ordering::SeqCst) - calls_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;

    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations.pipeline_calls.load(Ordering::SeqCst) != calls_before + 1 {
        return Err("query did not cross exactly one native whole-statement pipeline".to_owned());
    }
    if observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before {
        return Err(
            "query entered a generic, host-oriented, or unpinned execution route".to_owned(),
        );
    }
    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one whole-statement request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "whole-statement request disappeared".to_owned())?
    };
    assert_native_request(case, &request)?;
    assert_expected_output(fixture, case, &native)?;
    assert_matches_reference(&reference, &native)
}

fn run_all_native_cases(
    backend: &mut ObservedWholeStatementBackend,
) -> std::result::Result<(), Vec<String>> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = match Fixture::new(case) {
            Ok(fixture) => fixture,
            Err(error) => {
                failures.push(format!("{}: fixture failed: {error}", case.label()));
                continue;
            }
        };
        match fixture
            .image()
            .and_then(|image| backend.replace_all_projects(vec![image]))
        {
            Ok(()) => {}
            Err(error) => {
                failures.push(format!(
                    "{}: resident replacement failed: {error}",
                    case.label()
                ));
                continue;
            }
        }
        if let Err(error) = execute_native_case(&fixture, backend, case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures)
    }
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} post-write acceptance had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_is_exactly_the_42_verified_remove3_and_set6_scenarios() {
    // Keep these zero-based report IDs and exact expanded names literal. The ignored external
    // identity gate below resolves them against the current certified report.
    let cases = all_cases().collect::<Vec<_>>();
    assert_eq!(cases.len(), 42);
    assert_eq!(
        cases.iter().map(|case| case.id).collect::<Vec<_>>(),
        (673_u16..=693).chain(855_u16..=875).collect::<Vec<_>>()
    );
    assert!(
        cases[..21]
            .iter()
            .all(|case| case.feature == REMOVE_FEATURE)
    );
    assert!(cases[21..].iter().all(|case| case.feature == SET_FEATURE));
    assert!(
        cases
            .iter()
            .all(|case| !case.name.is_empty() && !case.query.is_empty())
    );
}

#[test]
#[ignore = "external identity gate: requires the current certified 3,897-scenario report"]
fn manifest_resolves_uniquely_to_exact_zero_based_report_indices() -> Result<()> {
    let report = certified_report()?;
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);
    assert_eq!(report.metal_passed, 3_184);

    for case in all_cases() {
        let matches = report
            .scenarios
            .iter()
            .enumerate()
            .filter(|(_, scenario)| {
                scenario.path.ends_with(case.feature) && scenario.name == case.name
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "{} did not resolve uniquely in the certified report",
            case.label()
        );
        let (report_index, certified) = matches[0];
        assert_eq!(
            report_index,
            usize::from(case.id),
            "{} resolved at the wrong zero-based report array index",
            case.label()
        );
        assert!(certified.cpu_passed, "{} is not CPU-green", case.label());
        assert!(
            certified.metal_passed,
            "{} is not Metal-green",
            case.label()
        );
        assert!(
            certified.cpu_metal_matched,
            "{} has a CPU/Metal divergence",
            case.label()
        );
        assert!(
            certified.fully_conformant,
            "{} is not conformant",
            case.label()
        );
        assert!(certified.shared_failures.is_empty(), "{}", case.label());
        assert!(certified.cpu_failures.is_empty(), "{}", case.label());
        assert!(certified.metal_failures.is_empty(), "{}", case.label());
        assert!(certified.divergences.is_empty(), "{}", case.label());
    }
    Ok(())
}

#[test]
fn cpu_oracle_proves_exact_rows_and_side_effects_for_all_42_scenarios() -> Result<()> {
    let mut failures = Vec::new();
    for case in all_cases() {
        let fixture = Fixture::new(case)?;
        match QueryEngine.execute(case.query, &mut context(&fixture, None)) {
            Ok(output) => {
                if let Err(error) = assert_expected_output(&fixture, case, &output) {
                    failures.push(format!("{}: {error}", case.label()));
                }
            }
            Err(error) => failures.push(format!(
                "{}: CPU oracle failed with {:?}: {error}",
                case.label(),
                error.code
            )),
        }
    }
    assert_no_failures("CPU oracle", failures);
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: post-write result stages are not yet one pinned native mutation pipeline"]
fn strict_cpu_reference_runs_all_42_through_one_pinned_native_boundary() -> Result<()> {
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("post-write manifest is empty"))?;
    let fixture = Fixture::new(first)?;
    let mut backend = fixture.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    if let Err(failures) = run_all_native_cases(&mut backend) {
        assert_no_failures("strict CPU reference", failures);
    }
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: requires post-write native composition before receipt fault injection"]
fn post_write_receipts_reject_wrong_device_provenance_before_publication() -> Result<()> {
    let case = all_cases()
        .next()
        .ok_or_else(|| Error::internal("post-write manifest is empty"))?;
    let fixture = Fixture::new(case)?;
    let backend = ObservedWholeStatementBackend::wrong_receipt_provenance(fixture.cpu()?)?;
    let observations = backend.observations();
    let error = QueryEngine
        .execute(case.query, &mut context(&fixture, Some(&backend)))
        .err()
        .ok_or_else(|| Error::internal("wrong-device post-write receipts were published"))?;
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pipeline_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
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
#[ignore = "red acceptance gate: all 42 post-write scenarios must compose on real Metal"]
fn real_metal_matches_cpu_for_all_42_with_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = all_cases()
        .next()
        .ok_or_else(|| Error::internal("post-write manifest is empty"))?;
    let fixture = Fixture::new(first)?;
    let mut backend = fixture.real_metal_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    if let Err(failures) = run_all_native_cases(&mut backend) {
        assert_no_failures("real Metal", failures);
    }
    Ok(())
}
