// Test-only module. Clippy's `allow-expect-in-tests` covers `#[test]` bodies but not the helper
// functions those tests call, and a failed expectation in a fixture is the intended way for a test
// to fail. The production denial of `expect` is unaffected.
#![allow(clippy::expect_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodeBinding,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectImage,
        ResidentPropertyFilterInstruction, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 64;
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";

fn assert_certified_report_identities<'a>(
    identities: impl IntoIterator<Item = (usize, &'a str, &'a str)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = std::collections::BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature, expanded_name)),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Default)]
struct PipelineObservations {
    pins: AtomicUsize,
    pipeline_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNodePipelineRequest>>,
}

/// Strict route observer which implements no Cypher semantics.
///
/// Before pinning, the CPU reference deliberately advertises Metal so strict native admission is
/// mandatory. After pinning it exposes its honest CPU kind and delegates only the complete
/// resident node pipeline. Every generic primitive and the newer typed-row route fail closed. A
/// real Metal wrapper reports Metal throughout and delegates the same single pipeline boundary.
struct ObservedPipelineBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    observations: Arc<PipelineObservations>,
}

impl ObservedPipelineBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            advertised_kind: BackendKind::Metal,
            pinned_kind: BackendKind::Cpu,
            actual_kind: BackendKind::Cpu,
            pinned: false,
            observations: Arc::new(PipelineObservations::default()),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "native string-predicate test did not construct a real Metal backend",
            ));
        }
        Ok(Self {
            inner: Box::new(inner),
            advertised_kind: BackendKind::Metal,
            pinned_kind: BackendKind::Metal,
            actual_kind: BackendKind::Metal,
            pinned: false,
            observations: Arc::new(PipelineObservations::default()),
        })
    }

    fn observations(&self) -> Arc<PipelineObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict string-predicate test rejected `{route}` execution"),
        ))
    }
}

impl ExecutionBackend for ObservedPipelineBackend {
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
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

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_query_route("execute_row_program")
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
enum FixtureKind {
    Core,
    ContainsWhitespace,
    BoundaryWhitespace,
    Unicode,
}

impl FixtureKind {
    const ALL: [Self; 4] = [
        Self::Core,
        Self::ContainsWhitespace,
        Self::BoundaryWhitespace,
        Self::Unicode,
    ];

    fn names(self) -> &'static [Option<&'static str>] {
        match self {
            Self::Core => &[
                Some("ABCDEF"),
                Some("AB"),
                Some("abcdef"),
                Some("ab"),
                Some(""),
                None,
            ],
            Self::ContainsWhitespace => &[
                Some("ABCDEF"),
                Some("AB"),
                Some("abcdef"),
                Some("ab"),
                Some(""),
                None,
                Some("Foo Foo"),
                Some("Foo\nFoo"),
                Some("Foo\tFoo"),
            ],
            Self::BoundaryWhitespace => &[
                Some("ABCDEF"),
                Some("AB"),
                Some("abcdef"),
                Some("ab"),
                Some(""),
                None,
                Some(" Foo "),
                Some("\nFoo\n"),
                Some("\tFoo\t"),
            ],
            Self::Unicode => &[
                Some("éclair"),
                Some("é"),
                Some("βeta"),
                Some("🙂graph🙂"),
                Some("🙂"),
                Some("aé🙂z"),
                None,
                Some("éclair"),
            ],
        }
    }
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    name_property: PropertyId,
}

impl Fixture {
    fn new(kind: FixtureKind) -> Result<Self> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("TheLabel")?;
        let name_property = graph.catalog_mut().intern_property("name")?;
        for (offset, name) in kind.names().iter().copied().enumerate() {
            let id = u64::try_from(offset + 1)
                .map_err(|_| Error::internal("string-predicate fixture ID overflowed"))?;
            let properties = name
                .map(|name| vec![(name_property, ScalarValue::String(Arc::<str>::from(name)))])
                .unwrap_or_default();
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties,
            })?;
        }
        let bookmark = Bookmark {
            term: 31,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            name_property,
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

    fn strict_cpu_backend(&self) -> Result<ObservedPipelineBackend> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(ObservedPipelineBackend::strict_cpu_reference(cpu))
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedPipelineBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedPipelineBackend::real_metal(metal)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedRows {
    Nodes(&'static [u64]),
    Strings(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug)]
struct StringPredicateCase {
    tck_identity: Option<(u16, &'static str)>,
    name: &'static str,
    fixture: FixtureKind,
    query: &'static str,
    column: &'static str,
    expected: ExpectedRows,
    leaf_count: usize,
    and_count: usize,
    not_count: usize,
}

impl StringPredicateCase {
    fn label(self) -> String {
        self.tck_identity.map_or_else(
            || self.name.to_owned(),
            |(id, feature)| format!("TCK {id} {feature} {}", self.name),
        )
    }
}

const OFFICIAL_CASES: &[StringPredicateCase] = &[
    StringPredicateCase {
        tck_identity: Some((2785, "features/expressions/string/String10.feature")),
        name: "[1] Finding exact matches with non-proper substring",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name CONTAINS 'ABCDEF' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2786, "features/expressions/string/String10.feature")),
        name: "[2] Finding substring of string",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name CONTAINS 'CD' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2787, "features/expressions/string/String10.feature")),
        name: "[3] Finding the empty substring",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name CONTAINS '' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 3, 4, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2788, "features/expressions/string/String10.feature")),
        name: "[4] Finding strings containing whitespace",
        fixture: FixtureKind::ContainsWhitespace,
        query: "MATCH (a) WHERE a.name CONTAINS ' ' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&["Foo Foo"]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2789, "features/expressions/string/String10.feature")),
        name: "[5] Finding strings containing newline",
        fixture: FixtureKind::ContainsWhitespace,
        query: r"MATCH (a) WHERE a.name CONTAINS '\n' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&["Foo\nFoo"]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2790, "features/expressions/string/String10.feature")),
        name: "[6] No string contains null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name CONTAINS null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2791, "features/expressions/string/String10.feature")),
        name: "[7] No string does not contain null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name CONTAINS null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
    StringPredicateCase {
        tck_identity: Some((2793, "features/expressions/string/String10.feature")),
        name: "[9] NOT with CONTAINS",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name CONTAINS 'b' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
    StringPredicateCase {
        tck_identity: Some((2794, "features/expressions/string/String11.feature")),
        name: "[1] Combining prefix and suffix search",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH 'a' AND a.name ENDS WITH 'f' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[3]),
        leaf_count: 2,
        and_count: 1,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2795, "features/expressions/string/String11.feature")),
        name: "[2] Combining prefix, suffix, and substring search",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH 'A' AND a.name CONTAINS 'C' AND a.name ENDS WITH 'EF' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 3,
        and_count: 2,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2798, "features/expressions/string/String8.feature")),
        name: "[1] Finding exact matches with non-proper prefix",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH 'ABCDEF' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2799, "features/expressions/string/String8.feature")),
        name: "[2] Finding beginning of string",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH 'ABC' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2800, "features/expressions/string/String8.feature")),
        name: "[3] Finding the empty prefix",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH '' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 3, 4, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2801, "features/expressions/string/String8.feature")),
        name: "[4] Finding strings starting with whitespace",
        fixture: FixtureKind::BoundaryWhitespace,
        query: "MATCH (a) WHERE a.name STARTS WITH ' ' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&[" Foo "]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2802, "features/expressions/string/String8.feature")),
        name: "[5] Finding strings starting with newline",
        fixture: FixtureKind::BoundaryWhitespace,
        query: r"MATCH (a) WHERE a.name STARTS WITH '\n' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&["\nFoo\n"]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2803, "features/expressions/string/String8.feature")),
        name: "[6] No string starts with null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name STARTS WITH null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2804, "features/expressions/string/String8.feature")),
        name: "[7] No string does not start with null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name STARTS WITH null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
    StringPredicateCase {
        tck_identity: Some((2806, "features/expressions/string/String8.feature")),
        name: "[9] NOT with STARTS WITH",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name STARTS WITH 'ab' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
    StringPredicateCase {
        tck_identity: Some((2807, "features/expressions/string/String9.feature")),
        name: "[1] Finding exact matches with non-proper suffix",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name ENDS WITH 'AB' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[2]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2808, "features/expressions/string/String9.feature")),
        name: "[2] Finding end of string",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name ENDS WITH 'DEF' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2809, "features/expressions/string/String9.feature")),
        name: "[3] Finding the empty suffix",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name ENDS WITH '' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 3, 4, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2810, "features/expressions/string/String9.feature")),
        name: "[4] Finding strings ending with whitespace",
        fixture: FixtureKind::BoundaryWhitespace,
        query: "MATCH (a) WHERE a.name ENDS WITH ' ' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&[" Foo "]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2811, "features/expressions/string/String9.feature")),
        name: "[5] Finding strings ending with newline",
        fixture: FixtureKind::BoundaryWhitespace,
        query: r"MATCH (a) WHERE a.name ENDS WITH '\n' RETURN a.name AS name",
        column: "name",
        expected: ExpectedRows::Strings(&["\nFoo\n"]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2812, "features/expressions/string/String9.feature")),
        name: "[6] No string ends with null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE a.name ENDS WITH null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: Some((2813, "features/expressions/string/String9.feature")),
        name: "[7] No string does not end with null",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name ENDS WITH null RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
    StringPredicateCase {
        tck_identity: Some((2815, "features/expressions/string/String9.feature")),
        name: "[9] NOT with ENDS WITH",
        fixture: FixtureKind::Core,
        query: "MATCH (a) WHERE NOT a.name ENDS WITH 'def' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 4, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
];

const UNICODE_CASES: &[StringPredicateCase] = &[
    StringPredicateCase {
        tck_identity: None,
        name: "UTF-8 prefix with stable duplicate rows",
        fixture: FixtureKind::Unicode,
        query: "MATCH (a) WHERE a.name STARTS WITH 'é' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 8]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: None,
        name: "UTF-8 suffix",
        fixture: FixtureKind::Unicode,
        query: "MATCH (a) WHERE a.name ENDS WITH '🙂' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[4, 5]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: None,
        name: "UTF-8 substring",
        fixture: FixtureKind::Unicode,
        query: "MATCH (a) WHERE a.name CONTAINS 'é🙂' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[6]),
        leaf_count: 1,
        and_count: 0,
        not_count: 0,
    },
    StringPredicateCase {
        tck_identity: None,
        name: "UTF-8 NOT and missing-property unknown",
        fixture: FixtureKind::Unicode,
        query: "MATCH (a) WHERE NOT a.name CONTAINS 'β' RETURN a",
        column: "a",
        expected: ExpectedRows::Nodes(&[1, 2, 4, 5, 6, 8]),
        leaf_count: 1,
        and_count: 0,
        not_count: 1,
    },
];

fn all_cases() -> impl Iterator<Item = &'static StringPredicateCase> {
    OFFICIAL_CASES.iter().chain(UNICODE_CASES)
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
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 2,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ObservedRows {
    Nodes(Vec<u64>),
    Strings(Vec<String>),
}

type CaseResult<T = ()> = std::result::Result<T, String>;

fn observe_rows(output: &ExecutionOutput, case: StringPredicateCase) -> CaseResult<ObservedRows> {
    if output.result.schema.len() != 1 || output.result.schema[0].0 != case.column {
        return Err(format!(
            "schema mismatch: expected one `{}` column, got {:?}",
            case.column, output.result.schema
        ));
    }
    let mut values = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() {
            return Err("result batch has misaligned columns".to_owned());
        }
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == case.column)
            .ok_or_else(|| format!("result batch omitted `{}`", case.column))?;
        values.extend(column.values.iter());
    }
    match case.expected {
        ExpectedRows::Nodes(_) => values
            .into_iter()
            .map(|value| match value {
                ResultValue::Node(node) => Ok(node.id.0),
                _ => Err(format!(
                    "`{}` contained a non-node value: {value:?}",
                    case.column
                )),
            })
            .collect::<CaseResult<Vec<_>>>()
            .map(ObservedRows::Nodes),
        ExpectedRows::Strings(_) => values
            .into_iter()
            .map(|value| match value {
                ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.to_string()),
                _ => Err(format!(
                    "`{}` contained a non-string value: {value:?}",
                    case.column
                )),
            })
            .collect::<CaseResult<Vec<_>>>()
            .map(ObservedRows::Strings),
    }
}

fn expected_rows(case: StringPredicateCase) -> ObservedRows {
    match case.expected {
        ExpectedRows::Nodes(ids) => ObservedRows::Nodes(ids.to_vec()),
        ExpectedRows::Strings(values) => {
            ObservedRows::Strings(values.iter().map(|value| (*value).to_owned()).collect())
        }
    }
}

fn assert_request(
    fixture: &Fixture,
    case: StringPredicateCase,
    request: &ResidentNodePipelineRequest,
) -> CaseResult {
    if request.project != PROJECT
        || request.layers != LayerMask::AUTHORITY
        || !request.labels.is_empty()
        || request.initial_optional
        || request.expansion.is_some()
        || !request.continuations.is_empty()
        || request.correlated_optional.is_some()
        || request.relationship_null_filter.is_some()
        || !request.predicates.is_empty()
        || request.value_matrix.is_some()
        || request.mutation.is_some()
        || !request.orders.is_empty()
        || request.offset != 0
        || request.limit != MAX_RESULT_ROWS + 1
        || !request.integer_projections.is_empty()
        || !request.property_null_projections.is_empty()
        || request.max_output_rows != MAX_RESULT_ROWS
    {
        return Err(format!(
            "query did not compile to the one bounded property-filter pipeline: {request:#?}"
        ));
    }
    let [program] = request.property_filters.as_slice() else {
        return Err(format!(
            "expected exactly one resident property-filter program, got {}",
            request.property_filters.len()
        ));
    };
    program
        .validate()
        .map_err(|error| format!("resident property-filter program is invalid: {error}"))?;
    if program.output as usize + 1 != program.instructions.len() {
        return Err(format!(
            "property-filter output is not its final SSA register: {program:#?}"
        ));
    }

    let mut leaf_count = 0_usize;
    let mut and_count = 0_usize;
    let mut or_count = 0_usize;
    let mut not_count = 0_usize;
    for instruction in &program.instructions {
        match instruction {
            ResidentPropertyFilterInstruction::And { .. } => and_count += 1,
            ResidentPropertyFilterInstruction::Or { .. } => or_count += 1,
            ResidentPropertyFilterInstruction::Not { .. } => not_count += 1,
            ResidentPropertyFilterInstruction::CompareString {
                binding, property, ..
            } => {
                leaf_count += 1;
                if *binding != ResidentNodeBinding::Start || *property != fixture.name_property {
                    return Err(format!(
                        "string leaf read the wrong resident binding/property: {instruction:?}"
                    ));
                }
            }
            ResidentPropertyFilterInstruction::IsNull {
                binding, property, ..
            } => {
                leaf_count += 1;
                if *binding != ResidentNodeBinding::Start
                    || property.is_some_and(|property| property != fixture.name_property)
                {
                    return Err(format!(
                        "property leaf read the wrong resident binding/property: {instruction:?}"
                    ));
                }
            }
        }
    }
    if (leaf_count, and_count, or_count, not_count)
        != (case.leaf_count, case.and_count, 0, case.not_count)
    {
        return Err(format!(
            "wrong resident Boolean shape: expected leaves/AND/OR/NOT={}/{}/0/{}, got {leaf_count}/{and_count}/{or_count}/{not_count}: {program:#?}",
            case.leaf_count, case.and_count, case.not_count
        ));
    }
    Ok(())
}

fn execute_case(
    fixture: &Fixture,
    backend: &ObservedPipelineBackend,
    case: StringPredicateCase,
) -> CaseResult<ObservedRows> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.pipeline_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let requests_before = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();

    let output = QueryEngine
        .execute(case.query, &mut context(fixture, backend))
        .map_err(|error| {
            format!(
                "query failed with {:?}: {error}; query: {}",
                error.code, case.query
            )
        })?;

    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one resident graph generation".to_owned());
    }
    if observations.pipeline_calls.load(Ordering::SeqCst) != calls_before + 1 {
        return Err("query did not cross exactly one resident node-pipeline boundary".to_owned());
    }
    if observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before {
        return Err(
            "query entered a generic, host-oriented, or typed-row backend route".to_owned(),
        );
    }

    let request = {
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            return Err("observer did not retain exactly one resident request".to_owned());
        }
        requests
            .last()
            .cloned()
            .ok_or_else(|| "resident request disappeared".to_owned())?
    };
    assert_request(fixture, case, &request)?;

    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err("read-only string predicate produced side effects or truncation".to_owned());
    }

    let actual = observe_rows(&output, case)?;
    let expected = expected_rows(case);
    if actual != expected {
        return Err(format!(
            "stable result mismatch: expected {expected:?}, got {actual:?}"
        ));
    }
    Ok(actual)
}

fn run_fixture_cases(
    fixture_kind: FixtureKind,
    fixture: &Fixture,
    backend: &ObservedPipelineBackend,
    failures: &mut Vec<String>,
) {
    for case in all_cases().filter(|case| case.fixture == fixture_kind) {
        if let Err(error) = execute_case(fixture, backend, *case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} native string-predicate suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn scenario_manifest_is_exactly_the_26_coherent_tck_cases() {
    let ids = OFFICIAL_CASES
        .iter()
        .filter_map(|case| case.tck_identity.map(|(id, _)| id))
        .collect::<Vec<_>>();
    assert_eq!(
        ids,
        vec![
            2785, 2786, 2787, 2788, 2789, 2790, 2791, 2793, 2794, 2795, 2798, 2799, 2800, 2801,
            2802, 2803, 2804, 2806, 2807, 2808, 2809, 2810, 2811, 2812, 2813, 2815,
        ]
    );
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_string_predicate_selector() {
    assert_certified_report_identities(OFFICIAL_CASES.iter().map(|case| {
        let (report_id, feature) = case
            .tck_identity
            .expect("official string-predicate case has a TCK identity");
        (usize::from(report_id), feature, case.name)
    }));
}

#[test]
fn strict_cpu_reference_runs_all_string_predicates_through_one_native_boundary() -> Result<()> {
    let mut failures = Vec::new();
    for fixture_kind in FixtureKind::ALL {
        let fixture = Fixture::new(fixture_kind)?;
        let backend = fixture.strict_cpu_backend()?;
        assert_eq!(backend.kind(), BackendKind::Metal);
        assert_eq!(backend.actual_kind, BackendKind::Cpu);
        run_fixture_cases(fixture_kind, &fixture, &backend, &mut failures);
    }
    assert_no_failures("strict CPU reference", failures);
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
#[ignore = "requires an available physical Metal device"]
fn real_metal_runs_all_string_predicates_with_cpu_exact_results_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = Fixture::new(FixtureKind::Core)?;
    let mut metal = first.real_metal_backend()?;
    assert_eq!(metal.kind(), BackendKind::Metal);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for fixture_kind in FixtureKind::ALL {
        let fixture = Fixture::new(fixture_kind)?;
        metal.replace_all_projects(vec![fixture.image()?])?;
        run_fixture_cases(fixture_kind, &fixture, &metal, &mut failures);
    }
    assert_no_failures("real Metal", failures);
    Ok(())
}
