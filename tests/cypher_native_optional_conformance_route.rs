// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance gate for the 52 read-only, fixed-length OPTIONAL MATCH scenarios
//! identified by `/tmp/irongraph-tck-full-post-temporal-array-last.json`.
//!
//! The manifest below is intentionally literal: report ordinal, pinned feature path, and exact
//! expanded scenario name.  The passing tests prove that these are precisely the report's
//! CPU-green/Metal-admission-failed OPTIONAL cases after excluding writes, MERGE, and variable-
//! length relationships.  The ignored tests are the red production gate: the generic row engine
//! is used only to construct the CPU oracle; strict CPU-reference and real Metal must each execute
//! the primary query as one pinned resident command with no legacy host graph operators.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentDeviceCompletion, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentNullableRelationMatchMode, ResidentNullableRelationObligationKind,
        ResidentNullableRelationPredicate, ResidentNullableRelationPredicateValue,
        ResidentNullableRelationReceipt, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentNullableRelationStage,
        ResidentNullableRelationshipEndpoint, ResidentProjectImage, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const PINNED_FEATURE_ROOT: &str =
    "/private/tmp/irongraph-opencypher-debug.6MXlLm/openCypher/tck/features";
const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4f50_5449_4f4e_414c_5f35_325f_5443_4b,
));
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;

/// `report-index|feature|expanded scenario name`. Report indexes are the exact zero-based array
/// positions in the certified 3,897-scenario JSON report, not feature-local scenario numbers.
const EXACT_MANIFEST: &str = r#"
369|clauses/match/Match3.feature|[27] Matching from null nodes should return no results owing to finding no matches
370|clauses/match/Match3.feature|[28] Matching from null nodes should return no results owing to matches being filtered out
510|clauses/match/Match7.feature|[2] OPTIONAL MATCH with previously bound nodes
511|clauses/match/Match7.feature|[3] OPTIONAL MATCH and bound nodes
512|clauses/match/Match7.feature|[4] Optionally matching relationship with bound nodes in reverse direction
513|clauses/match/Match7.feature|[5] Optionally matching relationship with a relationship that is already bound
514|clauses/match/Match7.feature|[6] Optionally matching relationship with a relationship and node that are both already bound
515|clauses/match/Match7.feature|[7] MATCH with OPTIONAL MATCH in longer pattern
516|clauses/match/Match7.feature|[8] Longer pattern with bound nodes without matches
517|clauses/match/Match7.feature|[9] Longer pattern with bound nodes
518|clauses/match/Match7.feature|[10] Optionally matching from null nodes should return null
519|clauses/match/Match7.feature|[11] Return two subgraphs with bound undirected relationship and optional relationship
524|clauses/match/Match7.feature|[16] Optionally matching named paths - null result
525|clauses/match/Match7.feature|[17] Optionally matching named paths - existing result
526|clauses/match/Match7.feature|[18] Named paths inside optional matches with node predicates
529|clauses/match/Match7.feature|[21] Handling optional matches between nulls
530|clauses/match/Match7.feature|[22] MATCH after OPTIONAL MATCH
533|clauses/match/Match7.feature|[25] Optionally matching self-loops without matches
534|clauses/match/Match7.feature|[26] Handling correlated optional matches; first does not match implies second does not match
535|clauses/match/Match7.feature|[27] Handling optional matches between optionally matched entities
536|clauses/match/Match7.feature|[28] Handling optional matches with inline label predicate
537|clauses/match/Match7.feature|[29] Satisfies the open world assumption, relationships between same nodes
538|clauses/match/Match7.feature|[30] Satisfies the open world assumption, single relationship
539|clauses/match/Match7.feature|[31] Satisfies the open world assumption, relationships between different nodes
578|clauses/match-where/MatchWhere6.feature|[1] Filter node with node label predicate on multi variables with multiple bindings after MATCH and OPTIONAL MATCH
579|clauses/match-where/MatchWhere6.feature|[2] Filter node with false node label predicate after OPTIONAL MATCH
581|clauses/match-where/MatchWhere6.feature|[4] Do not fail when predicates on optionally matched and missed nodes are invalid
582|clauses/match-where/MatchWhere6.feature|[5] Matching and optionally matching with unbound nodes and equality predicate in reverse direction
583|clauses/match-where/MatchWhere6.feature|[6] Join nodes on non-equality of properties – OPTIONAL MATCH and WHERE
584|clauses/match-where/MatchWhere6.feature|[7] Join nodes on non-equality of properties – OPTIONAL MATCH on two relationships and WHERE
585|clauses/match-where/MatchWhere6.feature|[8] Join nodes on non-equality of properties – Two OPTIONAL MATCH clauses and WHERE
906|clauses/with/With1.feature|[5] Forwarding null
907|clauses/with/With1.feature|[6] Forwarding a node variable possibly null
1234|clauses/with-where/WithWhere1.feature|[3] Filter for an unbound relationship variable
1235|clauses/with-where/WithWhere1.feature|[4] Filter for an unbound node variable
1267|expressions/aggregation/Aggregation5.feature|[1] `collect()` filtering nulls
1268|expressions/aggregation/Aggregation5.feature|[2] OPTIONAL MATCH and `collect()` on node property
1282|expressions/aggregation/Aggregation8.feature|[1] Distinct on unbound node
1537|expressions/graph/Graph3.feature|[7] `labels()` on null node
1542|expressions/graph/Graph4.feature|[3] `type()` on null relationship
1543|expressions/graph/Graph4.feature|[4] `type()` on mixed null and non-null relationships
1559|expressions/graph/Graph5.feature|[5] Label expression on null
1561|expressions/graph/Graph6.feature|[2] Statically access a property of a optional non-null node
1562|expressions/graph/Graph6.feature|[3] Statically access a property of a null node
1565|expressions/graph/Graph6.feature|[6] Statically access a property of a optional non-null relationship
1566|expressions/graph/Graph6.feature|[7] Statically access a property of a null relationship
1580|expressions/graph/Graph8.feature|[4] Using `keys()` on an optionally matched node
1583|expressions/graph/Graph8.feature|[7] Using `keys()` on an optionally matched relationship
1587|expressions/graph/Graph9.feature|[3] `properties()` on null
1684|expressions/list/List12.feature|[3] Size of list comprehension
2002|expressions/path/Path1.feature|[1] `nodes()` on null path
2005|expressions/path/Path2.feature|[3] `relationships()` on null path
"#;

const EXACT_FIRST_NINE_IDS: [usize; 9] = [369, 370, 510, 518, 529, 535, 536, 906, 907];
const EXACT_PREDICATE_IDS: [usize; 6] = [533, 579, 583, 585, 1234, 1235];
const EXACT_BOUND_RELATIONSHIP_FIRST_IDS: [usize; 2] = [512, 514];
const EXACT_BOUND_RELATIONSHIP_IDS: [usize; 3] = [512, 513, 514];
const EXACT_FIXED_MULTI_HOP_IDS: [usize; 3] = [515, 516, 517];
const EXACT_FIXED_MULTI_HOP_METAL_IDS: [usize; 2] = [516, 517];

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestCase {
    report_id: usize,
    feature: String,
    name: String,
}

#[derive(Clone, Debug, Deserialize)]
struct CertifiedReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Clone, Debug, Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
}

#[derive(Clone, Debug)]
struct SourceScenario {
    setup_queries: Vec<String>,
    query: String,
}

fn manifest() -> Result<Vec<ManifestCase>> {
    EXACT_MANIFEST
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.splitn(3, '|');
            let report_id = fields
                .next()
                .ok_or_else(|| Error::internal("OPTIONAL manifest omitted report ID"))?
                .parse::<usize>()
                .map_err(|error| Error::internal(format!("invalid OPTIONAL report ID: {error}")))?;
            let feature = fields
                .next()
                .ok_or_else(|| Error::internal("OPTIONAL manifest omitted feature"))?;
            let name = fields
                .next()
                .ok_or_else(|| Error::internal("OPTIONAL manifest omitted scenario name"))?;
            Ok(ManifestCase {
                report_id,
                feature: feature.to_owned(),
                name: name.to_owned(),
            })
        })
        .collect()
}

fn exact_first_nine() -> Result<Vec<ManifestCase>> {
    EXACT_FIRST_NINE_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn exact_predicate_tranche() -> Result<Vec<ManifestCase>> {
    EXACT_PREDICATE_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn exact_bound_relationship_first_tranche() -> Result<Vec<ManifestCase>> {
    EXACT_BOUND_RELATIONSHIP_FIRST_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn exact_bound_relationship_tranche() -> Result<Vec<ManifestCase>> {
    EXACT_BOUND_RELATIONSHIP_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn exact_fixed_multi_hop_tranche() -> Result<Vec<ManifestCase>> {
    EXACT_FIXED_MULTI_HOP_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn exact_fixed_multi_hop_metal_tranche() -> Result<Vec<ManifestCase>> {
    EXACT_FIXED_MULTI_HOP_METAL_IDS
        .iter()
        .map(|report_id| case_by_id(*report_id))
        .collect()
}

fn certified_report() -> Result<CertifiedReport> {
    let bytes = fs::read(REPORT_PATH).map_err(|error| {
        Error::internal(format!(
            "cannot read certified TCK report {REPORT_PATH}: {error}"
        ))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::internal(format!("cannot decode certified TCK report: {error}")))
}

fn normalize_query(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn query_from_failure(scenario: &CertifiedScenario) -> Result<String> {
    let failure = scenario
        .metal_failures
        .first()
        .ok_or_else(|| Error::internal("OPTIONAL report scenario has no Metal failure"))?;
    let start = failure.find('`').ok_or_else(|| {
        Error::internal("OPTIONAL report failure omitted opening query delimiter")
    })? + 1;
    let marker = "`: query unexpectedly failed";
    let end = failure[start..]
        .find(marker)
        .map(|offset| start + offset)
        .ok_or_else(|| Error::internal("OPTIONAL report failure omitted query delimiter"))?;
    Ok(failure[start..end].to_owned())
}

fn contains_variable_length_relationship(query: &str) -> bool {
    let mut in_relationship = false;
    for character in query.chars() {
        match character {
            '[' => in_relationship = true,
            ']' => in_relationship = false,
            '*' if in_relationship => return true,
            _ => {}
        }
    }
    false
}

fn scenario_block(source: &str, name: &str) -> Result<String> {
    let header = format!("Scenario: {name}");
    let start = source
        .find(&header)
        .ok_or_else(|| Error::internal(format!("pinned TCK omitted `{header}`")))?;
    let tail = &source[start..];
    let end = tail
        .lines()
        .skip(1)
        .scan(header.len() + 1, |offset, line| {
            let current = *offset;
            *offset += line.len() + 1;
            Some((current, line))
        })
        .find_map(|(offset, line)| line.trim_start().starts_with("Scenario:").then_some(offset))
        .unwrap_or(tail.len());
    Ok(tail[..end].to_owned())
}

fn docstring_after(block: &str, marker: &str) -> Result<String> {
    let start = block
        .find(marker)
        .ok_or_else(|| Error::internal(format!("scenario omitted `{marker}`")))?;
    let tail = &block[start + marker.len()..];
    let open = tail
        .find("\"\"\"")
        .ok_or_else(|| Error::internal(format!("`{marker}` omitted opening docstring")))?
        + 3;
    let close = tail[open..]
        .find("\"\"\"")
        .map(|offset| open + offset)
        .ok_or_else(|| Error::internal(format!("`{marker}` omitted closing docstring")))?;
    Ok(tail[open..close]
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" "))
}

fn source_scenario(case: &ManifestCase) -> Result<SourceScenario> {
    let path = Path::new(PINNED_FEATURE_ROOT).join(&case.feature);
    let source = fs::read_to_string(&path)
        .map_err(|error| Error::internal(format!("cannot read {}: {error}", path.display())))?;
    let block = scenario_block(&source, &case.name)?;
    if block.contains(" parameters are:")
        || block.contains(" parameter values are:")
        || block.contains("there exists a procedure ")
        || block.contains("executing control query:")
        || (block.contains("Given the ") && block.contains(" graph"))
    {
        return Err(Error::internal(format!(
            "{} requires an unsupported acceptance-fixture extension",
            case.name
        )));
    }
    let mut setup_queries = Vec::new();
    for marker in ["having executed:", "after having executed:"] {
        let mut rest = block.as_str();
        while let Some(offset) = rest.find(marker) {
            let tail = &rest[offset..];
            setup_queries.push(docstring_after(tail, marker)?);
            rest = &tail[marker.len()..];
        }
    }
    Ok(SourceScenario {
        setup_queries,
        query: docstring_after(&block, "executing query:")?,
    })
}

fn context<'a>(
    graph: &'a GraphStore,
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
        parameters: BTreeMap::new(),
        bookmark: Bookmark {
            term: 41,
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
        max_result_rows: 100_000,
        max_batch_rows: 1_024,
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

fn fixture_graph(source: &SourceScenario) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in &source.setup_queries {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        apply_mutations(&mut graph, &output.graph_mutations)?;
    }
    Ok(graph)
}

fn execute_generic_oracle(query: &str, graph: &GraphStore) -> Result<ExecutionOutput> {
    QueryEngine.execute(query, &mut context(graph, None, false))
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 41,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn cpu_backend(graph: &GraphStore) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(resident_image(graph)?)?;
    Ok(backend)
}

fn result_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(
            "read-only OPTIONAL acceptance query produced mutations",
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal(
                "OPTIONAL acceptance query produced an invalid result batch",
            ));
        }
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn assert_matches_oracle(
    case: &ManifestCase,
    oracle: &ExecutionOutput,
    actual: &ExecutionOutput,
) -> Result<()> {
    if oracle.result.schema != actual.result.schema
        || oracle.result.statistics != actual.result.statistics
        || oracle.result.truncated != actual.result.truncated
    {
        return Err(Error::internal(format!(
            "{} native result metadata differs from the generic CPU oracle",
            case.name
        )));
    }
    let mut expected = result_rows(oracle)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    let mut observed = result_rows(actual)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    expected.sort();
    observed.sort();
    if expected != observed {
        return Err(Error::internal(format!(
            "{} native rows differ from the generic CPU oracle: expected {expected:?}, got {observed:?}",
            case.name
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    CpuMasqueradesAsMetal,
    MutateRequestAfterFingerprint,
    MutatePredicateAfterFingerprint,
    StalePinnedGeneration,
    MissingReceipt,
    ForgedReceiptCardinality,
    ForgedPredicateReceiptCardinality,
    ForgedResultFingerprint,
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNullableRelationRequest>>,
    receipts: Mutex<Vec<Vec<ResidentNullableRelationReceipt>>>,
}

/// A fail-closed test boundary around the sealed nullable-relation command.
///
/// The root advertises Metal so `require_native_execution` cannot fall through to generic rows.
/// The pinned CPU-reference object reports CPU honestly; a separate fault mode deliberately lies
/// to prove that a CPU result cannot be accepted as Metal. Every host-visible scan, traversal,
/// filter, join, grouping, sort, and legacy node-pipeline entrypoint is rejected. The nullable
/// command may run only on the one immutable generation returned by `pin_project`.
struct StrictOptionalBackend {
    inner: Box<dyn ExecutionBackend>,
    actual_kind: BackendKind,
    pinned_kind: BackendKind,
    pinned: bool,
    fault: Fault,
    expected_bookmark: Bookmark,
    expected_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictOptionalBackend {
    fn cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(inner, BackendKind::Cpu, BackendKind::Cpu, Fault::None)
    }

    fn cpu_fault(inner: CpuBackend, fault: Fault) -> Result<Self> {
        let pinned_kind = if fault == Fault::CpuMasqueradesAsMetal {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        Self::new(inner, BackendKind::Cpu, pinned_kind, fault)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "OPTIONAL acceptance test did not construct a real Metal backend",
            ));
        }
        Self::new(inner, BackendKind::Metal, BackendKind::Metal, Fault::None)
    }

    fn new<B: ExecutionBackend + 'static>(
        inner: B,
        actual_kind: BackendKind,
        pinned_kind: BackendKind,
        fault: Fault,
    ) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("OPTIONAL backend omitted resident bookmark"))?;
        let expected_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("OPTIONAL backend omitted resident graph revision"))?;
        Ok(Self {
            inner: Box::new(inner),
            actual_kind,
            pinned_kind,
            pinned: false,
            fault,
            expected_bookmark,
            expected_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict OPTIONAL gate rejected host `{route}` route"),
        ))
    }
}

impl ExecutionBackend for StrictOptionalBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
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
            return self.reject("pin_project_twice_or_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "OPTIONAL pin did not preserve the admitted immutable generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        let expected_bookmark = if self.fault == Fault::StalePinnedGeneration {
            Bookmark {
                term: self.expected_bookmark.term,
                index: self.expected_bookmark.index.saturating_add(1),
            }
        } else {
            self.expected_bookmark
        };
        Ok(Box::new(Self {
            inner,
            actual_kind: self.actual_kind,
            pinned_kind: self.pinned_kind,
            pinned: true,
            fault: self.fault,
            expected_bookmark,
            expected_revision: self.expected_revision,
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
        if self.pinned && self.fault == Fault::StalePinnedGeneration {
            Some(self.expected_revision.saturating_add(1))
        } else {
            self.inner.resident_graph_revision(project)
        }
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        if self.pinned && self.fault == Fault::StalePinnedGeneration {
            Some(self.expected_bookmark)
        } else {
            self.inner.resident_bookmark(project)
        }
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
        self.inner.supports_nullable_relation_predicates()
    }

    fn supports_nullable_relation_scope_limit(&self) -> bool {
        self.inner.supports_nullable_relation_scope_limit()
    }

    fn supports_nullable_relation_existing_relationship(&self) -> bool {
        self.inner
            .supports_nullable_relation_existing_relationship()
    }

    fn supports_nullable_relation_relationship_endpoint_seed(&self) -> bool {
        self.inner
            .supports_nullable_relation_relationship_endpoint_seed()
    }

    fn supports_nullable_relation_string_property_equality(&self) -> bool {
        self.inner
            .supports_nullable_relation_string_property_equality()
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        if !self.pinned {
            return self.reject("execute_nullable_relation_on_root");
        }
        if request.generation.project != PROJECT
            || request.generation.bookmark != self.expected_bookmark
            || request.generation.graph_revision != self.expected_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "OPTIONAL command is not fenced to the pinned generation",
            ));
        }
        self.observations
            .complete_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        if self.fault == Fault::MutateRequestAfterFingerprint {
            let mut forged = request.clone();
            forged.capacities.max_output_rows = forged.capacities.max_output_rows.saturating_add(1);
            return self.inner.execute_nullable_relation(&forged, cancellation);
        }
        if self.fault == Fault::MutatePredicateAfterFingerprint {
            let mut forged = request.clone();
            let predicate = forged
                .predicate_program
                .filters
                .first_mut()
                .ok_or_else(|| Error::internal("predicate fault requires a predicate program"))?;
            predicate.predicate = ResidentNullableRelationPredicate::Constant(Some(true));
            return self.inner.execute_nullable_relation(&forged, cancellation);
        }
        let result = self
            .inner
            .execute_nullable_relation(request, cancellation)?;
        self.observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(result.clone().into_untrusted_parts().receipts);
        match self.fault {
            Fault::MissingReceipt => {
                let mut parts = result.into_untrusted_parts();
                parts.receipts.pop();
                Ok(ResidentNullableRelationResult::from_untrusted_parts(parts))
            }
            Fault::ForgedReceiptCardinality => {
                let mut parts = result.into_untrusted_parts();
                let receipt = parts
                    .receipts
                    .first_mut()
                    .ok_or_else(|| Error::internal("OPTIONAL result omitted every receipt"))?;
                receipt.output_cardinality = receipt.output_cardinality.saturating_add(1);
                Ok(ResidentNullableRelationResult::from_untrusted_parts(parts))
            }
            Fault::ForgedPredicateReceiptCardinality => {
                let mut parts = result.into_untrusted_parts();
                let receipt = parts
                    .receipts
                    .iter_mut()
                    .find(|receipt| {
                        matches!(
                            receipt.obligation.kind,
                            ResidentNullableRelationObligationKind::RelationFilter
                                | ResidentNullableRelationObligationKind::OptionalCandidateFilter
                        )
                    })
                    .ok_or_else(|| Error::internal("predicate result omitted a filter receipt"))?;
                receipt.output_cardinality = receipt.output_cardinality.saturating_add(1);
                Ok(ResidentNullableRelationResult::from_untrusted_parts(parts))
            }
            Fault::ForgedResultFingerprint => {
                let mut parts = result.into_untrusted_parts();
                parts.fingerprint.0[0] ^= 1;
                Ok(ResidentNullableRelationResult::from_untrusted_parts(parts))
            }
            Fault::None
            | Fault::CpuMasqueradesAsMetal
            | Fault::MutateRequestAfterFingerprint
            | Fault::MutatePredicateAfterFingerprint
            | Fault::StalePinnedGeneration => Ok(result),
        }
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

fn case_by_id(report_id: usize) -> Result<ManifestCase> {
    manifest()?
        .into_iter()
        .find(|case| case.report_id == report_id)
        .ok_or_else(|| Error::internal(format!("OPTIONAL manifest omitted report {report_id}")))
}

fn assert_route(case: &ManifestCase, observations: &RouteObservations) -> Result<()> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let calls = observations.complete_calls.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    if pins != 1 || calls != 1 || forbidden != 0 {
        return Err(Error::internal(format!(
            "{} did not use exactly one pinned complete resident command: pins={pins}, complete_calls={calls}, forbidden_calls={forbidden}",
            case.name
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let expected_proof_actions = requests.first().map_or(0, |request| {
        request
            .program
            .stages
            .len()
            .saturating_add(request.predicate_program.filters.len())
            .saturating_add(request.predicate_program.optional_groups.len())
    });
    if requests.len() != 1
        || requests[0].generation.project != PROJECT
        || requests[0].manifest.obligations.len() != expected_proof_actions
        || requests[0].capacities.stage_input_rows.len() != requests[0].program.stages.len()
        || requests[0].capacities.stage_candidate_rows.len() != requests[0].program.stages.len()
        || requests[0].capacities.stage_output_rows.len() != requests[0].program.stages.len()
    {
        return Err(Error::internal(format!(
            "{} did not retain exactly one complete project-fenced request",
            case.name
        )));
    }
    requests[0].validate()?;
    requests[0].scratch_bytes()?;
    Ok(())
}

fn assert_predicate_program(case: &ManifestCase, observations: &RouteObservations) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1 || requests[0].predicate_program.is_empty() {
        return Err(Error::internal(format!(
            "{} did not retain one nonempty nullable predicate program",
            case.name
        )));
    }
    Ok(())
}

fn assert_metal_receipts(case: &ManifestCase, observations: &RouteObservations) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1
        || receipts.len() != 1
        || receipts[0].len() != requests[0].manifest.obligations.len()
        || receipts[0]
            .iter()
            .any(|receipt| receipt.completion != ResidentDeviceCompletion::Metal)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} omitted exact device-authored Metal receipts", case.name),
        ));
    }
    Ok(())
}

fn assert_bound_relationship_limit_program(
    case: &ManifestCase,
    observations: &RouteObservations,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal(format!("{} omitted its nullable request", case.name)))?;
    if requests.len() != 1
        || !request.program.requires_stable_scope_limit()?
        || !request.program.requires_existing_relationship()?
    {
        return Err(Error::internal(format!(
            "{} omitted stable LIMIT or existing-relationship semantics",
            case.name
        )));
    }
    let requires_endpoint_seed = request
        .predicate_program
        .requires_relationship_endpoint_seed();
    if requires_endpoint_seed != (case.report_id == 513) {
        return Err(Error::internal(format!(
            "{} has an unexpected relationship-endpoint seed requirement",
            case.name
        )));
    }
    if case.report_id == 513 {
        let seed_stage = request
            .predicate_program
            .filters
            .iter()
            .find_map(|filter| match (&filter.placement, &filter.predicate) {
                (
                    irongraph::gpu::ResidentNullableRelationFilterPlacement::OptionalCandidates {
                        stage,
                    },
                    ResidentNullableRelationPredicate::RelationshipEndpoint { .. },
                ) => Some(usize::from(*stage)),
                _ => None,
            })
            .ok_or_else(|| {
                Error::internal(format!("{} omitted its endpoint-seed predicate", case.name))
            })?;
        if !matches!(
            request.program.stages.get(seed_stage),
            Some(ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Optional,
                ..
            })
        ) || !request
            .predicate_program
            .optional_groups
            .iter()
            .any(|group| {
                usize::from(group.first_stage) == seed_stage
                    && usize::from(group.last_stage) > seed_stage
                    && matches!(
                        request.program.stages.get(usize::from(group.last_stage)),
                        Some(ResidentNullableRelationStage::Expand {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            ..
                        })
                    )
            })
        {
            return Err(Error::internal(format!(
                "{} did not retain one atomic endpoint-seed OPTIONAL group",
                case.name
            )));
        }
        let mut forged = request.clone();
        let endpoint = forged
            .predicate_program
            .filters
            .iter_mut()
            .find_map(|filter| match &mut filter.predicate {
                ResidentNullableRelationPredicate::RelationshipEndpoint { endpoint, .. } => {
                    Some(endpoint)
                }
                _ => None,
            })
            .ok_or_else(|| Error::internal("nullable endpoint seed disappeared"))?;
        *endpoint = match endpoint {
            ResidentNullableRelationshipEndpoint::Source => {
                ResidentNullableRelationshipEndpoint::Target
            }
            ResidentNullableRelationshipEndpoint::Target
            | ResidentNullableRelationshipEndpoint::Either => {
                ResidentNullableRelationshipEndpoint::Source
            }
        };
        if forged.validate().is_ok() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{} accepted an endpoint-seed fingerprint forgery",
                    case.name
                ),
            ));
        }
    }
    let (stage, bindings) = request
        .program
        .stages
        .iter()
        .enumerate()
        .find_map(|(stage, candidate)| match candidate {
            ResidentNullableRelationStage::ScopeProject { bindings }
                if bindings.iter().all(|binding| binding.row_limit == Some(1)) =>
            {
                Some((stage, bindings))
            }
            _ => None,
        })
        .ok_or_else(|| Error::internal(format!("{} omitted WITH LIMIT 1", case.name)))?;
    if bindings.is_empty()
        || request.capacities.stage_output_rows[stage] > 1
        || request
            .capacities
            .stage_input_rows
            .get(stage.saturating_add(1))
            != Some(&request.capacities.stage_output_rows[stage])
    {
        return Err(Error::internal(format!(
            "{} has an invalid stable LIMIT capacity chain",
            case.name
        )));
    }
    let mut forged = request.clone();
    let ResidentNullableRelationStage::ScopeProject { bindings } =
        &mut forged.program.stages[stage]
    else {
        return Err(Error::internal("nullable LIMIT stage disappeared"));
    };
    for binding in bindings {
        binding.row_limit = None;
    }
    if forged.validate().is_ok() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} accepted a stable-LIMIT fingerprint forgery", case.name),
        ));
    }
    Ok(())
}

fn assert_fixed_multi_hop_program(
    case: &ManifestCase,
    graph: &GraphStore,
    observations: &RouteObservations,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal(format!("{} omitted its nullable request", case.name)))?;
    if requests.len() != 1
        || receipts.len() != 1
        || request.predicate_program.optional_groups.len() != 1
        || request
            .predicate_program
            .requires_string_property_equality()
            != (case.report_id == 515)
    {
        return Err(Error::internal(format!(
            "{} omitted its exact fixed multi-hop command contract",
            case.name
        )));
    }
    let group = request.predicate_program.optional_groups[0];
    let first = usize::from(group.first_stage);
    let last = usize::from(group.last_stage);
    if last != first.saturating_add(1)
        || !request.program.stages[first..=last].iter().all(|stage| {
            matches!(
                stage,
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    ..
                }
            )
        })
        || request.capacities.stage_input_rows[last] != request.capacities.stage_output_rows[first]
    {
        return Err(Error::internal(format!(
            "{} did not retain two capacity-chained OPTIONAL expansions",
            case.name
        )));
    }
    let group_receipt = receipts[0]
        .iter()
        .find(|receipt| {
            receipt.obligation.kind == ResidentNullableRelationObligationKind::AtomicOptionalGroup
        })
        .ok_or_else(|| Error::internal(format!("{} omitted its group receipt", case.name)))?;
    let expected_group = match case.report_id {
        515 | 517 => (1, 1, 1, 0, 1),
        516 => (1, 0, 0, 1, 1),
        _ => return Err(Error::internal("unexpected fixed multi-hop report ID")),
    };
    if (
        group_receipt.input_cardinality,
        group_receipt.matched_input_cardinality,
        group_receipt.candidate_cardinality,
        group_receipt.null_extension_cardinality,
        group_receipt.output_cardinality,
    ) != expected_group
        || group_receipt.completion != ResidentDeviceCompletion::CpuReference
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} has an invalid atomic OPTIONAL receipt", case.name),
        ));
    }
    if case.report_id == 515 {
        let expected_property = graph
            .catalog()
            .property("name")
            .ok_or_else(|| Error::internal("report 515 fixture omitted property `name`"))?;
        let filter = request
            .predicate_program
            .filters
            .first()
            .ok_or_else(|| Error::internal("report 515 omitted its string filter"))?;
        let ResidentNullableRelationPredicate::CompareString { left, right, .. } =
            &filter.predicate
        else {
            return Err(Error::internal(
                "report 515 did not compile exact string equality",
            ));
        };
        let values = [left, right];
        if values
            .iter()
            .filter(|value| {
                matches!(
                    value,
                    ResidentNullableRelationPredicateValue::String(value) if value.as_ref() == "A"
                )
            })
            .count()
            != 1
            || values
                .iter()
                .filter(|value| {
                    matches!(
                        value,
                        ResidentNullableRelationPredicateValue::StringProperty {
                            kind: irongraph::gpu::ResidentNullableRelationBindingKind::Node,
                            property,
                            ..
                        } if *property == expected_property
                    )
                })
                .count()
                != 1
            || !graph.node_property_is_string(expected_property)
        {
            return Err(Error::internal(
                "report 515 string predicate lost its literal, catalog property, or type",
            ));
        }
        let filter_receipt = receipts[0]
            .iter()
            .find(|receipt| {
                receipt.obligation.kind == ResidentNullableRelationObligationKind::RelationFilter
            })
            .ok_or_else(|| Error::internal("report 515 omitted its filter receipt"))?;
        if (
            filter_receipt.input_cardinality,
            filter_receipt.matched_input_cardinality,
            filter_receipt.candidate_cardinality,
            filter_receipt.null_extension_cardinality,
            filter_receipt.output_cardinality,
        ) != (3, 2, 1, 0, 1)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "report 515 string-filter receipt has the wrong 3VL cardinalities",
            ));
        }
        let mut forged = request.clone();
        let ResidentNullableRelationPredicate::CompareString { left, right, .. } =
            &mut forged.predicate_program.filters[0].predicate
        else {
            return Err(Error::internal("report 515 string predicate disappeared"));
        };
        let literal = [left, right]
            .into_iter()
            .find_map(|value| match value {
                ResidentNullableRelationPredicateValue::String(value) => Some(value),
                _ => None,
            })
            .ok_or_else(|| Error::internal("report 515 string literal disappeared"))?;
        *literal = Arc::<str>::from("forged");
        if forged.validate().is_ok() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "report 515 accepted a string-predicate fingerprint forgery",
            ));
        }
    } else if !request.predicate_program.filters.is_empty() {
        return Err(Error::internal(format!(
            "{} unexpectedly requires a scalar predicate kernel",
            case.name
        )));
    }
    let mut forged = request.clone();
    forged.predicate_program.optional_groups[0].last_stage = group.first_stage;
    if forged.validate().is_ok() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} accepted an atomic-group fingerprint forgery", case.name),
        ));
    }
    request.validate()?;
    request.scratch_bytes()?;
    Ok(())
}

fn execute_strict(
    case: &ManifestCase,
    source: &SourceScenario,
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(case, &oracle, &actual)?;
    assert_route(case, &observations)
}

fn execute_strict_ordered(
    name: &str,
    source: &SourceScenario,
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let case = ManifestCase {
        report_id: 0,
        feature: "adversarial/positive-optional".to_owned(),
        name: name.to_owned(),
    };
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let expected = result_rows(&oracle)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(&case, &oracle, &actual)?;
    let observed = result_rows(&actual)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    if observed != expected {
        return Err(Error::internal(format!(
            "{name} lost stable parent-major row order: expected {expected:?}, got {observed:?}"
        )));
    }
    assert_route(&case, &observations)
}

fn positive_mixed_optional_sources() -> Vec<SourceScenario> {
    let setup_queries = vec![
        "CREATE (a1:A {id: 1}), (a2:A {id: 2}), (b1:B {id: 1}), (b2:B {id: 2}), (b3:B {id: 3})"
            .to_owned(),
        "MATCH (a1:A {id: 1}), (a2:A {id: 2}), (b1:B {id: 1}), (b2:B {id: 2}), (b3:B {id: 3}) CREATE (a1)-[:T {id: 11}]->(b1), (a1)-[:T {id: 12}]->(b2), (a2)-[:DECOY]->(b3)"
            .to_owned(),
    ];
    vec![
        SourceScenario {
            setup_queries: setup_queries.clone(),
            query: "MATCH (a:A) OPTIONAL MATCH (a)-[r:T]->(b:B) RETURN a, b, r".to_owned(),
        },
        SourceScenario {
            setup_queries,
            query: "MATCH (a:A) MATCH (b:B) OPTIONAL MATCH (a)-[r:T]->(b) RETURN a, b, r"
                .to_owned(),
        },
    ]
}

fn predicate_optional_adversarial_sources() -> Vec<SourceScenario> {
    let setup_queries = vec![
        "CREATE (a1:A {id: 1}), (a2:A {id: 2}), (b0:B {id: 0}), (b2:B {id: 2}), (bn:B {marker: 1}), (x1:X {val: 10}), (x2:X {val: 1}), (y11:Y {id: 11}), (y12:Y {id: 12}), (y2:Y {id: 2}), (z0:Z {val: 0}), (z5:Z {val: 5})"
            .to_owned(),
        "MATCH (a1:A {id: 1}), (b0:B {id: 0}), (b2:B {id: 2}), (bn:B {marker: 1}), (x1:X {val: 10}), (x2:X {val: 1}), (y11:Y {id: 11}), (y12:Y {id: 12}), (y2:Y {id: 2}), (z0:Z {val: 0}), (z5:Z {val: 5}) CREATE (a1)-[:T]->(b0), (a1)-[:T]->(b2), (a1)-[:T]->(bn), (x1)-[:E1]->(y11), (x1)-[:E1]->(y12), (x2)-[:E1]->(y2), (y2)-[:E2]->(z0), (y2)-[:E2]->(z5)"
            .to_owned(),
    ];
    vec![
        SourceScenario {
            setup_queries: setup_queries.clone(),
            query: "MATCH (a:A) WHERE a.id IS NOT NULL OPTIONAL MATCH (a)-[r:T]->(b:B) WHERE a.id < b.id RETURN a, b, r"
                .to_owned(),
        },
        SourceScenario {
            setup_queries: setup_queries.clone(),
            query: "MATCH (a:A) OPTIONAL MATCH (a)-[r:T]->(b:B) WHERE null RETURN a, b, r"
                .to_owned(),
        },
        SourceScenario {
            setup_queries: setup_queries.clone(),
            query: "MATCH (a:A) OPTIONAL MATCH (a)-[r:T]->(b:B) WITH b WHERE r IS NULL RETURN b"
                .to_owned(),
        },
        SourceScenario {
            setup_queries,
            query: "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y)-[:E2]->(z:Z) WHERE x.val < z.val RETURN x, y, z"
                .to_owned(),
        },
    ]
}

fn graph6_and_static_null_path_sources() -> Vec<(ManifestCase, SourceScenario)> {
    vec![
        (
            ManifestCase {
                report_id: 1565,
                feature: "expressions/graph/Graph6.feature".to_owned(),
                name: "[6] Statically access a property of a optional non-null relationship"
                    .to_owned(),
            },
            SourceScenario {
                setup_queries: vec!["CREATE ()-[:REL {existing: 42, missing: null}]->()"
                    .to_owned()],
                query:
                    "OPTIONAL MATCH ()-[r]->() RETURN r.missing, r.missingToo, r.existing"
                        .to_owned(),
            },
        ),
        (
            ManifestCase {
                report_id: 1566,
                feature: "expressions/graph/Graph6.feature".to_owned(),
                name: "[7] Statically access a property of a null relationship".to_owned(),
            },
            SourceScenario {
                setup_queries: Vec::new(),
                query: "OPTIONAL MATCH ()-[r]->() RETURN r.missing".to_owned(),
            },
        ),
        (
            ManifestCase {
                report_id: 2002,
                feature: "expressions/path/Path1.feature".to_owned(),
                name: "[1] `nodes()` on null path".to_owned(),
            },
            SourceScenario {
                setup_queries: Vec::new(),
                query: "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p), nodes(null)"
                    .to_owned(),
            },
        ),
        (
            ManifestCase {
                report_id: 2005,
                feature: "expressions/path/Path2.feature".to_owned(),
                name: "[3] `relationships()` on null path".to_owned(),
            },
            SourceScenario {
                setup_queries: Vec::new(),
                query: "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN relationships(p), relationships(null)"
                    .to_owned(),
            },
        ),
    ]
}

fn assert_graph6_or_static_null_path_program(
    case: &ManifestCase,
    observations: &RouteObservations,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(format!(
            "report {} omitted its single nullable request",
            case.report_id
        )));
    };
    match case.report_id {
        1565 | 1566 => {
            let [group] = request.predicate_program.optional_groups.as_slice() else {
                return Err(Error::internal(format!(
                    "report {} omitted its atomic unanchored OPTIONAL group",
                    case.report_id
                )));
            };
            if (group.first_stage, group.last_stage) != (0, 1)
                || !matches!(
                    request.program.stages.as_slice(),
                    [
                        ResidentNullableRelationStage::NodeScan {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            ..
                        },
                        ResidentNullableRelationStage::Expand {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            ..
                        },
                        ResidentNullableRelationStage::FinalProject { .. }
                    ]
                )
            {
                return Err(Error::internal(format!(
                    "report {} lost its exact two-stage atomic seed",
                    case.report_id
                )));
            }
        }
        2002 | 2005 => {
            if !request.predicate_program.optional_groups.is_empty()
                || !matches!(
                    request.program.stages.as_slice(),
                    [
                        ResidentNullableRelationStage::NodeScan {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            labels: irongraph::gpu::ResidentNullableNodeDomain::KnownEmpty,
                            ..
                        },
                        ResidentNullableRelationStage::Expand {
                            mode: ResidentNullableRelationMatchMode::Optional,
                            ..
                        },
                        ResidentNullableRelationStage::FinalProject { .. }
                    ]
                )
            {
                return Err(Error::internal(format!(
                    "report {} lost its device-native static-null path proof",
                    case.report_id
                )));
            }
        }
        _ => return Err(Error::internal("unexpected Graph6/path report ID")),
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the certified report and pinned openCypher TCK checkout"]
fn exact_manifest_matches_the_certified_report_and_pinned_tck_sources() -> Result<()> {
    let cases = manifest()?;
    let report = certified_report()?;
    assert_eq!(cases.len(), 52, "OPTIONAL manifest cardinality changed");
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);
    assert_eq!(report.metal_passed, 3_184);

    let manifest_ids = cases.iter().map(|case| case.report_id).collect::<Vec<_>>();
    let mut unique_ids = manifest_ids.clone();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    assert_eq!(unique_ids.len(), 52, "OPTIONAL report IDs are not unique");

    for case in &cases {
        let matches = report
            .scenarios
            .iter()
            .enumerate()
            .filter(|(_, scenario)| {
                scenario.name == case.name && Path::new(&scenario.path).ends_with(&case.feature)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "report must contain exactly one path/name match for {} / {}",
            case.feature,
            case.name
        );
        let (actual_index, scenario) = matches[0];
        assert_eq!(
            case.report_id, actual_index,
            "stored OPTIONAL ID drifted from its authoritative report array index for {} / {}",
            case.feature, case.name
        );
        let indexed = report
            .scenarios
            .get(case.report_id)
            .ok_or_else(|| Error::internal(format!("report omitted index {}", case.report_id)))?;
        assert_eq!(indexed.name, case.name, "report index {}", case.report_id);
        assert!(
            Path::new(&scenario.path).ends_with(&case.feature),
            "report index {} path mismatch: {}",
            case.report_id,
            scenario.path
        );
        assert!(
            scenario.cpu_passed,
            "{} lost its CPU oracle pass",
            case.name
        );
        assert!(scenario.shared_failures.is_empty());
        assert!(scenario.cpu_failures.is_empty());
        let source = source_scenario(case)?;
        if scenario.metal_passed {
            assert!(scenario.fully_conformant);
            assert!(scenario.metal_failures.is_empty());
        } else {
            assert!(!scenario.fully_conformant);
            assert_eq!(scenario.metal_failures.len(), 1);
            assert!(scenario.metal_failures[0].contains(
                "GpuAdmissionFailure: active GPU execution class has no complete resident implementation for this query plan"
            ));
            let report_query = query_from_failure(scenario)?;
            assert_eq!(
                normalize_query(&source.query),
                normalize_query(&report_query)
            );
        }
        assert!(source.query.contains("OPTIONAL MATCH"));
        assert!(!contains_variable_length_relationship(&source.query));
        assert!(
            !source
                .query
                .split_whitespace()
                .any(|token| token == "MERGE")
        );
        assert!(
            !source
                .query
                .split_whitespace()
                .any(|token| { matches!(token, "CREATE" | "SET" | "REMOVE" | "DELETE") })
        );
    }
    Ok(())
}

#[test]
fn exact_manifest_covers_the_required_optional_semantic_families() -> Result<()> {
    let ids = manifest()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<std::collections::BTreeSet<_>>();
    let families: &[(&str, &[usize])] = &[
        (
            "null input and per-row null extension",
            &[369, 370, 518, 583],
        ),
        ("attached OPTIONAL WHERE", &[578, 579, 581, 582, 583, 584]),
        ("later WHERE after OPTIONAL", &[1234, 1235]),
        (
            "bound nodes and relationships",
            &[510, 511, 512, 513, 514, 519],
        ),
        (
            "zero, one, and multiple candidates",
            &[510, 517, 519, 537, 538, 539],
        ),
        (
            "directed, reverse, undirected, and self-loop",
            &[512, 519, 533, 536],
        ),
        ("fixed multi-hop", &[515, 516, 517, 584]),
        (
            "multiple OPTIONAL boundaries",
            &[529, 530, 534, 535, 585, 1268, 1587],
        ),
        (
            "nullable node/relationship properties",
            &[1561, 1562, 1565, 1566],
        ),
        (
            "label and relationship-type functions",
            &[1537, 1542, 1543, 1559],
        ),
        (
            "aggregation and collection tails",
            &[537, 538, 539, 1267, 1268, 1282, 1684],
        ),
        (
            "map/list/path tails",
            &[524, 525, 526, 1580, 1583, 1587, 2002, 2005],
        ),
    ];
    for (family, required) in families {
        assert!(
            required.iter().all(|id| ids.contains(id)),
            "OPTIONAL manifest lost coverage for {family}"
        );
    }
    Ok(())
}

/// Replays every exact fixture and primary query through the generic engine. This is deliberately
/// only the CPU oracle; it does not satisfy the native CPU or Metal acceptance gates below.
#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn generic_cpu_oracle_replays_all_52_exact_tck_scenarios() -> Result<()> {
    for case in manifest()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let output = execute_generic_oracle(&source.query, &graph).map_err(|error| {
            Error::internal(format!(
                "generic CPU oracle failed report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
        result_rows(&output)?;
    }
    Ok(())
}

#[test]
fn exact_first_nine_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let all = manifest()?;
    assert_eq!(
        all.len(),
        52,
        "OPTIONAL literal manifest cardinality changed"
    );
    let selected = exact_first_nine()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_FIRST_NINE_IDS);
    Ok(())
}

#[test]
fn exact_predicate_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let selected = exact_predicate_tranche()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_PREDICATE_IDS);
    Ok(())
}

#[test]
fn exact_512_514_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let selected = exact_bound_relationship_first_tranche()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_BOUND_RELATIONSHIP_FIRST_IDS);
    Ok(())
}

#[test]
fn exact_512_through_514_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let selected = exact_bound_relationship_tranche()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_BOUND_RELATIONSHIP_IDS);
    Ok(())
}

#[test]
fn exact_515_through_517_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let selected = exact_fixed_multi_hop_tranche()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_FIXED_MULTI_HOP_IDS);
    Ok(())
}

#[test]
fn exact_516_517_metal_selector_preserves_the_literal_52_case_manifest() -> Result<()> {
    let selected = exact_fixed_multi_hop_metal_tranche()?
        .into_iter()
        .map(|case| case.report_id)
        .collect::<Vec<_>>();
    assert_eq!(selected, EXACT_FIXED_MULTI_HOP_METAL_IDS);
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_matches_the_generic_oracle_for_exact_first_nine() -> Result<()> {
    for case in exact_first_nine()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "exact-nine native CPU failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_certifies_report_534() -> Result<()> {
    let case = case_by_id(534)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
    execute_strict(&case, &source, &graph, &backend).map_err(|error| {
        Error::internal(format!(
            "report 534 native CPU failure at {}: {}",
            case.name, error.message
        ))
    })
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_certifies_report_511() -> Result<()> {
    let case = case_by_id(511)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
    execute_strict(&case, &source, &graph, &backend).map_err(|error| {
        Error::internal(format!(
            "report 511 native CPU failure at {}: {}",
            case.name, error.message
        ))
    })
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_certifies_exact_512_and_514() -> Result<()> {
    for case in exact_bound_relationship_first_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "bound-relationship native CPU failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
        assert_bound_relationship_limit_program(&case, &observations)?;
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_certifies_exact_512_through_514() -> Result<()> {
    for case in exact_bound_relationship_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "bound-relationship native CPU failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
        assert_bound_relationship_limit_program(&case, &observations)?;
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn native_cpu_reference_certifies_exact_515_through_517() -> Result<()> {
    let mut failures = Vec::new();
    for case in exact_fixed_multi_hop_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        if let Err(error) = execute_strict(&case, &source, &graph, &backend)
            .and_then(|()| assert_fixed_multi_hop_program(&case, &graph, &observations))
        {
            failures.push(format!(
                "report {} / {}: {}",
                case.report_id, case.name, error.message
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "fixed multi-hop native CPU passed {}/{}: {}",
            EXACT_FIXED_MULTI_HOP_IDS
                .len()
                .saturating_sub(failures.len()),
            EXACT_FIXED_MULTI_HOP_IDS.len(),
            failures.join(" | ")
        )))
    }
}

#[test]
#[ignore = "red acceptance gate: nullable predicate stages must lower natively"]
fn native_cpu_reference_certifies_exact_predicate_tranche() -> Result<()> {
    for case in exact_predicate_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "nullable predicate CPU failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
        assert_predicate_program(&case, &observations)?;
    }
    Ok(())
}

#[test]
fn native_cpu_proves_positive_mixed_optional_expansion_and_stable_order() -> Result<()> {
    for (index, source) in positive_mixed_optional_sources().into_iter().enumerate() {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        execute_strict_ordered(
            &format!("positive mixed OPTIONAL CPU adversary {index}"),
            &source,
            &graph,
            &backend,
        )?;
    }
    Ok(())
}

#[test]
fn native_cpu_proves_predicate_3vl_scope_and_atomic_optional_semantics() -> Result<()> {
    for (index, source) in predicate_optional_adversarial_sources()
        .into_iter()
        .enumerate()
    {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        let name = format!("nullable predicate CPU adversary {index}");
        execute_strict_ordered(&name, &source, &graph, &backend)
            .map_err(|error| Error::internal(format!("{name} failed: {}", error.message)))?;
        let case = ManifestCase {
            report_id: index,
            feature: "adversarial/nullable-predicate".to_owned(),
            name,
        };
        assert_predicate_program(&case, &observations)?;
        if index == 3 {
            let requests = observations
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(requests[0].predicate_program.optional_groups.len(), 1);
        }
    }
    Ok(())
}

#[test]
fn native_cpu_proves_inline_string_property_3vl_before_atomic_optional() -> Result<()> {
    let source = SourceScenario {
        setup_queries: vec![
            "CREATE (:Probe {name: 'A'}), (:Probe {name: 'B'}), (:Probe)".to_owned(),
        ],
        query: "MATCH (a:Probe {name: 'A'}) OPTIONAL MATCH (a)-[:R]->()-[:R]->(z) RETURN a, z"
            .to_owned(),
    };
    let graph = fixture_graph(&source)?;
    let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
    let observations = backend.observations();
    execute_strict_ordered(
        "inline string property 3VL before atomic OPTIONAL",
        &source,
        &graph,
        &backend,
    )?;
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let filter_receipt = receipts
        .first()
        .and_then(|receipts| {
            receipts.iter().find(|receipt| {
                receipt.obligation.kind == ResidentNullableRelationObligationKind::RelationFilter
            })
        })
        .ok_or_else(|| Error::internal("string 3VL regression omitted its filter receipt"))?;
    if requests.len() != 1
        || !requests[0]
            .predicate_program
            .requires_string_property_equality()
        || requests[0].predicate_program.optional_groups.len() != 1
        || (
            filter_receipt.input_cardinality,
            filter_receipt.matched_input_cardinality,
            filter_receipt.candidate_cardinality,
            filter_receipt.null_extension_cardinality,
            filter_receipt.output_cardinality,
        ) != (3, 1, 1, 1, 1)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "string 3VL regression lost False/True/Null receipt cardinalities",
        ));
    }
    Ok(())
}

#[test]
fn native_cpu_certifies_graph6_and_static_null_path_remaining_cluster() -> Result<()> {
    for (case, source) in graph6_and_static_null_path_sources() {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "report {} strict CPU failure at {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
        assert_graph6_or_static_null_path_program(&case, &observations)?;
    }
    Ok(())
}

#[test]
fn graph6_and_static_null_path_near_misses_fail_before_native_execution() -> Result<()> {
    for query in [
        "OPTIONAL MATCH ()-[r:T]->() RETURN r",
        "OPTIONAL MATCH ()-[r]-() RETURN r",
        "WITH null AS a OPTIONAL MATCH p = (a)<-[r]-() RETURN nodes(p)",
        "WITH null AS a OPTIONAL MATCH p = (a)-[r:T]->() RETURN relationships(p)",
        "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN length(p)",
    ] {
        let source = SourceScenario {
            setup_queries: Vec::new(),
            query: query.to_owned(),
        };
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(&graph, Some(&backend), true))
            .err()
            .ok_or_else(|| {
                Error::internal(format!(
                    "near-miss unexpectedly used a native route: {query}"
                ))
            })?;
        if error.code != ErrorCode::GpuAdmissionFailure
            || observations.complete_calls.load(Ordering::SeqCst) != 0
            || observations.forbidden_calls.load(Ordering::SeqCst) != 0
        {
            return Err(Error::internal(format!(
                "near-miss did not fail closed before execution: {query}: {error:?}"
            )));
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device"]
fn real_metal_proves_positive_mixed_optional_expansion_and_stable_order() -> Result<()> {
    for (index, source) in positive_mixed_optional_sources().into_iter().enumerate() {
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        execute_strict_ordered(
            &format!("positive mixed OPTIONAL Metal adversary {index}"),
            &source,
            &graph,
            &backend,
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device"]
fn real_metal_certifies_graph6_and_static_null_path_remaining_cluster() -> Result<()> {
    for (case, source) in graph6_and_static_null_path_sources() {
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        let observations = backend.observations();
        execute_strict(&case, &source, &graph, &backend)
            .and_then(|()| assert_graph6_or_static_null_path_program(&case, &observations))
            .and_then(|()| assert_metal_receipts(&case, &observations))
            .map_err(|error| {
                Error::internal(format!(
                    "report {} real-Metal failure at {}: {}",
                    case.report_id, case.name, error.message
                ))
            })?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external acceptance gate: requires pinned TCK fixtures and a real Metal device"]
fn real_metal_matches_the_generic_oracle_for_exact_first_nine() -> Result<()> {
    for case in exact_first_nine()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "exact-nine real-Metal failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external acceptance gate: requires pinned TCK fixtures and a real Metal device"]
fn real_metal_certifies_exact_516_and_517() -> Result<()> {
    let mut failures = Vec::new();
    for case in exact_fixed_multi_hop_metal_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        let observations = backend.observations();
        match execute_strict(&case, &source, &graph, &backend)
            .and_then(|()| assert_metal_receipts(&case, &observations))
        {
            Ok(()) => {}
            Err(error) => failures.push(format!(
                "report {} / {}: {}",
                case.report_id, case.name, error.message
            )),
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "fixed multi-hop real Metal passed {}/{}: {}",
            EXACT_FIXED_MULTI_HOP_METAL_IDS
                .len()
                .saturating_sub(failures.len()),
            EXACT_FIXED_MULTI_HOP_METAL_IDS.len(),
            failures.join(" | ")
        )))
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red acceptance gate: requires native Metal string-property predicate support"]
fn real_metal_certifies_report_515() -> Result<()> {
    let case = case_by_id(515)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_project(resident_image(&graph)?)?;
    let backend = StrictOptionalBackend::metal(metal)?;
    let observations = backend.observations();
    execute_strict(&case, &source, &graph, &backend)
        .and_then(|()| assert_metal_receipts(&case, &observations))
        .map_err(|error| {
            Error::internal(format!(
                "report 515 real-Metal failure at {}: {}",
                case.name, error.message
            ))
        })
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external acceptance gate: requires pinned TCK fixtures and a real Metal device"]
fn real_metal_certifies_report_534() -> Result<()> {
    let case = case_by_id(534)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_project(resident_image(&graph)?)?;
    let backend = StrictOptionalBackend::metal(metal)?;
    execute_strict(&case, &source, &graph, &backend).map_err(|error| {
        Error::internal(format!(
            "report 534 real-Metal failure at {}: {}",
            case.name, error.message
        ))
    })
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "external acceptance gate: requires pinned TCK fixtures and a real Metal device"]
fn real_metal_certifies_report_511() -> Result<()> {
    let case = case_by_id(511)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_project(resident_image(&graph)?)?;
    let backend = StrictOptionalBackend::metal(metal)?;
    execute_strict(&case, &source, &graph, &backend).map_err(|error| {
        Error::internal(format!(
            "report 511 real-Metal failure at {}: {}",
            case.name, error.message
        ))
    })
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red acceptance gate: nullable predicate stages must lower on real Metal"]
fn real_metal_certifies_exact_predicate_tranche() -> Result<()> {
    for case in exact_predicate_tranche()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "nullable predicate real-Metal failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: all 52 OPTIONAL scenarios require one complete native CPU-reference command"]
fn native_cpu_reference_matches_the_generic_oracle_for_all_52() -> Result<()> {
    for case in manifest()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "native CPU-reference failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "red acceptance gate: all 52 OPTIONAL scenarios require one complete real-Metal command"]
fn real_metal_matches_the_generic_oracle_for_all_52() -> Result<()> {
    for case in manifest()? {
        let source = source_scenario(&case)?;
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        execute_strict(&case, &source, &graph, &backend).map_err(|error| {
            Error::internal(format!(
                "real-Metal failure at report {} / {}: {}",
                case.report_id, case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
#[ignore = "real Metal OPTIONAL acceptance must run on macOS with --features accelerator"]
fn real_metal_matches_the_generic_oracle_for_all_52_requires_metal() -> Result<()> {
    Err(Error::internal(
        "run the 52-scenario OPTIONAL gate on a real Metal device",
    ))
}

fn relationship_isomorphism_sources() -> Vec<(ManifestCase, SourceScenario)> {
    let setup_queries =
        vec!["CREATE (:A)-[:T1]->(l:Looper), (l)-[:LOOP]->(l), (l)-[:T2]->(:B)".to_owned()];
    vec![
        (
            ManifestCase {
                report_id: 357,
                feature: "clauses/match/Match3.feature".to_owned(),
                name: "[15] Mixing directed and undirected pattern parts with self-relationship, simple"
                    .to_owned(),
            },
            SourceScenario {
                setup_queries: setup_queries.clone(),
                query: "MATCH (x:A)-[r1]->(y)-[r2]-(z) RETURN x, r1, y, r2, z"
                    .to_owned(),
            },
        ),
        (
            ManifestCase {
                report_id: 358,
                feature: "clauses/match/Match3.feature".to_owned(),
                name: "[16] Mixing directed and undirected pattern parts with self-relationship, undirected"
                    .to_owned(),
            },
            SourceScenario {
                setup_queries: setup_queries.clone(),
                query: "MATCH (x)-[r1]-(y)-[r2]-(z) RETURN x, r1, y, r2, z".to_owned(),
            },
        ),
        (
            ManifestCase {
                report_id: usize::MAX,
                feature: "adversarial/anonymous-relationship-isomorphism".to_owned(),
                name: "anonymous relationship slots retain DifferentRelationships semantics"
                    .to_owned(),
            },
            SourceScenario {
                setup_queries,
                query: "MATCH (x)-[]-(y)-[]-(z) RETURN x, y, z".to_owned(),
            },
        ),
    ]
}

fn assert_relationship_isomorphism_program(
    case: &ManifestCase,
    observations: &RouteObservations,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests.first().ok_or_else(|| {
        Error::internal(format!(
            "{} omitted its sealed nullable-relation request",
            case.name
        ))
    })?;
    if requests.len() != 1 || !request.predicate_program.is_empty() {
        return Err(Error::internal(format!(
            "{} did not compile as one predicate-free nullable relation",
            case.name
        )));
    }
    let expansions = request
        .program
        .stages
        .iter()
        .filter_map(|stage| match stage {
            ResidentNullableRelationStage::Expand {
                mode,
                uniqueness_group,
                relationship,
                different_from,
                ..
            } => Some((
                *mode,
                *uniqueness_group,
                *relationship,
                different_from.clone(),
            )),
            _ => None,
        })
        .collect::<Vec<_>>();
    if expansions.len() != 2
        || expansions.iter().any(|(mode, _, relationship, _)| {
            *mode != ResidentNullableRelationMatchMode::Mandatory || relationship.is_none()
        })
        || expansions[0].1 != expansions[1].1
        || expansions[0].2 == expansions[1].2
        || !expansions[0].3.is_empty()
        || expansions[1].3 != vec![expansions[0].2.expect("relationship slot checked above")]
    {
        return Err(Error::internal(format!(
            "{} did not retain two distinct relationship slots across both native expansions",
            case.name
        )));
    }
    request.validate()?;
    request.scratch_bytes()?;
    Ok(())
}

fn assert_relationship_isomorphism_rows(
    case: &ManifestCase,
    output: &ExecutionOutput,
) -> Result<()> {
    let rows = result_rows(output)?;
    let expected_rows = match case.report_id {
        357 => 2,
        358 | usize::MAX => 6,
        _ => {
            return Err(Error::internal(format!(
                "{} is not a relationship-isomorphism gate",
                case.name
            )));
        }
    };
    if rows.len() != expected_rows {
        return Err(Error::internal(format!(
            "{} produced {} rows instead of the exact relationship-isomorphism count {expected_rows}",
            case.name,
            rows.len()
        )));
    }
    if case.report_id == usize::MAX {
        return Ok(());
    }

    let mut observed_pairs = BTreeSet::new();
    for row in &rows {
        let (Some(ResultValue::Relationship(first)), Some(ResultValue::Relationship(second))) =
            (row.get(1), row.get(3))
        else {
            return Err(Error::internal(format!(
                "{} did not publish relationships in columns r1/r2",
                case.name
            )));
        };
        if first.id == second.id {
            return Err(Error::internal(format!(
                "{} reused relationship {:?} in both fixed hops",
                case.name, first.id
            )));
        }
        observed_pairs.insert((
            first.relationship_type.clone(),
            second.relationship_type.clone(),
        ));
    }
    let expected_pairs = match case.report_id {
        357 => [("T1", "LOOP"), ("T1", "T2")]
            .into_iter()
            .map(|(first, second)| (first.to_owned(), second.to_owned()))
            .collect(),
        358 => [
            ("T1", "LOOP"),
            ("T1", "T2"),
            ("LOOP", "T1"),
            ("LOOP", "T2"),
            ("T2", "T1"),
            ("T2", "LOOP"),
        ]
        .into_iter()
        .map(|(first, second)| (first.to_owned(), second.to_owned()))
        .collect(),
        _ => unreachable!("relationship-isomorphism report ID checked above"),
    };
    if observed_pairs != expected_pairs {
        return Err(Error::internal(format!(
            "{} produced relationship-type pairs {observed_pairs:?} instead of {expected_pairs:?}",
            case.name
        )));
    }
    Ok(())
}

fn assert_relationship_isomorphism_receipts(
    case: &ManifestCase,
    observations: &RouteObservations,
) -> Result<()> {
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let expansion_cardinalities = receipts
        .first()
        .ok_or_else(|| Error::internal(format!("{} omitted native receipts", case.name)))?
        .iter()
        .filter(|receipt| {
            receipt.obligation.kind == ResidentNullableRelationObligationKind::MandatoryExpand
        })
        .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
        .collect::<Vec<_>>();
    let expected = match case.report_id {
        357 => vec![(1, 1), (1, 2)],
        358 | usize::MAX => vec![(3, 5), (5, 6)],
        _ => unreachable!("relationship-isomorphism report ID checked by source manifest"),
    };
    if expansion_cardinalities != expected {
        return Err(Error::internal(format!(
            "{} expansion receipts were {expansion_cardinalities:?} instead of {expected:?}",
            case.name
        )));
    }
    Ok(())
}

fn execute_strict_relationship_isomorphism(
    case: &ManifestCase,
    source: &SourceScenario,
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(case, &oracle, &actual)?;
    assert_relationship_isomorphism_rows(case, &actual)?;
    assert_relationship_isomorphism_receipts(case, &observations)?;
    assert_route(case, &observations)
}

#[test]
fn native_cpu_reference_enforces_relationship_isomorphism_across_fixed_hops() -> Result<()> {
    for (case, source) in relationship_isomorphism_sources() {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        let observations = backend.observations();
        execute_strict_relationship_isomorphism(&case, &source, &graph, &backend).map_err(
            |error| {
                Error::internal(format!(
                    "CPU relationship-isomorphism failure for {}: {}",
                    case.name, error.message
                ))
            },
        )?;
        assert_relationship_isomorphism_program(&case, &observations)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires the real-Metal relationship-isomorphism acceptance lane"]
fn real_metal_enforces_relationship_isomorphism_across_fixed_hops() -> Result<()> {
    for (case, source) in relationship_isomorphism_sources() {
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        let observations = backend.observations();
        execute_strict_relationship_isomorphism(&case, &source, &graph, &backend).map_err(
            |error| {
                Error::internal(format!(
                    "Metal relationship-isomorphism failure for {}: {}",
                    case.name, error.message
                ))
            },
        )?;
        assert_relationship_isomorphism_program(&case, &observations)?;
        assert_metal_receipts(&case, &observations)?;
    }
    Ok(())
}

fn separate_match_relationship_reuse_source() -> SourceScenario {
    SourceScenario {
        setup_queries: vec!["CREATE (:A)-[:T]->(:B)".to_owned()],
        query: "MATCH (a:A)-[r1:T]->(b:B) MATCH (a)-[r2:T]->(b) RETURN r1, r2".to_owned(),
    }
}

fn execute_strict_separate_match_relationship_reuse(
    source: &SourceScenario,
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let case = ManifestCase {
        report_id: usize::MAX - 1,
        feature: "adversarial/separate-match-relationship-reuse".to_owned(),
        name: "separate MATCH clauses may reuse one relationship".to_owned(),
    };
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(&case, &oracle, &actual)?;

    let rows = result_rows(&actual)?;
    let [row] = rows.as_slice() else {
        return Err(Error::internal(format!(
            "separate MATCH relationship reuse produced {} rows instead of one",
            rows.len()
        )));
    };
    let (Some(ResultValue::Relationship(first)), Some(ResultValue::Relationship(second))) =
        (row.first(), row.get(1))
    else {
        return Err(Error::internal(
            "separate MATCH relationship reuse did not publish r1 and r2",
        ));
    };
    if first.id != second.id {
        return Err(Error::internal(format!(
            "separate MATCH clauses returned different relationships {:?} and {:?}",
            first.id, second.id
        )));
    }

    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let expansion_cardinalities = receipts
        .first()
        .ok_or_else(|| Error::internal("separate MATCH reuse omitted native receipts"))?
        .iter()
        .filter(|receipt| {
            receipt.obligation.kind == ResidentNullableRelationObligationKind::MandatoryExpand
        })
        .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
        .collect::<Vec<_>>();
    if expansion_cardinalities != [(1, 1), (1, 1)] {
        return Err(Error::internal(format!(
            "separate MATCH expansion receipts were {expansion_cardinalities:?} instead of [(1, 1), (1, 1)]"
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal("separate MATCH reuse omitted its sealed request"))?;
    let expansions = request
        .program
        .stages
        .iter()
        .filter_map(|stage| match stage {
            ResidentNullableRelationStage::Expand {
                uniqueness_group,
                different_from,
                ..
            } => Some((*uniqueness_group, different_from.as_slice())),
            _ => None,
        })
        .collect::<Vec<_>>();
    if expansions.len() != 2
        || expansions[0].0 == expansions[1].0
        || expansions
            .iter()
            .any(|(_, exclusions)| !exclusions.is_empty())
    {
        return Err(Error::internal(format!(
            "separate MATCH clauses did not seal two empty uniqueness domains: {expansions:?}"
        )));
    }
    drop(requests);
    assert_route(&case, &observations)
}

#[test]
fn native_cpu_reference_allows_relationship_reuse_across_separate_match_clauses() -> Result<()> {
    let source = separate_match_relationship_reuse_source();
    let graph = fixture_graph(&source)?;
    let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
    execute_strict_separate_match_relationship_reuse(&source, &graph, &backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires the real-Metal relationship-uniqueness acceptance lane"]
fn real_metal_allows_relationship_reuse_across_separate_match_clauses() -> Result<()> {
    let source = separate_match_relationship_reuse_source();
    let graph = fixture_graph(&source)?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_project(resident_image(&graph)?)?;
    let backend = StrictOptionalBackend::metal(metal)?;
    execute_strict_separate_match_relationship_reuse(&source, &graph, &backend)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationshipUniquenessShape {
    SameClause,
    SeparateClauses,
    ExistingRelationship,
}

fn comma_relationship_uniqueness_sources() -> Vec<(
    ManifestCase,
    SourceScenario,
    RelationshipUniquenessShape,
    Vec<&'static str>,
)> {
    let setup_queries =
        vec!["CREATE (a:A {name: 'A'})-[:R]->(b:B {name: 'B'})-[:R]->(c:C {name: 'C'})".to_owned()];
    vec![
        (
            ManifestCase {
                report_id: usize::MAX - 2,
                feature: "adversarial/comma-relationship-isomorphism".to_owned(),
                name: "comma patterns share one named relationship-uniqueness domain".to_owned(),
            },
            SourceScenario {
                setup_queries: setup_queries.clone(),
                query: "MATCH (a:A)-[r1:R]-(b:B), (b)-[r2:R]-(c) RETURN c, r1, r2".to_owned(),
            },
            RelationshipUniquenessShape::SameClause,
            vec!["C"],
        ),
        (
            ManifestCase {
                report_id: usize::MAX - 3,
                feature: "adversarial/comma-relationship-isomorphism".to_owned(),
                name: "later MATCH starts a new named relationship-uniqueness domain".to_owned(),
            },
            SourceScenario {
                setup_queries: setup_queries.clone(),
                query: "MATCH (a:A)-[r1:R]-(b:B) MATCH (b)-[r2:R]-(c) RETURN c, r1, r2".to_owned(),
            },
            RelationshipUniquenessShape::SeparateClauses,
            vec!["A", "C"],
        ),
        (
            ManifestCase {
                report_id: usize::MAX - 4,
                feature: "adversarial/comma-relationship-isomorphism".to_owned(),
                name: "comma patterns retain anonymous relationship identity".to_owned(),
            },
            SourceScenario {
                setup_queries: setup_queries.clone(),
                query: "MATCH (a:A)-[]-(b:B), (b)-[]-(c) RETURN c".to_owned(),
            },
            RelationshipUniquenessShape::SameClause,
            vec!["C"],
        ),
        (
            ManifestCase {
                report_id: usize::MAX - 5,
                feature: "adversarial/comma-relationship-isomorphism".to_owned(),
                name: "later MATCH may reuse an anonymous relationship".to_owned(),
            },
            SourceScenario {
                setup_queries,
                query: "MATCH (a:A)-[]-(b:B) MATCH (b)-[]-(c) RETURN c".to_owned(),
            },
            RelationshipUniquenessShape::SeparateClauses,
            vec!["A", "C"],
        ),
    ]
}

fn existing_relationship_reuse_sources() -> Vec<(ManifestCase, SourceScenario, usize)> {
    vec![
        (
            ManifestCase {
                // Zero-based index of Match3 [24] in the pinned 3,897-scenario report.
                report_id: 366,
                feature: "clauses/match/Match3.feature".to_owned(),
                name: "[24] Matching twice with duplicate relationship types on same relationship"
                    .to_owned(),
            },
            SourceScenario {
                setup_queries: vec!["CREATE (:A)-[:T]->(:B)".to_owned()],
                query: "MATCH (a1)-[r:T]->() WITH r, a1 MATCH (a1)-[r:T]->(b2) RETURN a1, r, b2"
                    .to_owned(),
            },
            1,
        ),
        (
            ManifestCase {
                report_id: 367,
                feature: "clauses/match/Match3.feature".to_owned(),
                name: "[25] Matching twice with an additional node label".to_owned(),
            },
            SourceScenario {
                setup_queries: vec!["CREATE ()-[:T]->()".to_owned()],
                query: "MATCH (a1)-[r]->() WITH r, a1 MATCH (a1:X)-[r]->(b2) RETURN a1, r, b2"
                    .to_owned(),
            },
            0,
        ),
        (
            ManifestCase {
                report_id: 368,
                feature: "clauses/match/Match3.feature".to_owned(),
                name: "[26] Matching twice with a duplicate predicate".to_owned(),
            },
            SourceScenario {
                setup_queries: vec!["CREATE (:X:Y)-[:T]->()".to_owned()],
                query: "MATCH (a1:X:Y)-[r]->() WITH r, a1 MATCH (a1:Y)-[r]->(b2) RETURN a1, r, b2"
                    .to_owned(),
            },
            1,
        ),
    ]
}

fn assert_relationship_uniqueness_program(
    case: &ManifestCase,
    observations: &RouteObservations,
    expected_shape: RelationshipUniquenessShape,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests.first().ok_or_else(|| {
        Error::internal(format!(
            "{} omitted its sealed relationship-uniqueness request",
            case.name
        ))
    })?;
    let expansions = request
        .program
        .stages
        .iter()
        .filter_map(|stage| match stage {
            ResidentNullableRelationStage::Expand {
                uniqueness_group,
                relationship,
                different_from,
                ..
            } => Some((*uniqueness_group, *relationship, different_from.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    if expansions.len() != 2 {
        return Err(Error::internal(format!(
            "{} sealed {} expansions instead of two",
            case.name,
            expansions.len()
        )));
    }
    let existing_relationship_projection = request.program.stages.iter().any(|stage| {
        matches!(
            stage,
            ResidentNullableRelationStage::ScopeProject { bindings }
                if bindings.iter().any(|binding| {
                    binding.variable == "r"
                        && Some(binding.source) == expansions[0].1
                        && Some(binding.output) == expansions[1].1
                })
        )
    });
    let valid = match expected_shape {
        RelationshipUniquenessShape::SameClause => {
            expansions[0].0 == expansions[1].0
                && expansions[0].1.is_some()
                && expansions[1].1.is_some()
                && expansions[0].1 != expansions[1].1
                && expansions[0].2.is_empty()
                && expansions[1].2 == vec![expansions[0].1.expect("first slot checked above")]
        }
        RelationshipUniquenessShape::SeparateClauses => {
            expansions[0].0 != expansions[1].0
                && expansions[0].2.is_empty()
                && expansions[1].2.is_empty()
        }
        RelationshipUniquenessShape::ExistingRelationship => {
            expansions[0].0 != expansions[1].0
                && expansions[0].1.is_some()
                && expansions[1].1.is_some()
                && existing_relationship_projection
                && expansions[0].2.is_empty()
                && expansions[1].2.is_empty()
        }
    };
    if !valid {
        return Err(Error::internal(format!(
            "{} sealed the wrong relationship-uniqueness domains: {expansions:?}",
            case.name
        )));
    }
    request.validate()?;
    request.scratch_bytes()?;
    Ok(())
}

fn assert_existing_relationship_source_labels(
    case: &ManifestCase,
    graph: &GraphStore,
    observations: &RouteObservations,
) -> Result<()> {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal(format!("{} omitted its sealed request", case.name)))?;
    let source_labels = request
        .program
        .stages
        .iter()
        .filter_map(|stage| match stage {
            ResidentNullableRelationStage::Expand { source_labels, .. } => Some(source_labels),
            _ => None,
        })
        .nth(1)
        .ok_or_else(|| Error::internal(format!("{} omitted its second expansion", case.name)))?;
    let expected = match case.report_id {
        366 => irongraph::gpu::ResidentNullableNodeDomain::Any,
        367 => irongraph::gpu::ResidentNullableNodeDomain::KnownEmpty,
        368 => irongraph::gpu::ResidentNullableNodeDomain::Known(vec![
            graph
                .catalog()
                .label("Y")
                .ok_or_else(|| Error::internal("Match3 [26] fixture omitted label Y"))?,
        ]),
        _ => {
            return Err(Error::internal(format!(
                "{} is not an existing-relationship Match3 gate",
                case.name
            )));
        }
    };
    if source_labels != &expected {
        return Err(Error::internal(format!(
            "{} sealed source labels {source_labels:?} instead of {expected:?}",
            case.name
        )));
    }
    Ok(())
}

fn assert_node_name_column(output: &ExecutionOutput, expected: &[&str]) -> Result<()> {
    let mut actual = result_rows(output)?
        .into_iter()
        .map(|row| match row.first() {
            Some(ResultValue::Node(node)) => match node.properties.get("name") {
                Some(irongraph::ScalarValue::String(value)) => Ok(value.to_string()),
                other => Err(Error::internal(format!(
                    "relationship-uniqueness node omitted string property `name`: {other:?}"
                ))),
            },
            other => Err(Error::internal(format!(
                "relationship-uniqueness result omitted node column `c`: {other:?}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    actual.sort();
    let mut expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    expected.sort();
    if actual != expected {
        return Err(Error::internal(format!(
            "relationship-uniqueness names were {actual:?} instead of {expected:?}"
        )));
    }
    Ok(())
}

fn execute_strict_comma_relationship_uniqueness(
    case: &ManifestCase,
    source: &SourceScenario,
    expected_shape: RelationshipUniquenessShape,
    expected_names: &[&str],
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(case, &oracle, &actual)?;
    assert_node_name_column(&actual, expected_names)?;
    assert_relationship_uniqueness_program(case, &observations, expected_shape)?;
    assert_route(case, &observations)
}

#[test]
fn native_cpu_reference_scopes_comma_relationship_uniqueness_to_one_match_clause() -> Result<()> {
    for (case, source, expected_shape, expected_names) in comma_relationship_uniqueness_sources() {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        execute_strict_comma_relationship_uniqueness(
            &case,
            &source,
            expected_shape,
            &expected_names,
            &graph,
            &backend,
        )
        .map_err(|error| {
            Error::internal(format!(
                "CPU comma relationship-uniqueness failure for {}: {}",
                case.name, error.message
            ))
        })?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires the real-Metal relationship-uniqueness acceptance lane"]
fn real_metal_scopes_comma_relationship_uniqueness_to_one_match_clause() -> Result<()> {
    for (case, source, expected_shape, expected_names) in comma_relationship_uniqueness_sources() {
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        let observations = backend.observations();
        execute_strict_comma_relationship_uniqueness(
            &case,
            &source,
            expected_shape,
            &expected_names,
            &graph,
            &backend,
        )
        .map_err(|error| {
            Error::internal(format!(
                "Metal comma relationship-uniqueness failure for {}: {}",
                case.name, error.message
            ))
        })?;
        assert_metal_receipts(&case, &observations)?;
    }
    Ok(())
}

fn execute_strict_existing_relationship_reuse(
    case: &ManifestCase,
    source: &SourceScenario,
    expected_rows: usize,
    graph: &GraphStore,
    backend: &StrictOptionalBackend,
) -> Result<()> {
    let oracle = execute_generic_oracle(&source.query, graph)?;
    let observations = backend.observations();
    let actual = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true))?;
    assert_matches_oracle(case, &oracle, &actual)?;
    let rows = result_rows(&actual)?;
    if rows.len() != expected_rows {
        return Err(Error::internal(format!(
            "{} returned {} rows instead of {expected_rows}",
            case.name,
            rows.len()
        )));
    }
    if let Some(row) = rows.first() {
        let Some(ResultValue::Relationship(relationship)) = row.get(1) else {
            return Err(Error::internal(format!(
                "{} did not preserve the existing relationship output",
                case.name
            )));
        };
        if relationship.id.0 != 1 {
            return Err(Error::internal(format!(
                "{} returned relationship {:?} instead of EdgeId(1)",
                case.name, relationship.id
            )));
        }
    }
    assert_relationship_uniqueness_program(
        case,
        &observations,
        RelationshipUniquenessShape::ExistingRelationship,
    )?;
    assert_existing_relationship_source_labels(case, graph, &observations)?;
    assert_route(case, &observations)
}

#[test]
fn native_cpu_reference_reuses_existing_relationships_with_bound_label_constraints() -> Result<()> {
    for (case, source, expected_rows) in existing_relationship_reuse_sources() {
        let graph = fixture_graph(&source)?;
        let backend = StrictOptionalBackend::cpu_reference(cpu_backend(&graph)?)?;
        execute_strict_existing_relationship_reuse(&case, &source, expected_rows, &graph, &backend)
            .map_err(|error| {
                Error::internal(format!(
                    "CPU existing-relationship failure for {}: {}",
                    case.name, error.message
                ))
            })?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires the real-Metal relationship-uniqueness acceptance lane"]
fn real_metal_reuses_existing_relationships_with_bound_label_constraints() -> Result<()> {
    for (case, source, expected_rows) in existing_relationship_reuse_sources() {
        let graph = fixture_graph(&source)?;
        let governor =
            irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        let mut metal = MetalBackend::with_governor(0, governor)?;
        metal.admit_project(resident_image(&graph)?)?;
        let backend = StrictOptionalBackend::metal(metal)?;
        let observations = backend.observations();
        execute_strict_existing_relationship_reuse(&case, &source, expected_rows, &graph, &backend)
            .map_err(|error| {
                Error::internal(format!(
                    "Metal existing-relationship failure for {}: {}",
                    case.name, error.message
                ))
            })?;
        assert_metal_receipts(&case, &observations)?;
    }
    Ok(())
}

fn execute_fault(case_id: usize, fault: Fault) -> Result<(Error, Arc<RouteObservations>)> {
    let case = case_by_id(case_id)?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let backend = StrictOptionalBackend::cpu_fault(cpu_backend(&graph)?, fault)?;
    let observations = backend.observations();
    let error = QueryEngine
        .execute(&source.query, &mut context(&graph, Some(&backend), true))
        .err()
        .ok_or_else(|| {
            Error::internal(format!(
                "report {case_id} published a result despite injected {fault:?}"
            ))
        })?;
    Ok((error, observations))
}

#[test]
#[ignore = "red acceptance gate: requires device-authored completion receipts on OPTIONAL output"]
fn cpu_cannot_masquerade_as_metal_or_publish_missing_receipts() -> Result<()> {
    let (error, observations) = execute_fault(529, Fault::CpuMasqueradesAsMetal)?;
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        error.code,
        ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
    ));
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: requires complete request fingerprints on OPTIONAL output"]
fn request_fingerprint_forgery_is_rejected_before_result_publication() -> Result<()> {
    let (error, observations) = execute_fault(529, Fault::MutateRequestAfterFingerprint)?;
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 1);
    assert!(matches!(
        error.code,
        ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
    ));
    Ok(())
}

#[test]
#[ignore = "red acceptance gate: requires exact immutable generation fences on OPTIONAL output"]
fn stale_generation_is_rejected_before_result_publication() -> Result<()> {
    let (error, observations) = execute_fault(529, Fault::StalePinnedGeneration)?;
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        error.code,
        ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
    ));
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn missing_or_forged_nullable_receipts_and_result_fingerprints_are_rejected() -> Result<()> {
    for fault in [
        Fault::MissingReceipt,
        Fault::ForgedReceiptCardinality,
        Fault::ForgedResultFingerprint,
    ] {
        let (error, observations) = execute_fault(529, fault)?;
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
        assert_eq!(error.code, ErrorCode::CorruptStorage, "fault {fault:?}");
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn predicate_requests_seal_receipts_fingerprint_generation_and_capacity() -> Result<()> {
    for fault in [
        Fault::MutateRequestAfterFingerprint,
        Fault::MutatePredicateAfterFingerprint,
        Fault::MissingReceipt,
        Fault::ForgedPredicateReceiptCardinality,
        Fault::ForgedResultFingerprint,
    ] {
        let (error, observations) = execute_fault(579, fault)?;
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
        assert!(
            matches!(
                error.code,
                ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
            ),
            "predicate fault {fault:?} escaped with {error:?}"
        );
    }

    let (error, observations) = execute_fault(579, Fault::StalePinnedGeneration)?;
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.complete_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    assert!(matches!(
        error.code,
        ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
    ));
    Ok(())
}
