// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance gate for the 21 direct resident DELETE scenarios identified by
//! `/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json`.
//!
//! The ordinary manifest test and ignored external identity gate keep the literal zero-based report
//! manifest tied to the pinned TCK feature text and current certified report. CPU-reference and
//! real Metal each execute one complete, immutable, generation-pinned resident DELETE command.
//! Every legacy graph primitive is closed by the observer, so host traversal, connectivity checks,
//! target deduplication, or a generic post-delete tail cannot accidentally turn the gate green.

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

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, EntityDependency, ExecutionContext, ExecutionOutput,
        QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentDeleteRequest,
        ResidentDeleteResult, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentProjectImage, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
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
    0x4445_4c45_5445_5f32_315f_5443_4b5f_3031,
));
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 100_000;

/// `zero-based-report-id|feature|expanded scenario name`.
const EXACT_MANIFEST: &str = r#"
130|clauses/delete/Delete1.feature|[1] Delete nodes
131|clauses/delete/Delete1.feature|[2] Detach delete node
132|clauses/delete/Delete1.feature|[3] Detach deleting connected nodes and relationships
136|clauses/delete/Delete1.feature|[7] Failing when deleting connected nodes
138|clauses/delete/Delete2.feature|[1] Delete relationships
140|clauses/delete/Delete2.feature|[3] Delete relationship with bidirectional matching
145|clauses/delete/Delete4.feature|[1] Undirected expand followed by delete and count
157|clauses/delete/Delete6.feature|[1] Limiting to zero results after deleting nodes affects the result set but not the side effects
158|clauses/delete/Delete6.feature|[2] Skipping all results after deleting nodes affects the result set but not the side effects
159|clauses/delete/Delete6.feature|[3] Skipping and limiting to a few results after deleting nodes affects the result set but not the side effects
160|clauses/delete/Delete6.feature|[4] Skipping zero results and limiting to all results after deleting nodes does not affect the result set nor the side effects
161|clauses/delete/Delete6.feature|[5] Filtering after deleting nodes affects the result set but not the side effects
162|clauses/delete/Delete6.feature|[6] Aggregating in `RETURN` after deleting nodes affects the result set but not the side effects
163|clauses/delete/Delete6.feature|[7] Aggregating in `WITH` after deleting nodes affects the result set but not the side effects
164|clauses/delete/Delete6.feature|[8] Limiting to zero results after deleting relationships affects the result set but not the side effects
165|clauses/delete/Delete6.feature|[9] Skipping all results after deleting relationships affects the result set but not the side effects
166|clauses/delete/Delete6.feature|[10] Skipping and limiting to a few results after deleting relationships affects the result set but not the side effects
167|clauses/delete/Delete6.feature|[11] Skipping zero result and limiting to all results after deleting relationships does not affect the result set nor the side effects
168|clauses/delete/Delete6.feature|[12] Filtering after deleting relationships affects the result set but not the side effects
169|clauses/delete/Delete6.feature|[13] Aggregating in `RETURN` after deleting relationships affects the result set but not the side effects
170|clauses/delete/Delete6.feature|[14] Aggregating in `WITH` after deleting relationships affects the result set but not the side effects
"#;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ManifestCase {
    report_id: usize,
    feature: String,
    name: String,
}

impl ManifestCase {
    fn label(&self) -> String {
        format!("TCK id={} {} {}", self.report_id, self.feature, self.name)
    }
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
    operation_count: usize,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
    divergences: Vec<String>,
}

#[derive(Clone, Debug)]
struct SourceScenario {
    setup_queries: Vec<String>,
    query: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExpectedSideEffects {
    nodes_removed: usize,
    relationships_removed: usize,
    properties_removed: usize,
    labels_removed: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ExpectedOutcome {
    Success {
        column: Option<&'static str>,
        column_type: Option<ColumnType>,
        values: Vec<i64>,
        statistics: StatementStats,
        side_effects: ExpectedSideEffects,
    },
    DeleteConnectedNode,
}

fn manifest() -> Result<Vec<ManifestCase>> {
    EXACT_MANIFEST
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut fields = line.splitn(3, '|');
            let report_id = fields
                .next()
                .ok_or_else(|| Error::internal("DELETE manifest omitted report ID"))?
                .parse::<usize>()
                .map_err(|error| Error::internal(format!("invalid DELETE report ID: {error}")))?;
            let feature = fields
                .next()
                .ok_or_else(|| Error::internal("DELETE manifest omitted feature"))?;
            let name = fields
                .next()
                .ok_or_else(|| Error::internal("DELETE manifest omitted scenario name"))?;
            Ok(ManifestCase {
                report_id,
                feature: feature.to_owned(),
                name: name.to_owned(),
            })
        })
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

fn uniquely_resolved_scenario<'a>(
    report: &'a CertifiedReport,
    case: &ManifestCase,
) -> Result<(usize, &'a CertifiedScenario)> {
    let matches = report
        .scenarios
        .iter()
        .enumerate()
        .filter(|(_, scenario)| {
            scenario.path.ends_with(&case.feature) && scenario.name == case.name
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(Error::internal(format!(
            "{} resolved to {} certified report entries instead of exactly one",
            case.label(),
            matches.len()
        )));
    }
    Ok(matches[0])
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

fn docstrings_after(block: &str, marker: &str) -> Result<Vec<String>> {
    let mut values = Vec::new();
    let mut rest = block;
    while let Some(marker_offset) = rest.find(marker) {
        let tail = &rest[marker_offset + marker.len()..];
        let open = tail
            .find("\"\"\"")
            .ok_or_else(|| Error::internal(format!("`{marker}` omitted opening docstring")))?
            + 3;
        let close = tail[open..]
            .find("\"\"\"")
            .map(|offset| open + offset)
            .ok_or_else(|| Error::internal(format!("`{marker}` omitted closing docstring")))?;
        values.push(
            tail[open..close]
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        );
        rest = &tail[close + 3..];
    }
    Ok(values)
}

fn source_scenario(case: &ManifestCase) -> Result<SourceScenario> {
    let path = Path::new(PINNED_FEATURE_ROOT).join(&case.feature);
    let source = fs::read_to_string(&path)
        .map_err(|error| Error::internal(format!("cannot read {}: {error}", path.display())))?;
    let block = scenario_block(&source, &case.name)?;
    let setup_queries = docstrings_after(&block, "having executed:")?;
    let query = docstrings_after(&block, "executing query:")?
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal(format!("{} omitted its primary query", case.label())))?;
    Ok(SourceScenario {
        setup_queries,
        query,
    })
}

fn expected_outcome(id: usize) -> Result<ExpectedOutcome> {
    let success = |column, column_type, values, nodes, relationships, properties, labels| {
        ExpectedOutcome::Success {
            column,
            column_type,
            values,
            statistics: StatementStats {
                nodes_deleted: nodes as u64,
                relationships_deleted: relationships as u64,
                ..StatementStats::default()
            },
            side_effects: ExpectedSideEffects {
                nodes_removed: nodes,
                relationships_removed: relationships,
                properties_removed: properties,
                labels_removed: labels,
            },
        }
    };
    Ok(match id {
        130 | 131 => success(None, None, Vec::new(), 1, 0, 0, 0),
        132 => ExpectedOutcome::Success {
            column: None,
            column_type: None,
            values: Vec::new(),
            // DETACH removes incident edges as a consequence of deleting the node. They are
            // observable TCK side effects, but are not explicit DELETE relationship targets in
            // the statement statistics.
            statistics: StatementStats {
                nodes_deleted: 1,
                ..StatementStats::default()
            },
            side_effects: ExpectedSideEffects {
                nodes_removed: 1,
                relationships_removed: 3,
                properties_removed: 0,
                labels_removed: 1,
            },
        },
        136 => ExpectedOutcome::DeleteConnectedNode,
        138 => success(None, None, Vec::new(), 0, 3, 0, 0),
        140 => success(None, None, Vec::new(), 0, 1, 1, 0),
        145 => success(Some("c"), Some(ColumnType::Integer), vec![2], 2, 1, 0, 0),
        157 | 158 => success(Some("num"), Some(ColumnType::Null), Vec::new(), 1, 0, 1, 1),
        159 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![42, 42],
            5,
            0,
            5,
            1,
        ),
        160 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![42; 5],
            5,
            0,
            5,
            1,
        ),
        161 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![2, 4],
            5,
            0,
            5,
            1,
        ),
        162 | 163 => success(Some("sum"), Some(ColumnType::Integer), vec![15], 5, 0, 5, 1),
        164 | 165 => success(Some("num"), Some(ColumnType::Null), Vec::new(), 0, 1, 1, 0),
        166 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![42, 42],
            0,
            5,
            5,
            0,
        ),
        167 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![42; 5],
            0,
            5,
            5,
            0,
        ),
        168 => success(
            Some("num"),
            Some(ColumnType::Integer),
            vec![2, 4],
            0,
            5,
            5,
            0,
        ),
        169 | 170 => success(Some("sum"), Some(ColumnType::Integer), vec![15], 0, 5, 5, 0),
        _ => return Err(Error::internal(format!("unexpected DELETE report ID {id}"))),
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
            term: 43,
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

fn fixture_graph(source: &SourceScenario) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in &source.setup_queries {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        apply_mutations(&mut graph, &output.graph_mutations)?;
    }
    Ok(graph)
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 43,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphMetrics {
    nodes: usize,
    relationships: usize,
    properties: usize,
    represented_labels: usize,
}

fn graph_metrics(graph: &GraphStore) -> GraphMetrics {
    let represented_labels = graph
        .nodes()
        .flat_map(|node| node.labels().iter().copied())
        .collect::<BTreeSet<_>>()
        .len();
    GraphMetrics {
        nodes: graph.node_count(),
        relationships: graph.edge_count(),
        properties: graph
            .nodes()
            .map(|node| node.properties().len())
            .sum::<usize>()
            + graph
                .edges()
                .map(|edge| edge.properties().len())
                .sum::<usize>(),
        represented_labels,
    }
}

fn observed_side_effects(before: GraphMetrics, after: GraphMetrics) -> ExpectedSideEffects {
    ExpectedSideEffects {
        nodes_removed: before.nodes.saturating_sub(after.nodes),
        relationships_removed: before.relationships.saturating_sub(after.relationships),
        properties_removed: before.properties.saturating_sub(after.properties),
        labels_removed: before
            .represented_labels
            .saturating_sub(after.represented_labels),
    }
}

fn integer_values(output: &ExecutionOutput, column: &str) -> Result<Vec<i64>> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() {
            return Err(Error::internal("DELETE result batch is misaligned"));
        }
        let result_column = batch
            .columns
            .iter()
            .find(|candidate| candidate.name == column)
            .ok_or_else(|| Error::internal(format!("DELETE result omitted `{column}`")))?;
        for value in &result_column.values {
            match value {
                ResultValue::Scalar(ScalarValue::Integer(value)) => values.push(*value),
                _ => {
                    return Err(Error::internal(format!(
                        "DELETE result `{column}` contains {value:?}"
                    )));
                }
            }
        }
    }
    values.sort_unstable();
    Ok(values)
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

fn assert_success(
    graph: &GraphStore,
    output: &ExecutionOutput,
    expected: &ExpectedOutcome,
) -> Result<()> {
    let ExpectedOutcome::Success {
        column,
        column_type,
        values,
        statistics,
        side_effects,
    } = expected
    else {
        return Err(Error::internal("expected a DELETE constraint error"));
    };
    let expected_schema = column
        .zip(column_type.clone())
        .map(|(name, value_type)| vec![(name.to_owned(), value_type)])
        .unwrap_or_default();
    if output.result.schema != expected_schema {
        return Err(Error::internal(format!(
            "DELETE schema mismatch: expected {expected_schema:?}, got {:?}",
            output.result.schema
        )));
    }
    let actual_values = if let Some(column) = column {
        integer_values(output, column)?
    } else {
        Vec::new()
    };
    let mut expected_values = values.clone();
    expected_values.sort_unstable();
    if actual_values != expected_values {
        return Err(Error::internal(format!(
            "DELETE rows mismatch: expected {expected_values:?}, got {actual_values:?}"
        )));
    }
    if output.result.statistics != *statistics
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
    {
        return Err(Error::internal(format!(
            "DELETE result metadata mismatch: {:?}",
            output.result
        )));
    }
    let targets = mutation_targets(output);
    if targets != output.dependencies.write_targets {
        return Err(Error::internal(format!(
            "DELETE dependency targets differ from published mutations: targets={targets:?}, dependencies={:?}",
            output.dependencies.write_targets
        )));
    }
    if targets.len() != output.graph_mutations.len() {
        return Err(Error::internal(format!(
            "DELETE emitted duplicate or non-delete graph mutations: {:?}",
            output.graph_mutations
        )));
    }
    let mut committed = graph.clone();
    apply_mutations(&mut committed, &output.graph_mutations)?;
    let observed = observed_side_effects(graph_metrics(graph), graph_metrics(&committed));
    if observed != *side_effects {
        return Err(Error::internal(format!(
            "DELETE side effects mismatch: expected {side_effects:?}, got {observed:?}"
        )));
    }
    Ok(())
}

fn execution_debug(output: &ExecutionOutput) -> (Vec<String>, String) {
    (
        output
            .graph_mutations
            .iter()
            .map(|mutation| format!("{mutation:?}"))
            .collect(),
        format!("{:#?}", output.dependencies),
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum FaultMode {
    #[default]
    None,
    WrongGeneration,
    WrongFingerprint,
    WrongReceipt,
    WrongBackend,
    Capacity,
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    delete_calls: AtomicUsize,
    rejected_routes: AtomicUsize,
    requests: Mutex<Vec<ResidentDeleteRequest>>,
    raw_results: Mutex<Vec<String>>,
}

struct StrictDeleteBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    fault: FaultMode,
    observations: Arc<RouteObservations>,
}

impl StrictDeleteBackend {
    fn strict_cpu(inner: CpuBackend) -> Result<Self> {
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            FaultMode::None,
        )
    }

    fn faulted_cpu(inner: CpuBackend, fault: FaultMode) -> Result<Self> {
        let pinned_kind = if fault == FaultMode::WrongBackend {
            BackendKind::Metal
        } else {
            BackendKind::Cpu
        };
        Self::new(
            inner,
            BackendKind::Metal,
            pinned_kind,
            BackendKind::Cpu,
            fault,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "DELETE acceptance gate did not construct a real Metal backend",
            ));
        }
        Self::new(
            inner,
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
            FaultMode::None,
        )
    }

    fn new<B: ExecutionBackend + 'static>(
        inner: B,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
        fault: FaultMode,
    ) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("DELETE backend has no resident bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("DELETE backend has no resident graph revision"))?;
        Ok(Self {
            inner: Box::new(inner),
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            fault,
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
            format!("strict DELETE gate rejected `{route}`"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self
            .inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("replacement DELETE image has no bookmark"))?;
        self.expected_graph_revision = self
            .inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("replacement DELETE image has no revision"))?;
        Ok(())
    }

    fn device_request(&self, request: &ResidentDeleteRequest) -> Result<ResidentDeleteRequest> {
        let mut device = request.clone();
        match self.fault {
            FaultMode::None | FaultMode::WrongBackend => {}
            FaultMode::WrongGeneration => {
                device.expected_graph_revision = device.expected_graph_revision.saturating_add(1);
                device.seal()?;
            }
            FaultMode::WrongFingerprint => {
                device.fingerprint.0[0] ^= 0x80;
            }
            FaultMode::WrongReceipt => {
                device.selection_obligation.id ^= 1_u64 << 63;
                device.seal()?;
            }
            FaultMode::Capacity => {
                device.maximum_delete_intents = 0;
                device.seal()?;
            }
        }
        Ok(device)
    }
}

impl ExecutionBackend for StrictDeleteBackend {
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
            return self.reject("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "DELETE pin did not preserve the immutable resident generation",
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
            fault: self.fault,
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
                "DELETE command is not tied to the pinned resident generation",
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
        let device = self.device_request(request)?;
        let result = self.inner.execute_delete_pipeline(&device, cancellation);
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

fn admitted_cpu(graph: &GraphStore) -> Result<CpuBackend> {
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(resident_image(graph)?)?;
    Ok(cpu)
}

fn assert_request_is_complete(case: &ManifestCase, request: &ResidentDeleteRequest) -> Result<()> {
    let command_debug = format!("{:#?}", request.commands);
    if command_debug.is_empty() || !command_debug.contains("target") {
        return Err(Error::internal(format!(
            "{} did not encode DELETE in the backend command: {command_debug}",
            case.label()
        )));
    }
    if request.selection_obligation.id == 0 || request.maximum_delete_intents == 0 {
        return Err(Error::internal(
            "resident DELETE omitted selection proof or intent capacity",
        ));
    }
    let has_post_delete_tail = case.report_id == 145 || (157..=170).contains(&case.report_id);
    if has_post_delete_tail {
        let continuation = request
            .continuation
            .as_ref()
            .ok_or_else(|| Error::internal("post-delete work escaped the resident command"))?;
        if request.expected_bookmark.term != 43
            || request.expected_graph_revision == 0
            || request.fingerprint.0 == [0; 32]
            || continuation.maximum_output_rows > MAX_RESULT_ROWS
        {
            return Err(Error::internal(
                "resident DELETE continuation has an invalid fence, fingerprint, or capacity",
            ));
        }
        let continuation_debug = format!("{continuation:#?}");
        let required = match case.report_id {
            157 | 164 => "Limit",
            158 | 165 => "Skip",
            159 | 160 | 166 | 167 => "Limit",
            161 | 168 => "Filter",
            162 | 163 | 169 | 170 => "Sum",
            145 => "Count",
            _ => unreachable!(),
        };
        if !continuation_debug.contains(required) {
            return Err(Error::internal(format!(
                "{} omitted native post-delete {required}: {continuation_debug}",
                case.label()
            )));
        }
        if (161..=163).contains(&case.report_id) || (168..=170).contains(&case.report_id) {
            if !continuation_debug.contains("Property") {
                return Err(Error::internal(
                    "post-delete filter/SUM did not carry the captured pre-delete property column",
                ));
            }
        }
    }
    if case.report_id == 157 || case.report_id == 164 {
        if request.selection.offset != 0 || request.selection.limit != usize::MAX {
            return Err(Error::internal(
                "LIMIT 0 was applied to DELETE selection instead of only the final relation",
            ));
        }
    }
    Ok(())
}

fn assert_raw_delete_intents(case: &ManifestCase, raw: &str) -> Result<()> {
    if !raw.contains("ResidentDeleteResult") || !raw.contains("intents") {
        return Err(Error::internal(format!(
            "{} returned no native DELETE intent frame: {raw}",
            case.label()
        )));
    }
    if case.report_id == 132 {
        let first_relationship = raw
            .find("target_kind: Relationship")
            .ok_or_else(|| Error::internal("DETACH omitted incident relationship intents"))?;
        let node = raw
            .find("target_kind: Node")
            .ok_or_else(|| Error::internal("DETACH omitted its node intent"))?;
        if first_relationship >= node || raw.matches("target_kind: Relationship").count() != 3 {
            return Err(Error::internal(
                "DETACH did not deduplicate and order all incident relationships before the node",
            ));
        }
    }
    if case.report_id == 145 {
        if raw.matches("target_kind: Relationship").count() != 1
            || raw.matches("target_kind: Node").count() != 2
        {
            return Err(Error::internal(
                "undirected DELETE did not normalize duplicate rows to three stable entity intents",
            ));
        }
    }
    Ok(())
}

fn assert_single_boundary(
    observations: &RouteObservations,
    pins_before: usize,
    calls_before: usize,
    rejected_before: usize,
) -> Result<(ResidentDeleteRequest, String)> {
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1
        || observations.delete_calls.load(Ordering::SeqCst) != calls_before + 1
        || observations.rejected_routes.load(Ordering::SeqCst) != rejected_before
    {
        return Err(Error::internal(format!(
            "DELETE did not use exactly one pinned resident command: pins={}, calls={}, rejected={}",
            observations.pins.load(Ordering::SeqCst) - pins_before,
            observations.delete_calls.load(Ordering::SeqCst) - calls_before,
            observations.rejected_routes.load(Ordering::SeqCst) - rejected_before,
        )));
    }
    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| Error::internal("DELETE observer lost its request"))?;
    let raw = observations
        .raw_results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| Error::internal("DELETE observer lost its raw result"))?;
    Ok((request, raw))
}

fn execute_native_case(
    backend: &StrictDeleteBackend,
    case: &ManifestCase,
    source: &SourceScenario,
    graph: &GraphStore,
) -> Result<()> {
    let expected = expected_outcome(case.report_id)?;
    let oracle = QueryEngine.execute(&source.query, &mut context(graph, None, false));
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let calls_before = observations.delete_calls.load(Ordering::SeqCst);
    let rejected_before = observations.rejected_routes.load(Ordering::SeqCst);
    let native = QueryEngine.execute(&source.query, &mut context(graph, Some(backend), true));
    let (request, raw) =
        assert_single_boundary(&observations, pins_before, calls_before, rejected_before)?;
    assert_request_is_complete(case, &request)?;
    match (expected, oracle, native) {
        (ExpectedOutcome::DeleteConnectedNode, Err(reference), Err(actual)) => {
            if reference.code != ErrorCode::QueryType
                || actual.code != ErrorCode::QueryType
                || !reference.message.contains("DeleteConnectedNode")
                || !actual.message.contains("DeleteConnectedNode")
                || !raw.contains("DeleteConnectedNode")
            {
                return Err(Error::internal(format!(
                    "connected-node failure did not originate in the backend: oracle={reference}, native={actual}, raw={raw}"
                )));
            }
        }
        (expected @ ExpectedOutcome::Success { .. }, Ok(reference), Ok(actual)) => {
            assert_success(graph, &reference, &expected)?;
            assert_success(graph, &actual, &expected)?;
            if actual.result != reference.result
                || execution_debug(&actual) != execution_debug(&reference)
            {
                return Err(Error::internal(format!(
                    "native DELETE differs from CPU semantics:\nCPU={reference:#?}\nnative={actual:#?}"
                )));
            }
            assert_raw_delete_intents(case, &raw)?;
        }
        (_, reference, native) => {
            return Err(Error::internal(format!(
                "{} outcome mismatch: oracle={reference:#?}, native={native:#?}",
                case.label()
            )));
        }
    }
    Ok(())
}

fn run_native_manifest(backend: &mut StrictDeleteBackend) -> Result<()> {
    let mut failures = Vec::new();
    for case in manifest()? {
        let result = (|| {
            let source = source_scenario(&case)?;
            let graph = fixture_graph(&source)?;
            backend.replace_all_projects(vec![resident_image(&graph)?])?;
            execute_native_case(backend, &case, &source, &graph)
        })();
        if let Err(error) = result {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "strict DELETE gate had {} failure(s):\n{}",
            failures.len(),
            failures.join("\n")
        )))
    }
}

#[test]
fn literal_manifest_has_the_exact_21_zero_based_report_ids() -> Result<()> {
    let cases = manifest()?;
    assert_eq!(cases.len(), 21);
    assert_eq!(
        cases.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        vec![
            130, 131, 132, 136, 138, 140, 145, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
            167, 168, 169, 170,
        ]
    );
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the certified report and pinned openCypher TCK checkout"]
fn manifest_is_exactly_the_21_certified_zero_based_delete_scenarios() -> Result<()> {
    let cases = manifest()?;
    let report = certified_report()?;
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);
    assert_eq!(report.metal_passed, 3_184);
    assert_eq!(cases.len(), 21);
    assert_eq!(
        cases.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        vec![
            130, 131, 132, 136, 138, 140, 145, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
            167, 168, 169, 170,
        ]
    );
    for case in &cases {
        let (report_index, certified) = uniquely_resolved_scenario(&report, case)?;
        assert_eq!(
            report_index,
            case.report_id,
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
        assert_eq!(certified.operation_count, 1, "{}", case.label());
        assert!(certified.shared_failures.is_empty(), "{}", case.label());
        assert!(certified.cpu_failures.is_empty(), "{}", case.label());
        assert!(certified.metal_failures.is_empty(), "{}", case.label());
        assert!(certified.divergences.is_empty(), "{}", case.label());
        let source = source_scenario(case)?;
        assert!(!source.query.trim().is_empty(), "{}", case.label());
    }
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn generic_cpu_oracle_proves_rows_side_effects_stats_and_dependencies_for_all_21() -> Result<()> {
    let mut failures = Vec::new();
    for case in manifest()? {
        let result = (|| {
            let source = source_scenario(&case)?;
            let graph = fixture_graph(&source)?;
            let expected = expected_outcome(case.report_id)?;
            match (
                expected.clone(),
                QueryEngine.execute(&source.query, &mut context(&graph, None, false)),
            ) {
                (ExpectedOutcome::DeleteConnectedNode, Err(error))
                    if error.code == ErrorCode::QueryType
                        && error.message.contains("DeleteConnectedNode") => {}
                (expected @ ExpectedOutcome::Success { .. }, Ok(output)) => {
                    assert_success(&graph, &output, &expected)?;
                }
                (expected, actual) => {
                    return Err(Error::internal(format!(
                        "expected {expected:?}, got {actual:#?}"
                    )));
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert!(
        failures.is_empty(),
        "generic CPU DELETE oracle had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn strict_cpu_reference_executes_all_21_without_fallback() -> Result<()> {
    let first = manifest()?
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("DELETE manifest is empty"))?;
    let source = source_scenario(&first)?;
    let graph = fixture_graph(&source)?;
    let mut backend = StrictDeleteBackend::strict_cpu(admitted_cpu(&graph)?)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    run_native_manifest(&mut backend)
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn connected_node_constraint_uses_pinned_backend_graph_not_host_connectivity() -> Result<()> {
    let case = manifest()?
        .into_iter()
        .find(|case| case.report_id == 136)
        .ok_or_else(|| Error::internal("DELETE1 [7] disappeared"))?;
    let connected_source = source_scenario(&case)?;
    let connected_graph = fixture_graph(&connected_source)?;
    let disconnected_source = SourceScenario {
        setup_queries: vec!["CREATE (x:X)".to_owned()],
        query: connected_source.query.clone(),
    };
    let disconnected_host_graph = fixture_graph(&disconnected_source)?;
    let backend = StrictDeleteBackend::strict_cpu(admitted_cpu(&connected_graph)?)?;
    let error = QueryEngine
        .execute(
            &connected_source.query,
            &mut context(&disconnected_host_graph, Some(&backend), true),
        )
        .err()
        .ok_or_else(|| Error::internal("host connectivity incorrectly allowed DELETE"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("DeleteConnectedNode"));
    let observations = backend.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.delete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.rejected_routes.load(Ordering::SeqCst), 0);
    assert!(
        observations
            .raw_results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last()
            .is_some_and(|raw| raw.contains("DeleteConnectedNode"))
    );
    Ok(())
}

#[test]
#[ignore = "external acceptance gate: requires the pinned openCypher TCK fixture checkout"]
fn wrong_generation_fingerprint_receipt_backend_and_capacity_publish_nothing() -> Result<()> {
    let case = manifest()?
        .into_iter()
        .find(|case| case.report_id == 162)
        .ok_or_else(|| Error::internal("DELETE6 [6] disappeared"))?;
    let source = source_scenario(&case)?;
    let graph = fixture_graph(&source)?;
    let before = graph_metrics(&graph);
    for fault in [
        FaultMode::WrongGeneration,
        FaultMode::WrongFingerprint,
        FaultMode::WrongReceipt,
        FaultMode::WrongBackend,
        FaultMode::Capacity,
    ] {
        let backend = StrictDeleteBackend::faulted_cpu(admitted_cpu(&graph)?, fault)?;
        let error = QueryEngine
            .execute(&source.query, &mut context(&graph, Some(&backend), true))
            .err()
            .ok_or_else(|| Error::internal(format!("{fault:?} DELETE fault was published")))?;
        assert!(
            matches!(
                error.code,
                ErrorCode::CorruptStorage | ErrorCode::GpuAdmissionFailure
            ),
            "{fault:?} returned unexpected error: {error}"
        );
        assert_eq!(
            graph_metrics(&graph),
            before,
            "{fault:?} changed canonical graph"
        );
        let observations = backend.observations();
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{fault:?}");
        assert_eq!(
            observations.delete_calls.load(Ordering::SeqCst),
            1,
            "{fault:?}"
        );
        assert_eq!(
            observations.rejected_routes.load(Ordering::SeqCst),
            0,
            "{fault:?}"
        );
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
#[ignore = "hardware acceptance gate: requires real Metal"]
fn real_metal_executes_all_21_with_honest_provenance_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let first = manifest()?
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("DELETE manifest is empty"))?;
    let source = source_scenario(&first)?;
    let graph = fixture_graph(&source)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(resident_image(&graph)?)?;
    let mut backend = StrictDeleteBackend::real_metal(metal)?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    run_native_manifest(&mut backend)
}
