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
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentScalarCellTag,
        ResidentScalarProgramOpcode, ResidentScalarProgramOperand, ResidentScalarProgramRequest,
        ResidentScalarProgramResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const OFFICIAL_MAP_IDENTITIES: [(usize, &str, &str); 5] = [
    (
        1588,
        "features/expressions/graph/Graph9.feature",
        "[4] `properties()` on a map",
    ),
    (
        1908,
        "features/expressions/map/Map1.feature",
        "[1] Statically access a field of a non-null map",
    ),
    (
        1909,
        "features/expressions/map/Map1.feature",
        "[2] Statically access a field of a null map",
    ),
    (
        1910,
        "features/expressions/map/Map1.feature",
        "[3] Statically access a field of a map resulting from an expression",
    ),
    (
        1921,
        "features/expressions/map/Map1.feature",
        "[6] Fail when performing property access on a non-map [1922]",
    ),
];

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
struct ScalarObservations {
    calls: AtomicUsize,
    last_request: Mutex<Option<ResidentScalarProgramRequest>>,
}

/// Strict-route harness around the real CPU scalar backend. Reporting a non-CPU execution class
/// makes `require_native_execution` fail closed if the complete scalar route is not admitted; the
/// only successful path delegates one canonical request to `CpuBackend::execute_scalar_program`.
struct ObservedScalarBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<ScalarObservations>,
    force_null_outputs: bool,
}

impl ObservedScalarBackend {
    fn strict_cpu(force_null_outputs: bool) -> Self {
        Self {
            inner: Box::new(CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024)),
            reported_kind: BackendKind::Metal,
            observations: Arc::new(ScalarObservations::default()),
            force_null_outputs,
        }
    }

    fn observations(&self) -> Arc<ScalarObservations> {
        Arc::clone(&self.observations)
    }
}

impl ExecutionBackend for ObservedScalarBackend {
    fn kind(&self) -> BackendKind {
        self.reported_kind
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
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            reported_kind: self.reported_kind,
            observations: Arc::clone(&self.observations),
            force_null_outputs: self.force_null_outputs,
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
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner.scan_nodes(project, label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_node_i64(project, property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_project_in(project, targets, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.inner.search_vectors(request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.inner.sort_rows(request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.inner.join_node_i64(request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.inner.group_node_i64(request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_scalar_program(&self) -> bool {
        true
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        self.observations.calls.fetch_add(1, Ordering::SeqCst);
        *self
            .observations
            .last_request
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(request.clone());
        let mut result = self.inner.execute_scalar_program(request, cancellation)?;
        if self.force_null_outputs {
            result.output_cells.fill(request.null_cell);
        }
        Ok(result)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_i64(values, validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.inner.expand_out(sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.inner
            .exact_l2(matrix, rows, dimension, query, cancellation)
    }
}

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: ProjectId::random(),
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
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_single_row(backend: &dyn ExecutionBackend, query: &str) -> Result<Vec<ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph, backend))?;
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal("native map query did not return one batch"));
    };
    if batch.row_count != 1 || batch.columns.iter().any(|column| column.values.len() != 1) {
        return Err(Error::internal(
            "native map query did not return exactly one row",
        ));
    }
    Ok(batch
        .columns
        .iter()
        .map(|column| column.values[0].clone())
        .collect())
}

fn execute_query_error(backend: &dyn ExecutionBackend, query: &str) -> Error {
    let graph = GraphStore::default();
    QueryEngine
        .execute(query, &mut context(&graph, backend))
        .expect_err("query must produce its native runtime type error")
}

fn null() -> ResultValue {
    ResultValue::Scalar(ScalarValue::Null)
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn string(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn input_string(
    request: &ResidentScalarProgramRequest,
    operand: ResidentScalarProgramOperand,
) -> Result<String> {
    let ResidentScalarProgramOperand::Cell(cell) = operand else {
        return Err(Error::internal(
            "static property key was not an immutable cell",
        ));
    };
    let value = request
        .scalar_cells
        .get(cell as usize)
        .copied()
        .ok_or_else(|| Error::internal("static property key cell is undefined"))?;
    if value.tag != ResidentScalarCellTag::String {
        return Err(Error::internal("static property key cell is not a string"));
    }
    let identity = usize::try_from(value.payload)
        .map_err(|_| Error::internal("static property key identity is unaddressable"))?;
    let start = request
        .string_offsets
        .get(identity.saturating_sub(1))
        .copied()
        .ok_or_else(|| Error::internal("static property key start is undefined"))?
        as usize;
    let end = start
        .checked_add(value.auxiliary as usize)
        .ok_or_else(|| Error::internal("static property key length overflowed"))?;
    let bytes = request
        .string_bytes
        .get(start..end)
        .ok_or_else(|| Error::internal("static property key bytes are undefined"))?;
    String::from_utf8(bytes.to_vec())
        .map_err(|_| Error::internal("static property key is not valid UTF-8"))
}

#[test]
fn strict_cpu_static_property_route_is_one_native_scalar_call() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu(false);
    let observations = backend.observations();
    let values = execute_single_row(
        &backend,
        "WITH {existing: 42, notMissing: null} AS m \
         RETURN m.missing, m.notMissing, m.existing",
    )?;
    assert_eq!(values, vec![null(), null(), integer(42)]);
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);

    let request = observations
        .last_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("native scalar request was not observed"))?;
    assert_eq!(request.instructions.len(), 3);
    assert!(
        request
            .instructions
            .iter()
            .all(|instruction| instruction.opcode == ResidentScalarProgramOpcode::ListIndex)
    );
    assert_eq!(
        request
            .instructions
            .iter()
            .map(|instruction| input_string(&request, instruction.right))
            .collect::<Result<Vec<_>>>()?,
        vec!["missing", "notMissing", "existing"]
    );
    Ok(())
}

#[test]
fn backend_returned_frame_is_the_only_final_value_source() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu(true);
    let observations = backend.observations();
    let values = execute_single_row(
        &backend,
        "WITH {existing: 42} AS map RETURN map.existing AS result",
    )?;

    // Deliberately replacing the backend root with its canonical NULL cell changes the public
    // result. If Rust had looked up the compiler-owned map, this would incorrectly remain 42.
    assert_eq!(values, vec![null()]);
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn strict_cpu_static_property_semantics_cover_nested_null_and_identifiers() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu(false);
    let observations = backend.observations();
    let cases = [
        ("WITH null AS m RETURN m.missing", vec![null()]),
        (
            "WITH [123, {existing: 42, notMissing: null}] AS list \
             RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing",
            vec![null(), null(), integer(42)],
        ),
        (
            "WITH {name: 'Mats', Name: 'Pontus', null: 'lower', NULL: 'upper'} AS map \
             RETURN map.name, map.Name, map.nAMe, map.`null`, map.`NULL`",
            vec![
                string("Mats"),
                string("Pontus"),
                null(),
                string("lower"),
                string("upper"),
            ],
        ),
    ];
    let case_count = cases.len();
    for (query, expected) in cases {
        assert_eq!(execute_single_row(&backend, query)?, expected, "{query}");
    }
    assert_eq!(observations.calls.load(Ordering::SeqCst), case_count);
    Ok(())
}

#[test]
fn strict_cpu_properties_map_identity_stays_in_the_native_scalar_frame() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu(false);
    let observations = backend.observations();
    let values = execute_single_row(
        &backend,
        "RETURN properties({name: 'Popeye', level: 9001}) AS m",
    )?;
    assert_eq!(
        values,
        vec![ResultValue::Map(BTreeMap::from([
            ("level".to_owned(), integer(9001)),
            ("name".to_owned(), string("Popeye")),
        ]))]
    );
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    let request = observations
        .last_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("native properties(map) request was not observed"))?;
    assert!(request.instructions.is_empty());
    let [ResidentScalarProgramOperand::Cell(output)] = request.output_values.as_slice() else {
        return Err(Error::internal(
            "native properties(map) did not publish one immutable cell",
        ));
    };
    assert_eq!(
        request.scalar_cells[*output as usize].tag,
        ResidentScalarCellTag::Map
    );
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_exact_map_property_selectors() {
    assert_certified_report_identities(OFFICIAL_MAP_IDENTITIES);
}

#[test]
fn non_map_property_still_fails_before_native_dispatch() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu(false);
    let observations = backend.observations();
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "WITH 123 AS nonMap RETURN nonMap.num",
            &mut context(&graph, &backend),
        )
        .expect_err("non-map property access must remain a static type error");
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(observations.calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_static_property_matches_native_cpu_reference() -> Result<()> {
    let cpu = ObservedScalarBackend::strict_cpu(false);
    let observations = cpu.observations();
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 32 * 1024 * 1024)?;
    let queries = [
        "WITH {existing: 42, notMissing: null} AS m \
         RETURN m.missing, m.notMissing, m.existing",
        "WITH null AS m RETURN m.missing",
        "WITH [123, {existing: 42, notMissing: null}] AS list \
         RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing",
        "WITH {name: 'Mats', Name: 'Pontus', null: 'lower', NULL: 'upper'} AS map \
         RETURN map.name, map.Name, map.nAMe, map.`null`, map.`NULL`",
        "RETURN properties({name: 'Popeye', level: 9001}) AS m",
    ];
    for query in queries {
        let cpu_values = execute_single_row(&cpu, query)?;
        let metal_values = execute_single_row(&metal, query)?;
        assert_eq!(metal_values, cpu_values, "{query}");
    }
    assert_eq!(observations.calls.load(Ordering::SeqCst), queries.len());
    Ok(())
}

#[test]
fn strict_cpu_dynamic_index_preserves_distinct_runtime_type_errors() {
    let backend = ObservedScalarBackend::strict_cpu(false);
    let cases = [
        (
            "WITH true AS value, 0 AS idx RETURN value[idx]",
            "InvalidArgumentType: indexing requires a LIST, MAP, NODE, or RELATIONSHIP",
        ),
        (
            "WITH 123 AS value, 0 AS idx RETURN value[idx]",
            "InvalidArgumentType: indexing requires a LIST, MAP, NODE, or RELATIONSHIP",
        ),
        (
            "WITH 4.7 AS value, 0 AS idx RETURN value[idx]",
            "InvalidArgumentType: indexing requires a LIST, MAP, NODE, or RELATIONSHIP",
        ),
        (
            "WITH 'value' AS value, 0 AS idx RETURN value[idx]",
            "InvalidArgumentType: indexing requires a LIST, MAP, NODE, or RELATIONSHIP",
        ),
        (
            "WITH [1, 2, 3] AS value, true AS idx RETURN value[idx]",
            "InvalidArgumentType: list index requires an INTEGER",
        ),
        (
            "WITH [1, 2, 3] AS value, 4.7 AS idx RETURN value[idx]",
            "InvalidArgumentType: list index requires an INTEGER",
        ),
        (
            "WITH [1, 2, 3] AS value, '1' AS idx RETURN value[idx]",
            "InvalidArgumentType: list index requires an INTEGER",
        ),
        (
            "WITH [1, 2, 3] AS value, [1] AS idx RETURN value[idx]",
            "InvalidArgumentType: list index requires an INTEGER",
        ),
        (
            "WITH [1, 2, 3] AS value, {x: 1} AS idx RETURN value[idx]",
            "InvalidArgumentType: list index requires an INTEGER",
        ),
    ];
    for (query, expected) in cases {
        let error = execute_query_error(&backend, query);
        assert_eq!(error.code, ErrorCode::QueryType, "{query}");
        assert_eq!(error.message, expected, "{query}");
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_dynamic_index_errors_exactly_match_cpu() -> Result<()> {
    let cpu = ObservedScalarBackend::strict_cpu(false);
    let metal = MetalBackend::new(0, 128 * 1024 * 1024, 32 * 1024 * 1024)?;
    let queries = [
        "WITH true AS value, 0 AS idx RETURN value[idx]",
        "WITH 123 AS value, 0 AS idx RETURN value[idx]",
        "WITH 4.7 AS value, 0 AS idx RETURN value[idx]",
        "WITH 'value' AS value, 0 AS idx RETURN value[idx]",
        "WITH [1, 2, 3] AS value, true AS idx RETURN value[idx]",
        "WITH [1, 2, 3] AS value, 4.7 AS idx RETURN value[idx]",
        "WITH [1, 2, 3] AS value, '1' AS idx RETURN value[idx]",
        "WITH [1, 2, 3] AS value, [1] AS idx RETURN value[idx]",
        "WITH [1, 2, 3] AS value, {x: 1} AS idx RETURN value[idx]",
    ];
    for query in queries {
        let cpu_error = execute_query_error(&cpu, query);
        let metal_error = execute_query_error(&metal, query);
        assert_eq!(metal_error.code, cpu_error.code, "{query}");
        assert_eq!(metal_error.message, cpu_error.message, "{query}");
    }
    Ok(())
}
