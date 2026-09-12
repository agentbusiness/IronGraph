//! Metal-first complete-residency backend using Candle's production Metal kernels.

use std::{collections::BTreeMap, mem, sync::Arc};

use candle_core::{Device, Tensor};
use parking_lot::ReentrantMutexGuard;
use tokio_util::sync::CancellationToken;

use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    graph::{IvfPqBuildPlan, IvfPqConfig, IvfPqIndex, LayerMask, VectorIndex},
    types::{LabelId, PropertyId},
};

use super::{
    BackendKind, CompareOp, DeviceMemoryGovernor, DistanceBatch, ExecutionBackend,
    ResidentBooleanAggregateProgramRequest, ResidentBooleanProgramRequest,
    ResidentBooleanProgramResult, ResidentCreateNodeRequest, ResidentCreateNodeResult,
    ResidentDeleteRequest, ResidentDeleteResult, ResidentGraphProcedure,
    ResidentGraphProcedureRequest, ResidentGraphProcedureResult, ResidentGroup,
    ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
    ResidentMultiwayIntersectionRequest, ResidentNodeGroupPipelineRequest,
    ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentNullableRelationRequest,
    ResidentNullableRelationResult, ResidentPatternCountRequest, ResidentPatternCountResult,
    ResidentPatternPairPredicateRequest, ResidentPatternPairPredicateResult,
    ResidentPatternPredicateRequest, ResidentPatternPredicateResult, ResidentProcedureTableRequest,
    ResidentProcedureTableResult, ResidentProjectDelta, ResidentProjectImage,
    ResidentQuantifierProgramRequest, ResidentQuantifierProgramResult, ResidentRangeProgramRequest,
    ResidentRangeProgramResult, ResidentRowMutationRequest, ResidentRowMutationResult,
    ResidentRowProgramRequest, ResidentRowProgramResult, ResidentScalarInput,
    ResidentScalarProgramRequest, ResidentScalarProgramResult, ResidentSegmentedAggregationRequest,
    ResidentSegmentedAggregationResult, ResidentSortRequest, ResidentSortResult,
    ResidentTemporalArithmeticRequest, ResidentTemporalArithmeticResult,
    ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult,
    ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
    ResidentToBooleanRequest, ResidentToBooleanResult, ResidentVariablePathRequest,
    ResidentVariablePathResult, ResidentVectorQuery, ResidentVectorResult,
    accelerator::{
        CandleIvfPqBuildKernel, CandleResident, candle_error, exact_l2,
        execute_metal_boolean_aggregate_program, execute_metal_boolean_program,
        execute_metal_create_node, execute_metal_procedure_table, execute_metal_quantifier_program,
        execute_metal_range_program, execute_metal_scalar_program,
        execute_metal_segmented_aggregation, execute_metal_temporal_arithmetic_program,
        execute_metal_temporal_value_program, execute_metal_to_boolean,
        execute_metal_variable_path, filter_tensor, intersect_sorted_node_sets,
        metal_breadth_first_scratch_bytes, metal_degree_scratch_bytes,
        metal_depth_first_scratch_bytes, metal_graph_metrics_scratch_bytes,
        metal_kcore_scratch_bytes, metal_louvain_scratch_bytes, metal_pagerank_scratch_bytes,
        metal_quantifier_program_scratch_bytes, metal_resident_row_program_scratch_bytes,
        metal_scc_scratch_bytes, metal_shortest_path_scratch_bytes,
        metal_temporal_arithmetic_scratch_bytes, metal_unit_dijkstra_scratch_bytes,
        metal_wcc_scratch_bytes, metal_weighted_dijkstra_scratch_bytes,
        prepare_metal_graph_algorithm_kernels, prepare_metal_segmented_aggregation,
        prepare_metal_sort_kernels, prepare_metal_temporal_value_program,
    },
    device::construct_metal_device,
    ensure_graph_execution, ensure_not_cancelled, metal_sort_scratch_bytes, operator_scratch_bytes,
    validate_matrix, vector_query_scratch_bytes,
};

/// Production Metal backend. Construction fails unless a real Metal device is available.
pub struct MetalBackend {
    device: Device,
    /// Serialises every command stream on `device`. See [`crate::metal_gate`]: Candle's Metal
    /// backend shares one buffer pool pair and one `MTLResidencySet` per device and mutates them
    /// without a covering lock, so two threads on one device is undefined behaviour — and
    /// `pin_project` hands this exact device to every concurrent read.
    gate: &'static parking_lot::ReentrantMutex<()>,
    governor: DeviceMemoryGovernor,
    resident: BTreeMap<ProjectId, Arc<CandleResident>>,
    /// The device's maximum single-buffer length, used as the native command-buffer scratch ceiling
    /// so a high-memory GPU is allowed the largest command it can physically hold.
    max_buffer_length: usize,
}

const METAL_KERNEL_SOURCES: [(&str, &str); 4] = [
    (
        "operators.metal",
        include_str!("../../../kernels/metal/operators.metal"),
    ),
    (
        "graph_paths.metal",
        include_str!("../../../kernels/metal/graph_paths.metal"),
    ),
    (
        "graph_components_metrics.metal",
        include_str!("../../../kernels/metal/graph_components_metrics.metal"),
    ),
    (
        "graph_louvain.metal",
        include_str!("../../../kernels/metal/graph_louvain.metal"),
    ),
];

fn metal_kernel_sources_hash(sources: &[(&str, &str)]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new();
    for (name, source) in sources {
        hasher.update(&(name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update(&(source.len() as u64).to_le_bytes());
        hasher.update(source.as_bytes());
    }
    hasher.finalize()
}

impl std::fmt::Debug for MetalBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetalBackend")
            .field("project_count", &self.resident.len())
            .field("resident_revision", &self.resident_revision())
            .finish_non_exhaustive()
    }
}

/// Incremental peak introduced by device-authored mutation effects: canonical membership/value
/// lanes, one effect bit per attempted intent, and the additional sealed descriptor/receipt for
/// every command. The ordinary pipeline and continuation reservations account for the pre-effect
/// packet; this delta keeps admission honest after widening that packet.
fn metal_mutation_effect_scratch_bytes(request: &ResidentNodePipelineRequest) -> Result<usize> {
    let Some(program) = &request.mutation else {
        return Ok(0);
    };
    let overflow = || {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal mutation effect scratch accounting overflow",
        )
    };
    let static_membership_lanes = program
        .commands
        .iter()
        .try_fold(0_usize, |total, command| {
            let lanes = match &command.operation {
                super::ResidentMutationOperation::AddLabels { labels, .. }
                | super::ResidentMutationOperation::RemoveLabels { labels, .. } => labels.len(),
                super::ResidentMutationOperation::RemoveProperty { .. } => 1,
                _ => 0,
            };
            total.checked_add(lanes).ok_or_else(overflow)
        })?;
    let static_effect_lanes = if static_membership_lanes == 0 {
        0
    } else {
        static_membership_lanes
            .checked_add(1)
            .ok_or_else(overflow)?
    };
    // Typed SET widened each target from identity-only (three lanes) to identity plus the
    // canonical destination cell (three additional lanes).
    let typed_current_lanes = program
        .commands
        .iter()
        .filter(|command| {
            matches!(
                command.operation,
                super::ResidentMutationOperation::SetProperty { .. }
            )
        })
        .count()
        .checked_mul(3)
        .ok_or_else(overflow)?;
    let canonical_words = request
        .max_output_rows
        .checked_mul(static_effect_lanes.max(typed_current_lanes))
        .ok_or_else(overflow)?;
    let intent_words = program.max_intents;
    // Nine output words, six typed descriptor words, and one static command word per effect
    // obligation. Taking all three is conservative across the mutually exclusive packet paths.
    let receipt_words = program
        .commands
        .len()
        .checked_mul(16)
        .ok_or_else(overflow)?;
    canonical_words
        .checked_add(intent_words)
        .and_then(|words| words.checked_add(receipt_words))
        .and_then(|words| words.checked_mul(mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(overflow)
}

impl MetalBackend {
    pub fn new(
        device_ordinal: usize,
        memory_limit_bytes: usize,
        reserved_bytes: usize,
    ) -> Result<Self> {
        Self::with_governor(
            device_ordinal,
            DeviceMemoryGovernor::new(memory_limit_bytes, reserved_bytes),
        )
    }

    pub fn with_governor(
        device_ordinal: usize,
        mut governor: DeviceMemoryGovernor,
    ) -> Result<Self> {
        let ordinal = u32::try_from(device_ordinal).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal device ordinal exceeds u32",
            )
        })?;
        let device = construct_metal_device(ordinal)
            .map_err(|error| Error::new(ErrorCode::GpuAdmissionFailure, error))?;
        let recommended = device
            .as_metal_device()
            .map_err(candle_error)?
            .metal_device()
            .recommended_max_working_set_size();
        if recommended > 0 {
            governor.cap_limit(recommended)?;
        }
        // The cap above is computed once, here, from what the DEVICE can hold. It is correct at this
        // instant and never revisited, so a process that starts on an idle machine keeps the whole
        // budget after other tenants have taken the memory. Measured on a 48 GB host: 37.4 GB
        // admitted against 19.3 GB actually free, with swap 91% used — 1.9x over-admission, which is
        // more than enough to have the process killed while its own ledger still says there is room.
        //
        // The probe closes that by bounding each admission's INCREASE against free memory at the
        // moment of admission. It lives here rather than in the execution crate because reading host
        // memory is platform-specific and that crate is not.
        governor.set_host_available_probe(std::sync::Arc::new(host_available_bytes));
        // The device's recommended working set is the largest amount of memory it wants in use, so
        // it is also the natural ceiling for one command's scratch buffer: a big-memory GPU is
        // allowed a proportionally big command instead of a fixed sub-hardware constant.
        let max_buffer_length = usize::try_from(recommended).unwrap_or(usize::MAX);
        prepare_metal_sort_kernels(&device)?;
        prepare_metal_graph_algorithm_kernels(&device)?;
        let gate = crate::metal_gate::metal_device_gate(&device).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal backend constructed a non-Metal Candle device",
            )
        })?;
        Ok(Self {
            device,
            gate,
            governor,
            resident: BTreeMap::new(),
            max_buffer_length,
        })
    }

    /// Hash of every backend-specific Metal source library shipped for tuned deployments.
    ///
    /// Keep the library name and byte length in the transcript so moving code between source
    /// units, adding a new unit, or changing any graph kernel invalidates the same production
    /// fingerprint used by deployment/cache compatibility checks.
    #[must_use]
    pub fn kernel_source_hash() -> blake3::Hash {
        metal_kernel_sources_hash(&METAL_KERNEL_SOURCES)
    }

    /// Builds deterministic flat IVF-PQ pages with device-side assignment and bounded admitted
    /// scratch. The returned pages remain derived and are published by the normal index lifecycle.
    pub fn build_ivf_pq(&self, source: &VectorIndex, config: IvfPqConfig) -> Result<IvfPqIndex> {
        self.build_ivf_pq_cancellable(source, config, &CancellationToken::new())
    }

    fn build_ivf_pq_cancellable(
        &self,
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<IvfPqIndex> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        let plan = IvfPqBuildPlan::for_shape(source.len() as u64, source.dimension(), config)?;
        let scratch = usize::try_from(plan.peak_scratch_bytes).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "IVF-PQ scratch estimate exceeds this process address space",
            )
        })?;
        let derived = usize::try_from(plan.derived_bytes).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "IVF-PQ derived pages exceed this process address space",
            )
        })?;
        let _derived = self.governor.reserve_staging(derived)?;
        let _reservation = self.governor.reserve_scratch(scratch)?;
        let distance_tile = plan.assignment_tile_bytes.max(size_of::<f32>());
        let mut kernel = CandleIvfPqBuildKernel::new(&self.device, distance_tile)?;
        IvfPqIndex::build_with_kernel_cancellable(source, config, &mut kernel, cancellation)
    }

    fn project(&self, project: ProjectId) -> Result<&CandleResident> {
        self.resident.get(&project).map(Arc::as_ref).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })
    }

    /// Holds this backend's Candle device for one whole operation, readback included.
    ///
    /// Candle sweeps and frees its buffer pools inside the readback (`flush_and_wait_current` ->
    /// `drop_unused_buffers`), so a guard that ends before the readback does not serialise the
    /// part that actually races. The gate is reentrant: a gated method may call another.
    fn gate(&self) -> ReentrantMutexGuard<'static, ()> {
        self.gate.lock()
    }

    fn total_after(&self, project: ProjectId, replacement_bytes: usize) -> usize {
        self.resident
            .iter()
            .filter(|(id, _)| **id != project)
            .map(|(_, resident)| resident.allocated_bytes)
            .fold(replacement_bytes, usize::saturating_add)
    }
}

impl ExecutionBackend for MetalBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn build_ivf_pq(
        &self,
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<IvfPqIndex> {
        self.build_ivf_pq_cancellable(source, config, cancellation)
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.governor.available_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<super::ScratchReservation> {
        self.governor.reserve_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.resident
            .get(&project)
            .map(|resident| resident.allocated_bytes)
    }

    fn shared_project_backing(&self, project: ProjectId) -> Option<super::ResidentSharedBacking> {
        let resident = self.resident.get(&project)?;
        Some(super::ResidentSharedBacking {
            graph: resident.shared_graph_backing()?,
            temporal: resident.shared_temporal_backings(),
            vectors: resident.shared_vector_backings(),
        })
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        let resident = self.resident.get(&project).cloned().ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })?;
        let governor = self.governor.pin_generation(resident.allocated_bytes)?;
        Ok(Box::new(Self {
            device: self.device.clone(),
            gate: self.gate,
            governor,
            resident: BTreeMap::from([(project, resident)]),
            max_buffer_length: self.max_buffer_length,
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        let _gate = self.gate();
        let planned = CandleResident::planned_bytes(&image);
        let staging = self.governor.reserve_staging(planned)?;
        let project = image.project;
        let staged = CandleResident::upload(image, &self.device)?;
        let total = self.total_after(project, staged.allocated_bytes);
        let mut governor = self.governor.clone();
        governor.publish_staging(staging, staged.allocated_bytes, total, || {
            let old = self.resident.insert(project, Arc::new(staged));
            drop(old);
            self.device.synchronize().map_err(candle_error)
        })?;
        Ok(())
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        let _gate = self.gate();
        let planned = images.iter().try_fold(0_usize, |total, image| {
            total
                .checked_add(CandleResident::planned_bytes(image))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "replacement Metal project bytes overflow",
                    )
                })
        })?;
        let staging = self.governor.reserve_staging(planned)?;
        let mut replacement = BTreeMap::new();
        for image in images {
            let project = image.project;
            let resident = CandleResident::upload(image, &self.device)?;
            if replacement.insert(project, Arc::new(resident)).is_some() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot contains a duplicate project ID",
                ));
            }
        }
        let actual = replacement.values().try_fold(0_usize, |total, resident| {
            total.checked_add(resident.allocated_bytes).ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "replacement Metal project bytes overflow",
                )
            })
        })?;
        let mut governor = self.governor.clone();
        governor.publish_staging(staging, actual, actual, || {
            let old = mem::replace(&mut self.resident, replacement);
            drop(old);
            self.device.synchronize().map_err(candle_error)
        })?;
        Ok(())
    }

    fn apply_project_delta(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        let _gate = self.gate();
        let project = delta.project;
        let current = self.resident.get(&project).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete Metal-resident image"),
            )
        })?;
        let host_complete = current.delta_publish_is_host_complete(&delta);
        let staged_bytes = current.planned_delta_staging_bytes(&delta)?;
        let reservation = self.governor.reserve_staging(staged_bytes)?;
        let staged = current.stage_delta(&delta, &self.device)?;
        let total = self.total_after(project, staged.allocated_bytes);
        let mut governor = self.governor.clone();
        governor.publish_staging(reservation, staged_bytes, total, || {
            let old = self.resident.insert(project, Arc::new(staged));
            drop(old);
            if host_complete {
                Ok(())
            } else {
                self.device.synchronize().map_err(candle_error)
            }
        })?;
        Ok(())
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        let _gate = self.gate();
        let total = self
            .resident
            .iter()
            .filter(|(resident_project, _)| **resident_project != project)
            .map(|(_, resident)| resident)
            .map(|resident| resident.allocated_bytes)
            .fold(0_usize, usize::saturating_add);
        let staging = self.governor.reserve_staging(0)?;
        let mut governor = self.governor.clone();
        governor.publish_staging(staging, 0, total, || {
            let old = self.resident.remove(&project);
            drop(old);
            self.device.synchronize().map_err(candle_error)
        })?;
        Ok(())
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        for resident in self.resident.values_mut() {
            let resident = Arc::make_mut(resident);
            resident.bookmark = bookmark;
        }
    }

    fn resident_revision(&self) -> Option<u64> {
        self.resident
            .values()
            .map(|resident| resident.revision)
            .max()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.resident
            .get(&project)
            .map(|resident| resident.revision)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.resident
            .get(&project)
            .map(|resident| resident.bookmark)
    }

    fn scan_nodes(
        &self,
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(project)?
            .scan_nodes(label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(project)?
            .filter_node_i64(property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_out(&self.device, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_in(&self.device, targets, cancellation)
    }

    fn expand_project_out_bounded(
        &self,
        project: ProjectId,
        sources: &[u32],
        maximum_rows: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_out_bounded(sources, maximum_rows, cancellation)
    }

    fn expand_project_in_bounded(
        &self,
        project: ProjectId,
        targets: &[u32],
        maximum_rows: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.project(project)?
            .expand_in_bounded(targets, maximum_rows, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(resident.vector_row_count(request.property))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal vector-pipeline scratch row count overflow",
                )
            })?;
        let output_rows = request
            .limit
            .checked_mul(request.query_count)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal vector result shape overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(vector_query_scratch_bytes(
            input_rows,
            resident.vector_dimension(request.property),
            request.query_count,
            output_rows,
        )?)?;
        resident.search_vectors(&self.device, request, cancellation)
    }

    fn intersect_sorted_node_sets(
        &self,
        request: &ResidentMultiwayIntersectionRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        self.project(request.project)?;
        let input_rows = request.sorted_sets.iter().try_fold(0_usize, |total, set| {
            total.checked_add(set.len()).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal multiway-intersection input size overflow",
                )
            })
        })?;
        let maximum_output = request
            .sorted_sets
            .iter()
            .map(Vec::len)
            .min()
            .unwrap_or(0)
            .min(request.max_output_rows.saturating_add(1));
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            160,
            maximum_output,
            24,
        )?)?;
        intersect_sorted_node_sets(&self.device, request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self
            .governor
            .reserve_scratch(metal_sort_scratch_bytes(request)?)?;
        self.project(request.project)?
            .sort_rows(&self.device, request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        let _gate = self.gate();
        let input_rows = request
            .left_rows
            .len()
            .checked_add(request.right_rows.len())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal join scratch accounting overflow",
                )
            })?;
        let maximum_output = request
            .left_rows
            .len()
            .checked_mul(request.right_rows.len())
            .map_or(request.max_pairs, |pairs| pairs.min(request.max_pairs));
        let maximum_output = maximum_output.min(u32::MAX as usize - 1);
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            256,
            maximum_output,
            64,
        )?)?;
        self.project(request.project)?
            .join_node_i64(&self.device, request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            request.rows.len(),
            384,
            request.rows.len(),
            96,
        )?)?;
        self.project(request.project)?
            .group_node_i64(&self.device, request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let relationship_count = request.relationship_column_count();
        let maximum_rows =
            request.maximum_pipeline_rows(resident.node_count(), resident.edge_count())?;
        let projected_width = request
            .integer_projections
            .len()
            .saturating_mul(size_of::<i64>() + size_of::<u8>())
            .saturating_add(request.property_null_projections.len())
            .saturating_add(
                relationship_count
                    .saturating_mul(2)
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(size_of::<u32>());
        let input_rows = resident
            .node_count()
            .checked_add(maximum_rows)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-pipeline scratch row count overflow",
                )
            })?;
        let pipeline_scratch =
            operator_scratch_bytes(input_rows, 160, maximum_rows, projected_width)?;
        let effect_scratch = metal_mutation_effect_scratch_bytes(request)?;
        let segmented_value_scratch = request
            .mutation
            .as_ref()
            .and_then(|program| program.continuation.as_ref())
            .and_then(|continuation| continuation.segmented_value_program.as_ref())
            .map_or(Ok(0), |program| {
                prepare_metal_segmented_aggregation(resident, program)
                    .map(|prepared| prepared.scratch_bytes())
            })?;
        let unit_value_scratch = request
            .mutation
            .as_ref()
            .and_then(|program| program.continuation.as_ref())
            .and_then(|continuation| continuation.value_program.as_ref())
            .map_or(Ok(0), |program| {
                metal_quantifier_program_scratch_bytes(resident, program)
            })?;
        let scratch_bytes = pipeline_scratch
            .checked_add(request.mutation_continuation_scratch_bytes()?)
            .and_then(|bytes| bytes.checked_add(effect_scratch))
            .and_then(|bytes| bytes.checked_add(segmented_value_scratch))
            .and_then(|bytes| bytes.checked_add(unit_value_scratch))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal mutation/effect scratch reservation overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_node_pipeline(&self.device, request, cancellation)
    }

    fn execute_row_program(
        &self,
        request: &ResidentRowProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.project)?;
        let maximum_rows = request
            .input
            .maximum_pipeline_rows(resident.node_count(), resident.edge_count())?;
        let admitted_rows = maximum_rows.min(request.input.max_output_rows);
        let relationship_count = request.input.relationship_column_count();
        let projected_width = relationship_count
            .checked_mul(2)
            .and_then(|columns| columns.checked_mul(size_of::<u32>()))
            .and_then(|bytes| bytes.checked_add(size_of::<u32>()))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal typed-row wrapped-pipeline width overflow",
                )
            })?;
        let pipeline_input_rows =
            resident
                .node_count()
                .checked_add(maximum_rows)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Metal typed-row wrapped-pipeline row count overflow",
                    )
                })?;
        let pipeline_scratch =
            operator_scratch_bytes(pipeline_input_rows, 160, maximum_rows, projected_width)?;
        let row_scratch = metal_resident_row_program_scratch_bytes(request, admitted_rows)?;
        let combined_scratch = pipeline_scratch.checked_add(row_scratch).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal typed-row combined scratch accounting overflow",
            )
        })?;
        let mut scratch = self.governor.reserve_scratch(combined_scratch)?;
        resident.execute_row_program(&self.device, request, &mut scratch, cancellation)
    }

    fn supports_native_quantifier_program(&self) -> bool {
        true
    }

    fn supports_native_quantifier_entity_source(&self) -> bool {
        true
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        if resident.bookmark != request.generation.bookmark
            || resident.revision != request.generation.graph_revision
            || resident.layout_version != request.generation.layout_version
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier program expected a different Metal bookmark, graph revision, or layout version",
            ));
        }
        let scratch_bytes = metal_quantifier_program_scratch_bytes(resident, request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        execute_metal_quantifier_program(&self.device, resident, request, cancellation)
    }

    fn supports_nullable_relation_existing_relationship(&self) -> bool {
        true
    }

    fn supports_nullable_relation_relationship_endpoint_seed(&self) -> bool {
        true
    }

    fn supports_nullable_relation_predicates(&self) -> bool {
        true
    }

    fn supports_nullable_relation_scope_limit(&self) -> bool {
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
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        let scratch_bytes = resident.metal_nullable_relation_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_nullable_relation(&self.device, request, cancellation)
    }

    fn execute_delete_pipeline(
        &self,
        request: &ResidentDeleteRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentDeleteResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.project)?;
        let scratch_bytes = resident.metal_delete_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_delete_pipeline_metal(&self.device, request, cancellation)
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        let _gate = self.gate();
        ensure_not_cancelled(cancellation)?;
        request.validate()?;
        let resident = self.project(request.generation.project)?;
        let scratch_bytes = resident.metal_row_mutation_scratch_bytes(request)?;
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_row_mutation_metal(&self.device, request, cancellation)
    }

    fn execute_pattern_predicate(
        &self,
        request: &ResidentPatternPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPredicateResult> {
        let _gate = self.gate();
        let resident = self.project(request.project)?;
        let _scratch = self
            .governor
            .reserve_scratch(resident.metal_pattern_predicate_scratch_bytes(request)?)?;
        resident.execute_pattern_predicate(&self.device, request, cancellation)
    }

    fn execute_pattern_count(
        &self,
        request: &ResidentPatternCountRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternCountResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        self.project(request.project)?
            .execute_pattern_count(&self.device, request, cancellation)
    }

    fn execute_pattern_predicate_pairs(
        &self,
        request: &ResidentPatternPairPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPairPredicateResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        self.project(request.project)?
            .execute_pattern_predicate_pairs(&self.device, request, cancellation)
    }

    fn supports_native_variable_path(&self) -> bool {
        true
    }

    fn execute_variable_path(
        &self,
        request: &ResidentVariablePathRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        let _gate = self.gate();
        request.validate()?;
        // This is where a real Metal command buffer is built, so the fixed command-buffer scratch
        // ABI ceiling applies here (and only here — host executors have no such bound).
        request.enforce_native_scratch_abi(self.max_buffer_length)?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_variable_path(
            &self.device,
            self.project(request.project)?,
            request,
            cancellation,
        )
    }

    fn supports_native_to_boolean(&self) -> bool {
        true
    }

    fn execute_to_boolean(
        &self,
        request: &ResidentToBooleanRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentToBooleanResult> {
        let _gate = self.gate();
        let scratch = request.inputs.iter().try_fold(0_usize, |bytes, input| {
            let payload = match input {
                ResidentScalarInput::String(value) => value.len(),
                ResidentScalarInput::Null
                | ResidentScalarInput::Boolean(_)
                | ResidentScalarInput::Other => 0,
            };
            bytes
                .checked_add(payload.saturating_add(size_of::<u32>() + 2))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "Metal toBoolean scratch accounting overflow",
                    )
                })
        })?;
        let _scratch = self.governor.reserve_scratch(scratch)?;
        execute_metal_to_boolean(&self.device, request, cancellation)
    }

    fn supports_native_create_node(&self) -> bool {
        true
    }

    fn execute_create_node(
        &self,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_create_node(
            &self.device,
            self.project(request.project)?,
            request,
            cancellation,
        )
    }

    fn supports_native_temporal_value_program(&self) -> bool {
        true
    }

    fn supports_native_temporal_arithmetic_program(&self) -> bool {
        true
    }

    fn execute_temporal_arithmetic_program(
        &self,
        request: &ResidentTemporalArithmeticRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalArithmeticResult> {
        let _gate = self.gate();
        request.validate()?;
        ensure_not_cancelled(cancellation)?;
        let _scratch = self
            .governor
            .reserve_scratch(metal_temporal_arithmetic_scratch_bytes(request)?)?;
        execute_metal_temporal_arithmetic_program(
            &self.device,
            self.project(request.generation.project)?,
            request,
            cancellation,
        )
    }

    fn execute_temporal_value_program(
        &self,
        request: &ResidentTemporalValueProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let prepared = prepare_metal_temporal_value_program(request)?;
        let _scratch = self.governor.reserve_scratch(prepared.scratch_bytes())?;
        execute_metal_temporal_value_program(&self.device, request, &prepared, cancellation)
    }

    fn supports_native_boolean_program(&self) -> bool {
        true
    }

    fn execute_boolean_program(
        &self,
        request: &ResidentBooleanProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_boolean_program(&self.device, request, cancellation)
    }

    fn supports_native_scalar_program(&self) -> bool {
        true
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_scalar_program(&self.device, request, cancellation)
    }

    fn supports_native_boolean_aggregate_program(&self) -> bool {
        true
    }

    fn execute_boolean_aggregate_program(
        &self,
        request: &ResidentBooleanAggregateProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentBooleanProgramResult> {
        let _gate = self.gate();
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_boolean_aggregate_program(&self.device, request, cancellation)
    }

    fn supports_native_segmented_aggregation(&self) -> bool {
        true
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        let _gate = self.gate();
        request.validate()?;
        let resident = self.project(request.project)?;
        let prepared = prepare_metal_segmented_aggregation(resident, request)?;
        let _scratch = self.governor.reserve_scratch(prepared.scratch_bytes())?;
        execute_metal_segmented_aggregation(
            &self.device,
            resident,
            request,
            &prepared,
            cancellation,
        )
    }

    fn supports_native_procedure_table(&self) -> bool {
        true
    }

    fn execute_procedure_table(
        &self,
        request: &ResidentProcedureTableRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentProcedureTableResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_procedure_table(&self.device, request, cancellation)
    }

    fn supports_native_range_program(&self) -> bool {
        true
    }

    fn execute_range_program(
        &self,
        request: &ResidentRangeProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRangeProgramResult> {
        let _gate = self.gate();
        request.validate()?;
        let _scratch = self.governor.reserve_scratch(request.scratch_bytes()?)?;
        execute_metal_range_program(&self.device, request, cancellation)
    }

    fn execute_node_group_pipeline(
        &self,
        request: &ResidentNodeGroupPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        let _gate = self.gate();
        let resident = self.project(request.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                request
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-group scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            448,
            request.max_groups.min(input_rows),
            96,
        )?)?;
        resident.execute_node_group_pipeline(&self.device, request, cancellation)
    }

    fn execute_node_group_pipelines(
        &self,
        first: &ResidentNodeGroupPipelineRequest,
        additional: &[ResidentNodeGroupPipelineRequest],
        cancellation: &CancellationToken,
    ) -> Result<Vec<Vec<ResidentGroup>>> {
        let _gate = self.gate();
        let resident = self.project(first.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                first
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-group scratch row count overflow",
                )
            })?;
        let maximum_groups = std::iter::once(first)
            .chain(additional)
            .map(|request| request.max_groups)
            .max()
            .unwrap_or(0)
            .min(input_rows);
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            448,
            maximum_groups,
            96,
        )?)?;
        resident.execute_node_group_pipelines(&self.device, first, additional, cancellation)
    }

    fn execute_temporal_pipeline(
        &self,
        request: &ResidentTemporalPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
        let _gate = self.gate();
        let resident = self.project(request.input.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(
                request
                    .input
                    .expansion
                    .as_ref()
                    .map_or(0, |_| resident.edge_count()),
            )
            .and_then(|rows| {
                rows.checked_add(resident.temporal_row_count(request.target, request.property))
            })
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal resident-temporal scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            416,
            request.max_output_rows.min(input_rows),
            64,
        )?)?;
        resident.execute_temporal_pipeline(&self.device, request, cancellation)
    }

    fn execute_graph_procedure(
        &self,
        request: &ResidentGraphProcedureRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentGraphProcedureResult> {
        let _gate = self.gate();
        ensure_graph_execution(cancellation, request.deadline)?;
        let resident = self.project(request.project)?;
        let scratch_bytes = match request.procedure {
            ResidentGraphProcedure::Degree => metal_degree_scratch_bytes(resident.node_count())?,
            ResidentGraphProcedure::BreadthFirst { .. } => {
                metal_breadth_first_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DepthFirst { .. } => {
                metal_depth_first_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::ShortestPath { .. } => {
                metal_shortest_path_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DijkstraWeighted { .. } => {
                metal_weighted_dijkstra_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::DijkstraUnit { .. } => {
                metal_unit_dijkstra_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::Louvain => {
                metal_louvain_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
            ResidentGraphProcedure::PageRank { .. } => {
                metal_pagerank_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::WeaklyConnectedComponents => {
                metal_wcc_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::StronglyConnectedComponents => {
                metal_scc_scratch_bytes(resident.node_count())?
            }
            ResidentGraphProcedure::TriangleCount
            | ResidentGraphProcedure::ClusteringCoefficient => {
                metal_graph_metrics_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
            ResidentGraphProcedure::KCore => {
                metal_kcore_scratch_bytes(resident.node_count(), resident.edge_count())?
            }
        };
        let _scratch = self.governor.reserve_scratch(scratch_bytes)?;
        resident.execute_graph_procedure(&self.device, request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        let _gate = self.gate();
        if values.len() != validity.len() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "filter value and validity lengths differ",
            ));
        }
        if values.is_empty() {
            return Ok(Vec::new());
        }
        ensure_not_cancelled(cancellation)?;
        let _scratch = self.governor.reserve_scratch(
            values
                .len()
                .saturating_mul(size_of::<i64>() + size_of::<u8>() * 2),
        )?;
        let values =
            Tensor::from_slice(values, values.len(), &self.device).map_err(candle_error)?;
        let validity = Tensor::from_slice(
            &validity
                .iter()
                .map(|valid| u8::from(*valid))
                .collect::<Vec<_>>(),
            validity.len(),
            &self.device,
        )
        .map_err(candle_error)?;
        filter_tensor(&values, &validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        let _gate = self.gate();
        self.expand_project_out(ProjectId(uuid::Uuid::nil()), sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        let _gate = self.gate();
        validate_matrix(matrix, rows, dimension, query)?;
        ensure_not_cancelled(cancellation)?;
        let _scratch = self.governor.reserve_scratch(
            matrix
                .len()
                .saturating_add(query.len())
                .saturating_add(rows)
                .saturating_mul(size_of::<f32>()),
        )?;
        let distances = exact_l2(&self.device, matrix, rows, dimension, query)?;
        ensure_not_cancelled(cancellation)?;
        Ok(DistanceBatch { distances })
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use crate::graph::{IvfPqConfig, Similarity, VectorIndex};

    use super::*;

    #[test]
    fn production_kernel_fingerprint_covers_every_metal_library() {
        assert_eq!(METAL_KERNEL_SOURCES.len(), 4);
        assert_eq!(
            MetalBackend::kernel_source_hash(),
            metal_kernel_sources_hash(&METAL_KERNEL_SOURCES)
        );
        for changed in 0..METAL_KERNEL_SOURCES.len() {
            let mut sources = METAL_KERNEL_SOURCES;
            sources[changed].1 = "deliberately changed source";
            assert_ne!(
                MetalBackend::kernel_source_hash(),
                metal_kernel_sources_hash(&sources),
                "source library {changed} is absent from the production fingerprint"
            );
        }
    }

    #[test]
    #[ignore = "requires an available physical Metal device"]
    fn metal_builds_flat_ivf_pq_pages_with_bounded_device_assignment() -> Result<()> {
        let mut vectors = VectorIndex::new(4, Similarity::Euclidean)?;
        for row in 0..64_u64 {
            let value = row as f32 / 64.0;
            vectors.upsert(row + 1, &[value, 1.0 - value, value * value, 0.5], row + 1)?;
        }
        let backend = MetalBackend::new(0, 256 * 1024 * 1024, 8 * 1024 * 1024)?;
        let index = backend.build_ivf_pq(
            &vectors,
            IvfPqConfig {
                size_class_version: crate::graph::IVF_PQ_SIZE_CLASS_VERSION,
                coarse_centroids: 4,
                subquantizers: 2,
                bits_per_code: 4,
                probes: 2,
                candidate_budget: 32,
                iterations: 3,
                seed: 7,
            },
        )?;
        let hits = index.search(&vectors, &[0.5, 0.5, 0.25, 0.5], 4)?;
        assert_eq!(hits.len(), 4);
        Ok(())
    }
}

/// Bytes the host still has free, or `None` when it cannot be determined.
///
/// Free + inactive + speculative: inactive and speculative pages are reclaimable on demand, so
/// counting only `free` would understate what is available by most of the file cache and refuse
/// work that would have fit.
///
/// **The page size is read, never assumed.** Apple Silicon uses 16 KiB pages where x86 used 4 KiB,
/// and hardcoding 4096 here understates availability by exactly 4x — which is precisely the error
/// that produced a wrong headline number while this was being characterised.
///
/// `None` on any failure. The caller treats that as "no opinion" and admits as it would without a
/// probe: a memory check that cannot read memory must not become a new way to refuse work.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn host_available_bytes() -> Option<usize> {
    use std::process::Command;

    let page_size = Command::new("/usr/sbin/sysctl")
        .args(["-n", "vm.pagesize"])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|text| text.trim().parse::<usize>().ok())?;
    let output = Command::new("/usr/bin/vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let mut pages = 0_usize;
    let mut seen = 0_usize;
    for line in text.lines() {
        let Some((label, value)) = line.split_once(':') else {
            continue;
        };
        let label = label.trim().trim_matches('"');
        if !matches!(label, "Pages free" | "Pages inactive" | "Pages speculative") {
            continue;
        }
        let Ok(count) = value.trim().trim_end_matches('.').parse::<usize>() else {
            continue;
        };
        pages = pages.saturating_add(count);
        seen += 1;
    }
    // All three or nothing: a partial parse would silently understate availability, which is the
    // failure mode that refuses work that would have fit.
    (seen == 3).then(|| pages.saturating_mul(page_size))
}

#[cfg(not(any(target_os = "macos", target_os = "ios")))]
fn host_available_bytes() -> Option<usize> {
    None
}

#[cfg(test)]
mod host_memory_tests {
    #[test]
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn the_host_probe_reads_a_believable_number_from_the_real_machine() {
        // This exists because the bug it guards against already happened: the page size was assumed
        // to be 4096 while Apple Silicon uses 16384, understating availability by exactly 4x and
        // producing a wrong headline figure. An assumed-too-LARGE page size is detectable — it makes
        // "available" exceed physical RAM — so bound it against the real hardware size rather than
        // against a constant.
        let physical = std::process::Command::new("/usr/sbin/sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|out| String::from_utf8(out.stdout).ok())
            .and_then(|text| text.trim().parse::<usize>().ok())
            .expect("hw.memsize");
        let available = super::host_available_bytes().expect("probe should read this machine");
        assert!(
            available > 0 && available <= physical,
            "probe reported {available} bytes available on a machine with {physical} bytes of RAM"
        );
        // Not an assertion — the value itself, so a human can compare it against `vm_stat` when this
        // is being trusted for the first time on new hardware.
        println!(
            "host_available_bytes = {available} ({:.1} GB) of {:.1} GB physical",
            available as f64 / 2f64.powi(30),
            physical as f64 / 2f64.powi(30)
        );
    }
}
