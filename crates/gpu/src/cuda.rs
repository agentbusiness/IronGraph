//! CUDA complete-residency backend using Candle's production CUDA kernels.

use std::{collections::BTreeMap, sync::Arc};

use candle_core::{Device, Tensor};
use tokio_util::sync::CancellationToken;

use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result,
    graph::{IvfPqBuildPlan, IvfPqConfig, IvfPqIndex, LayerMask, VectorIndex},
    types::{LabelId, PropertyId},
};

use super::{
    BackendKind, CompareOp, DeviceMemoryGovernor, DistanceBatch, ExecutionBackend,
    ResidentAggregate, ResidentGraphProcedureRequest, ResidentGraphProcedureResult, ResidentGroup,
    ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
    ResidentMultiwayIntersectionRequest, ResidentNodeGroupPipelineRequest,
    ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectDelta,
    ResidentProjectImage, ResidentSortRequest, ResidentSortResult, ResidentTemporalPipelineRequest,
    ResidentTemporalPipelineResult, ResidentVectorQuery, ResidentVectorResult,
    accelerator::{
        CandleIvfPqBuildKernel, CandleResident, candle_error, exact_l2, filter_tensor,
        intersect_sorted_node_sets,
    },
    ensure_graph_execution, ensure_not_cancelled, operator_scratch_bytes, validate_matrix,
    vector_query_scratch_bytes,
};

/// Complete-residency CUDA execution backend.
pub struct CudaBackend {
    device: Device,
    governor: DeviceMemoryGovernor,
    resident: BTreeMap<ProjectId, Arc<CandleResident>>,
}

impl std::fmt::Debug for CudaBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaBackend")
            .field("project_count", &self.resident.len())
            .field("resident_revision", &self.resident_revision())
            .finish_non_exhaustive()
    }
}

impl CudaBackend {
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

    pub fn with_governor(device_ordinal: usize, governor: DeviceMemoryGovernor) -> Result<Self> {
        let device = Device::new_cuda(device_ordinal).map_err(candle_error)?;
        Ok(Self {
            device,
            governor,
            resident: BTreeMap::new(),
        })
    }

    /// Builds deterministic flat IVF-PQ pages with device-side assignment and bounded admitted
    /// scratch. CUDA hardware and recall/throughput remain separate acceptance gates.
    pub fn build_ivf_pq(&self, source: &VectorIndex, config: IvfPqConfig) -> Result<IvfPqIndex> {
        self.build_ivf_pq_cancellable(source, config, &CancellationToken::new())
    }

    fn build_ivf_pq_cancellable(
        &self,
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<IvfPqIndex> {
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
                format!("project {project} has no complete CUDA-resident image"),
            )
        })
    }

    fn total_after(&self, project: ProjectId, replacement_bytes: usize) -> usize {
        self.resident
            .iter()
            .filter(|(id, _)| **id != project)
            .map(|(_, resident)| resident.allocated_bytes)
            .fold(replacement_bytes, usize::saturating_add)
    }
}

impl ExecutionBackend for CudaBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cuda
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

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        let resident = self.resident.get(&project).cloned().ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete CUDA-resident image"),
            )
        })?;
        let governor = self.governor.pin_generation(resident.allocated_bytes)?;
        Ok(Box::new(Self {
            device: self.device.clone(),
            governor,
            resident: BTreeMap::from([(project, resident)]),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        let planned = CandleResident::planned_bytes(&image);
        let staging = self.governor.reserve_staging(planned)?;
        let project = image.project;
        let staged = CandleResident::upload(image, &self.device)?;
        let total = self.total_after(project, staged.allocated_bytes);
        self.governor
            .commit_staging(staging, staged.allocated_bytes, total)?;
        self.resident.insert(project, Arc::new(staged));
        Ok(())
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        let planned = images.iter().try_fold(0_usize, |total, image| {
            total
                .checked_add(CandleResident::planned_bytes(image))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "replacement CUDA project bytes overflow",
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
        self.device.synchronize().map_err(candle_error)?;
        let actual = replacement.values().try_fold(0_usize, |total, resident| {
            total.checked_add(resident.allocated_bytes).ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "replacement CUDA project bytes overflow",
                )
            })
        })?;
        self.governor.commit_staging(staging, actual, actual)?;
        self.resident = replacement;
        Ok(())
    }

    fn apply_project_delta(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        let project = delta.project;
        let current = self.resident.get(&project).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!("project {project} has no complete CUDA-resident image"),
            )
        })?;
        let staged_bytes = current
            .allocated_bytes
            .checked_add(delta.staging_bytes())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "CUDA delta peak-memory accounting overflow",
                )
            })?;
        let reservation = self.governor.reserve_staging(staged_bytes)?;
        let staged = current.stage_delta(&delta, &self.device)?;
        let total = self.total_after(project, staged.allocated_bytes);
        self.governor
            .commit_staging(reservation, staged_bytes, total)?;
        self.resident.insert(project, Arc::new(staged));
        Ok(())
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.resident.remove(&project);
        let total = self
            .resident
            .values()
            .map(|resident| resident.allocated_bytes)
            .fold(0_usize, usize::saturating_add);
        self.governor.admit_persistent(total)
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
        self.project(project)?
            .filter_node_i64(property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.project(project)?
            .expand_out(&self.device, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
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
        self.project(project)?
            .expand_in_bounded(targets, maximum_rows, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        let resident = self.project(request.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(resident.vector_row_count(request.property))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "CUDA vector-pipeline scratch row count overflow",
                )
            })?;
        let output_rows = request
            .limit
            .checked_mul(request.query_count)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "CUDA vector result shape overflow",
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
        self.project(request.project)?;
        let input_rows = request.sorted_sets.iter().try_fold(0_usize, |total, set| {
            total.checked_add(set.len()).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "CUDA multiway-intersection input size overflow",
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
        request.validate()?;
        let output_rows = request
            .limit
            .unwrap_or(request.row_count)
            .min(request.row_count);
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            request.row_count,
            256,
            output_rows,
            16,
        )?)?;
        self.project(request.project)?
            .sort_rows(&self.device, request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        let input_rows = request
            .left_rows
            .len()
            .checked_add(request.right_rows.len())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "CUDA join scratch accounting overflow",
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
            320,
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
        // The CUDA integer group kernel cannot accumulate the f64 sum of squares dispersion needs;
        // decline so the executor runs it on the host.
        if matches!(request.aggregate, ResidentAggregate::Dispersion { .. }) {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "CUDA resident group aggregation does not support dispersion",
            ));
        }
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            request.rows.len(),
            512,
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
                    "CUDA resident-pipeline scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            160,
            maximum_rows,
            projected_width,
        )?)?;
        resident.execute_node_pipeline(&self.device, request, cancellation)
    }

    fn execute_node_group_pipeline(
        &self,
        request: &ResidentNodeGroupPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
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
                    "CUDA resident-group scratch row count overflow",
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

    fn execute_temporal_pipeline(
        &self,
        request: &ResidentTemporalPipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
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
                    "CUDA resident-temporal scratch row count overflow",
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
        ensure_graph_execution(cancellation, request.deadline)?;
        let resident = self.project(request.project)?;
        let input_rows = resident
            .node_count()
            .checked_add(resident.edge_count())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "CUDA graph-procedure scratch row count overflow",
                )
            })?;
        let _scratch = self.governor.reserve_scratch(operator_scratch_bytes(
            input_rows,
            512,
            request.max_output_rows.min(resident.node_count()),
            32,
        )?)?;
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
