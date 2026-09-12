//! Deterministic, device-resident multilevel Louvain for Apple Metal.
//!
//! The public graph is unweighted, so every weight produced by coarsening is an exact integer.
//! Apple GPUs do not expose native binary64 arithmetic; modularity-gain comparisons therefore use
//! exact wide-integer cross products in `graph_louvain.metal` rather than silently weakening the
//! CPU contract to FP32. Each pass proposes every strictly beneficial move, then applies a
//! deterministic maximal matching whose moves touch disjoint pairs of communities. Their
//! modularity deltas are additive, proving that every applied parallel batch is non-decreasing.

use std::{sync::OnceLock, time::Instant};

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor,
    backend::BackendStorage,
};
use objc2_metal::MTLDevice;
use tokio_util::sync::CancellationToken;

use crate::{Error, ErrorCode, Result, graph::LayerMask};

use super::{CandleResident, candle_error};
use crate::ResidentGraphProcedureResult;
use crate::ensure_graph_execution;

const THREADS: usize = 256;
const WORK_TILE_ROWS: usize = 262_144;
const BOUNDED_SCAN_NODE_TILE_ROWS: usize = 4_096;
const RADIX_BLOCKS_PER_SUBMISSION: usize = 256;
const DIRECT_NEIGHBOR_FAST_LIMIT: u64 = 128;
const DIRECT_NEIGHBOR_HARD_LIMIT: u64 = 256;
const SORTED_DECISION_CHUNK_ROWS: u32 = 1_024;
const SORTED_CANDIDATE_FIXED_WORK: u64 = 1_000_000;
const SORTED_CANDIDATE_RADIX_PASSES: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct GraphArgs {
    edge_capacity: u64,
    pair_capacity: u64,
    oriented_capacity: u64,
    total_weight: u64,
    work_offset: u64,
    work_count: u64,
    node_count: u32,
    edge_count: u32,
    layer_mask: u32,
    reduce_mode: u32,
}

impl GraphArgs {
    fn new(
        node_count: usize,
        edge_count: usize,
        pair_capacity: usize,
        total_weight: u64,
        layers: LayerMask,
        reduce_mode: u32,
    ) -> Result<Self> {
        let oriented_capacity = pair_capacity.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal Louvain oriented capacity overflow",
            )
        })?;
        Ok(Self {
            edge_capacity: u64::try_from(edge_count).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain edge capacity exceeds u64",
                )
            })?,
            pair_capacity: u64::try_from(pair_capacity).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain pair capacity exceeds u64",
                )
            })?,
            oriented_capacity: u64::try_from(oriented_capacity).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain oriented capacity exceeds u64",
                )
            })?,
            total_weight,
            work_offset: 0,
            work_count: 0,
            node_count: u32::try_from(node_count).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain node capacity exceeds u32",
                )
            })?,
            edge_count: u32::try_from(edge_count).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain edge capacity exceeds u32",
                )
            })?,
            layer_mask: u32::from(layers.bits()),
            reduce_mode,
        })
    }

    fn with_work(self, offset: usize, count: usize) -> candle_core::Result<Self> {
        Ok(Self {
            work_offset: u64::try_from(offset).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain work offset exceeds u64".to_owned())
            })?,
            work_count: u64::try_from(count).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain work count exceeds u64".to_owned())
            })?,
            ..self
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Validate,
    BasePairs,
    ReducePairs,
    OrientPairs,
    Degree,
    HighDegree,
    InitializeLevel,
    CandidateKeys,
    CandidateAggregate,
    SortedInitialize,
    SortedChunk,
    Decide,
    DecideSorted,
    MatchingInitialize,
    MatchingMinimum,
    MatchingWinners,
    MatchingAdvance,
    ApplyMatching,
    UpdateCommunityWeights,
    RebuildActive,
    MapOriginal,
    CoarsenPairs,
    CanonicalFirst,
    CanonicalKeys,
    CanonicalPublish,
    CsrOffsets,
    CsrNeighbors,
}

#[derive(Clone)]
struct Pipelines {
    arange_i64: candle_metal_kernels::metal::ComputePipeline,
    radix_clear: candle_metal_kernels::metal::ComputePipeline,
    radix_histogram: candle_metal_kernels::metal::ComputePipeline,
    radix_accumulate_totals: candle_metal_kernels::metal::ComputePipeline,
    radix_initialize_bases: candle_metal_kernels::metal::ComputePipeline,
    radix_offsets: candle_metal_kernels::metal::ComputePipeline,
    radix_scatter: candle_metal_kernels::metal::ComputePipeline,
    gather_i64: candle_metal_kernels::metal::ComputePipeline,
    gather_u32: candle_metal_kernels::metal::ComputePipeline,
    read_u32: candle_metal_kernels::metal::ComputePipeline,
    sum_u32_clear: candle_metal_kernels::metal::ComputePipeline,
    sum_u32: candle_metal_kernels::metal::ComputePipeline,
    sum_i64_clear: candle_metal_kernels::metal::ComputePipeline,
    sum_i64: candle_metal_kernels::metal::ComputePipeline,
    zero_u32: candle_metal_kernels::metal::ComputePipeline,
    u8_to_u32: candle_metal_kernels::metal::ComputePipeline,
    validate_clear: candle_metal_kernels::metal::ComputePipeline,
    validate: candle_metal_kernels::metal::ComputePipeline,
    base_pairs: candle_metal_kernels::metal::ComputePipeline,
    reduce_pairs_clear: candle_metal_kernels::metal::ComputePipeline,
    reduce_pairs: candle_metal_kernels::metal::ComputePipeline,
    orient_pairs: candle_metal_kernels::metal::ComputePipeline,
    degree_clear: candle_metal_kernels::metal::ComputePipeline,
    degree: candle_metal_kernels::metal::ComputePipeline,
    high_degree_clear: candle_metal_kernels::metal::ComputePipeline,
    high_degree: candle_metal_kernels::metal::ComputePipeline,
    initialize_level: candle_metal_kernels::metal::ComputePipeline,
    candidate_keys: candle_metal_kernels::metal::ComputePipeline,
    candidate_aggregate_clear: candle_metal_kernels::metal::ComputePipeline,
    candidate_aggregate: candle_metal_kernels::metal::ComputePipeline,
    sorted_initialize: candle_metal_kernels::metal::ComputePipeline,
    sorted_chunk: candle_metal_kernels::metal::ComputePipeline,
    decide_clear: candle_metal_kernels::metal::ComputePipeline,
    decide: candle_metal_kernels::metal::ComputePipeline,
    decide_sorted: candle_metal_kernels::metal::ComputePipeline,
    matching_initialize: candle_metal_kernels::metal::ComputePipeline,
    clear_u32_max: candle_metal_kernels::metal::ComputePipeline,
    matching_priority: candle_metal_kernels::metal::ComputePipeline,
    matching_minimum: candle_metal_kernels::metal::ComputePipeline,
    matching_clear: candle_metal_kernels::metal::ComputePipeline,
    matching_winners: candle_metal_kernels::metal::ComputePipeline,
    matching_advance: candle_metal_kernels::metal::ComputePipeline,
    apply_matching: candle_metal_kernels::metal::ComputePipeline,
    copy_community_weights: candle_metal_kernels::metal::ComputePipeline,
    update_community_weights: candle_metal_kernels::metal::ComputePipeline,
    active_clear: candle_metal_kernels::metal::ComputePipeline,
    rebuild_active: candle_metal_kernels::metal::ComputePipeline,
    map_original: candle_metal_kernels::metal::ComputePipeline,
    coarsen_pairs: candle_metal_kernels::metal::ComputePipeline,
    canonical_first_clear: candle_metal_kernels::metal::ComputePipeline,
    canonical_first_scatter: candle_metal_kernels::metal::ComputePipeline,
    canonical_keys: candle_metal_kernels::metal::ComputePipeline,
    canonical_publish_clear: candle_metal_kernels::metal::ComputePipeline,
    canonical_publish: candle_metal_kernels::metal::ComputePipeline,
    csr_offsets: candle_metal_kernels::metal::ComputePipeline,
    csr_neighbors: candle_metal_kernels::metal::ComputePipeline,
}

#[allow(clippy::too_many_lines)]
fn pipelines(device: &candle_core::MetalDevice) -> candle_core::Result<Pipelines> {
    static PIPELINES: OnceLock<Pipelines> = OnceLock::new();
    if let Some(pipelines) = PIPELINES.get() {
        return Ok(pipelines.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(
            include_str!("../../../../kernels/metal/graph_louvain.metal"),
            None,
        )
        .map_err(|error| {
            candle_core::Error::Msg(format!("compiling Metal Louvain kernels failed: {error}"))
        })?;
    let pipeline = |name: &str| -> candle_core::Result<_> {
        let function = library.get_function(name, None).map_err(|error| {
            candle_core::Error::Msg(format!(
                "loading Metal Louvain kernel {name} failed: {error}"
            ))
        })?;
        let raw = device
            .metal_device()
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "creating Metal Louvain pipeline {name} failed: {error:?}"
                ))
            })?;
        Ok(candle_metal_kernels::metal::ComputePipeline::new(raw))
    };
    let result = Pipelines {
        arange_i64: pipeline("ig_louvain_arange_i64")?,
        radix_clear: pipeline("ig_louvain_radix_clear")?,
        radix_histogram: pipeline("ig_louvain_radix_histogram")?,
        radix_accumulate_totals: pipeline("ig_louvain_radix_accumulate_totals")?,
        radix_initialize_bases: pipeline("ig_louvain_radix_initialize_bases")?,
        radix_offsets: pipeline("ig_louvain_radix_offsets")?,
        radix_scatter: pipeline("ig_louvain_radix_scatter")?,
        gather_i64: pipeline("ig_louvain_gather_i64")?,
        gather_u32: pipeline("ig_louvain_gather_u32")?,
        read_u32: pipeline("ig_louvain_read_u32")?,
        sum_u32_clear: pipeline("ig_louvain_sum_u32_clear")?,
        sum_u32: pipeline("ig_louvain_sum_u32")?,
        sum_i64_clear: pipeline("ig_louvain_sum_i64_clear")?,
        sum_i64: pipeline("ig_louvain_sum_i64")?,
        zero_u32: pipeline("ig_louvain_zero_u32")?,
        u8_to_u32: pipeline("ig_louvain_u8_to_u32")?,
        validate_clear: pipeline("ig_louvain_validate_clear")?,
        validate: pipeline("ig_louvain_validate_graph")?,
        base_pairs: pipeline("ig_louvain_base_pairs")?,
        reduce_pairs_clear: pipeline("ig_louvain_reduce_pairs_clear")?,
        reduce_pairs: pipeline("ig_louvain_reduce_pairs")?,
        orient_pairs: pipeline("ig_louvain_orient_pairs")?,
        degree_clear: pipeline("ig_louvain_degree_clear")?,
        degree: pipeline("ig_louvain_degree")?,
        high_degree_clear: pipeline("ig_louvain_high_degree_clear")?,
        high_degree: pipeline("ig_louvain_high_degree")?,
        initialize_level: pipeline("ig_louvain_initialize_level")?,
        candidate_keys: pipeline("ig_louvain_candidate_keys")?,
        candidate_aggregate_clear: pipeline("ig_louvain_candidate_aggregate_clear")?,
        candidate_aggregate: pipeline("ig_louvain_candidate_aggregate")?,
        sorted_initialize: pipeline("ig_louvain_sorted_initialize")?,
        sorted_chunk: pipeline("ig_louvain_sorted_chunk")?,
        decide_clear: pipeline("ig_louvain_decide_clear")?,
        decide: pipeline("ig_louvain_decide")?,
        decide_sorted: pipeline("ig_louvain_decide_sorted")?,
        matching_initialize: pipeline("ig_louvain_matching_initialize")?,
        clear_u32_max: pipeline("ig_louvain_clear_u32_max")?,
        matching_priority: pipeline("ig_louvain_matching_priority")?,
        matching_minimum: pipeline("ig_louvain_matching_minimum")?,
        matching_clear: pipeline("ig_louvain_matching_clear")?,
        matching_winners: pipeline("ig_louvain_matching_winners")?,
        matching_advance: pipeline("ig_louvain_matching_advance")?,
        apply_matching: pipeline("ig_louvain_apply_matching")?,
        copy_community_weights: pipeline("ig_louvain_copy_community_weights")?,
        update_community_weights: pipeline("ig_louvain_update_community_weights")?,
        active_clear: pipeline("ig_louvain_active_clear")?,
        rebuild_active: pipeline("ig_louvain_rebuild_active")?,
        map_original: pipeline("ig_louvain_map_original")?,
        coarsen_pairs: pipeline("ig_louvain_coarsen_pairs")?,
        canonical_first_clear: pipeline("ig_louvain_canonical_first_clear")?,
        canonical_first_scatter: pipeline("ig_louvain_canonical_first_scatter")?,
        canonical_keys: pipeline("ig_louvain_canonical_keys")?,
        canonical_publish_clear: pipeline("ig_louvain_canonical_publish_clear")?,
        canonical_publish: pipeline("ig_louvain_canonical_publish")?,
        csr_offsets: pipeline("ig_louvain_csr_offsets")?,
        csr_neighbors: pipeline("ig_louvain_csr_neighbors")?,
    };
    for candidate in [
        &result.arange_i64,
        &result.radix_clear,
        &result.radix_histogram,
        &result.radix_accumulate_totals,
        &result.radix_initialize_bases,
        &result.radix_offsets,
        &result.radix_scatter,
        &result.gather_i64,
        &result.gather_u32,
        &result.read_u32,
        &result.sum_u32_clear,
        &result.sum_u32,
        &result.sum_i64_clear,
        &result.sum_i64,
        &result.zero_u32,
        &result.u8_to_u32,
        &result.validate_clear,
        &result.validate,
        &result.base_pairs,
        &result.reduce_pairs_clear,
        &result.reduce_pairs,
        &result.orient_pairs,
        &result.degree_clear,
        &result.degree,
        &result.high_degree_clear,
        &result.high_degree,
        &result.initialize_level,
        &result.candidate_keys,
        &result.candidate_aggregate_clear,
        &result.candidate_aggregate,
        &result.sorted_initialize,
        &result.sorted_chunk,
        &result.decide_clear,
        &result.decide,
        &result.decide_sorted,
        &result.matching_initialize,
        &result.clear_u32_max,
        &result.matching_priority,
        &result.matching_minimum,
        &result.matching_clear,
        &result.matching_winners,
        &result.matching_advance,
        &result.apply_matching,
        &result.copy_community_weights,
        &result.update_community_weights,
        &result.active_clear,
        &result.rebuild_active,
        &result.map_original,
        &result.coarsen_pairs,
        &result.canonical_first_clear,
        &result.canonical_first_scatter,
        &result.canonical_keys,
        &result.canonical_publish_clear,
        &result.canonical_publish,
        &result.csr_offsets,
        &result.csr_neighbors,
    ] {
        if candidate.max_total_threads_per_threadgroup() < THREADS {
            return Err(candle_core::Error::Msg(
                "selected Metal device cannot run 256-thread Louvain groups".to_owned(),
            ));
        }
    }
    let _ = PIPELINES.set(result.clone());
    Ok(PIPELINES.get().cloned().unwrap_or(result))
}

pub fn prepare(device: &Device) -> Result<()> {
    let Device::Metal(device) = device else {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "Metal Louvain pipeline preparation requires a Metal device",
        ));
    };
    pipelines(device).map_err(candle_error)?;
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RadixArgs {
    row_count: u64,
    block_count: u64,
    work_offset: u64,
    work_count: u64,
    digit_pass: u32,
    reserved_0: u32,
    reserved_1: u32,
    reserved_2: u32,
}

impl RadixArgs {
    fn new(row_count: usize, block_count: usize, digit_pass: u32) -> candle_core::Result<Self> {
        Ok(Self {
            row_count: u64::try_from(row_count).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain radix row count exceeds u64".to_owned())
            })?,
            block_count: u64::try_from(block_count).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain radix block count exceeds u64".to_owned())
            })?,
            work_offset: 0,
            work_count: 0,
            digit_pass,
            reserved_0: 0,
            reserved_1: 0,
            reserved_2: 0,
        })
    }

    fn with_work(self, offset: usize, count: usize) -> candle_core::Result<Self> {
        Ok(Self {
            work_offset: u64::try_from(offset).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain radix work offset exceeds u64".to_owned())
            })?,
            work_count: u64::try_from(count).map_err(|_| {
                candle_core::Error::Msg("Metal Louvain radix work count exceeds u64".to_owned())
            })?,
            ..self
        })
    }
}

fn metal_checkpoint(
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> candle_core::Result<()> {
    ensure_graph_execution(cancellation, deadline)
        .map_err(|error| candle_core::Error::Msg(error.to_string()))
}

fn louvain_candle_error(
    error: candle_core::Error,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Error {
    if cancellation.is_cancelled() {
        Error::new(ErrorCode::Cancelled, "GPU operation cancelled")
    } else if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
        Error::new(
            ErrorCode::DeadlineExceeded,
            "GPU graph operation deadline exceeded",
        )
    } else {
        candle_error(error)
    }
}

#[derive(Clone, Debug)]
struct LouvainArange {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp1 for LouvainArange {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-arange"
    }

    fn cpu_fwd(
        &self,
        _primary: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain arange cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        primary: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = layout.shape().elem_count();
        if rows == 0 || primary.dtype() != DType::I64 || !layout.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "Metal Louvain arange tensor contract is invalid".to_owned(),
            ));
        }
        let device = primary.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows, DType::I64)
            .with_label("irongraph Metal Louvain radix initial positions")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(rows, rows.div_ceil(RADIX_TILE_ROWS), 0)?;
        for offset in (0..rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.arange_i64);
                encoder.set_output_buffer(0, Some(&output), 0);
                encoder.set_bytes(1, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), rows, DType::I64),
            Shape::from(rows),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainStableRadixPass {
    digit_pass: u32,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp2 for LouvainStableRadixPass {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-stable-radix-pass"
    }

    fn cpu_fwd(
        &self,
        _values: &CpuStorage,
        _values_layout: &Layout,
        _positions: &CpuStorage,
        _positions_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain radix cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        values: &MetalStorage,
        values_layout: &Layout,
        positions: &MetalStorage,
        positions_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = values_layout.shape().elem_count();
        if rows == 0
            || self.digit_pass >= 8
            || values.dtype() != DType::I64
            || positions.dtype() != DType::I64
            || !values_layout.is_contiguous()
            || !positions_layout.is_contiguous()
            || positions_layout.shape().elem_count() != rows
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain radix tensor contract is invalid".to_owned(),
            ));
        }
        let block_count = rows.div_ceil(RADIX_TILE_ROWS);
        let table_elements = block_count.checked_mul(RADIX_BUCKETS).ok_or_else(|| {
            candle_core::Error::Msg("Metal Louvain radix table shape overflow".to_owned())
        })?;
        let device = values.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows, DType::I64)
            .with_label("irongraph Metal Louvain radix positions")
            .build()?;
        let histograms = device
            .new_buffer_builder()
            .with_size_for(table_elements, DType::U32)
            .with_label("irongraph Metal Louvain radix histograms")
            .build()?;
        let offsets = device
            .new_buffer_builder()
            .with_size_for(table_elements, DType::I64)
            .with_label("irongraph Metal Louvain radix offsets")
            .build()?;
        let totals = device
            .new_buffer_builder()
            .with_size_for(RADIX_BUCKETS, DType::I64)
            .with_label("irongraph Metal Louvain radix bucket totals")
            .build()?;
        let running = device
            .new_buffer_builder()
            .with_size_for(RADIX_BUCKETS, DType::I64)
            .with_label("irongraph Metal Louvain radix running offsets")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(rows, block_count, self.digit_pass)?;
        let value_offset = values_layout.start_offset() * DType::I64.size_in_bytes();
        let position_offset = positions_layout.start_offset() * DType::I64.size_in_bytes();

        metal_checkpoint(&self.cancellation, self.deadline)?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipelines.radix_clear);
            encoder.set_output_buffer(0, Some(&totals), 0);
            encoder.set_output_buffer(1, Some(&running), 0);
            encoder.dispatch_thread_groups(groups(RADIX_BUCKETS), threads());
        }
        // Each command covers at most 256 fixed 1,024-row radix blocks. Histogram production and
        // its bounded bucket reduction are ordered in the same command buffer.
        for block_offset in (0..block_count).step_by(RADIX_BLOCKS_PER_SUBMISSION) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let block_work = RADIX_BLOCKS_PER_SUBMISSION.min(block_count - block_offset);
            let args = base_args.with_work(block_offset, block_work)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.radix_histogram);
                encoder.set_input_buffer(0, Some(values.buffer()), value_offset);
                encoder.set_input_buffer(1, Some(positions.buffer()), position_offset);
                encoder.set_output_buffer(2, Some(&histograms), 0);
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(radix_groups(block_work), threads());
                encoder.set_compute_pipeline_state(&pipelines.radix_accumulate_totals);
                encoder.set_input_buffer(0, Some(&histograms), 0);
                encoder.set_output_buffer(1, Some(&totals), 0);
                encoder.set_bytes(2, &args);
                encoder.dispatch_thread_groups(groups(RADIX_BUCKETS), threads());
            }
        }

        metal_checkpoint(&self.cancellation, self.deadline)?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipelines.radix_initialize_bases);
            encoder.set_input_buffer(0, Some(&totals), 0);
            encoder.set_output_buffer(1, Some(&running), 0);
            encoder.dispatch_thread_groups(groups(RADIX_BUCKETS), threads());
        }
        // Running bucket positions make chunks resumable without host-visible keys or positions.
        // The offset phase precedes scatter in each command, preserving global stable order.
        for block_offset in (0..block_count).step_by(RADIX_BLOCKS_PER_SUBMISSION) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let block_work = RADIX_BLOCKS_PER_SUBMISSION.min(block_count - block_offset);
            let args = base_args.with_work(block_offset, block_work)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.radix_offsets);
                encoder.set_input_buffer(0, Some(&histograms), 0);
                encoder.set_output_buffer(1, Some(&offsets), 0);
                encoder.set_output_buffer(2, Some(&running), 0);
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(groups(RADIX_BUCKETS), threads());
                encoder.set_compute_pipeline_state(&pipelines.radix_scatter);
                encoder.set_input_buffer(0, Some(values.buffer()), value_offset);
                encoder.set_input_buffer(1, Some(positions.buffer()), position_offset);
                encoder.set_input_buffer(2, Some(&offsets), 0);
                encoder.set_output_buffer(3, Some(&output), 0);
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(radix_groups(block_work), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), rows, DType::I64),
            Shape::from(rows),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainGather {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp2 for LouvainGather {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-gather"
    }

    fn cpu_fwd(
        &self,
        _values: &CpuStorage,
        _values_layout: &Layout,
        _positions: &CpuStorage,
        _positions_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain gather cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        values: &MetalStorage,
        values_layout: &Layout,
        positions: &MetalStorage,
        positions_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = values_layout.shape().elem_count();
        if rows == 0
            || values.dtype() != DType::I64
            || positions.dtype() != DType::I64
            || !values_layout.is_contiguous()
            || !positions_layout.is_contiguous()
            || positions_layout.shape().elem_count() != rows
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain gather tensor contract is invalid".to_owned(),
            ));
        }
        let device = values.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows, DType::I64)
            .with_label("irongraph Metal Louvain gathered column")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(rows, rows.div_ceil(RADIX_TILE_ROWS), 0)?;
        for offset in (0..rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.gather_i64);
                encoder.set_input_buffer(
                    0,
                    Some(values.buffer()),
                    values_layout.start_offset() * DType::I64.size_in_bytes(),
                );
                encoder.set_input_buffer(
                    1,
                    Some(positions.buffer()),
                    positions_layout.start_offset() * DType::I64.size_in_bytes(),
                );
                encoder.set_output_buffer(2, Some(&output), 0);
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), rows, DType::I64),
            Shape::from(rows),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainGatherU32 {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp2 for LouvainGatherU32 {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-gather-u32"
    }

    fn cpu_fwd(
        &self,
        _values: &CpuStorage,
        _values_layout: &Layout,
        _positions: &CpuStorage,
        _positions_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain U32 gather cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        values: &MetalStorage,
        values_layout: &Layout,
        positions: &MetalStorage,
        positions_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = positions_layout.shape().elem_count();
        if rows == 0
            || values.dtype() != DType::U32
            || positions.dtype() != DType::U32
            || !values_layout.is_contiguous()
            || !positions_layout.is_contiguous()
            || positions_layout.dims().len() != 1
            || values_layout.dims().len() != 1
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain U32 gather tensor contract is invalid".to_owned(),
            ));
        }
        let device = values.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows, DType::U32)
            .with_label("irongraph Metal Louvain gathered labels")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(rows, 0, 0)?;
        for offset in (0..rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.gather_u32);
                encoder.set_input_buffer(
                    0,
                    Some(values.buffer()),
                    values_layout.start_offset() * DType::U32.size_in_bytes(),
                );
                encoder.set_input_buffer(
                    1,
                    Some(positions.buffer()),
                    positions_layout.start_offset() * DType::U32.size_in_bytes(),
                );
                encoder.set_output_buffer(2, Some(&output), 0);
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), rows, DType::U32),
            Shape::from(rows),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainZeroU32 {
    rows: usize,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp1 for LouvainZeroU32 {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-zero-u32"
    }

    fn cpu_fwd(
        &self,
        _primary: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain U32 zero fill cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        primary: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        if self.rows == 0 || !layout.is_contiguous() {
            return Err(candle_core::Error::Msg(
                "Metal Louvain U32 zero-fill contract is invalid".to_owned(),
            ));
        }
        let device = primary.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(self.rows, DType::U32)
            .with_label("irongraph Metal Louvain zero U32")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(self.rows, 0, 0)?;
        for offset in (0..self.rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(self.rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.zero_u32);
                encoder.set_output_buffer(0, Some(&output), 0);
                encoder.set_bytes(1, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), self.rows, DType::U32),
            Shape::from(self.rows),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainU8ToU32 {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp1 for LouvainU8ToU32 {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-u8-to-u32"
    }

    fn cpu_fwd(
        &self,
        _primary: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain U8-to-U32 conversion cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        primary: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = layout.shape().elem_count();
        if rows == 0
            || primary.dtype() != DType::U8
            || !layout.is_contiguous()
            || layout.dims().len() != 1
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain U8-to-U32 tensor contract is invalid".to_owned(),
            ));
        }
        let device = primary.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(rows, DType::U32)
            .with_label("irongraph Metal Louvain active nodes")
            .build()?;
        let pipelines = pipelines(device)?;
        let base_args = RadixArgs::new(rows, 0, 0)?;
        for offset in (0..rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.u8_to_u32);
                encoder.set_input_buffer(
                    0,
                    Some(primary.buffer()),
                    layout.start_offset() * DType::U8.size_in_bytes(),
                );
                encoder.set_output_buffer(1, Some(&output), 0);
                encoder.set_bytes(2, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), rows, DType::U32),
            Shape::from(rows),
        ))
    }
}

#[derive(Clone, Copy, Debug)]
enum U32ScalarMode {
    Read(usize),
    Sum,
}

#[derive(Clone, Debug)]
struct LouvainU32Scalar {
    mode: U32ScalarMode,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp1 for LouvainU32Scalar {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-u32-scalar"
    }

    fn cpu_fwd(
        &self,
        _values: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain U32 scalar cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        values: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = layout.shape().elem_count();
        if rows == 0
            || values.dtype() != DType::U32
            || !layout.is_contiguous()
            || layout.dims().len() != 1
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain U32 scalar tensor contract is invalid".to_owned(),
            ));
        }
        if matches!(self.mode, U32ScalarMode::Read(offset) if offset >= rows) {
            return Err(candle_core::Error::Msg(
                "Metal Louvain scalar read offset is out of bounds".to_owned(),
            ));
        }
        let device = values.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(1, DType::U32)
            .with_label("irongraph Metal Louvain U32 scalar")
            .build()?;
        let pipelines = pipelines(device)?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        match self.mode {
            U32ScalarMode::Read(offset) => {
                let args = RadixArgs::new(rows, 0, 0)?.with_work(offset, 1)?;
                {
                    let encoder = device.command_encoder()?;
                    let encoder = encoder.as_ref();
                    encoder.set_compute_pipeline_state(&pipelines.read_u32);
                    encoder.set_input_buffer(
                        0,
                        Some(values.buffer()),
                        layout.start_offset() * DType::U32.size_in_bytes(),
                    );
                    encoder.set_output_buffer(1, Some(&output), 0);
                    encoder.set_bytes(2, &args);
                    encoder.dispatch_thread_groups(groups(1), threads());
                }
            }
            U32ScalarMode::Sum => {
                {
                    let encoder = device.command_encoder()?;
                    let encoder = encoder.as_ref();
                    encoder.set_compute_pipeline_state(&pipelines.sum_u32_clear);
                    encoder.set_output_buffer(0, Some(&output), 0);
                    encoder.dispatch_thread_groups(groups(1), threads());
                }
                let base_args = RadixArgs::new(rows, 0, 0)?;
                for offset in (0..rows).step_by(WORK_TILE_ROWS) {
                    metal_checkpoint(&self.cancellation, self.deadline)?;
                    let count = WORK_TILE_ROWS.min(rows - offset);
                    let args = base_args.with_work(offset, count)?;
                    {
                        let encoder = device.command_encoder()?;
                        let encoder = encoder.as_ref();
                        encoder.set_compute_pipeline_state(&pipelines.sum_u32);
                        encoder.set_input_buffer(
                            0,
                            Some(values.buffer()),
                            layout.start_offset() * DType::U32.size_in_bytes(),
                        );
                        encoder.set_output_buffer(1, Some(&output), 0);
                        encoder.set_bytes(2, &args);
                        encoder.dispatch_thread_groups(groups(count), threads());
                    }
                }
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), 1, DType::U32),
            Shape::from(1),
        ))
    }
}

#[derive(Clone, Debug)]
struct LouvainI64Sum {
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl CustomOp1 for LouvainI64Sum {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain-i64-sum"
    }

    fn cpu_fwd(
        &self,
        _values: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain I64 sum cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        values: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let rows = layout.shape().elem_count();
        if rows == 0
            || values.dtype() != DType::I64
            || !layout.is_contiguous()
            || layout.dims().len() != 1
        {
            return Err(candle_core::Error::Msg(
                "Metal Louvain I64 sum tensor contract is invalid".to_owned(),
            ));
        }
        let device = values.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(1, DType::I64)
            .with_label("irongraph Metal Louvain I64 sum")
            .build()?;
        let pipelines = pipelines(device)?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        {
            let encoder = device.command_encoder()?;
            let encoder = encoder.as_ref();
            encoder.set_compute_pipeline_state(&pipelines.sum_i64_clear);
            encoder.set_output_buffer(0, Some(&output), 0);
            encoder.dispatch_thread_groups(groups(1), threads());
        }
        let base_args = RadixArgs::new(rows, 0, 0)?;
        for offset in (0..rows).step_by(WORK_TILE_ROWS) {
            metal_checkpoint(&self.cancellation, self.deadline)?;
            let count = WORK_TILE_ROWS.min(rows - offset);
            let args = base_args.with_work(offset, count)?;
            {
                let encoder = device.command_encoder()?;
                let encoder = encoder.as_ref();
                encoder.set_compute_pipeline_state(&pipelines.sum_i64);
                encoder.set_input_buffer(
                    0,
                    Some(values.buffer()),
                    layout.start_offset() * DType::I64.size_in_bytes(),
                );
                encoder.set_output_buffer(1, Some(&output), 0);
                encoder.set_bytes(2, &args);
                encoder.dispatch_thread_groups(groups(count), threads());
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), 1, DType::I64),
            Shape::from(1),
        ))
    }
}

fn read_u32_scalar(
    tensor: &Tensor,
    offset: usize,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<u32> {
    let values = tensor
        .apply_op1_no_bwd(&LouvainU32Scalar {
            mode: U32ScalarMode::Read(offset),
            cancellation: cancellation.clone(),
            deadline,
        })
        .and_then(|scalar| scalar.to_vec1::<u32>())
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    values
        .first()
        .copied()
        .ok_or_else(|| Error::internal("Metal Louvain scalar read returned no value"))
}

fn sum_u32(
    tensor: &Tensor,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<u32> {
    let values = tensor
        .apply_op1_no_bwd(&LouvainU32Scalar {
            mode: U32ScalarMode::Sum,
            cancellation: cancellation.clone(),
            deadline,
        })
        .and_then(|scalar| scalar.to_vec1::<u32>())
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    values
        .first()
        .copied()
        .ok_or_else(|| Error::internal("Metal Louvain U32 sum returned no value"))
}

fn sum_i64(
    tensor: &Tensor,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<i64> {
    let values = tensor
        .apply_op1_no_bwd(&LouvainI64Sum {
            cancellation: cancellation.clone(),
            deadline,
        })
        .and_then(|scalar| scalar.to_vec1::<i64>())
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    values
        .first()
        .copied()
        .ok_or_else(|| Error::internal("Metal Louvain I64 sum returned no value"))
}

fn groups(rows: usize) -> objc2_metal::MTLSize {
    objc2_metal::MTLSize {
        width: rows.max(1).div_ceil(THREADS),
        height: 1,
        depth: 1,
    }
}

fn radix_groups(blocks: usize) -> objc2_metal::MTLSize {
    objc2_metal::MTLSize {
        width: blocks.max(1),
        height: 1,
        depth: 1,
    }
}

const fn threads() -> objc2_metal::MTLSize {
    objc2_metal::MTLSize {
        width: THREADS,
        height: 1,
        depth: 1,
    }
}

#[derive(Clone, Debug)]
struct LouvainOp {
    stage: Stage,
    auxiliary: Vec<Tensor>,
    args: GraphArgs,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl LouvainOp {
    fn output_contract(&self) -> candle_core::Result<(usize, DType)> {
        let nodes = self.args.node_count as usize;
        let pairs = usize::try_from(self.args.pair_capacity)
            .map_err(|_| candle_core::Error::Msg("Louvain pair capacity overflow".to_owned()))?;
        let oriented = usize::try_from(self.args.oriented_capacity).map_err(|_| {
            candle_core::Error::Msg("Louvain oriented capacity overflow".to_owned())
        })?;
        match self.stage {
            Stage::Validate | Stage::HighDegree => Ok((1, DType::U32)),
            Stage::BasePairs | Stage::ReducePairs | Stage::CoarsenPairs => pairs
                .checked_mul(2)
                .map(|rows| (rows, DType::I64))
                .ok_or_else(|| candle_core::Error::Msg("Louvain pair output overflow".to_owned())),
            Stage::OrientPairs => pairs
                .checked_mul(4)
                .map(|rows| (rows, DType::I64))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain adjacency output overflow".to_owned())
                }),
            Stage::Degree | Stage::UpdateCommunityWeights | Stage::CanonicalKeys => {
                Ok((nodes, DType::I64))
            }
            Stage::CandidateKeys => pairs
                .checked_mul(2)
                .map(|rows| (rows, DType::I64))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain candidate output overflow".to_owned())
                }),
            Stage::CandidateAggregate => Ok((oriented, DType::I64)),
            Stage::SortedInitialize | Stage::SortedChunk => nodes
                .checked_mul(3)
                .map(|rows| (rows, DType::I64))
                .ok_or_else(|| candle_core::Error::Msg("Louvain sorted state overflow".to_owned())),
            Stage::InitializeLevel
            | Stage::MatchingInitialize
            | Stage::ApplyMatching
            | Stage::MapOriginal
            | Stage::CanonicalPublish => Ok((nodes, DType::U32)),
            Stage::MatchingMinimum => nodes
                .checked_mul(2)
                .map(|rows| (rows, DType::U32))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain matching minimum overflow".to_owned())
                }),
            Stage::Decide | Stage::DecideSorted | Stage::RebuildActive | Stage::CanonicalFirst => {
                nodes
                    .checked_add(1)
                    .map(|rows| (rows, DType::U32))
                    .ok_or_else(|| {
                        candle_core::Error::Msg("Louvain control output overflow".to_owned())
                    })
            }
            Stage::MatchingWinners => nodes
                .checked_mul(2)
                .and_then(|rows| rows.checked_add(1))
                .map(|rows| (rows, DType::U32))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain matching output overflow".to_owned())
                }),
            Stage::MatchingAdvance => nodes
                .checked_mul(2)
                .map(|rows| (rows, DType::U32))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain matching state overflow".to_owned())
                }),
            Stage::CsrOffsets => nodes
                .checked_add(1)
                .map(|rows| (rows, DType::U32))
                .ok_or_else(|| {
                    candle_core::Error::Msg("Louvain CSR offset shape overflow".to_owned())
                }),
            Stage::CsrNeighbors => usize::try_from(self.args.total_weight)
                .map(|rows| (rows, DType::U32))
                .map_err(|_| {
                    candle_core::Error::Msg("Louvain CSR neighbor shape overflow".to_owned())
                }),
        }
    }
}

impl CustomOp1 for LouvainOp {
    fn name(&self) -> &'static str {
        "irongraph-metal-louvain"
    }

    fn cpu_fwd(
        &self,
        _storage: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal Louvain cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(
        clippy::branches_sharing_code,
        clippy::significant_drop_tightening,
        clippy::too_many_lines
    )]
    fn metal_fwd(
        &self,
        primary: &MetalStorage,
        primary_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        if !primary_layout.is_contiguous() || primary_layout.dims().len() != 1 {
            return Err(candle_core::Error::Msg(
                "Metal Louvain primary tensor is not contiguous rank one".to_owned(),
            ));
        }
        let nodes = self.args.node_count as usize;
        let edges = self.args.edge_count as usize;
        let pairs = usize::try_from(self.args.pair_capacity)
            .map_err(|_| candle_core::Error::Msg("Louvain pair capacity overflow".to_owned()))?;
        let oriented = usize::try_from(self.args.oriented_capacity).map_err(|_| {
            candle_core::Error::Msg("Louvain oriented capacity overflow".to_owned())
        })?;
        let two_nodes = nodes.checked_mul(2).ok_or_else(|| {
            candle_core::Error::Msg("Louvain two-node input shape overflow".to_owned())
        })?;
        let three_nodes = nodes.checked_mul(3).ok_or_else(|| {
            candle_core::Error::Msg("Louvain three-node input shape overflow".to_owned())
        })?;
        let winner_rows = two_nodes.checked_add(1).ok_or_else(|| {
            candle_core::Error::Msg("Louvain winner input shape overflow".to_owned())
        })?;
        let require_primary = |dtype: DType, rows: usize| -> candle_core::Result<()> {
            if primary.dtype() != dtype || primary_layout.shape().elem_count() != rows {
                return Err(candle_core::Error::Msg(
                    "Metal Louvain primary tensor contract is invalid".to_owned(),
                ));
            }
            Ok(())
        };
        let require_auxiliary =
            |index: usize, dtype: DType, rows: usize| -> candle_core::Result<()> {
                let tensor = self.auxiliary.get(index).ok_or_else(|| {
                    candle_core::Error::Msg("Metal Louvain auxiliary tensor is absent".to_owned())
                })?;
                if tensor.dtype() != dtype || tensor.rank() != 1 || tensor.elem_count() != rows {
                    return Err(candle_core::Error::Msg(format!(
                        "Metal Louvain auxiliary tensor {index} contract is invalid"
                    )));
                }
                Ok(())
            };
        let require_auxiliary_count = |count: usize| -> candle_core::Result<()> {
            if self.auxiliary.len() != count {
                return Err(candle_core::Error::Msg(
                    "Metal Louvain auxiliary tensor count is invalid".to_owned(),
                ));
            }
            Ok(())
        };
        match self.stage {
            Stage::Validate | Stage::BasePairs => {
                require_primary(DType::U8, nodes)?;
                require_auxiliary_count(4)?;
                require_auxiliary(0, DType::U8, edges)?;
                require_auxiliary(1, DType::U8, edges)?;
                require_auxiliary(2, DType::U32, edges)?;
                require_auxiliary(3, DType::U32, edges)?;
            }
            Stage::ReducePairs | Stage::OrientPairs => {
                require_primary(DType::I64, pairs)?;
                require_auxiliary_count(1)?;
                require_auxiliary(0, DType::I64, pairs)?;
            }
            Stage::Degree => {
                require_primary(DType::I64, oriented)?;
                require_auxiliary_count(2)?;
                require_auxiliary(0, DType::I64, oriented)?;
                require_auxiliary(1, DType::U32, nodes)?;
            }
            Stage::HighDegree | Stage::CandidateKeys => {
                require_primary(DType::I64, oriented)?;
                require_auxiliary_count(1)?;
                require_auxiliary(0, DType::U32, nodes)?;
            }
            Stage::InitializeLevel | Stage::CanonicalKeys => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(0)?;
            }
            Stage::CandidateAggregate | Stage::CsrOffsets | Stage::CsrNeighbors => {
                require_primary(DType::I64, oriented)?;
                require_auxiliary_count(1)?;
                require_auxiliary(0, DType::I64, oriented)?;
            }
            Stage::SortedInitialize | Stage::RebuildActive | Stage::MapOriginal => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(1)?;
                require_auxiliary(0, DType::U32, nodes)?;
            }
            Stage::SortedChunk => {
                require_primary(DType::I64, three_nodes)?;
                require_auxiliary_count(6)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
                require_auxiliary(2, DType::I64, nodes)?;
                require_auxiliary(3, DType::I64, nodes)?;
                require_auxiliary(4, DType::I64, oriented)?;
                require_auxiliary(5, DType::I64, oriented)?;
            }
            Stage::Decide => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(5)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::I64, nodes)?;
                require_auxiliary(2, DType::I64, nodes)?;
                require_auxiliary(3, DType::I64, oriented)?;
                require_auxiliary(4, DType::I64, oriented)?;
            }
            Stage::DecideSorted => {
                require_primary(DType::I64, three_nodes)?;
                require_auxiliary_count(4)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
                require_auxiliary(2, DType::I64, nodes)?;
                require_auxiliary(3, DType::I64, nodes)?;
            }
            Stage::MatchingInitialize | Stage::MatchingMinimum | Stage::ApplyMatching => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(2)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
            }
            Stage::MatchingWinners => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(3)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
                require_auxiliary(2, DType::U32, two_nodes)?;
            }
            Stage::MatchingAdvance => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(4)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
                require_auxiliary(2, DType::U32, nodes)?;
                require_auxiliary(3, DType::U32, winner_rows)?;
            }
            Stage::UpdateCommunityWeights => {
                require_primary(DType::I64, nodes)?;
                require_auxiliary_count(4)?;
                require_auxiliary(0, DType::U32, nodes)?;
                require_auxiliary(1, DType::U32, nodes)?;
                require_auxiliary(2, DType::U32, nodes)?;
                require_auxiliary(3, DType::I64, nodes)?;
            }
            Stage::CoarsenPairs => {
                require_primary(DType::I64, pairs)?;
                require_auxiliary_count(2)?;
                require_auxiliary(0, DType::I64, pairs)?;
                require_auxiliary(1, DType::U32, nodes)?;
            }
            Stage::CanonicalFirst => {
                require_primary(DType::U32, nodes)?;
                require_auxiliary_count(1)?;
                require_auxiliary(0, DType::U32, pairs)?;
            }
            Stage::CanonicalPublish => {
                require_primary(DType::I64, nodes)?;
                require_auxiliary_count(0)?;
            }
        }
        macro_rules! auxiliary {
            ($index:expr, $storage:ident, $layout:ident, $metal:ident) => {
                let tensor = self.auxiliary.get($index).ok_or_else(|| {
                    candle_core::Error::Msg("Metal Louvain auxiliary tensor is absent".to_owned())
                })?;
                let ($storage, $layout) = tensor.storage_and_layout();
                let Storage::Metal($metal) = &*$storage else {
                    return Err(candle_core::Error::Msg(
                        "Metal Louvain auxiliary tensor moved off device".to_owned(),
                    ));
                };
                if !$layout.is_contiguous() || $layout.dims().len() != 1 {
                    return Err(candle_core::Error::Msg(
                        "Metal Louvain auxiliary tensor is not contiguous rank one".to_owned(),
                    ));
                }
            };
        }
        let (output_rows, output_dtype) = self.output_contract()?;
        let device = primary.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(output_rows, output_dtype)
            .with_label("irongraph Metal Louvain stage")
            .build()?;
        let pipeline = pipelines(device)?;
        let bytes = |layout: &Layout, dtype: DType| layout.start_offset() * dtype.size_in_bytes();
        let groups = |rows: usize| objc2_metal::MTLSize {
            width: rows.max(1).div_ceil(THREADS),
            height: 1,
            depth: 1,
        };
        let threads = objc2_metal::MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        };
        let phase_count = match self.stage {
            Stage::Validate
            | Stage::ReducePairs
            | Stage::Degree
            | Stage::HighDegree
            | Stage::CandidateAggregate
            | Stage::Decide
            | Stage::DecideSorted
            | Stage::MatchingWinners
            | Stage::UpdateCommunityWeights
            | Stage::RebuildActive
            | Stage::CanonicalFirst
            | Stage::CanonicalPublish => 2,
            Stage::MatchingMinimum => 3,
            _ => 1,
        };
        for phase in 0..phase_count {
            let (phase_rows, tile_rows) = match (self.stage, phase) {
                (Stage::Validate | Stage::HighDegree | Stage::Decide | Stage::DecideSorted, 0) => {
                    (1, 1)
                }
                (Stage::Validate, 1) => (nodes.max(edges), WORK_TILE_ROWS),
                (Stage::Degree, 1) | (Stage::CandidateKeys | Stage::CandidateAggregate, _) => {
                    (oriented, WORK_TILE_ROWS)
                }
                (Stage::CsrOffsets, _) => (
                    nodes.checked_add(1).ok_or_else(|| {
                        candle_core::Error::Msg("Louvain CSR offset work shape overflow".to_owned())
                    })?,
                    WORK_TILE_ROWS,
                ),
                (Stage::CsrNeighbors, _) => (
                    usize::try_from(self.args.total_weight).map_err(|_| {
                        candle_core::Error::Msg(
                            "Louvain CSR neighbor work shape overflow".to_owned(),
                        )
                    })?,
                    WORK_TILE_ROWS,
                ),
                (Stage::SortedChunk, _) | (Stage::Decide, 1) => {
                    (nodes, BOUNDED_SCAN_NODE_TILE_ROWS)
                }
                (Stage::CanonicalFirst, 1)
                | (
                    Stage::BasePairs
                    | Stage::ReducePairs
                    | Stage::OrientPairs
                    | Stage::CoarsenPairs,
                    _,
                ) => (pairs, WORK_TILE_ROWS),
                _ => (nodes, WORK_TILE_ROWS),
            };
            for work_offset in (0..phase_rows).step_by(tile_rows) {
                metal_checkpoint(&self.cancellation, self.deadline)?;
                let work_count = tile_rows.min(phase_rows - work_offset);
                let chunk_args = self.args.with_work(work_offset, work_count)?;
                {
                    let encoder = device.command_encoder()?;
                    let encoder = encoder.as_ref();
                    match self.stage {
                        Stage::Validate => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.validate_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.dispatch_thread_groups(groups(1), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.validate);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::U8),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U8),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::U8),
                                );
                                encoder.set_input_buffer(
                                    3,
                                    Some(a2.buffer()),
                                    bytes(a2_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    4,
                                    Some(a3.buffer()),
                                    bytes(a3_layout, DType::U32),
                                );
                                encoder.set_output_buffer(5, Some(&output), 0);
                                encoder.set_bytes(6, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::BasePairs => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            encoder.set_compute_pipeline_state(&pipeline.base_pairs);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U8),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U8),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U8),
                            );
                            encoder.set_input_buffer(
                                3,
                                Some(a2.buffer()),
                                bytes(a2_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                4,
                                Some(a3.buffer()),
                                bytes(a3_layout, DType::U32),
                            );
                            encoder.set_output_buffer(5, Some(&output), 0);
                            encoder.set_output_buffer(
                                6,
                                Some(&output),
                                pairs * DType::I64.size_in_bytes(),
                            );
                            encoder.set_bytes(7, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::ReducePairs => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.reduce_pairs_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_output_buffer(
                                    1,
                                    Some(&output),
                                    pairs * DType::I64.size_in_bytes(),
                                );
                                encoder.set_bytes(2, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.reduce_pairs);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::I64),
                                );
                                encoder.set_output_buffer(2, Some(&output), 0);
                                encoder.set_output_buffer(
                                    3,
                                    Some(&output),
                                    pairs * DType::I64.size_in_bytes(),
                                );
                                encoder.set_bytes(4, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::OrientPairs => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.orient_pairs);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::I64),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_output_buffer(
                                3,
                                Some(&output),
                                oriented * DType::I64.size_in_bytes(),
                            );
                            encoder.set_bytes(4, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::Degree => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.degree_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_bytes(1, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.degree);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::U32),
                                );
                                encoder.set_output_buffer(3, Some(&output), 0);
                                encoder.set_bytes(4, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::HighDegree => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.high_degree_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.dispatch_thread_groups(groups(1), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.high_degree);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_output_buffer(2, Some(&output), 0);
                                encoder.set_bytes(3, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::InitializeLevel => {
                            encoder.set_compute_pipeline_state(&pipeline.initialize_level);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_output_buffer(1, Some(&output), 0);
                            encoder.set_bytes(2, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CandidateKeys => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.candidate_keys);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_bytes(3, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CandidateAggregate => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(
                                    &pipeline.candidate_aggregate_clear,
                                );
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_bytes(1, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.candidate_aggregate);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::I64),
                                );
                                encoder.set_output_buffer(2, Some(&output), 0);
                                encoder.set_bytes(3, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::SortedInitialize => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.sorted_initialize);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_bytes(3, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::SortedChunk => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            auxiliary!(4, a4_storage, a4_layout, a4);
                            auxiliary!(5, a5_storage, a5_layout, a5);
                            encoder.set_compute_pipeline_state(&pipeline.sorted_chunk);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                3,
                                Some(a2.buffer()),
                                bytes(a2_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                4,
                                Some(a3.buffer()),
                                bytes(a3_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                5,
                                Some(a4.buffer()),
                                bytes(a4_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                6,
                                Some(a5.buffer()),
                                bytes(a5_layout, DType::I64),
                            );
                            encoder.set_output_buffer(7, Some(&output), 0);
                            encoder.set_bytes(8, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::Decide => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            auxiliary!(4, a4_storage, a4_layout, a4);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.decide_clear);
                                encoder.set_output_buffer(
                                    0,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.dispatch_thread_groups(groups(1), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.decide);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    3,
                                    Some(a2.buffer()),
                                    bytes(a2_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    4,
                                    Some(a3.buffer()),
                                    bytes(a3_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    5,
                                    Some(a4.buffer()),
                                    bytes(a4_layout, DType::I64),
                                );
                                encoder.set_output_buffer(6, Some(&output), 0);
                                encoder.set_output_buffer(
                                    7,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(8, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::DecideSorted => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.decide_clear);
                                encoder.set_output_buffer(
                                    0,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.dispatch_thread_groups(groups(1), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.decide_sorted);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    3,
                                    Some(a2.buffer()),
                                    bytes(a2_layout, DType::I64),
                                );
                                encoder.set_input_buffer(
                                    4,
                                    Some(a3.buffer()),
                                    bytes(a3_layout, DType::I64),
                                );
                                encoder.set_output_buffer(5, Some(&output), 0);
                                encoder.set_output_buffer(
                                    6,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(7, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::MatchingInitialize => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            encoder.set_compute_pipeline_state(&pipeline.matching_initialize);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U32),
                            );
                            encoder.set_output_buffer(3, Some(&output), 0);
                            encoder.set_bytes(4, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::MatchingMinimum => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            match phase {
                                0 => {
                                    encoder.set_compute_pipeline_state(&pipeline.clear_u32_max);
                                    encoder.set_output_buffer(0, Some(&output), 0);
                                    encoder.set_bytes(1, &chunk_args);
                                    encoder.dispatch_thread_groups(groups(work_count), threads);
                                }
                                1 => {
                                    encoder.set_compute_pipeline_state(&pipeline.matching_priority);
                                    encoder.set_input_buffer(
                                        0,
                                        Some(primary.buffer()),
                                        bytes(primary_layout, DType::U32),
                                    );
                                    encoder.set_input_buffer(
                                        1,
                                        Some(a0.buffer()),
                                        bytes(a0_layout, DType::U32),
                                    );
                                    encoder.set_input_buffer(
                                        2,
                                        Some(a1.buffer()),
                                        bytes(a1_layout, DType::U32),
                                    );
                                    encoder.set_output_buffer(3, Some(&output), 0);
                                    encoder.set_bytes(4, &chunk_args);
                                    encoder.dispatch_thread_groups(groups(work_count), threads);
                                }
                                _ => {
                                    encoder.set_compute_pipeline_state(&pipeline.matching_minimum);
                                    encoder.set_input_buffer(
                                        0,
                                        Some(primary.buffer()),
                                        bytes(primary_layout, DType::U32),
                                    );
                                    encoder.set_input_buffer(
                                        1,
                                        Some(a0.buffer()),
                                        bytes(a0_layout, DType::U32),
                                    );
                                    encoder.set_input_buffer(
                                        2,
                                        Some(a1.buffer()),
                                        bytes(a1_layout, DType::U32),
                                    );
                                    encoder.set_input_buffer(3, Some(&output), 0);
                                    encoder.set_output_buffer(
                                        4,
                                        Some(&output),
                                        nodes * DType::U32.size_in_bytes(),
                                    );
                                    encoder.set_bytes(5, &chunk_args);
                                    encoder.dispatch_thread_groups(groups(work_count), threads);
                                }
                            }
                        }
                        Stage::MatchingWinners => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.matching_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_output_buffer(
                                    1,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_output_buffer(
                                    2,
                                    Some(&output),
                                    nodes * 2 * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(3, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.matching_winners);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    3,
                                    Some(a2.buffer()),
                                    bytes(a2_layout, DType::U32),
                                );
                                encoder.set_output_buffer(4, Some(&output), 0);
                                encoder.set_output_buffer(
                                    5,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_output_buffer(
                                    6,
                                    Some(&output),
                                    nodes * 2 * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(7, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::MatchingAdvance => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            encoder.set_compute_pipeline_state(&pipeline.matching_advance);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                3,
                                Some(a2.buffer()),
                                bytes(a2_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                4,
                                Some(a3.buffer()),
                                bytes(a3_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                5,
                                Some(a3.buffer()),
                                bytes(a3_layout, DType::U32) + nodes * DType::U32.size_in_bytes(),
                            );
                            encoder.set_output_buffer(6, Some(&output), 0);
                            encoder.set_output_buffer(
                                7,
                                Some(&output),
                                nodes * DType::U32.size_in_bytes(),
                            );
                            encoder.set_bytes(8, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::ApplyMatching => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            encoder.set_compute_pipeline_state(&pipeline.apply_matching);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U32),
                            );
                            encoder.set_output_buffer(3, Some(&output), 0);
                            encoder.set_bytes(4, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::UpdateCommunityWeights => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            auxiliary!(2, a2_storage, a2_layout, a2);
                            auxiliary!(3, a3_storage, a3_layout, a3);
                            if phase == 0 {
                                encoder
                                    .set_compute_pipeline_state(&pipeline.copy_community_weights);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_output_buffer(1, Some(&output), 0);
                                encoder.set_bytes(2, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder
                                    .set_compute_pipeline_state(&pipeline.update_community_weights);
                                encoder.set_input_buffer(
                                    0,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a1.buffer()),
                                    bytes(a1_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    2,
                                    Some(a2.buffer()),
                                    bytes(a2_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    3,
                                    Some(a3.buffer()),
                                    bytes(a3_layout, DType::I64),
                                );
                                encoder.set_output_buffer(4, Some(&output), 0);
                                encoder.set_bytes(5, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::RebuildActive => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.active_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_output_buffer(
                                    1,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(2, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.rebuild_active);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_output_buffer(2, Some(&output), 0);
                                encoder.set_output_buffer(
                                    3,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(4, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::MapOriginal => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.map_original);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::U32),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_bytes(3, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CoarsenPairs => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            auxiliary!(1, a1_storage, a1_layout, a1);
                            encoder.set_compute_pipeline_state(&pipeline.coarsen_pairs);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                2,
                                Some(a1.buffer()),
                                bytes(a1_layout, DType::U32),
                            );
                            encoder.set_output_buffer(3, Some(&output), 0);
                            encoder.set_output_buffer(
                                4,
                                Some(&output),
                                pairs * DType::I64.size_in_bytes(),
                            );
                            encoder.set_bytes(5, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CanonicalFirst => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            if phase == 0 {
                                encoder.set_compute_pipeline_state(&pipeline.canonical_first_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_output_buffer(
                                    1,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(2, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder
                                    .set_compute_pipeline_state(&pipeline.canonical_first_scatter);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::U32),
                                );
                                encoder.set_input_buffer(
                                    1,
                                    Some(a0.buffer()),
                                    bytes(a0_layout, DType::U32),
                                );
                                encoder.set_output_buffer(2, Some(&output), 0);
                                encoder.set_output_buffer(
                                    3,
                                    Some(&output),
                                    nodes * DType::U32.size_in_bytes(),
                                );
                                encoder.set_bytes(4, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::CanonicalKeys => {
                            encoder.set_compute_pipeline_state(&pipeline.canonical_keys);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::U32),
                            );
                            encoder.set_output_buffer(1, Some(&output), 0);
                            encoder.set_bytes(2, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CanonicalPublish => {
                            if phase == 0 {
                                encoder
                                    .set_compute_pipeline_state(&pipeline.canonical_publish_clear);
                                encoder.set_output_buffer(0, Some(&output), 0);
                                encoder.set_bytes(1, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            } else {
                                encoder.set_compute_pipeline_state(&pipeline.canonical_publish);
                                encoder.set_input_buffer(
                                    0,
                                    Some(primary.buffer()),
                                    bytes(primary_layout, DType::I64),
                                );
                                encoder.set_output_buffer(1, Some(&output), 0);
                                encoder.set_bytes(2, &chunk_args);
                                encoder.dispatch_thread_groups(groups(work_count), threads);
                            }
                        }
                        Stage::CsrOffsets => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.csr_offsets);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::I64),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_bytes(3, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                        Stage::CsrNeighbors => {
                            auxiliary!(0, a0_storage, a0_layout, a0);
                            encoder.set_compute_pipeline_state(&pipeline.csr_neighbors);
                            encoder.set_input_buffer(
                                0,
                                Some(primary.buffer()),
                                bytes(primary_layout, DType::I64),
                            );
                            encoder.set_input_buffer(
                                1,
                                Some(a0.buffer()),
                                bytes(a0_layout, DType::I64),
                            );
                            encoder.set_output_buffer(2, Some(&output), 0);
                            encoder.set_bytes(3, &chunk_args);
                            encoder.dispatch_thread_groups(groups(work_count), threads);
                        }
                    }
                }
            }
        }
        device.wait_until_completed()?;
        metal_checkpoint(&self.cancellation, self.deadline)?;
        Ok((
            MetalStorage::new(output, device.clone(), output_rows, output_dtype),
            Shape::from(output_rows),
        ))
    }
}

fn apply(
    stage: Stage,
    primary: &Tensor,
    auxiliary: Vec<Tensor>,
    args: GraphArgs,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<Tensor> {
    primary
        .apply_op1_no_bwd(&LouvainOp {
            stage,
            auxiliary,
            args,
            cancellation: cancellation.clone(),
            deadline,
        })
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))
}

fn split_i64(workspace: &Tensor, rows: usize) -> Result<(Tensor, Tensor)> {
    Ok((
        workspace.narrow(0, 0, rows).map_err(candle_error)?,
        workspace.narrow(0, rows, rows).map_err(candle_error)?,
    ))
}

fn stable_sort(
    keys: &Tensor,
    values: &Tensor,
    _device: &Device,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<(Tensor, Tensor)> {
    if keys.elem_count() != values.elem_count() || keys.dtype() != DType::I64 {
        return Err(Error::internal("Metal Louvain sort columns are misaligned"));
    }
    let mut positions = keys
        .apply_op1_no_bwd(&LouvainArange {
            cancellation: cancellation.clone(),
            deadline,
        })
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    for digit_pass in 0..8 {
        ensure_graph_execution(cancellation, deadline)?;
        positions = keys
            .apply_op2_no_bwd(
                &positions,
                &LouvainStableRadixPass {
                    digit_pass,
                    cancellation: cancellation.clone(),
                    deadline,
                },
            )
            .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    }
    let gather = LouvainGather {
        cancellation: cancellation.clone(),
        deadline,
    };
    Ok((
        keys.apply_op2_no_bwd(&positions, &gather)
            .map_err(|error| louvain_candle_error(error, cancellation, deadline))?,
        values
            .apply_op2_no_bwd(&positions, &gather)
            .map_err(|error| louvain_candle_error(error, cancellation, deadline))?,
    ))
}

fn use_sorted_candidate_path(
    maximum_degree: u32,
    total_weight: u64,
    oriented_capacity: usize,
) -> Result<bool> {
    let maximum_degree = u64::from(maximum_degree);
    if maximum_degree <= DIRECT_NEIGHBOR_FAST_LIMIT {
        return Ok(false);
    }
    // This hard ceiling is also enforced by the direct Metal kernel. It bounds cancellation
    // latency and makes every direct lane's quadratic work finite independent of graph size.
    if maximum_degree > DIRECT_NEIGHBOR_HARD_LIMIT {
        return Ok(true);
    }
    // Direct row aggregation is bounded by max(d)*sum(d). The fallback performs eight stable
    // radix passes across the fixed oriented capacity and pays substantial command-buffer setup.
    // The fixed work term is calibrated by the real-Metal crossover gates: a 160-edge hub remains
    // direct, while the 24,963-edge dense fixture routes sorted. This avoids both tiny-hub launch
    // domination and unbounded O(sum d^2) work on dense/power-law levels.
    let direct_upper = maximum_degree.checked_mul(total_weight).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal Louvain direct-work estimate overflow",
        )
    })?;
    let sorted_work = u64::try_from(oriented_capacity)
        .ok()
        .and_then(|rows| rows.checked_mul(SORTED_CANDIDATE_RADIX_PASSES))
        .and_then(|work| work.checked_add(SORTED_CANDIDATE_FIXED_WORK))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal Louvain sorted-work estimate overflow",
            )
        })?;
    Ok(direct_upper > sorted_work)
}

const RADIX_TILE_ROWS: usize = 1_024;
const RADIX_BUCKETS: usize = 256;

fn scratch_overflow(detail: &'static str) -> Error {
    Error::new(ErrorCode::ResultBudgetExceeded, detail)
}

fn checked_product(left: usize, right: usize, detail: &'static str) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| scratch_overflow(detail))
}

fn checked_sum<const N: usize>(parts: [usize; N], detail: &'static str) -> Result<usize> {
    parts.into_iter().try_fold(0_usize, |total, part| {
        total
            .checked_add(part)
            .ok_or_else(|| scratch_overflow(detail))
    })
}

/// Candle 0.11's pooled private/shared Metal allocator requests the next power-of-two byte size.
/// There is no 4 KiB floor on the selected Apple device.
fn pooled_bytes(logical: usize, detail: &'static str) -> Result<usize> {
    if logical == 0 {
        return Ok(0);
    }
    logical
        .checked_next_power_of_two()
        .ok_or_else(|| scratch_overflow(detail))
}

fn pooled_rows(rows: usize, element_bytes: usize, detail: &'static str) -> Result<usize> {
    pooled_bytes(checked_product(rows, element_bytes, detail)?, detail)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StableSortScratchBreakdown {
    first_radix_pass: usize,
    later_radix_pass: usize,
    gather: usize,
}

impl StableSortScratchBreakdown {
    fn required_bytes(self) -> usize {
        self.first_radix_pass
            .max(self.later_radix_pass)
            .max(self.gather)
    }
}

/// Transient allocations inside `stable_sort`, excluding its externally owned key/value inputs.
/// The tiled device arange owns one position buffer. Every synchronized radix pass owns its input
/// and output position buffers, one histogram table, one offset table, and two 256-entry bucket
/// state vectors. Scratch from different digits never overlaps. The final bounded gather owns one
/// position and two independently pooled I64 outputs.
fn stable_sort_scratch(rows: usize) -> Result<StableSortScratchBreakdown> {
    if rows == 0 {
        return Ok(StableSortScratchBreakdown {
            first_radix_pass: 0,
            later_radix_pass: 0,
            gather: 0,
        });
    }
    let positions = pooled_rows(
        rows,
        std::mem::size_of::<i64>(),
        "Metal Louvain radix position allocation overflow",
    )?;
    let table_elements = rows
        .div_ceil(RADIX_TILE_ROWS)
        .checked_mul(RADIX_BUCKETS)
        .ok_or_else(|| scratch_overflow("Metal Louvain radix table shape overflow"))?;
    let histograms = pooled_rows(
        table_elements,
        std::mem::size_of::<u32>(),
        "Metal Louvain radix histogram allocation overflow",
    )?;
    let offsets = pooled_rows(
        table_elements,
        std::mem::size_of::<i64>(),
        "Metal Louvain radix offset allocation overflow",
    )?;
    let bucket_state = pooled_rows(
        RADIX_BUCKETS,
        std::mem::size_of::<i64>(),
        "Metal Louvain radix bucket-state allocation overflow",
    )?;
    let radix_pass = checked_sum(
        [
            positions,
            positions,
            histograms,
            offsets,
            bucket_state,
            bucket_state,
        ],
        "Metal Louvain radix-pass scratch overflow",
    )?;
    Ok(StableSortScratchBreakdown {
        first_radix_pass: radix_pass,
        later_radix_pass: radix_pass,
        gather: checked_product(
            positions,
            3,
            "Metal Louvain radix gather position scratch overflow",
        )?,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LouvainScratchBreakdown {
    selection_and_node_readback: usize,
    base_preprocess: usize,
    oriented_sort_and_level_setup: usize,
    direct_local_move: usize,
    sorted_local_move: usize,
    coarsening: usize,
    result_readback: usize,
}

impl LouvainScratchBreakdown {
    fn required_bytes(self) -> usize {
        self.selection_and_node_readback
            .max(self.base_preprocess)
            .max(self.oriented_sort_and_level_setup)
            .max(self.direct_local_move)
            .max(self.sorted_local_move)
            .max(self.coarsening)
            .max(self.result_readback)
    }
}

#[allow(clippy::too_many_lines)]
fn scratch_breakdown(node_count: usize, edge_count: usize) -> Result<LouvainScratchBreakdown> {
    if node_count == 0 && edge_count == 0 {
        return Ok(LouvainScratchBreakdown {
            selection_and_node_readback: 0,
            base_preprocess: 0,
            oriented_sort_and_level_setup: 0,
            direct_local_move: 0,
            sorted_local_move: 0,
            coarsening: 0,
            result_readback: 0,
        });
    }

    let u8_nodes = pooled_rows(
        node_count,
        std::mem::size_of::<u8>(),
        "Metal Louvain U8 node allocation overflow",
    )?;
    let u32_nodes = pooled_rows(
        node_count,
        std::mem::size_of::<u32>(),
        "Metal Louvain U32 node allocation overflow",
    )?;
    let i64_nodes = pooled_rows(
        node_count,
        std::mem::size_of::<i64>(),
        "Metal Louvain I64 node allocation overflow",
    )?;
    let host_u32_nodes = checked_product(
        node_count,
        std::mem::size_of::<u32>(),
        "Metal Louvain host node vector overflow",
    )?;
    let control = pooled_rows(
        node_count
            .checked_add(1)
            .ok_or_else(|| scratch_overflow("Metal Louvain node control shape overflow"))?,
        std::mem::size_of::<u32>(),
        "Metal Louvain node control allocation overflow",
    )?;
    let u32_two_nodes = pooled_rows(
        node_count
            .checked_mul(2)
            .ok_or_else(|| scratch_overflow("Metal Louvain two-node shape overflow"))?,
        std::mem::size_of::<u32>(),
        "Metal Louvain two-node allocation overflow",
    )?;
    let winner_rows = node_count
        .checked_mul(2)
        .and_then(|rows| rows.checked_add(1))
        .ok_or_else(|| scratch_overflow("Metal Louvain winner shape overflow"))?;
    let winners = pooled_rows(
        winner_rows,
        std::mem::size_of::<u32>(),
        "Metal Louvain winner allocation overflow",
    )?;
    let scalar_u32 = pooled_rows(
        1,
        std::mem::size_of::<u32>(),
        "Metal Louvain U32 scalar allocation overflow",
    )?;
    let i64_reduction_scalar = pooled_rows(
        1,
        std::mem::size_of::<i64>(),
        "Metal Louvain I64 scalar allocation overflow",
    )?;

    // `node_selection_mask` has at most four simultaneously live U8 columns. The generic
    // compactor's Hillis-Steele scan keeps its I64 input plus four scan-generation buffers; its
    // final scatter keeps two I64 columns and six U32 columns. `to_vec1` then overlaps the visible
    // rows, one shared readback allocation, and the exact host result vector.
    let mask_build = checked_product(
        u8_nodes,
        4,
        "Metal Louvain visibility-mask scratch overflow",
    )?;
    let selection_scan = checked_sum(
        [
            u8_nodes,
            u32_nodes,
            checked_product(
                i64_nodes,
                5,
                "Metal Louvain visibility scan scratch overflow",
            )?,
        ],
        "Metal Louvain visibility scan peak overflow",
    )?;
    let selection_scatter = checked_sum(
        [
            u8_nodes,
            checked_product(
                u32_nodes,
                6,
                "Metal Louvain visibility scatter U32 scratch overflow",
            )?,
            checked_product(
                i64_nodes,
                2,
                "Metal Louvain visibility scatter I64 scratch overflow",
            )?,
        ],
        "Metal Louvain visibility scatter peak overflow",
    )?;
    let node_readback = checked_sum(
        [u8_nodes, u32_nodes, u32_nodes, host_u32_nodes],
        "Metal Louvain visible-node readback overflow",
    )?;
    let selection_and_node_readback = mask_build
        .max(selection_scan)
        .max(selection_scatter)
        .max(node_readback);

    let front = checked_sum(
        [u8_nodes, u32_nodes, host_u32_nodes],
        "Metal Louvain persistent visible-row scratch overflow",
    )?;
    if edge_count == 0 {
        let result_readback = checked_sum(
            [front, host_u32_nodes],
            "Metal Louvain edgeless result scratch overflow",
        )?;
        return Ok(LouvainScratchBreakdown {
            selection_and_node_readback,
            base_preprocess: 0,
            oriented_sort_and_level_setup: 0,
            direct_local_move: 0,
            sorted_local_move: 0,
            coarsening: 0,
            result_readback,
        });
    }

    let pair_workspace = pooled_rows(
        edge_count,
        2 * std::mem::size_of::<i64>(),
        "Metal Louvain pair workspace overflow",
    )?;
    let edge_i64 = pooled_rows(
        edge_count,
        std::mem::size_of::<i64>(),
        "Metal Louvain edge-column allocation overflow",
    )?;
    let oriented_count = edge_count
        .checked_mul(2)
        .ok_or_else(|| scratch_overflow("Metal Louvain oriented row shape overflow"))?;
    let oriented_i64 = pooled_rows(
        oriented_count,
        std::mem::size_of::<i64>(),
        "Metal Louvain oriented-column allocation overflow",
    )?;
    let oriented_workspace = pooled_rows(
        oriented_count,
        2 * std::mem::size_of::<i64>(),
        "Metal Louvain oriented workspace overflow",
    )?;

    let base_sort = checked_sum(
        [
            front,
            pair_workspace,
            stable_sort_scratch(edge_count)?.required_bytes(),
        ],
        "Metal Louvain base-sort peak overflow",
    )?;
    let base_reduce = checked_sum(
        [front, edge_i64, edge_i64, pair_workspace],
        "Metal Louvain base-reduce peak overflow",
    )?;
    let base_weight_readback = checked_sum(
        [
            front,
            pair_workspace,
            i64_reduction_scalar,
            i64_reduction_scalar,
            8,
        ],
        "Metal Louvain base-weight readback peak overflow",
    )?;
    let base_initialize = checked_sum(
        [front, pair_workspace, u32_nodes, u32_nodes],
        "Metal Louvain base initialization peak overflow",
    )?;
    let base_preprocess = base_sort
        .max(base_reduce)
        .max(base_weight_readback)
        .max(base_initialize);

    // After the first coarsening, `active` is a view into a V+1 control allocation, so the
    // allocator-rounded control size (not merely 4V logical bytes) is the persistent upper bound.
    let level_persistent = checked_sum(
        [front, pair_workspace, control, u32_nodes],
        "Metal Louvain level-persistent scratch overflow",
    )?;
    let oriented_sort = checked_sum(
        [
            level_persistent,
            oriented_workspace,
            stable_sort_scratch(oriented_count)?.required_bytes(),
        ],
        "Metal Louvain oriented-sort peak overflow",
    )?;
    let level_setup = checked_sum(
        [
            level_persistent,
            oriented_i64,
            oriented_i64,
            i64_nodes,
            u32_nodes,
            scalar_u32,
            scalar_u32,
            4,
        ],
        "Metal Louvain level-setup peak overflow",
    )?;
    let oriented_sort_and_level_setup = oriented_sort.max(level_setup);

    // `degree`, an independently updated community-weight vector, and membership all remain live
    // across subsequent local passes. Tensor views/clones that share one allocation are counted
    // once; every listed entry below is a distinct live Metal buffer.
    let local_common = checked_sum(
        [
            level_persistent,
            oriented_i64,
            oriented_i64,
            i64_nodes,
            i64_nodes,
            u32_nodes,
        ],
        "Metal Louvain local-move common scratch overflow",
    )?;
    let control_readback = checked_sum(
        [control, scalar_u32, scalar_u32, 4],
        "Metal Louvain proposal readback overflow",
    )?;
    let matching_minimum = checked_sum(
        [control, u32_nodes, u32_nodes, u32_two_nodes],
        "Metal Louvain matching-minimum scratch overflow",
    )?;
    let matching_winners = checked_sum(
        [
            control,
            u32_nodes,
            u32_nodes,
            u32_two_nodes,
            winners,
            scalar_u32,
            scalar_u32,
            4,
        ],
        "Metal Louvain matching-winner scratch overflow",
    )?;
    let matching_advance = checked_sum(
        [control, u32_nodes, u32_nodes, winners, u32_two_nodes],
        "Metal Louvain matching-advance scratch overflow",
    )?;
    let update_community = checked_sum(
        [control, u32_nodes, i64_nodes],
        "Metal Louvain community-update scratch overflow",
    )?;
    let apply_membership = checked_sum(
        [control, u32_nodes, u32_nodes],
        "Metal Louvain membership-update scratch overflow",
    )?;
    let direct_extra = control_readback
        .max(matching_minimum)
        .max(matching_winners)
        .max(matching_advance)
        .max(update_community)
        .max(apply_membership);
    let direct_local_move = checked_sum(
        [local_common, direct_extra],
        "Metal Louvain direct local-move peak overflow",
    )?;

    let candidate_i64 = oriented_i64;
    let candidate_sort = checked_sum(
        [
            candidate_i64,
            stable_sort_scratch(oriented_count)?.required_bytes(),
        ],
        "Metal Louvain candidate-sort scratch overflow",
    )?;
    let sorted_state = pooled_rows(
        node_count
            .checked_mul(3)
            .ok_or_else(|| scratch_overflow("Metal Louvain sorted-state shape overflow"))?,
        std::mem::size_of::<i64>(),
        "Metal Louvain sorted-state allocation overflow",
    )?;
    let candidate_aggregate = checked_product(
        candidate_i64,
        3,
        "Metal Louvain candidate-aggregate scratch overflow",
    )?;
    let sorted_initialize = checked_sum(
        [candidate_i64, candidate_i64, sorted_state],
        "Metal Louvain sorted-state initialization scratch overflow",
    )?;
    let sorted_chunk = checked_sum(
        [candidate_i64, candidate_i64, sorted_state, sorted_state],
        "Metal Louvain sorted-chunk scratch overflow",
    )?;
    let sorted_finalize = checked_sum(
        [sorted_state, control, scalar_u32, scalar_u32, 4],
        "Metal Louvain sorted-finalize scratch overflow",
    )?;
    let sorted_local_move = checked_sum(
        [
            local_common,
            candidate_sort
                .max(candidate_aggregate)
                .max(sorted_initialize)
                .max(sorted_chunk)
                .max(sorted_finalize)
                .max(direct_extra),
        ],
        "Metal Louvain sorted local-move peak overflow",
    )?;

    let coarsen_nodes = checked_sum(
        [front, u32_nodes, u32_nodes, control],
        "Metal Louvain coarsening node-state overflow",
    )?;
    let map_original = checked_sum(
        [coarsen_nodes, pair_workspace, u32_nodes],
        "Metal Louvain map-original peak overflow",
    )?;
    let rebuild_active = checked_sum(
        [
            coarsen_nodes,
            pair_workspace,
            control,
            scalar_u32,
            scalar_u32,
            4,
        ],
        "Metal Louvain active-rebuild peak overflow",
    )?;
    let map_pairs = checked_sum(
        [coarsen_nodes, pair_workspace, pair_workspace],
        "Metal Louvain pair-map peak overflow",
    )?;
    let coarsen_sort = checked_sum(
        [
            front,
            u32_nodes,
            control,
            pair_workspace,
            stable_sort_scratch(edge_count)?.required_bytes(),
        ],
        "Metal Louvain coarsening-sort peak overflow",
    )?;
    let coarsen_reduce = checked_sum(
        [
            front,
            u32_nodes,
            control,
            edge_i64,
            edge_i64,
            pair_workspace,
        ],
        "Metal Louvain coarsening-reduce peak overflow",
    )?;
    let coarsening = map_original
        .max(rebuild_active)
        .max(map_pairs)
        .max(coarsen_sort)
        .max(coarsen_reduce);

    // Pair/oriented/local state is explicitly dropped before canonical publication. Community
    // meaning stays on device: first-visible positions are atomically collected, sorted, published
    // by opaque label, and gathered. Only the final compact canonical column crosses to the host.
    let final_persistent = checked_sum(
        [front, u32_nodes],
        "Metal Louvain final persistent state overflow",
    )?;
    let canonical_first = checked_sum(
        [final_persistent, control, scalar_u32, scalar_u32, 4],
        "Metal Louvain canonical-first scratch overflow",
    )?;
    let canonical_keys = checked_sum(
        [final_persistent, control, i64_nodes],
        "Metal Louvain canonical-key scratch overflow",
    )?;
    let canonical_sort = checked_sum(
        [
            final_persistent,
            i64_nodes,
            stable_sort_scratch(node_count)?.required_bytes(),
        ],
        "Metal Louvain canonical-sort scratch overflow",
    )?;
    let canonical_publish = checked_sum(
        [final_persistent, i64_nodes, u32_nodes],
        "Metal Louvain canonical-publish scratch overflow",
    )?;
    let canonical_gather = checked_sum(
        [front, u32_nodes, u32_nodes, u32_nodes],
        "Metal Louvain canonical-gather scratch overflow",
    )?;
    let label_readback = checked_sum(
        [front, u32_nodes, u32_nodes, host_u32_nodes],
        "Metal Louvain label readback scratch overflow",
    )?;
    let result_readback = canonical_first
        .max(canonical_keys)
        .max(canonical_sort)
        .max(canonical_publish)
        .max(canonical_gather)
        .max(label_readback);

    Ok(LouvainScratchBreakdown {
        selection_and_node_readback,
        base_preprocess,
        oriented_sort_and_level_setup,
        direct_local_move,
        sorted_local_move,
        coarsening,
        result_readback,
    })
}

pub fn scratch_bytes(node_count: usize, edge_count: usize) -> Result<usize> {
    Ok(scratch_breakdown(node_count, edge_count)?.required_bytes())
}

/// Peak transient plus returned-output bytes for the shared unique-undirected CSR builder.
/// Resident graph columns and the caller-owned visibility mask are excluded.
pub fn unique_undirected_csr_scratch_bytes(node_count: usize, edge_count: usize) -> Result<usize> {
    let offsets = pooled_rows(
        node_count
            .checked_add(1)
            .ok_or_else(|| scratch_overflow("unique CSR offset shape overflow"))?,
        std::mem::size_of::<u32>(),
        "unique CSR offset allocation overflow",
    )?;
    if edge_count == 0 {
        return Ok(offsets);
    }
    let pair_workspace = pooled_rows(
        edge_count,
        2 * std::mem::size_of::<i64>(),
        "unique CSR pair workspace overflow",
    )?;
    let edge_i64 = pooled_rows(
        edge_count,
        std::mem::size_of::<i64>(),
        "unique CSR edge column overflow",
    )?;
    let oriented_count = edge_count
        .checked_mul(2)
        .ok_or_else(|| scratch_overflow("unique CSR oriented shape overflow"))?;
    let oriented_i64 = pooled_rows(
        oriented_count,
        std::mem::size_of::<i64>(),
        "unique CSR oriented column overflow",
    )?;
    let oriented_workspace = pooled_rows(
        oriented_count,
        2 * std::mem::size_of::<i64>(),
        "unique CSR oriented workspace overflow",
    )?;
    let neighbors = pooled_rows(
        oriented_count,
        std::mem::size_of::<u32>(),
        "unique CSR neighbor allocation overflow",
    )?;
    let scalar_i64 = pooled_rows(
        1,
        std::mem::size_of::<i64>(),
        "unique CSR scalar allocation overflow",
    )?;
    let validation = checked_sum(
        [
            pooled_rows(1, std::mem::size_of::<u32>(), "unique CSR status overflow")?,
            pooled_rows(1, std::mem::size_of::<u32>(), "unique CSR staging overflow")?,
            4,
        ],
        "unique CSR validation peak overflow",
    )?;
    let base_sort = checked_sum(
        [
            pair_workspace,
            stable_sort_scratch(edge_count)?.required_bytes(),
        ],
        "unique CSR base-sort peak overflow",
    )?;
    let base_reduce = checked_sum(
        [edge_i64, edge_i64, pair_workspace],
        "unique CSR base-reduce peak overflow",
    )?;
    let base_sum = checked_sum(
        [pair_workspace, scalar_i64, scalar_i64, 8],
        "unique CSR base-sum peak overflow",
    )?;
    let orient = checked_sum(
        [pair_workspace, oriented_workspace],
        "unique CSR orientation peak overflow",
    )?;
    let orient_sort = checked_sum(
        [
            oriented_workspace,
            stable_sort_scratch(oriented_count)?.required_bytes(),
        ],
        "unique CSR oriented-sort peak overflow",
    )?;
    let publication = checked_sum(
        [oriented_i64, oriented_i64, offsets, neighbors],
        "unique CSR publication peak overflow",
    )?;
    Ok(validation
        .max(base_sort)
        .max(base_reduce)
        .max(base_sum)
        .max(orient)
        .max(orient_sort)
        .max(publication))
}

/// Worst-case bytes retained by the returned CSR while a caller's algorithm workspace executes.
pub fn unique_undirected_csr_retained_bytes(node_count: usize, edge_count: usize) -> Result<usize> {
    let offsets = pooled_rows(
        node_count
            .checked_add(1)
            .ok_or_else(|| scratch_overflow("unique CSR retained offset shape overflow"))?,
        std::mem::size_of::<u32>(),
        "unique CSR retained offset allocation overflow",
    )?;
    let neighbors = pooled_rows(
        edge_count
            .checked_mul(2)
            .ok_or_else(|| scratch_overflow("unique CSR retained neighbor shape overflow"))?,
        std::mem::size_of::<u32>(),
        "unique CSR retained neighbor allocation overflow",
    )?;
    checked_sum(
        [offsets, neighbors],
        "unique CSR retained allocation overflow",
    )
}

struct PreparedUniquePairs {
    keys: Tensor,
    weights: Tensor,
    total_weight: u64,
}

pub struct UniqueUndirectedCsr {
    pub offsets: Tensor,
    pub neighbors: Tensor,
    pub adjacency_count: usize,
}

fn csr_adjacency_count(total_weight: u64) -> Result<usize> {
    let count = usize::try_from(total_weight).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal unique-undirected reciprocal row count exceeds usize",
        )
    })?;
    if count > u32::MAX as usize {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "unique-undirected adjacency exceeds canonical U32 CSR capacity",
        ));
    }
    Ok(count)
}

fn prepare_unique_pairs(
    resident: &CandleResident,
    visible_mask: &Tensor,
    layers: LayerMask,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<PreparedUniquePairs> {
    ensure_graph_execution(cancellation, deadline)?;
    let node_count = resident.node_count;
    let edge_count = resident.edge_count;
    let Some(edge_active) = resident.edge_active.as_ref() else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident unique-undirected builder edge activity column is absent",
        ));
    };
    let Some(edge_layers) = resident.edge_layers.as_ref() else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident unique-undirected builder edge layer column is absent",
        ));
    };
    let Some(edge_sources) = resident.edge_sources.as_ref() else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident unique-undirected builder edge source column is absent",
        ));
    };
    let Some(edge_targets) = resident.edge_targets.as_ref() else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "resident unique-undirected builder edge target column is absent",
        ));
    };
    let device = visible_mask.device();
    let args = GraphArgs::new(node_count, edge_count, edge_count, 0, layers, 0)?;
    let graph_columns = vec![
        edge_active.clone(),
        edge_layers.clone(),
        edge_sources.clone(),
        edge_targets.clone(),
    ];
    let status = apply(
        Stage::Validate,
        visible_mask,
        graph_columns.clone(),
        args,
        cancellation,
        deadline,
    )?
    .to_vec1::<u32>()
    .map_err(candle_error)?
    .first()
    .copied()
    .unwrap_or(3);
    if status != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal unique-undirected graph validation failed with status {status}"),
        ));
    }
    let (base_keys, base_weights) = {
        let base = apply(
            Stage::BasePairs,
            visible_mask,
            graph_columns,
            args,
            cancellation,
            deadline,
        )?;
        let (keys, weights) = split_i64(&base, edge_count)?;
        stable_sort(&keys, &weights, device, cancellation, deadline)?
    };
    let reduced = apply(
        Stage::ReducePairs,
        &base_keys,
        vec![base_weights],
        args,
        cancellation,
        deadline,
    )?;
    drop(base_keys);
    let (keys, weights) = split_i64(&reduced, edge_count)?;
    let unique_edges = sum_i64(&weights, cancellation, deadline)?;
    let total_weight = u64::try_from(unique_edges)
        .ok()
        .and_then(|edges| edges.checked_mul(2))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal unique-undirected reciprocal row count overflow",
            )
        })?;
    Ok(PreparedUniquePairs {
        keys,
        weights,
        total_weight,
    })
}

#[allow(clippy::too_many_lines)]
pub fn build_unique_undirected_csr(
    resident: &CandleResident,
    visible_mask: &Tensor,
    layers: LayerMask,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<UniqueUndirectedCsr> {
    ensure_graph_execution(cancellation, deadline)?;
    let Device::Metal(_) = visible_mask.device() else {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "unique-undirected CSR construction requires the Metal graph backend",
        ));
    };
    if resident.edge_count == 0 {
        let offset_rows = resident.node_count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "unique-undirected empty CSR offset shape overflow",
            )
        })?;
        let offsets = visible_mask
            .apply_op1_no_bwd(&LouvainZeroU32 {
                rows: offset_rows,
                cancellation: cancellation.clone(),
                deadline,
            })
            .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
        let neighbors =
            Tensor::zeros(0, DType::U32, visible_mask.device()).map_err(candle_error)?;
        return Ok(UniqueUndirectedCsr {
            offsets,
            neighbors,
            adjacency_count: 0,
        });
    }
    let prepared = prepare_unique_pairs(resident, visible_mask, layers, cancellation, deadline)?;
    let adjacency_count = csr_adjacency_count(prepared.total_weight)?;
    let args = GraphArgs::new(
        resident.node_count,
        resident.edge_count,
        resident.edge_count,
        prepared.total_weight,
        layers,
        0,
    )?;
    let oriented_count = resident.edge_count.checked_mul(2).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal unique-undirected oriented shape overflow",
        )
    })?;
    if adjacency_count == 0 {
        let offset_rows = resident.node_count.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "unique-undirected zero-adjacency CSR offset shape overflow",
            )
        })?;
        let offsets = visible_mask
            .apply_op1_no_bwd(&LouvainZeroU32 {
                rows: offset_rows,
                cancellation: cancellation.clone(),
                deadline,
            })
            .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
        let neighbors =
            Tensor::zeros(0, DType::U32, visible_mask.device()).map_err(candle_error)?;
        ensure_graph_execution(cancellation, deadline)?;
        return Ok(UniqueUndirectedCsr {
            offsets,
            neighbors,
            adjacency_count: 0,
        });
    }
    let oriented = apply(
        Stage::OrientPairs,
        &prepared.keys,
        vec![prepared.weights],
        args,
        cancellation,
        deadline,
    )?;
    drop(prepared.keys);
    let (oriented_keys, oriented_weights) = split_i64(&oriented, oriented_count)?;
    let (oriented_keys, oriented_weights) = stable_sort(
        &oriented_keys,
        &oriented_weights,
        visible_mask.device(),
        cancellation,
        deadline,
    )?;
    let offsets = apply(
        Stage::CsrOffsets,
        &oriented_keys,
        vec![oriented_weights.clone()],
        args,
        cancellation,
        deadline,
    )?;
    let neighbors = apply(
        Stage::CsrNeighbors,
        &oriented_keys,
        vec![oriented_weights],
        args,
        cancellation,
        deadline,
    )?;
    ensure_graph_execution(cancellation, deadline)?;
    Ok(UniqueUndirectedCsr {
        offsets,
        neighbors,
        adjacency_count,
    })
}

#[allow(clippy::too_many_lines)]
pub fn execute(
    resident: &CandleResident,
    visible_mask: &Tensor,
    visible_rows: &Tensor,
    layers: LayerMask,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<ResidentGraphProcedureResult> {
    execute_with_progress(
        resident,
        visible_mask,
        visible_rows,
        layers,
        cancellation,
        deadline,
        None,
    )
}

#[derive(Default)]
struct LouvainProgress {
    total_local_passes: usize,
    max_level_local_passes: usize,
    sorted_candidate_levels: usize,
    sorted_candidate_chunks: usize,
    #[cfg(test)]
    cancel_after_sorted_candidate_chunks: Option<usize>,
    #[cfg(test)]
    cancellation_triggered_at: Option<std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>>>,
}

#[allow(clippy::too_many_lines)]
fn execute_with_progress(
    resident: &CandleResident,
    visible_mask: &Tensor,
    visible_rows: &Tensor,
    layers: LayerMask,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
    mut progress: Option<&mut LouvainProgress>,
) -> Result<ResidentGraphProcedureResult> {
    ensure_graph_execution(cancellation, deadline)?;
    let Device::Metal(_) = visible_mask.device() else {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "resident Louvain requires the Metal graph backend",
        ));
    };
    let node_count = resident.node_count;
    let edge_count = resident.edge_count;
    let node_rows = visible_rows.to_vec1::<u32>().map_err(candle_error)?;
    ensure_graph_execution(cancellation, deadline)?;
    if node_rows.is_empty() {
        return Ok(ResidentGraphProcedureResult::Louvain {
            node_rows,
            community: Vec::new(),
        });
    }
    if edge_count == 0 {
        let community = (0..node_rows.len())
            .map(|index| {
                u32::try_from(index).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Metal Louvain community ID exceeds u32",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure_graph_execution(cancellation, deadline)?;
        return Ok(ResidentGraphProcedureResult::Louvain {
            node_rows,
            community,
        });
    }
    let device = visible_mask.device();
    let prepared = prepare_unique_pairs(resident, visible_mask, layers, cancellation, deadline)?;
    let mut pair_keys = prepared.keys;
    let mut pair_weights = prepared.weights;
    let total_weight = prepared.total_weight;
    let args = GraphArgs::new(node_count, edge_count, edge_count, total_weight, layers, 0)?;
    let mut active = visible_mask
        .apply_op1_no_bwd(&LouvainU8ToU32 {
            cancellation: cancellation.clone(),
            deadline,
        })
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    let mut active_count = node_rows.len();
    let mut original_map = apply(
        Stage::InitializeLevel,
        &active,
        Vec::new(),
        args,
        cancellation,
        deadline,
    )?;

    if total_weight != 0 {
        for _level in 0..node_count {
            ensure_graph_execution(cancellation, deadline)?;
            let oriented_count = edge_count.checked_mul(2).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal Louvain oriented row count overflow",
                )
            })?;
            let (oriented_keys, oriented_weights) = {
                let oriented = apply(
                    Stage::OrientPairs,
                    &pair_keys,
                    vec![pair_weights.clone()],
                    args,
                    cancellation,
                    deadline,
                )?;
                let (oriented_keys, oriented_weights) = split_i64(&oriented, oriented_count)?;
                stable_sort(
                    &oriented_keys,
                    &oriented_weights,
                    device,
                    cancellation,
                    deadline,
                )?
            };
            let degree = apply(
                Stage::Degree,
                &oriented_keys,
                vec![oriented_weights.clone(), active.clone()],
                args,
                cancellation,
                deadline,
            )?;
            let maximum_degree = apply(
                Stage::HighDegree,
                &oriented_keys,
                vec![active.clone()],
                args,
                cancellation,
                deadline,
            )?
            .to_vec1::<u32>()
            .map_err(candle_error)?
            .first()
            .copied()
            .unwrap_or(u32::MAX);
            let use_sorted_candidates =
                use_sorted_candidate_path(maximum_degree, total_weight, oriented_count)?;
            if use_sorted_candidates && let Some(progress) = progress.as_deref_mut() {
                progress.sorted_candidate_levels = progress
                    .sorted_candidate_levels
                    .checked_add(1)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "Metal Louvain sorted-level count overflow",
                        )
                    })?;
            }
            let mut membership = apply(
                Stage::InitializeLevel,
                &active,
                Vec::new(),
                args,
                cancellation,
                deadline,
            )?;
            let mut community_weights = degree.clone();
            let mut level_local_passes = 0_usize;

            // No heuristic pass cap: every continuing pass applies at least one exact,
            // strictly beneficial move. The accepted moves touch disjoint source/target
            // communities, so their modularity deltas are additive and strictly positive.
            // Since the membership state space is finite, that is the termination invariant.
            loop {
                ensure_graph_execution(cancellation, deadline)?;
                let decision = if use_sorted_candidates {
                    let (candidate_keys, candidate_weights) = {
                        let candidate_keys = apply(
                            Stage::CandidateKeys,
                            &oriented_keys,
                            vec![membership.clone()],
                            args,
                            cancellation,
                            deadline,
                        )?;
                        stable_sort(
                            &candidate_keys,
                            &oriented_weights,
                            device,
                            cancellation,
                            deadline,
                        )?
                    };
                    let candidate_aggregate = apply(
                        Stage::CandidateAggregate,
                        &candidate_keys,
                        vec![candidate_weights],
                        args,
                        cancellation,
                        deadline,
                    )?;
                    let mut sorted_state = apply(
                        Stage::SortedInitialize,
                        &active,
                        vec![membership.clone()],
                        args,
                        cancellation,
                        deadline,
                    )?;
                    let maximum_degree = usize::try_from(maximum_degree).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "Metal Louvain maximum degree exceeds usize",
                        )
                    })?;
                    for chunk_base in
                        (0..maximum_degree).step_by(SORTED_DECISION_CHUNK_ROWS as usize)
                    {
                        ensure_graph_execution(cancellation, deadline)?;
                        let chunk_args = GraphArgs {
                            reduce_mode: u32::try_from(chunk_base).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "Metal Louvain sorted chunk offset exceeds u32",
                                )
                            })?,
                            ..args
                        };
                        sorted_state = apply(
                            Stage::SortedChunk,
                            &sorted_state,
                            vec![
                                active.clone(),
                                membership.clone(),
                                degree.clone(),
                                community_weights.clone(),
                                candidate_keys.clone(),
                                candidate_aggregate.clone(),
                            ],
                            chunk_args,
                            cancellation,
                            deadline,
                        )?;
                        if let Some(progress) = progress.as_deref_mut() {
                            progress.sorted_candidate_chunks = progress
                                .sorted_candidate_chunks
                                .checked_add(1)
                                .ok_or_else(|| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "Metal Louvain sorted-chunk count overflow",
                                    )
                                })?;
                            #[cfg(test)]
                            if progress.cancel_after_sorted_candidate_chunks
                                == Some(progress.sorted_candidate_chunks)
                            {
                                if let Some(triggered_at) = &progress.cancellation_triggered_at {
                                    *triggered_at.lock().map_err(|_| {
                                        Error::internal(
                                            "Metal Louvain cancellation test clock was poisoned",
                                        )
                                    })? = Some(std::time::Instant::now());
                                }
                                cancellation.cancel();
                            }
                        }
                    }
                    drop(candidate_keys);
                    drop(candidate_aggregate);
                    apply(
                        Stage::DecideSorted,
                        &sorted_state,
                        vec![
                            active.clone(),
                            membership.clone(),
                            degree.clone(),
                            community_weights.clone(),
                        ],
                        args,
                        cancellation,
                        deadline,
                    )?
                } else {
                    apply(
                        Stage::Decide,
                        &active,
                        vec![
                            membership.clone(),
                            degree.clone(),
                            community_weights.clone(),
                            oriented_keys.clone(),
                            oriented_weights.clone(),
                        ],
                        args,
                        cancellation,
                        deadline,
                    )?
                };
                let proposals = decision.narrow(0, 0, node_count).map_err(candle_error)?;
                let has_proposal = read_u32_scalar(&decision, node_count, cancellation, deadline)?;
                if has_proposal > 1 {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "Metal Louvain direct-candidate router exceeded its hard row bound",
                    ));
                }
                if has_proposal == 0 {
                    break;
                }

                // Proposals are edges in a community-conflict graph. A deterministic parallel
                // greedy matching accepts batches whose source/target community pairs are
                // disjoint. The exact individual modularity deltas are therefore additive.
                let mut active_proposals = apply(
                    Stage::MatchingInitialize,
                    &active,
                    vec![membership.clone(), proposals.clone()],
                    args,
                    cancellation,
                    deadline,
                )?;
                let mut accepted = visible_mask
                    .apply_op1_no_bwd(&LouvainZeroU32 {
                        rows: node_count,
                        cancellation: cancellation.clone(),
                        deadline,
                    })
                    .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
                let mut accepted_any = false;
                for _round in 0..node_count {
                    ensure_graph_execution(cancellation, deadline)?;
                    let minimum = apply(
                        Stage::MatchingMinimum,
                        &active_proposals,
                        vec![membership.clone(), proposals.clone()],
                        args,
                        cancellation,
                        deadline,
                    )?;
                    let winners = apply(
                        Stage::MatchingWinners,
                        &active_proposals,
                        vec![membership.clone(), proposals.clone(), minimum],
                        args,
                        cancellation,
                        deadline,
                    )?;
                    let winner_count =
                        read_u32_scalar(&winners, node_count * 2, cancellation, deadline)?;
                    if winner_count == 0 {
                        let remaining = sum_u32(&active_proposals, cancellation, deadline)?;
                        if remaining != 0 {
                            return Err(Error::new(
                                ErrorCode::CorruptStorage,
                                "Metal Louvain proposal matching stalled with active moves",
                            ));
                        }
                        break;
                    }
                    accepted_any = true;
                    let advanced = apply(
                        Stage::MatchingAdvance,
                        &active_proposals,
                        vec![accepted, membership.clone(), proposals.clone(), winners],
                        args,
                        cancellation,
                        deadline,
                    )?;
                    active_proposals = advanced.narrow(0, 0, node_count).map_err(candle_error)?;
                    accepted = advanced
                        .narrow(0, node_count, node_count)
                        .map_err(candle_error)?;
                }
                if !accepted_any {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "Metal Louvain proposal matching made no progress",
                    ));
                }
                drop(active_proposals);
                community_weights = apply(
                    Stage::UpdateCommunityWeights,
                    &community_weights,
                    vec![
                        membership.clone(),
                        proposals.clone(),
                        accepted.clone(),
                        degree.clone(),
                    ],
                    args,
                    cancellation,
                    deadline,
                )?;
                membership = apply(
                    Stage::ApplyMatching,
                    &membership,
                    vec![proposals, accepted],
                    args,
                    cancellation,
                    deadline,
                )?;
                level_local_passes = level_local_passes.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Metal Louvain local-pass count overflow",
                    )
                })?;
                if let Some(progress) = progress.as_deref_mut() {
                    progress.total_local_passes =
                        progress.total_local_passes.checked_add(1).ok_or_else(|| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "Metal Louvain progress count overflow",
                            )
                        })?;
                    progress.max_level_local_passes =
                        progress.max_level_local_passes.max(level_local_passes);
                }
            }

            drop(oriented_keys);
            drop(oriented_weights);
            drop(degree);
            drop(community_weights);

            original_map = apply(
                Stage::MapOriginal,
                &original_map,
                vec![membership.clone()],
                args,
                cancellation,
                deadline,
            )?;
            let rebuilt = apply(
                Stage::RebuildActive,
                &active,
                vec![membership.clone()],
                args,
                cancellation,
                deadline,
            )?;
            let next_active = rebuilt.narrow(0, 0, node_count).map_err(candle_error)?;
            let next_count =
                read_u32_scalar(&rebuilt, node_count, cancellation, deadline)? as usize;
            if next_count == 0 || next_count == active_count || next_count <= 1 {
                break;
            }
            active = next_active;

            let (mapped_keys, mapped_weights) = {
                let mapped = apply(
                    Stage::CoarsenPairs,
                    &pair_keys,
                    vec![pair_weights.clone(), membership],
                    args,
                    cancellation,
                    deadline,
                )?;
                drop(pair_keys);
                drop(pair_weights);
                let (mapped_keys, mapped_weights) = split_i64(&mapped, edge_count)?;
                stable_sort(
                    &mapped_keys,
                    &mapped_weights,
                    device,
                    cancellation,
                    deadline,
                )?
            };
            let coarse_args = GraphArgs {
                reduce_mode: 1,
                ..args
            };
            let coarse = apply(
                Stage::ReducePairs,
                &mapped_keys,
                vec![mapped_weights],
                coarse_args,
                cancellation,
                deadline,
            )?;
            drop(mapped_keys);
            (pair_keys, pair_weights) = split_i64(&coarse, edge_count)?;
            active_count = next_count;
        }
    }

    ensure_graph_execution(cancellation, deadline)?;
    drop(pair_keys);
    drop(pair_weights);
    drop(active);
    let canonical_args = GraphArgs::new(node_count, 0, node_rows.len(), 0, layers, 0)?;
    let first_workspace = apply(
        Stage::CanonicalFirst,
        &original_map,
        vec![visible_rows.clone()],
        canonical_args,
        cancellation,
        deadline,
    )?;
    let canonical_status = read_u32_scalar(&first_workspace, node_count, cancellation, deadline)?;
    if canonical_status != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal Louvain canonical publication failed with status {canonical_status}"),
        ));
    }
    let first_positions = first_workspace
        .narrow(0, 0, node_count)
        .map_err(candle_error)?;
    let canonical_keys = apply(
        Stage::CanonicalKeys,
        &first_positions,
        Vec::new(),
        canonical_args,
        cancellation,
        deadline,
    )?;
    drop(first_positions);
    drop(first_workspace);
    let (sorted_canonical_keys, redundant_keys) = stable_sort(
        &canonical_keys,
        &canonical_keys,
        device,
        cancellation,
        deadline,
    )?;
    drop(canonical_keys);
    drop(redundant_keys);
    let canonical_by_label = apply(
        Stage::CanonicalPublish,
        &sorted_canonical_keys,
        Vec::new(),
        canonical_args,
        cancellation,
        deadline,
    )?;
    drop(sorted_canonical_keys);
    let selected_labels = original_map
        .apply_op2_no_bwd(
            visible_rows,
            &LouvainGatherU32 {
                cancellation: cancellation.clone(),
                deadline,
            },
        )
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    drop(original_map);
    let canonical_labels = canonical_by_label
        .apply_op2_no_bwd(
            &selected_labels,
            &LouvainGatherU32 {
                cancellation: cancellation.clone(),
                deadline,
            },
        )
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    drop(canonical_by_label);
    drop(selected_labels);
    let community = canonical_labels
        .to_vec1::<u32>()
        .map_err(|error| louvain_candle_error(error, cancellation, deadline))?;
    Ok(ResidentGraphProcedureResult::Louvain {
        node_rows,
        community,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_estimate_matches_allocator_classes_and_phase_liveness() -> Result<()> {
        assert_eq!(scratch_bytes(0, 0)?, 0);
        assert_eq!(pooled_bytes(1, "test")?, 1);
        assert_eq!(pooled_bytes(3, "test")?, 4);
        assert_eq!(pooled_bytes(4_095, "test")?, 4_096);
        assert_eq!(pooled_bytes(4_097, "test")?, 8_192);
        assert_eq!(
            stable_sort_scratch(3)?,
            StableSortScratchBreakdown {
                first_radix_pass: 7_232,
                later_radix_pass: 7_232,
                gather: 96,
            }
        );

        // Every number below is the hand-expanded sum of distinct simultaneously live buffers for
        // V=3,E=5 after applying Candle's allocator class to each allocation independently.
        let tiny = scratch_breakdown(3, 5)?;
        assert_eq!(
            tiny,
            LouvainScratchBreakdown {
                selection_and_node_readback: 180,
                base_preprocess: 7_456,
                oriented_sort_and_level_setup: 7_872,
                direct_local_move: 652,
                sorted_local_move: 8_080,
                coarsening: 7_488,
                result_readback: 7_312,
            }
        );
        assert_eq!(scratch_bytes(3, 5)?, 8_080);
        assert_eq!(unique_undirected_csr_scratch_bytes(3, 5)?, 7_680);
        assert_eq!(unique_undirected_csr_retained_bytes(3, 5)?, 80);
        let edgeless = scratch_breakdown(3, 0)?;
        assert_eq!(edgeless.selection_and_node_readback, 180);
        assert_eq!(edgeless.result_readback, 44);
        assert_eq!(scratch_bytes(3, 0)?, 180);

        let high_cardinality = scratch_breakdown(1_024, 8_192)?;
        assert!(high_cardinality.sorted_local_move > high_cardinality.direct_local_move);
        assert_eq!(
            scratch_bytes(1_024, 8_192)?,
            high_cardinality.required_bytes()
        );
        assert!(pooled_bytes(usize::MAX, "test").is_err());
        assert!(scratch_bytes(usize::MAX, usize::MAX).is_err());
        assert!(unique_undirected_csr_scratch_bytes(usize::MAX, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn adaptive_candidate_router_keeps_tiny_hubs_direct_and_dense_levels_sorted() -> Result<()> {
        assert!(!use_sorted_candidate_path(160, 320, 320)?);
        assert!(use_sorted_candidate_path(130, 49_926, 49_926)?);
        assert!(!use_sorted_candidate_path(50, 393_472, 393_472)?);
        assert!(use_sorted_candidate_path(257, 514, 514)?);
        Ok(())
    }

    #[test]
    fn tiled_dispatch_accepts_wide_derived_domains_up_to_actual_abi_limits() -> Result<()> {
        let maximum_u32 = u32::MAX as usize;
        assert!(GraphArgs::new(maximum_u32, 0, 0, 0, LayerMask::OBSERVED, 0).is_ok());
        assert!(GraphArgs::new(maximum_u32 + 1, 0, 0, 0, LayerMask::OBSERVED, 0).is_err());
        let widest = GraphArgs::new(1, maximum_u32, maximum_u32, 0, LayerMask::OBSERVED, 0)?;
        assert_eq!(widest.oriented_capacity, u64::from(u32::MAX) * 2);
        assert!(GraphArgs::new(1, maximum_u32 + 1, 0, 0, LayerMask::OBSERVED, 0).is_err());
        let wide_offset = widest
            .with_work(maximum_u32 + 1, WORK_TILE_ROWS)
            .map_err(candle_error)?;
        assert_eq!(wide_offset.work_offset, u64::from(u32::MAX) + 1);
        assert_eq!(wide_offset.work_count, WORK_TILE_ROWS as u64);
        Ok(())
    }

    #[test]
    fn unique_csr_admits_only_its_actual_u32_offset_domain() -> Result<()> {
        assert_eq!(csr_adjacency_count(u64::from(u32::MAX))?, u32::MAX as usize);
        let error = csr_adjacency_count(u64::from(u32::MAX) + 1)
            .expect_err("U32 CSR must reject one reciprocal row beyond its ABI limit");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        Ok(())
    }

    #[test]
    fn metal_source_hard_bounds_quadratic_rows_and_parallelizes_coarsening_runs() {
        let source = include_str!("../../../../kernels/metal/graph_louvain.metal");
        assert!(source.contains("IG_LV_DIRECT_NEIGHBOR_HARD_LIMIT = 256ul"));
        assert!(source.contains("end - begin > IG_LV_DIRECT_NEIGHBOR_HARD_LIMIT"));
        assert!(source.contains("kernel void ig_louvain_reduce_pairs_clear"));
        assert!(source.contains("ig_lv_lower_bound_key("));
        assert!(source.contains("ig_lv_atomic_add_u64("));
        assert!(source.contains("ulong word = ulong(position) * 2ul"));
        assert!(source.contains("degree_words + ulong(node) * 2ul"));
        assert!(source.contains("kernel void ig_louvain_radix_histogram"));
        assert!(source.contains("kernel void ig_louvain_radix_offsets"));
        assert!(source.contains("kernel void ig_louvain_radix_scatter"));
        assert!(source.contains("ulong position = args.work_offset + ulong(local)"));
        assert!(!source.contains(
            "Runs are disjoint, so total work remains linear even though one lane owns a run"
        ));
    }

    #[cfg(target_os = "macos")]
    fn metal_star_fixture(leaf_count: usize) -> Result<(CandleResident, Tensor, Tensor)> {
        let _guard = crate::metal_test_guard();
        use crate::{
            Bookmark, EdgeId, Layer, NodeId, ProjectId,
            execution::ResidentProjectImage,
            graph::{EdgeInput, GraphStore, IndexCatalog, NodeInput, TemporalStore},
        };

        let mut graph = GraphStore::default();
        let vertex = graph.catalog_mut().intern_label("StarVertex")?;
        let connected = graph.catalog_mut().intern_relationship_type("STAR_EDGE")?;
        for node in 0..=leaf_count {
            let id = u64::try_from(node + 1)
                .map_err(|_| Error::internal("Metal star node ID exceeds u64"))?;
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![vertex],
                properties: Vec::new(),
            })?;
        }
        for leaf in 1..=leaf_count {
            let id = u64::try_from(leaf_count + leaf + 1)
                .map_err(|_| Error::internal("Metal star edge ID exceeds u64"))?;
            let target = u64::try_from(leaf + 1)
                .map_err(|_| Error::internal("Metal star endpoint exceeds u64"))?;
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(1),
                target: NodeId(target),
                relationship_type: connected,
                layer: Layer::Observed,
                revision: id,
                properties: Vec::new(),
            })?;
        }
        let project = ProjectId(uuid::Uuid::nil());
        let image = ResidentProjectImage::build(
            project,
            Bookmark {
                term: 0,
                index: graph.revision(),
            },
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let Some(device) = crate::metal_test_device() else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Metal test device is unavailable",
            ));
        };
        let resident = CandleResident::upload(image, &device)?;
        let visible_mask =
            Tensor::ones(graph.node_count(), DType::U8, &device).map_err(candle_error)?;
        let visible_rows = Tensor::from_vec(
            (0..graph.node_count())
                .map(|row| {
                    u32::try_from(row).map_err(|_| Error::internal("Metal star row exceeds u32"))
                })
                .collect::<Result<Vec<_>>>()?,
            graph.node_count(),
            &device,
        )
        .map_err(candle_error)?;
        Ok((resident, visible_mask, visible_rows))
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_metal_unique_csr_collapses_parallel_reverse_and_filtered_edges() -> Result<()> {
        let _guard = crate::metal_test_guard();
        use crate::{
            Bookmark, EdgeId, Layer, NodeId, ProjectId,
            execution::ResidentProjectImage,
            graph::{EdgeInput, GraphStore, IndexCatalog, NodeInput, TemporalStore},
        };

        let mut graph = GraphStore::default();
        let vertex = graph.catalog_mut().intern_label("CsrVertex")?;
        let connected = graph.catalog_mut().intern_relationship_type("CSR_EDGE")?;
        for id in 1_u64..=5 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![vertex],
                properties: Vec::new(),
            })?;
        }
        for (index, (source, target, layer)) in [
            (1_u64, 2_u64, Layer::Observed),
            (2, 1, Layer::Observed),
            (1, 2, Layer::Observed),
            (2, 3, Layer::Observed),
            (3, 3, Layer::Observed),
            (3, 4, Layer::Knowledge),
            (4, 5, Layer::Observed),
        ]
        .into_iter()
        .enumerate()
        {
            graph.insert_edge(EdgeInput {
                id: EdgeId(100 + index as u64),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: connected,
                layer,
                revision: 100 + index as u64,
                properties: Vec::new(),
            })?;
        }
        let image = ResidentProjectImage::build(
            ProjectId(uuid::Uuid::nil()),
            Bookmark {
                term: 0,
                index: graph.revision(),
            },
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let Some(device) = crate::metal_test_device() else {
            return Ok(());
        };
        let resident = CandleResident::upload(image, &device)?;
        let visible_mask =
            Tensor::from_vec(vec![1_u8, 1, 1, 1, 0], 5, &device).map_err(candle_error)?;
        for _ in 0..2 {
            let csr = build_unique_undirected_csr(
                &resident,
                &visible_mask,
                LayerMask::OBSERVED,
                &CancellationToken::new(),
                None,
            )?;
            assert_eq!(csr.adjacency_count, 4);
            assert_eq!(
                csr.offsets.to_vec1::<u32>().map_err(candle_error)?,
                vec![0, 1, 3, 4, 4, 4]
            );
            assert_eq!(
                csr.neighbors.to_vec1::<u32>().map_err(candle_error)?,
                vec![1, 0, 2, 1]
            );
        }
        // The canonical edge table is non-empty, but filtering to an isolated visible node
        // produces zero reciprocal adjacency. This must not dispatch a zero-row CSR-neighbor
        // kernel on Metal.
        let isolate_mask =
            Tensor::from_vec(vec![0_u8, 0, 0, 1, 0], 5, &device).map_err(candle_error)?;
        let empty = build_unique_undirected_csr(
            &resident,
            &isolate_mask,
            LayerMask::OBSERVED,
            &CancellationToken::new(),
            None,
        )?;
        assert_eq!(empty.adjacency_count, 0);
        assert_eq!(
            empty.offsets.to_vec1::<u32>().map_err(candle_error)?,
            vec![0; 6]
        );
        assert!(
            empty
                .neighbors
                .to_vec1::<u32>()
                .map_err(candle_error)?
                .is_empty()
        );
        Ok(())
    }

    /// A Workspace relationship is ordinary data, not corruption.
    ///
    /// `Layer::Workspace = 2` was added after these kernels were written, and both the Louvain
    /// validation guard and its layer-visibility helper still bounded a layer index at 1. Every
    /// projection over a graph holding Workspace relationships therefore failed with
    /// `CORRUPT_STORAGE: Metal unique-undirected graph validation failed with status 2`, which is
    /// what `CALL graph.louvain()` reported. Nothing in this suite covered a third layer, which is
    /// why it shipped.
    #[cfg(target_os = "macos")]
    #[test]
    fn real_metal_unique_csr_accepts_a_workspace_layer_relationship() -> Result<()> {
        let _guard = crate::metal_test_guard();
        use crate::{
            Bookmark, EdgeId, Layer, NodeId, ProjectId,
            execution::ResidentProjectImage,
            graph::{EdgeInput, GraphStore, IndexCatalog, NodeInput, TemporalStore},
        };

        let mut graph = GraphStore::default();
        let vertex = graph.catalog_mut().intern_label("WorkspaceVertex")?;
        let connected = graph
            .catalog_mut()
            .intern_relationship_type("WORKSPACE_EDGE")?;
        for id in 1_u64..=3 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![vertex],
                properties: Vec::new(),
            })?;
        }
        for (index, (source, target, layer)) in
            [(1_u64, 2_u64, Layer::Observed), (2, 3, Layer::Workspace)]
                .into_iter()
                .enumerate()
        {
            graph.insert_edge(EdgeInput {
                id: EdgeId(200 + index as u64),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: connected,
                layer,
                revision: 200 + index as u64,
                properties: Vec::new(),
            })?;
        }
        let image = ResidentProjectImage::build(
            ProjectId(uuid::Uuid::nil()),
            Bookmark {
                term: 0,
                index: graph.revision(),
            },
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let Some(device) = crate::metal_test_device() else {
            return Ok(());
        };
        let resident = CandleResident::upload(image, &device)?;
        let visible_mask = Tensor::from_vec(vec![1_u8, 1, 1], 3, &device).map_err(candle_error)?;

        // Validation must accept the Workspace row rather than call it corrupt. Selecting only
        // OBSERVED still excludes it, so this asserts acceptance, not a widened projection.
        let observed = build_unique_undirected_csr(
            &resident,
            &visible_mask,
            LayerMask::OBSERVED,
            &CancellationToken::new(),
            None,
        )?;
        assert_eq!(observed.adjacency_count, 2);
        assert_eq!(
            observed.neighbors.to_vec1::<u32>().map_err(candle_error)?,
            vec![1, 0]
        );

        // Selecting the Workspace layer must actually reach the relationship the helper used to
        // drop, so a passing validation cannot be mistaken for a projection that silently omits it.
        let workspace = build_unique_undirected_csr(
            &resident,
            &visible_mask,
            LayerMask::WORKSPACE,
            &CancellationToken::new(),
            None,
        )?;
        assert_eq!(workspace.adjacency_count, 2);
        assert_eq!(
            workspace.neighbors.to_vec1::<u32>().map_err(candle_error)?,
            vec![2, 1]
        );
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_metal_star_executes_more_than_one_hundred_local_passes() -> Result<()> {
        if crate::metal_test_device().is_none() {
            return Ok(());
        }
        let leaf_count = 160_usize;
        let (resident, visible_mask, visible_rows) = metal_star_fixture(leaf_count)?;
        let mut progress = LouvainProgress::default();
        let result = execute_with_progress(
            &resident,
            &visible_mask,
            &visible_rows,
            LayerMask::OBSERVED,
            &CancellationToken::new(),
            None,
            Some(&mut progress),
        )?;
        let ResidentGraphProcedureResult::Louvain { community, .. } = result else {
            return Err(Error::internal(
                "Metal star returned the wrong result shape",
            ));
        };
        assert!(community.iter().all(|label| *label == community[0]));
        assert_eq!(progress.max_level_local_passes, leaf_count);
        assert_eq!(progress.total_local_passes, leaf_count);
        assert_eq!(progress.sorted_candidate_levels, 0);
        assert_eq!(progress.sorted_candidate_chunks, 0);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn real_metal_single_hub_cancels_between_bounded_sorted_chunks() -> Result<()> {
        if crate::metal_test_device().is_none() {
            return Ok(());
        }
        let leaf_count = 2_305_usize;
        let (resident, visible_mask, visible_rows) = metal_star_fixture(leaf_count)?;
        let cancellation = CancellationToken::new();
        let triggered_at = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut progress = LouvainProgress {
            cancel_after_sorted_candidate_chunks: Some(1),
            cancellation_triggered_at: Some(triggered_at.clone()),
            ..LouvainProgress::default()
        };
        let error = execute_with_progress(
            &resident,
            &visible_mask,
            &visible_rows,
            LayerMask::OBSERVED,
            &cancellation,
            None,
            Some(&mut progress),
        )
        .expect_err("bounded sorted decision must observe active cancellation");
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert_eq!(progress.sorted_candidate_levels, 1);
        assert_eq!(progress.sorted_candidate_chunks, 1);
        let triggered_at = triggered_at
            .lock()
            .map_err(|_| Error::internal("Metal Louvain cancellation clock was poisoned"))?
            .ok_or_else(|| Error::internal("Metal Louvain cancellation was never triggered"))?;
        assert!(
            triggered_at.elapsed() < std::time::Duration::from_secs(2),
            "Metal Louvain cancellation exceeded the two-second hard test bound"
        );
        Ok(())
    }
}
