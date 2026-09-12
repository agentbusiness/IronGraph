//! Native Metal connected-components and undirected graph metrics.
//!
//! Directed edge-linear phases scan the canonical source/target columns exactly once, without
//! joining CSR positions back to relationship rows. Undirected algorithms derive one bounded, ephemeral O(V+E)
//! unique reciprocal CSR on-device; it is never a second durable graph. Host code controls
//! dispatch chunks, cancellation, result budgets, and compact typed publication; it never
//! reconstructs adjacency or executes an algorithmic edge scan.

use std::{
    sync::{Arc, OnceLock},
    time::Instant,
};

use candle_core::{
    CpuStorage, CustomOp1, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor,
    backend::BackendStorage,
};
use objc2_metal::MTLDevice;
use tokio_util::sync::CancellationToken;

use crate::{
    Error, ErrorCode, Result,
    execution::{ResidentGraphProcedureResult, ensure_graph_execution},
    graph::LayerMask,
};

use super::{CandleResident, candle_error};

const THREADS: usize = 256;
const METRICS_EDGE_CHUNK: usize = 16 * 1024;
const METRICS_TRIANGLE_SCAN_QUANTUM: usize = 128;
const METRICS_CURSOR_WORDS: usize = 2 * METRICS_EDGE_CHUNK;
const KCORE_ROW_SCAN_QUANTUM: usize = 128;
const KCORE_TICKET_CHUNK: usize = 16 * 1024;
const UNREACHED: u32 = u32::MAX;
fn scratch_overflow(name: &str) -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        format!("Metal {name} scratch accounting overflow"),
    )
}

/// Candle 0.11's Metal buffer pool allocates the next power-of-two byte class for every non-empty
/// request. Admission uses those actual pool classes, not logical tensor bytes or an alignment
/// guess. Zero-length tensors do not allocate a device buffer.
pub fn pooled_allocation_bytes(logical_bytes: usize, name: &str) -> Result<usize> {
    if logical_bytes == 0 {
        return Ok(0);
    }
    logical_bytes
        .checked_next_power_of_two()
        .ok_or_else(|| scratch_overflow(name))
}

fn checked_sum(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| scratch_overflow(name))
    })
}

fn typed_allocation(node_count: usize, width: usize, name: &str) -> Result<usize> {
    pooled_allocation_bytes(
        node_count
            .checked_mul(width)
            .ok_or_else(|| scratch_overflow(name))?,
        name,
    )
}

/// Exact worst-case live pool classes for `selected_positions(mask, V, ...)`. The scan-loop peak
/// retains mask + selected bits + prefix input, then simultaneously owns old scan, shifted scan,
/// add output, minimum output, and the zero prefix. The final scatter peak is counted separately.
/// `selected_count == V` is the exact admission worst case and also the common all-visible path.
pub fn shared_selection_scratch_bytes(node_count: usize, name: &str) -> Result<usize> {
    if node_count == 0 {
        return Ok(0);
    }
    let mask = typed_allocation(node_count, 1, name)?;
    let u32s = typed_allocation(node_count, 4, name)?;
    let i64s = typed_allocation(node_count, 8, name)?;
    let initial_scan = checked_sum(&[mask, u32s, i64s, i64s], name)?;
    let loop_peak = if node_count == 1 {
        initial_scan
    } else {
        let last_offset = 1_usize << (usize::BITS - 1 - (node_count - 1).leading_zeros());
        let zero_prefix = typed_allocation(last_offset, 8, name)?;
        checked_sum(
            &[mask, u32s, i64s, i64s, i64s, i64s, i64s, zero_prefix],
            name,
        )?
    };
    let destination_peak = checked_sum(&[mask, u32s, i64s, i64s, i64s, u32s, u32s, u32s], name)?;
    let scatter_peak = checked_sum(
        &[
            mask, u32s, i64s, i64s, u32s, u32s, u32s, u32s, u32s, u32s, u32s,
        ],
        name,
    )?;
    Ok(initial_scan
        .max(loop_peak)
        .max(destination_peak)
        .max(scatter_peak))
}

fn retained_selection_bytes(node_count: usize, name: &str) -> Result<usize> {
    checked_sum(
        &[
            typed_allocation(node_count, 1, name)?,
            typed_allocation(node_count, 4, name)?,
        ],
        name,
    )
}

/// Peak once selection is complete: visible mask + selected-row tensor + resident procedure
/// workspace + compact result + its independent shared readback staging allocation.
pub fn compact_procedure_scratch_bytes(
    node_count: usize,
    workspace_logical_bytes: usize,
    output_width: usize,
    name: &str,
) -> Result<usize> {
    let retained = retained_selection_bytes(node_count, name)?;
    let workspace = pooled_allocation_bytes(workspace_logical_bytes, name)?;
    let output = typed_allocation(node_count, output_width, name)?;
    let procedure = checked_sum(&[retained, workspace, output, output], name)?;
    Ok(shared_selection_scratch_bytes(node_count, name)?.max(procedure))
}

/// Exact live peak for device-side component canonicalization. The scan path retains the visible
/// mask, selected rows, algorithm workspace, selected component minimums, the converted flag
/// column, and the five simultaneously-live binary64-sized i64 scan buffers. Publication then
/// retains three i64 compact buffers plus the compact u32 result and its readback staging class.
fn component_procedure_scratch_bytes(
    node_count: usize,
    workspace_logical_bytes: usize,
    name: &str,
) -> Result<usize> {
    let retained = retained_selection_bytes(node_count, name)?;
    let workspace = pooled_allocation_bytes(workspace_logical_bytes, name)?;
    let u32s = typed_allocation(node_count, 4, name)?;
    let i64s = typed_allocation(node_count, 8, name)?;
    let gather_peak = checked_sum(&[retained, workspace, u32s, u32s], name)?;
    let scan_peak = checked_sum(
        &[retained, workspace, u32s, i64s, i64s, i64s, i64s, i64s],
        name,
    )?;
    let publish_peak = checked_sum(&[retained, workspace, i64s, i64s, i64s, u32s, u32s], name)?;
    Ok(shared_selection_scratch_bytes(node_count, name)?
        .max(gather_peak)
        .max(scan_peak)
        .max(publish_peak))
}

/// `PageRank` keeps its software-binary64 publication bank inside the main workspace, then allocates
/// only the compact visible copy and that copy's independent shared readback staging buffer.
pub fn pagerank_procedure_scratch_bytes(
    node_count: usize,
    workspace_logical_bytes: usize,
) -> Result<usize> {
    let name = "PageRank";
    let retained = retained_selection_bytes(node_count, name)?;
    let workspace = pooled_allocation_bytes(workspace_logical_bytes, name)?;
    let full_rank = typed_allocation(node_count, 8, name)?;
    let procedure = checked_sum(&[retained, workspace, full_rank, full_rank], name)?;
    Ok(shared_selection_scratch_bytes(node_count, name)?.max(procedure))
}

/// Peak includes generic selection plus the 3V+2 algorithm/canonical workspace and GPU scan.
pub fn wcc_scratch_bytes(node_count: usize) -> Result<usize> {
    let words = node_count
        .checked_mul(3)
        .and_then(|words| words.checked_add(2))
        .ok_or_else(|| scratch_overflow("WCC"))?;
    component_procedure_scratch_bytes(
        node_count,
        words
            .checked_mul(4)
            .ok_or_else(|| scratch_overflow("WCC"))?,
        "WCC",
    )
}

/// Peak includes the generic selection scan plus the direct 2V directed-degree packet.
pub fn degree_scratch_bytes(node_count: usize) -> Result<usize> {
    let words = node_count
        .checked_mul(2)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| scratch_overflow("degree"))?;
    compact_procedure_scratch_bytes(
        node_count,
        words
            .checked_mul(4)
            .ok_or_else(|| scratch_overflow("degree"))?,
        4,
        "degree",
    )
}

/// Peak includes generic selection plus the 5V+4 color/trim/canonical workspace and GPU scan.
pub fn scc_scratch_bytes(node_count: usize) -> Result<usize> {
    let words = node_count
        .checked_mul(5)
        .and_then(|value| value.checked_add(4))
        .ok_or_else(|| scratch_overflow("SCC"))?;
    component_procedure_scratch_bytes(
        node_count,
        words
            .checked_mul(4)
            .ok_or_else(|| scratch_overflow("SCC"))?,
        "SCC",
    )
}

fn unique_projection_procedure_scratch_bytes(
    node_count: usize,
    edge_count: usize,
    workspace_logical_bytes: usize,
    output_width: usize,
    name: &str,
) -> Result<usize> {
    let selection_peak = shared_selection_scratch_bytes(node_count, name)?;
    let selection_retained = retained_selection_bytes(node_count, name)?;
    let builder =
        super::graph_louvain::unique_undirected_csr_scratch_bytes(node_count, edge_count)?;
    let builder_peak = checked_sum(&[selection_retained, builder], name)?;
    let unique_retained =
        super::graph_louvain::unique_undirected_csr_retained_bytes(node_count, edge_count)?;
    let workspace = pooled_allocation_bytes(workspace_logical_bytes, name)?;
    let output = typed_allocation(node_count, output_width, name)?;
    let procedure_peak = checked_sum(
        &[
            selection_retained,
            unique_retained,
            workspace,
            output,
            output,
        ],
        name,
    )?;
    Ok(selection_peak.max(builder_peak).max(procedure_peak))
}

/// Exact peak for shared unique-CSR construction plus the retained 3V packet, hierarchical u64
/// banks, fixed cursor bank, and compact selected binary64 publication.
pub fn metrics_scratch_bytes(node_count: usize, edge_count: usize) -> Result<usize> {
    let workspace = metrics_workspace_words(node_count)
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| scratch_overflow("undirected metrics"))?;
    unique_projection_procedure_scratch_bytes(
        node_count,
        edge_count,
        workspace,
        8,
        "undirected metrics",
    )
}

/// Peak includes generic selection plus the 5V+6 unique-CSR queue/degree/cursor workspace and
/// compact result. Each unique adjacency row is scanned once when removed.
pub fn kcore_scratch_bytes(node_count: usize, edge_count: usize) -> Result<usize> {
    let words = node_count
        .checked_mul(5)
        .and_then(|value| value.checked_add(6))
        .ok_or_else(|| scratch_overflow("k-core"))?;
    unique_projection_procedure_scratch_bytes(
        node_count,
        edge_count,
        words
            .checked_mul(4)
            .ok_or_else(|| scratch_overflow("k-core"))?,
        4,
        "k-core",
    )
}

#[derive(Clone)]
struct GraphImage {
    edge_targets: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    edge_sources: Tensor,
    undirected_offsets: Tensor,
    directed_incoming_offsets: Tensor,
    undirected_neighbors: Tensor,
    node_count: usize,
    edge_count: usize,
    adjacency_count: usize,
    undirected_adjacency_count: usize,
    layer_mask: u32,
}

impl GraphImage {
    fn from_resident(
        resident: &CandleResident,
        visible_nodes: &Tensor,
        layers: LayerMask,
    ) -> Result<Option<Self>> {
        let Some(edge_targets) = resident.edge_targets.as_ref() else {
            return Ok(None);
        };
        let edge_active = resident.edge_active.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident adjacency has no edge-active tensor",
            )
        })?;
        let edge_layers = resident.edge_layers.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident adjacency has no edge-layer tensor",
            )
        })?;
        let edge_sources = resident.edge_sources.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident adjacency has no canonical edge-source tensor",
            )
        })?;
        if edge_sources.elem_count() != resident.edge_count
            || edge_targets.elem_count() != resident.edge_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident canonical edge endpoint cardinalities differ",
            ));
        }
        Ok(Some(Self {
            edge_targets: edge_targets.clone(),
            visible_nodes: visible_nodes.clone(),
            edge_active: edge_active.clone(),
            edge_layers: edge_layers.clone(),
            edge_sources: edge_sources.clone(),
            undirected_offsets: resident.outgoing_offsets.clone(),
            directed_incoming_offsets: resident.incoming_offsets.clone(),
            undirected_neighbors: edge_targets.clone(),
            node_count: resident.node_count,
            edge_count: resident.edge_count,
            adjacency_count: resident.edge_count,
            undirected_adjacency_count: resident.edge_count,
            layer_mask: u32::from(layers.bits()),
        }))
    }

    fn with_unique_undirected(mut self, unique: super::graph_louvain::UniqueUndirectedCsr) -> Self {
        self.undirected_offsets = unique.offsets;
        self.undirected_neighbors = unique.neighbors;
        self.undirected_adjacency_count = unique.adjacency_count;
        self
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Args {
    node_count: u32,
    edge_count: u32,
    adjacency_count: u32,
    layer_mask: u32,
    scalar: u32,
    partial_count: u32,
    reserved_1: u32,
    reserved_2: u32,
}

impl Args {
    fn new(graph: &GraphImage, scalar: u32, span: u32) -> candle_core::Result<Self> {
        Ok(Self {
            node_count: u32::try_from(graph.node_count).map_err(|_| {
                candle_core::Error::Msg("Metal graph node count exceeds u32".to_owned())
            })?,
            edge_count: u32::try_from(graph.edge_count).map_err(|_| {
                candle_core::Error::Msg("Metal graph edge count exceeds u32".to_owned())
            })?,
            adjacency_count: u32::try_from(graph.adjacency_count).map_err(|_| {
                candle_core::Error::Msg("Metal graph adjacency count exceeds u32".to_owned())
            })?,
            layer_mask: graph.layer_mask,
            scalar,
            partial_count: u32::try_from(graph.node_count.div_ceil(THREADS)).map_err(|_| {
                candle_core::Error::Msg("Metal graph reduction tile count exceeds u32".to_owned())
            })?,
            reserved_1: span,
            reserved_2: 0,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct MetricsLayout {
    words: usize,
    partial_a: usize,
    partial_b: usize,
    result: usize,
    status: usize,
    cursors: usize,
}

impl MetricsLayout {
    fn new(node_count: usize) -> Option<Self> {
        let partial_words = node_count.div_ceil(THREADS).checked_mul(2)?;
        let partial_a = node_count.checked_mul(3)?;
        let partial_b = partial_a.checked_add(partial_words)?;
        let result = partial_b.checked_add(partial_words)?;
        let status = result.checked_add(2)?;
        let cursors = status.checked_add(2)?;
        Some(Self {
            words: cursors.checked_add(METRICS_CURSOR_WORDS)?,
            partial_a,
            partial_b,
            result,
            status,
            cursors,
        })
    }
}

fn metrics_workspace_words(node_count: usize) -> Option<usize> {
    MetricsLayout::new(node_count).map(|layout| layout.words)
}

#[derive(Clone, Copy, Debug)]
struct ComponentCanonicalLayout {
    words: usize,
    root_minimum: usize,
    minimum_flags: usize,
    status: usize,
}

impl ComponentCanonicalLayout {
    fn wcc(node_count: usize) -> Option<Self> {
        Some(Self {
            words: node_count.checked_mul(3)?.checked_add(2)?,
            root_minimum: node_count.checked_add(2)?,
            minimum_flags: node_count.checked_mul(2)?.checked_add(2)?,
            status: node_count.checked_add(1)?,
        })
    }

    fn scc(node_count: usize) -> Option<Self> {
        Some(Self {
            words: node_count.checked_mul(5)?.checked_add(4)?,
            root_minimum: node_count,
            minimum_flags: node_count.checked_mul(2)?,
            status: node_count.checked_mul(5)?.checked_add(1)?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum CommandKind {
    WccInitialize {
        visible_count: u32,
    },
    WccPrepare,
    WccEdgeChunk {
        start: u32,
        span: u32,
    },
    WccCompress,
    SccInitialize {
        visible_count: u32,
    },
    SccDegreeChunk {
        start: u32,
        span: u32,
    },
    SccTrimSelect,
    SccTrimEdgeChunk {
        start: u32,
        span: u32,
    },
    SccCycleProbeInitialize,
    SccCycleProbeEdgeChunk {
        start: u32,
        span: u32,
    },
    SccCycleProbeValidate,
    SccCycleFinish {
        rounds: u32,
    },
    SccColorInitialize,
    SccColorPrepare,
    SccColorEdgeChunk {
        start: u32,
        span: u32,
    },
    SccBackwardSeed,
    SccBackwardPrepare,
    SccBackwardEdgeChunk {
        start: u32,
        span: u32,
    },
    SccColorAssign,
    ComponentCanonicalInitialize {
        layout: ComponentCanonicalLayout,
        start: u32,
        span: u32,
    },
    ComponentCanonicalMark {
        layout: ComponentCanonicalLayout,
        start: u32,
        span: u32,
    },
    ComponentCanonicalFlags {
        layout: ComponentCanonicalLayout,
        start: u32,
        span: u32,
    },
    DegreeInitialize,
    DegreeCsr,
    DegreeChunk {
        start: u32,
        span: u32,
    },
    MetricsInitialize,
    MetricsDegreeChunk {
        start: u32,
        span: u32,
    },
    MetricsTriangleChunk {
        start: u32,
        span: u32,
        reset: bool,
    },
    MetricsTrianglePartials,
    MetricsTriangleReduce {
        input_count: u32,
        input_second: bool,
    },
    MetricsTriangleResult {
        input_second: bool,
    },
    MetricsClusteringChunk {
        start: u32,
        span: u32,
    },
    KCoreInitialize {
        visible_count: u32,
    },
    KCoreSeed {
        core: u32,
    },
    KCoreDrainPrepare,
    KCoreDrainChunk {
        core: u32,
        start: u32,
        span: u32,
    },
}

impl CommandKind {
    fn workspace_words(self, node_count: usize) -> candle_core::Result<usize> {
        let checked = |factor: usize, tail: usize, name: &str| {
            node_count
                .checked_mul(factor)
                .and_then(|words| words.checked_add(tail))
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("Metal {name} workspace size overflow"))
                })
        };
        match self {
            Self::WccInitialize { .. }
            | Self::WccPrepare
            | Self::WccEdgeChunk { .. }
            | Self::WccCompress => ComponentCanonicalLayout::wcc(node_count)
                .map(|layout| layout.words)
                .ok_or_else(|| {
                    candle_core::Error::Msg("Metal WCC workspace size overflow".to_owned())
                }),
            Self::DegreeInitialize | Self::DegreeCsr | Self::DegreeChunk { .. } => {
                checked(2, 1, "degree")
            }
            Self::SccInitialize { .. }
            | Self::SccDegreeChunk { .. }
            | Self::SccTrimSelect
            | Self::SccTrimEdgeChunk { .. }
            | Self::SccCycleProbeInitialize
            | Self::SccCycleProbeEdgeChunk { .. }
            | Self::SccCycleProbeValidate
            | Self::SccCycleFinish { .. }
            | Self::SccColorInitialize
            | Self::SccColorPrepare
            | Self::SccColorEdgeChunk { .. }
            | Self::SccBackwardSeed
            | Self::SccBackwardPrepare
            | Self::SccBackwardEdgeChunk { .. }
            | Self::SccColorAssign => checked(5, 4, "SCC"),
            Self::ComponentCanonicalInitialize { layout, .. }
            | Self::ComponentCanonicalMark { layout, .. }
            | Self::ComponentCanonicalFlags { layout, .. } => Ok(layout.words),
            Self::MetricsInitialize
            | Self::MetricsDegreeChunk { .. }
            | Self::MetricsTriangleChunk { .. }
            | Self::MetricsTrianglePartials
            | Self::MetricsTriangleReduce { .. }
            | Self::MetricsTriangleResult { .. }
            | Self::MetricsClusteringChunk { .. } => metrics_workspace_words(node_count)
                .ok_or_else(|| {
                    candle_core::Error::Msg(
                        "Metal undirected metrics workspace size overflow".to_owned(),
                    )
                }),
            Self::KCoreInitialize { .. }
            | Self::KCoreSeed { .. }
            | Self::KCoreDrainPrepare
            | Self::KCoreDrainChunk { .. } => checked(5, 6, "k-core"),
        }
    }
}

#[derive(Clone)]
struct Pipelines {
    control_clear: candle_metal_kernels::metal::ComputePipeline,
    clear_changed: candle_metal_kernels::metal::ComputePipeline,
    wcc_initialize: candle_metal_kernels::metal::ComputePipeline,
    wcc_relax_edges: candle_metal_kernels::metal::ComputePipeline,
    wcc_compress: candle_metal_kernels::metal::ComputePipeline,
    component_canonical_initialize: candle_metal_kernels::metal::ComputePipeline,
    component_canonical_mark: candle_metal_kernels::metal::ComputePipeline,
    component_canonical_flags: candle_metal_kernels::metal::ComputePipeline,
    scc_initialize: candle_metal_kernels::metal::ComputePipeline,
    scc_initial_degree_edges: candle_metal_kernels::metal::ComputePipeline,
    scc_trim_mark: candle_metal_kernels::metal::ComputePipeline,
    scc_assign_candidates: candle_metal_kernels::metal::ComputePipeline,
    scc_trim_decrement_edges: candle_metal_kernels::metal::ComputePipeline,
    scc_cycle_probe_initialize: candle_metal_kernels::metal::ComputePipeline,
    scc_cycle_probe_edges: candle_metal_kernels::metal::ComputePipeline,
    scc_cycle_probe_validate: candle_metal_kernels::metal::ComputePipeline,
    scc_cycle_jump: candle_metal_kernels::metal::ComputePipeline,
    scc_cycle_assign: candle_metal_kernels::metal::ComputePipeline,
    scc_color_initialize: candle_metal_kernels::metal::ComputePipeline,
    scc_color_edges: candle_metal_kernels::metal::ComputePipeline,
    scc_backward_seed: candle_metal_kernels::metal::ComputePipeline,
    scc_backward_edges: candle_metal_kernels::metal::ComputePipeline,
    scc_color_assign: candle_metal_kernels::metal::ComputePipeline,
    degree_prepare: candle_metal_kernels::metal::ComputePipeline,
    degree_csr: candle_metal_kernels::metal::ComputePipeline,
    degree_edges: candle_metal_kernels::metal::ComputePipeline,
    metrics_prepare: candle_metal_kernels::metal::ComputePipeline,
    metrics_degree_edges: candle_metal_kernels::metal::ComputePipeline,
    triangle_cursor_prepare: candle_metal_kernels::metal::ComputePipeline,
    triangle_oriented_edges: candle_metal_kernels::metal::ComputePipeline,
    triangle_tiles: candle_metal_kernels::metal::ComputePipeline,
    u64_reduce_tiles: candle_metal_kernels::metal::ComputePipeline,
    triangle_finalize: candle_metal_kernels::metal::ComputePipeline,
    clustering_publish: candle_metal_kernels::metal::ComputePipeline,
    kcore_initialize: candle_metal_kernels::metal::ComputePipeline,
    kcore_prepare: candle_metal_kernels::metal::ComputePipeline,
    kcore_seed: candle_metal_kernels::metal::ComputePipeline,
    kcore_drain_prepare: candle_metal_kernels::metal::ComputePipeline,
    kcore_drain: candle_metal_kernels::metal::ComputePipeline,
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
            include_str!("../../../../kernels/metal/graph_components_metrics.metal"),
            None,
        )
        .map_err(|error| {
            candle_core::Error::Msg(format!(
                "compiling Metal components/metrics kernels failed: {error}"
            ))
        })?;
    let pipeline = |name: &str| -> candle_core::Result<_> {
        let function = library.get_function(name, None).map_err(|error| {
            candle_core::Error::Msg(format!(
                "loading Metal components/metrics kernel {name} failed: {error}"
            ))
        })?;
        let raw = device
            .metal_device()
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "creating Metal components/metrics pipeline {name} failed: {error:?}"
                ))
            })?;
        Ok(candle_metal_kernels::metal::ComputePipeline::new(raw))
    };
    let value = Pipelines {
        control_clear: pipeline("ig_cm_control_clear")?,
        clear_changed: pipeline("ig_cm_clear_changed")?,
        wcc_initialize: pipeline("ig_cm_wcc_initialize")?,
        wcc_relax_edges: pipeline("ig_cm_wcc_relax_edges")?,
        wcc_compress: pipeline("ig_cm_wcc_compress")?,
        component_canonical_initialize: pipeline("ig_cm_component_canonical_initialize")?,
        component_canonical_mark: pipeline("ig_cm_component_canonical_mark")?,
        component_canonical_flags: pipeline("ig_cm_component_canonical_flags")?,
        scc_initialize: pipeline("ig_cm_scc_initialize")?,
        scc_initial_degree_edges: pipeline("ig_cm_scc_initial_degree_edges")?,
        scc_trim_mark: pipeline("ig_cm_scc_trim_mark")?,
        scc_assign_candidates: pipeline("ig_cm_scc_assign_candidates")?,
        scc_trim_decrement_edges: pipeline("ig_cm_scc_trim_decrement_edges")?,
        scc_cycle_probe_initialize: pipeline("ig_cm_scc_cycle_probe_initialize")?,
        scc_cycle_probe_edges: pipeline("ig_cm_scc_cycle_probe_edges")?,
        scc_cycle_probe_validate: pipeline("ig_cm_scc_cycle_probe_validate")?,
        scc_cycle_jump: pipeline("ig_cm_scc_cycle_jump")?,
        scc_cycle_assign: pipeline("ig_cm_scc_cycle_assign")?,
        scc_color_initialize: pipeline("ig_cm_scc_color_initialize")?,
        scc_color_edges: pipeline("ig_cm_scc_color_edges")?,
        scc_backward_seed: pipeline("ig_cm_scc_backward_seed")?,
        scc_backward_edges: pipeline("ig_cm_scc_backward_edges")?,
        scc_color_assign: pipeline("ig_cm_scc_color_assign")?,
        degree_prepare: pipeline("ig_cm_degree_prepare")?,
        degree_csr: pipeline("ig_cm_degree_csr")?,
        degree_edges: pipeline("ig_cm_degree_edges")?,
        metrics_prepare: pipeline("ig_cm_undirected_prepare")?,
        metrics_degree_edges: pipeline("ig_cm_undirected_degree_edges")?,
        triangle_cursor_prepare: pipeline("ig_cm_triangle_cursor_prepare")?,
        triangle_oriented_edges: pipeline("ig_cm_triangle_oriented_edges")?,
        triangle_tiles: pipeline("ig_cm_triangle_tiles")?,
        u64_reduce_tiles: pipeline("ig_cm_u64_reduce_tiles")?,
        triangle_finalize: pipeline("ig_cm_triangle_finalize")?,
        clustering_publish: pipeline("ig_cm_clustering_publish")?,
        kcore_initialize: pipeline("ig_cm_kcore_initialize")?,
        kcore_prepare: pipeline("ig_cm_kcore_prepare")?,
        kcore_seed: pipeline("ig_cm_kcore_seed")?,
        kcore_drain_prepare: pipeline("ig_cm_kcore_drain_prepare")?,
        kcore_drain: pipeline("ig_cm_kcore_drain")?,
    };
    for candidate in [
        &value.control_clear,
        &value.clear_changed,
        &value.wcc_initialize,
        &value.wcc_relax_edges,
        &value.wcc_compress,
        &value.component_canonical_initialize,
        &value.component_canonical_mark,
        &value.component_canonical_flags,
        &value.scc_initialize,
        &value.scc_initial_degree_edges,
        &value.scc_trim_mark,
        &value.scc_assign_candidates,
        &value.scc_trim_decrement_edges,
        &value.scc_cycle_probe_initialize,
        &value.scc_cycle_probe_edges,
        &value.scc_cycle_probe_validate,
        &value.scc_cycle_jump,
        &value.scc_cycle_assign,
        &value.scc_color_initialize,
        &value.scc_color_edges,
        &value.scc_backward_seed,
        &value.scc_backward_edges,
        &value.scc_color_assign,
        &value.degree_prepare,
        &value.degree_csr,
        &value.degree_edges,
        &value.metrics_prepare,
        &value.metrics_degree_edges,
        &value.triangle_cursor_prepare,
        &value.triangle_oriented_edges,
        &value.triangle_tiles,
        &value.u64_reduce_tiles,
        &value.triangle_finalize,
        &value.clustering_publish,
        &value.kcore_initialize,
        &value.kcore_prepare,
        &value.kcore_seed,
        &value.kcore_drain_prepare,
        &value.kcore_drain,
    ] {
        if candidate.max_total_threads_per_threadgroup() < THREADS {
            return Err(candle_core::Error::Msg(
                "selected Metal device cannot run 256-thread components/metrics groups".to_owned(),
            ));
        }
    }
    let _ = PIPELINES.set(value.clone());
    Ok(PIPELINES.get().cloned().unwrap_or(value))
}

/// Compile, cache, and validate all components/metrics pipelines during backend preparation.
pub fn prepare(device: &Device) -> Result<()> {
    let Device::Metal(device) = device else {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "Metal components/metrics pipeline preparation requires a Metal device",
        ));
    };
    pipelines(device).map_err(candle_error)?;
    Ok(())
}

#[derive(Clone)]
struct GraphCommand {
    graph: GraphImage,
    kind: CommandKind,
}

impl CustomOp1 for GraphCommand {
    fn name(&self) -> &'static str {
        "irongraph-metal-graph-components-metrics"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal graph components/metrics cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        macro_rules! metal_tensor {
            ($field:ident, $storage:ident, $layout:ident, $metal:ident, $message:literal) => {
                let ($storage, $layout) = self.graph.$field.storage_and_layout();
                let Storage::Metal($metal) = &*$storage else {
                    return Err(candle_core::Error::Msg($message.to_owned()));
                };
            };
        }
        metal_tensor!(
            edge_targets,
            edge_targets_storage,
            edge_targets_layout,
            edge_targets,
            "Metal components edge-target column moved off device"
        );
        metal_tensor!(
            visible_nodes,
            visible_nodes_storage,
            visible_nodes_layout,
            visible_nodes,
            "Metal components visibility moved off device"
        );
        metal_tensor!(
            edge_active,
            edge_active_storage,
            edge_active_layout,
            edge_active,
            "Metal components edge-active column moved off device"
        );
        metal_tensor!(
            edge_layers,
            edge_layers_storage,
            edge_layers_layout,
            edge_layers,
            "Metal components edge-layer column moved off device"
        );
        metal_tensor!(
            edge_sources,
            edge_sources_storage,
            edge_sources_layout,
            edge_sources,
            "Metal components edge-source column moved off device"
        );
        metal_tensor!(
            undirected_offsets,
            undirected_offsets_storage,
            undirected_offsets_layout,
            undirected_offsets,
            "Metal unique-undirected offsets moved off device"
        );
        metal_tensor!(
            undirected_neighbors,
            undirected_neighbors_storage,
            undirected_neighbors_layout,
            undirected_neighbors,
            "Metal unique-undirected neighbors moved off device"
        );
        metal_tensor!(
            directed_incoming_offsets,
            directed_incoming_offsets_storage,
            directed_incoming_offsets_layout,
            directed_incoming_offsets,
            "Metal directed incoming offsets moved off device"
        );

        let words = self.kind.workspace_words(self.graph.node_count)?;
        let layouts = [
            workspace_layout,
            edge_targets_layout,
            visible_nodes_layout,
            edge_active_layout,
            edge_layers_layout,
            edge_sources_layout,
            undirected_offsets_layout,
            undirected_neighbors_layout,
            directed_incoming_offsets_layout,
        ];
        if self.graph.node_count == 0
            || self.graph.layer_mask == 0
            || workspace.dtype() != DType::U32
            || self.graph.edge_targets.dtype() != DType::U32
            || self.graph.visible_nodes.dtype() != DType::U8
            || self.graph.edge_active.dtype() != DType::U8
            || self.graph.edge_layers.dtype() != DType::U8
            || self.graph.edge_sources.dtype() != DType::U32
            || self.graph.undirected_offsets.dtype() != DType::U32
            || self.graph.undirected_neighbors.dtype() != DType::U32
            || layouts
                .iter()
                .any(|layout| !layout.is_contiguous() || layout.dims().len() != 1)
            || workspace_layout.shape().elem_count() != words
            || edge_targets_layout.shape().elem_count() != self.graph.edge_count
            || visible_nodes_layout.shape().elem_count() != self.graph.node_count
            || edge_active_layout.shape().elem_count() != self.graph.edge_count
            || edge_layers_layout.shape().elem_count() != self.graph.edge_count
            || edge_sources_layout.shape().elem_count() != self.graph.edge_count
            || undirected_offsets_layout.shape().elem_count()
                != self.graph.node_count.saturating_add(1)
            || directed_incoming_offsets_layout.shape().elem_count()
                != self.graph.node_count.saturating_add(1)
            || undirected_neighbors_layout.shape().elem_count()
                != self.graph.undirected_adjacency_count
        {
            return Err(candle_core::Error::Msg(
                "Metal graph components/metrics tensor contract is invalid".to_owned(),
            ));
        }

        let device = workspace.device();
        let pipelines = pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        let word_bytes = DType::U32.size_in_bytes();
        let byte = |word: usize| {
            workspace_layout
                .start_offset()
                .saturating_add(word)
                .saturating_mul(word_bytes)
        };
        let node_groups = objc2_metal::MTLSize {
            width: self.graph.node_count.div_ceil(THREADS),
            height: 1,
            depth: 1,
        };
        let node_threads = objc2_metal::MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        };
        let one = objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let (scalar, span) = match self.kind {
            CommandKind::MetricsDegreeChunk { start, span }
            | CommandKind::MetricsTriangleChunk { start, span, .. }
            | CommandKind::WccEdgeChunk { start, span }
            | CommandKind::SccDegreeChunk { start, span }
            | CommandKind::SccTrimEdgeChunk { start, span }
            | CommandKind::SccCycleProbeEdgeChunk { start, span }
            | CommandKind::SccColorEdgeChunk { start, span }
            | CommandKind::SccBackwardEdgeChunk { start, span }
            | CommandKind::DegreeChunk { start, span }
            | CommandKind::MetricsClusteringChunk { start, span }
            | CommandKind::ComponentCanonicalInitialize { start, span, .. }
            | CommandKind::ComponentCanonicalMark { start, span, .. }
            | CommandKind::ComponentCanonicalFlags { start, span, .. } => (start, span),
            CommandKind::KCoreSeed { core } => (core, 0),
            CommandKind::MetricsTriangleReduce { input_count, .. } => (input_count, 0),
            CommandKind::KCoreDrainChunk { core, start, .. } => (core, start),
            CommandKind::WccInitialize { visible_count }
            | CommandKind::SccInitialize { visible_count }
            | CommandKind::KCoreInitialize { visible_count } => (visible_count, 0),
            CommandKind::MetricsInitialize => (
                0,
                u32::try_from(self.graph.node_count).map_err(|_| {
                    candle_core::Error::Msg("Metal metrics node count exceeds u32".to_owned())
                })?,
            ),
            _ => (0, 0),
        };
        let mut args = Args::new(&self.graph, scalar, span)?;
        if matches!(
            self.kind,
            CommandKind::MetricsInitialize
                | CommandKind::MetricsDegreeChunk { .. }
                | CommandKind::MetricsTriangleChunk { .. }
                | CommandKind::MetricsTrianglePartials
                | CommandKind::MetricsTriangleReduce { .. }
                | CommandKind::MetricsTriangleResult { .. }
                | CommandKind::MetricsClusteringChunk { .. }
                | CommandKind::KCoreInitialize { .. }
                | CommandKind::KCoreSeed { .. }
                | CommandKind::KCoreDrainPrepare
                | CommandKind::KCoreDrainChunk { .. }
        ) {
            args.adjacency_count =
                u32::try_from(self.graph.undirected_adjacency_count).map_err(|_| {
                    candle_core::Error::Msg(
                        "Metal unique-undirected adjacency count exceeds u32".to_owned(),
                    )
                })?;
        }
        if matches!(self.kind, CommandKind::MetricsTriangleChunk { .. }) {
            args.reserved_2 = u32::try_from(METRICS_TRIANGLE_SCAN_QUANTUM).map_err(|_| {
                candle_core::Error::Msg("Metal triangle scan quantum exceeds u32".to_owned())
            })?;
        }
        if let CommandKind::KCoreDrainChunk { span, .. } = self.kind {
            args.partial_count = span;
            args.reserved_2 = u32::try_from(KCORE_ROW_SCAN_QUANTUM).map_err(|_| {
                candle_core::Error::Msg("Metal k-core row quantum exceeds u32".to_owned())
            })?;
        }
        let edge_chunk_groups = objc2_metal::MTLSize {
            width: usize::try_from(span)
                .unwrap_or(usize::MAX)
                .div_ceil(THREADS)
                .max(1),
            height: 1,
            depth: 1,
        };
        let node_chunk_groups = objc2_metal::MTLSize {
            width: usize::try_from(span)
                .unwrap_or(usize::MAX)
                .div_ceil(THREADS)
                .max(1),
            height: 1,
            depth: 1,
        };
        let edge_targets_byte = edge_targets_layout.start_offset() * word_bytes;
        let visible_byte = visible_nodes_layout.start_offset() * DType::U8.size_in_bytes();
        let active_byte = edge_active_layout.start_offset() * DType::U8.size_in_bytes();
        let layers_byte = edge_layers_layout.start_offset() * DType::U8.size_in_bytes();
        let edge_sources_byte = edge_sources_layout.start_offset() * word_bytes;
        let undirected_offsets_byte = undirected_offsets_layout.start_offset() * word_bytes;
        let undirected_neighbors_byte = undirected_neighbors_layout.start_offset() * word_bytes;
        let directed_incoming_offsets_byte =
            directed_incoming_offsets_layout.start_offset() * word_bytes;

        macro_rules! clear_control {
            ($offset:expr, $values:expr) => {{
                let values: [u32; 4] = $values;
                encoder.set_compute_pipeline_state(&pipelines.control_clear);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte($offset));
                encoder.set_bytes(1, &values);
                encoder.dispatch_thread_groups(one, one);
                encoder.insert_memory_barrier();
            }};
        }
        macro_rules! clear_word {
            ($offset:expr) => {{
                encoder.set_compute_pipeline_state(&pipelines.clear_changed);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte($offset));
                encoder.dispatch_thread_groups(one, one);
                encoder.insert_memory_barrier();
            }};
        }

        let n = self.graph.node_count;
        match self.kind {
            CommandKind::WccInitialize { .. } => {
                clear_control!(n, [0, 0, UNREACHED, 0]);
                encoder.set_compute_pipeline_state(&pipelines.wcc_initialize);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::WccPrepare => clear_word!(n),
            CommandKind::WccEdgeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.wcc_relax_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_bytes(8, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::WccCompress => {
                encoder.set_compute_pipeline_state(&pipelines.wcc_compress);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccInitialize { visible_count } => {
                clear_control!(5 * n, [0, 0, visible_count, 0]);
                encoder.set_compute_pipeline_state(&pipelines.scc_initialize);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(5, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccDegreeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.scc_initial_degree_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_input_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(9, Some(workspace.buffer()), byte(5 * n + 1));
                encoder.set_bytes(10, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::SccTrimSelect => {
                clear_word!(5 * n);
                clear_word!(5 * n + 3);
                encoder.set_compute_pipeline_state(&pipelines.scc_trim_mark);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(2, Some(workspace.buffer()), byte(2 * n));
                encoder.set_input_buffer(3, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(5, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(6, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
                encoder.insert_memory_barrier();
                encoder.set_compute_pipeline_state(&pipelines.scc_assign_candidates);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(5 * n + 2));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(5 * n + 3));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccTrimEdgeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.scc_trim_decrement_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_input_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(9, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(10, Some(workspace.buffer()), byte(5 * n + 1));
                encoder.set_bytes(11, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::SccCycleProbeInitialize => {
                clear_word!(5 * n + 3);
                encoder.set_compute_pipeline_state(&pipelines.scc_cycle_probe_initialize);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(2, Some(workspace.buffer()), byte(2 * n));
                encoder.set_input_buffer(3, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(5, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(7, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccCycleProbeEdgeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.scc_cycle_probe_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_input_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(9, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::SccCycleProbeValidate => {
                encoder.set_compute_pipeline_state(&pipelines.scc_cycle_probe_validate);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccCycleFinish { rounds } => {
                let rounds = usize::try_from(rounds).map_err(|_| {
                    candle_core::Error::Msg("Metal SCC cycle round count exceeds usize".to_owned())
                })?;
                for round in 0..rounds {
                    let (current_successor, next_successor, current_label, next_label) =
                        if round.is_multiple_of(2) {
                            (n, 2 * n, 4 * n, 3 * n)
                        } else {
                            (2 * n, n, 3 * n, 4 * n)
                        };
                    encoder.set_compute_pipeline_state(&pipelines.scc_cycle_jump);
                    encoder.set_input_buffer(0, Some(workspace.buffer()), byte(current_successor));
                    encoder.set_input_buffer(1, Some(workspace.buffer()), byte(current_label));
                    encoder.set_output_buffer(2, Some(workspace.buffer()), byte(next_successor));
                    encoder.set_output_buffer(3, Some(workspace.buffer()), byte(next_label));
                    encoder.set_input_buffer(4, Some(workspace.buffer()), byte(0));
                    encoder.set_output_buffer(5, Some(workspace.buffer()), byte(5 * n + 1));
                    encoder.set_bytes(6, &args);
                    encoder.dispatch_thread_groups(node_groups, node_threads);
                    encoder.insert_memory_barrier();
                }
                let label = if rounds.is_multiple_of(2) {
                    4 * n
                } else {
                    3 * n
                };
                encoder.set_compute_pipeline_state(&pipelines.scc_cycle_assign);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(label));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(5 * n + 2));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(5 * n + 1));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccColorInitialize => {
                encoder.set_compute_pipeline_state(&pipelines.scc_color_initialize);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccColorPrepare | CommandKind::SccBackwardPrepare => clear_word!(5 * n),
            CommandKind::SccColorEdgeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.scc_color_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_input_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(9, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::SccBackwardSeed => {
                encoder.set_compute_pipeline_state(&pipelines.scc_backward_seed);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(4 * n));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::SccBackwardEdgeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.scc_backward_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_input_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(9, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(10, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::SccColorAssign => {
                clear_word!(5 * n + 3);
                encoder.set_compute_pipeline_state(&pipelines.scc_color_assign);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(5 * n + 2));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(5 * n + 3));
                encoder.set_bytes(5, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::ComponentCanonicalInitialize { layout, .. } => {
                encoder.set_compute_pipeline_state(&pipelines.component_canonical_initialize);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte(layout.root_minimum));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(layout.minimum_flags));
                encoder.set_bytes(2, &args);
                encoder.dispatch_thread_groups(node_chunk_groups, node_threads);
            }
            CommandKind::ComponentCanonicalMark { layout, .. } => {
                encoder.set_compute_pipeline_state(&pipelines.component_canonical_mark);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(layout.root_minimum));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_chunk_groups, node_threads);
            }
            CommandKind::ComponentCanonicalFlags { layout, .. } => {
                encoder.set_compute_pipeline_state(&pipelines.component_canonical_flags);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(layout.root_minimum));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(layout.minimum_flags));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(node_chunk_groups, node_threads);
            }
            CommandKind::DegreeInitialize => {
                clear_word!(2 * n);
                encoder.set_compute_pipeline_state(&pipelines.degree_prepare);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(2 * n));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::DegreeCsr => {
                clear_word!(2 * n);
                encoder.set_compute_pipeline_state(&pipelines.degree_csr);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(
                    1,
                    Some(directed_incoming_offsets.buffer()),
                    directed_incoming_offsets_byte,
                );
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(2 * n));
                encoder.set_bytes(5, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::DegreeChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.degree_edges);
                encoder.set_input_buffer(0, Some(edge_targets.buffer()), edge_targets_byte);
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(3, Some(edge_active.buffer()), active_byte);
                encoder.set_input_buffer(4, Some(edge_layers.buffer()), layers_byte);
                encoder.set_input_buffer(5, Some(edge_sources.buffer()), edge_sources_byte);
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(2 * n));
                encoder.set_bytes(9, &args);
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::MetricsInitialize => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                clear_word!(layout.status);
                encoder.set_compute_pipeline_state(&pipelines.metrics_prepare);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(1, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::MetricsDegreeChunk { .. } => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                encoder.set_compute_pipeline_state(&pipelines.metrics_degree_edges);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(
                    1,
                    Some(undirected_neighbors.buffer()),
                    undirected_neighbors_byte,
                );
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(5, &args);
                encoder.dispatch_thread_groups(node_chunk_groups, node_threads);
            }
            CommandKind::MetricsTriangleChunk { reset, .. } => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                if reset {
                    encoder.set_compute_pipeline_state(&pipelines.triangle_cursor_prepare);
                    encoder.set_output_buffer(0, Some(workspace.buffer()), byte(layout.cursors));
                    encoder.set_output_buffer(
                        1,
                        Some(workspace.buffer()),
                        byte(layout.cursors + METRICS_EDGE_CHUNK),
                    );
                    encoder.set_bytes(2, &args);
                    encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
                    encoder.insert_memory_barrier();
                }
                clear_word!(layout.status + 1);
                encoder.set_compute_pipeline_state(&pipelines.triangle_oriented_edges);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(
                    1,
                    Some(undirected_neighbors.buffer()),
                    undirected_neighbors_byte,
                );
                encoder.set_input_buffer(2, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(5, &args);
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(layout.cursors));
                encoder.set_output_buffer(
                    7,
                    Some(workspace.buffer()),
                    byte(layout.cursors + METRICS_EDGE_CHUNK),
                );
                encoder.set_output_buffer(8, Some(workspace.buffer()), byte(layout.status + 1));
                encoder.dispatch_thread_groups(edge_chunk_groups, node_threads);
            }
            CommandKind::MetricsTrianglePartials => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                let partial_count = n.div_ceil(THREADS);
                encoder.set_compute_pipeline_state(&pipelines.triangle_tiles);
                encoder.set_input_buffer(0, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(layout.partial_a));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(4, &args);
                encoder.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: partial_count.div_ceil(THREADS),
                        height: 1,
                        depth: 1,
                    },
                    node_threads,
                );
            }
            CommandKind::MetricsTriangleReduce {
                input_count,
                input_second,
            } => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                let (input, output) = if input_second {
                    (layout.partial_b, layout.partial_a)
                } else {
                    (layout.partial_a, layout.partial_b)
                };
                let output_count = usize::try_from(input_count)
                    .unwrap_or(usize::MAX)
                    .div_ceil(THREADS);
                encoder.set_compute_pipeline_state(&pipelines.u64_reduce_tiles);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(input));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(output));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: output_count.div_ceil(THREADS),
                        height: 1,
                        depth: 1,
                    },
                    node_threads,
                );
            }
            CommandKind::MetricsTriangleResult { input_second } => {
                let layout = MetricsLayout::new(n).ok_or_else(|| {
                    candle_core::Error::Msg("Metal metrics layout overflow".to_owned())
                })?;
                let input = if input_second {
                    layout.partial_b
                } else {
                    layout.partial_a
                };
                encoder.set_compute_pipeline_state(&pipelines.triangle_finalize);
                encoder.set_input_buffer(0, Some(workspace.buffer()), byte(input));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(layout.result));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(layout.status));
                encoder.set_bytes(3, &args);
                encoder.dispatch_thread_groups(one, one);
            }
            CommandKind::MetricsClusteringChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.clustering_publish);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte(0));
                encoder.set_bytes(1, &args);
                encoder.dispatch_thread_groups(node_chunk_groups, node_threads);
            }
            CommandKind::KCoreInitialize { visible_count } => {
                clear_control!(5 * n, [visible_count, 0, UNREACHED, 0]);
                encoder.set_compute_pipeline_state(&pipelines.kcore_initialize);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(1, Some(visible_nodes.buffer()), visible_byte);
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(5, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(8, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::KCoreSeed { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.kcore_prepare);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(1, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(2, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
                encoder.insert_memory_barrier();
                encoder.set_compute_pipeline_state(&pipelines.kcore_seed);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte(0));
                encoder.set_input_buffer(1, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(5, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(6, &args);
                encoder.dispatch_thread_groups(node_groups, node_threads);
            }
            CommandKind::KCoreDrainPrepare => {
                encoder.set_compute_pipeline_state(&pipelines.kcore_drain_prepare);
                encoder.set_output_buffer(0, Some(workspace.buffer()), byte(5 * n + 1));
                encoder.dispatch_thread_groups(one, one);
            }
            CommandKind::KCoreDrainChunk { .. } => {
                encoder.set_compute_pipeline_state(&pipelines.kcore_drain);
                encoder.set_input_buffer(
                    0,
                    Some(undirected_offsets.buffer()),
                    undirected_offsets_byte,
                );
                encoder.set_input_buffer(
                    1,
                    Some(undirected_neighbors.buffer()),
                    undirected_neighbors_byte,
                );
                encoder.set_output_buffer(2, Some(workspace.buffer()), byte(0));
                encoder.set_output_buffer(3, Some(workspace.buffer()), byte(n));
                encoder.set_output_buffer(4, Some(workspace.buffer()), byte(3 * n));
                encoder.set_output_buffer(5, Some(workspace.buffer()), byte(2 * n));
                encoder.set_output_buffer(6, Some(workspace.buffer()), byte(4 * n));
                encoder.set_output_buffer(7, Some(workspace.buffer()), byte(5 * n));
                encoder.set_bytes(8, &args);
                encoder.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: usize::try_from(args.partial_count)
                            .unwrap_or(usize::MAX)
                            .div_ceil(THREADS)
                            .max(1),
                        height: 1,
                        depth: 1,
                    },
                    node_threads,
                );
            }
        }
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                words,
                DType::U32,
            ),
            Shape::from(words),
        ))
    }
}

fn status_error(status: u32, algorithm: &str) -> Result<()> {
    match status {
        0 => Ok(()),
        1 => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} rejected corrupt CSR offsets"),
        )),
        2 => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} rejected corrupt CSR ordinals"),
        )),
        3 => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} rejected corrupt visibility metadata"),
        )),
        4 => Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "triangle count overflow",
        )),
        5 => Err(Error::new(
            ErrorCode::CorruptStorage,
            "Metal triangle incidence total is not divisible by three",
        )),
        6 => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} degree underflow"),
        )),
        other => Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} returned unknown status {other}"),
        )),
    }
}

fn visible_rows(visible_rows: &Tensor) -> Result<Vec<u32>> {
    visible_rows.to_vec1::<u32>().map_err(candle_error)
}

/// Candle narrow views retain their full backing. Gather control words into a distinct small Metal
/// allocation before host transfer so polling never stages the complete algorithm workspace.
fn read_u32_words(tensor: &Tensor, offset: usize, count: usize) -> Result<Vec<u32>> {
    let end = offset.checked_add(count).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal control-word read range overflow",
        )
    })?;
    if end > tensor.elem_count() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "Metal control-word read exceeds its workspace",
        ));
    }
    tensor
        .narrow(0, offset, count)
        .and_then(|words| words.copy())
        .and_then(|words| words.to_vec1::<u32>())
        .map_err(candle_error)
}

fn singleton_components(
    selected_rows: &Tensor,
    rows: Vec<u32>,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<ResidentGraphProcedureResult> {
    ensure_graph_execution(cancellation, deadline)?;
    if rows.is_empty() {
        return Ok(ResidentGraphProcedureResult::Components {
            node_rows: rows,
            component: Vec::new(),
        });
    }
    let end = u32::try_from(rows.len()).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "component ID space exhausted",
        )
    })?;
    let component = Tensor::arange(0_u32, end, selected_rows.device())
        .and_then(|values| values.to_vec1::<u32>())
        .map_err(candle_error)?;
    if component.len() != rows.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "Metal singleton component publication length mismatch",
        ));
    }
    ensure_graph_execution(cancellation, deadline)?;
    Ok(ResidentGraphProcedureResult::Components {
        node_rows: rows,
        component,
    })
}

#[allow(clippy::too_many_lines)]
fn canonical_component_ids(
    graph: &GraphImage,
    selected_rows: &Tensor,
    mut workspace: Tensor,
    layout: ComponentCanonicalLayout,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
    algorithm: &str,
) -> Result<Vec<u32>> {
    for start in (0..graph.node_count).step_by(METRICS_EDGE_CHUNK) {
        ensure_graph_execution(cancellation, deadline)?;
        let span = (graph.node_count - start).min(METRICS_EDGE_CHUNK);
        let start = u32::try_from(start).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                format!("Metal {algorithm} canonical node start exceeds u32"),
            )
        })?;
        let span = u32::try_from(span).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                format!("Metal {algorithm} canonical node span exceeds u32"),
            )
        })?;
        workspace = workspace
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::ComponentCanonicalInitialize {
                    layout,
                    start,
                    span,
                },
            })
            .map_err(candle_error)?;
    }
    for start in (0..graph.node_count).step_by(METRICS_EDGE_CHUNK) {
        ensure_graph_execution(cancellation, deadline)?;
        let span = (graph.node_count - start).min(METRICS_EDGE_CHUNK);
        workspace = workspace
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::ComponentCanonicalMark {
                    layout,
                    start: u32::try_from(start).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            format!("Metal {algorithm} canonical mark start exceeds u32"),
                        )
                    })?,
                    span: u32::try_from(span).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            format!("Metal {algorithm} canonical mark span exceeds u32"),
                        )
                    })?,
                },
            })
            .map_err(candle_error)?;
        let status = read_u32_words(&workspace, layout.status, 1)?;
        status_error(status.first().copied().unwrap_or(1), algorithm)?;
    }
    for start in (0..graph.node_count).step_by(METRICS_EDGE_CHUNK) {
        ensure_graph_execution(cancellation, deadline)?;
        let span = (graph.node_count - start).min(METRICS_EDGE_CHUNK);
        workspace = workspace
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::ComponentCanonicalFlags {
                    layout,
                    start: u32::try_from(start).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            format!("Metal {algorithm} canonical flag start exceeds u32"),
                        )
                    })?,
                    span: u32::try_from(span).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            format!("Metal {algorithm} canonical flag span exceeds u32"),
                        )
                    })?,
                },
            })
            .map_err(candle_error)?;
        let status = read_u32_words(&workspace, layout.status, 1)?;
        status_error(status.first().copied().unwrap_or(1), algorithm)?;
    }

    let selected_roots = workspace
        .narrow(0, 0, graph.node_count)
        .and_then(|assignment| assignment.index_select(selected_rows, 0))
        .map_err(candle_error)?;
    let selected_minimums = workspace
        .narrow(0, layout.root_minimum, graph.node_count)
        .and_then(|minimums| minimums.index_select(&selected_roots, 0))
        .map_err(candle_error)?;
    drop(selected_roots);
    let flags = workspace
        .narrow(0, layout.minimum_flags, graph.node_count)
        .and_then(|flags| flags.to_dtype(DType::I64))
        .map_err(candle_error)?;
    let prefix = super::bounded_inclusive_scan_i64_with_deadline(
        &flags,
        i64::MAX,
        workspace.device(),
        cancellation,
        deadline,
    )?;
    drop(flags);
    let selected_prefix = prefix
        .index_select(&selected_minimums, 0)
        .map_err(candle_error)?;
    drop(selected_minimums);
    drop(prefix);
    let one = Tensor::ones(selected_prefix.elem_count(), DType::I64, workspace.device())
        .map_err(candle_error)?;
    let component = selected_prefix
        .sub(&one)
        .and_then(|values| values.to_dtype(DType::U32))
        .and_then(|values| values.to_vec1::<u32>())
        .map_err(candle_error)?;
    if component.len() != selected_rows.elem_count() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("Metal {algorithm} canonical publication length mismatch"),
        ));
    }
    ensure_graph_execution(cancellation, deadline)?;
    Ok(component)
}

impl CandleResident {
    pub(super) fn degree_uses_no_filter_csr(&self, layers: LayerMask) -> bool {
        layers == LayerMask::ALL
            && self.active_node_count == self.node_count
            && self.active_edge_count == self.edge_count
            && self.outgoing_overlay.rows.is_empty()
            && self.incoming_overlay.rows.is_empty()
    }

    pub fn metal_degree(
        &self,
        visible_mask: &Tensor,
        selected_rows: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        ensure_graph_execution(cancellation, deadline)?;
        let no_filter_csr = self.degree_uses_no_filter_csr(layers);
        let node_rows = if no_filter_csr {
            (0..u32::try_from(self.node_count).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "degree node count exceeds u32",
                )
            })?)
                .collect()
        } else {
            visible_rows(selected_rows)?
        };
        ensure_graph_execution(cancellation, deadline)?;
        let Some(graph) = GraphImage::from_resident(self, visible_mask, layers)? else {
            return Ok(ResidentGraphProcedureResult::Degree {
                out_degree: vec![0; node_rows.len()],
                in_degree: vec![0; node_rows.len()],
                node_rows,
            });
        };
        let words = CommandKind::DegreeInitialize
            .workspace_words(self.node_count)
            .map_err(candle_error)?;
        let status = 2 * self.node_count;
        ensure_graph_execution(cancellation, deadline)?;
        let mut workspace =
            Tensor::zeros(words, DType::U32, visible_mask.device()).map_err(candle_error)?;
        if no_filter_csr {
            // Delta shape: the guard is maintained from changed activity rows and the overlay row
            // counts. This query reads the two resident offset columns once; it never rebuilds CSR
            // or scans unrelated relationship metadata.
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::DegreeCsr,
                })
                .map_err(candle_error)?;
        } else {
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::DegreeInitialize,
                })
                .map_err(candle_error)?;
            for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
                let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::DegreeChunk {
                            start: u32::try_from(start).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "degree edge chunk start exceeds u32",
                                )
                            })?,
                            span: u32::try_from(span).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "degree edge chunk length exceeds u32",
                                )
                            })?,
                        },
                    })
                    .map_err(candle_error)?;
            }
        }
        let chunk_status = read_u32_words(&workspace, status, 1)?;
        status_error(
            chunk_status.first().copied().unwrap_or(1),
            "directed degree",
        )?;
        let publish = |start| {
            let values = workspace.narrow(0, start, self.node_count)?;
            if no_filter_csr {
                values.to_vec1::<u32>()
            } else {
                values.index_select(selected_rows, 0)?.to_vec1::<u32>()
            }
        };
        let out_degree = publish(0).map_err(candle_error)?;
        let in_degree = publish(self.node_count).map_err(candle_error)?;
        ensure_graph_execution(cancellation, deadline)?;
        Ok(ResidentGraphProcedureResult::Degree {
            node_rows,
            out_degree,
            in_degree,
        })
    }

    pub fn metal_weakly_connected_components(
        &self,
        visible_mask: &Tensor,
        selected_rows: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        ensure_graph_execution(cancellation, deadline)?;
        let node_rows = visible_rows(selected_rows)?;
        if node_rows.is_empty() {
            return singleton_components(selected_rows, node_rows, cancellation, deadline);
        }
        let source = node_rows[0];
        let forward = self.metal_bfs_persistent_workspace_for_direction(
            visible_mask,
            source,
            layers,
            false,
            cancellation,
            deadline,
        )?;
        let reached = forward
            .narrow(0, 0, self.node_count)
            .and_then(|distance| distance.ne(UNREACHED))
            .and_then(|reached| reached.to_dtype(DType::U32))
            .and_then(|reached| reached.sum_all())
            .and_then(|count| count.to_scalar::<u32>())
            .map_err(candle_error)?;
        if reached as usize == node_rows.len() {
            ensure_graph_execution(cancellation, deadline)?;
            return Ok(ResidentGraphProcedureResult::Components {
                component: vec![0; node_rows.len()],
                node_rows,
            });
        }
        let Some(graph) = GraphImage::from_resident(self, visible_mask, layers)? else {
            return singleton_components(selected_rows, node_rows, cancellation, deadline);
        };
        let words = CommandKind::WccCompress
            .workspace_words(self.node_count)
            .map_err(candle_error)?;
        let mut workspace = Tensor::zeros(words, DType::U32, visible_mask.device())
            .map_err(candle_error)?
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::WccInitialize {
                    visible_count: u32::try_from(node_rows.len()).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "visible node count exceeds u32",
                        )
                    })?,
                },
            })
            .map_err(candle_error)?;
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::WccPrepare,
                })
                .map_err(candle_error)?;
            for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
                let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::WccEdgeChunk {
                            start: u32::try_from(start).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "WCC edge chunk start exceeds u32",
                                )
                            })?,
                            span: u32::try_from(span).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "WCC edge chunk length exceeds u32",
                                )
                            })?,
                        },
                    })
                    .map_err(candle_error)?;
                let control = read_u32_words(&workspace, self.node_count, 2)?;
                status_error(control.get(1).copied().unwrap_or(1), "WCC")?;
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::WccCompress,
                })
                .map_err(candle_error)?;
            let control = read_u32_words(&workspace, self.node_count, 2)?;
            status_error(control.get(1).copied().unwrap_or(1), "WCC")?;
            if control.first().copied().unwrap_or(1) == 0 {
                break;
            }
        }
        let layout = ComponentCanonicalLayout::wcc(self.node_count).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal WCC canonical workspace size overflow",
            )
        })?;
        let component = canonical_component_ids(
            &graph,
            selected_rows,
            workspace,
            layout,
            cancellation,
            deadline,
            "WCC",
        )?;
        Ok(ResidentGraphProcedureResult::Components {
            node_rows,
            component,
        })
    }

    #[allow(clippy::too_many_lines)]
    pub fn metal_strongly_connected_components(
        &self,
        visible_mask: &Tensor,
        selected_rows: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        ensure_graph_execution(cancellation, deadline)?;
        let node_rows = visible_rows(selected_rows)?;
        if node_rows.is_empty() {
            return singleton_components(selected_rows, node_rows, cancellation, deadline);
        }
        let Some(graph) = GraphImage::from_resident(self, visible_mask, layers)? else {
            return singleton_components(selected_rows, node_rows, cancellation, deadline);
        };
        let visible_count = u32::try_from(node_rows.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "visible node count exceeds u32",
            )
        })?;

        // A strongly connected visible graph is the common analytics case and needs no SCC
        // partitioning: one forward and one reverse traversal from any visible node prove that
        // every visible node reaches every other. Keep both proofs on Metal via the persistent
        // FIFO traversal. The former color-propagation path can otherwise require one
        // host-synchronized round per graph-diameter step before discovering the same single
        // component. A failed proof changes no state and falls through to the complete general
        // algorithm below.
        let source = node_rows[0];
        let reached_count = |workspace: &Tensor| -> Result<u32> {
            workspace
                .narrow(0, 0, self.node_count)
                .and_then(|distance| distance.ne(UNREACHED))
                .and_then(|reached| reached.to_dtype(DType::U32))
                .and_then(|reached| reached.sum_all())
                .and_then(|count| count.to_scalar::<u32>())
                .map_err(candle_error)
        };
        let forward = self.metal_bfs_persistent_workspace_for_direction(
            visible_mask,
            source,
            layers,
            false,
            cancellation,
            deadline,
        )?;
        if reached_count(&forward)? == visible_count {
            drop(forward);
            let reverse = self.metal_bfs_persistent_workspace_for_direction(
                visible_mask,
                source,
                layers,
                true,
                cancellation,
                deadline,
            )?;
            if reached_count(&reverse)? == visible_count {
                ensure_graph_execution(cancellation, deadline)?;
                return Ok(ResidentGraphProcedureResult::Components {
                    component: vec![0; node_rows.len()],
                    node_rows,
                });
            }
        }
        let words = CommandKind::SccInitialize { visible_count }
            .workspace_words(self.node_count)
            .map_err(candle_error)?;
        let mut workspace = Tensor::zeros(words, DType::U32, visible_mask.device())
            .map_err(candle_error)?
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::SccInitialize { visible_count },
            })
            .map_err(candle_error)?;
        let control_offset = 5 * self.node_count;
        for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
            ensure_graph_execution(cancellation, deadline)?;
            let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccDegreeChunk {
                        start: u32::try_from(start).map_err(|_| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "SCC edge chunk start exceeds u32",
                            )
                        })?,
                        span: u32::try_from(span).map_err(|_| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "SCC edge chunk length exceeds u32",
                            )
                        })?,
                    },
                })
                .map_err(candle_error)?;
            let control = read_u32_words(&workspace, control_offset, 2)?;
            status_error(control.get(1).copied().unwrap_or(1), "SCC degree build")?;
        }
        loop {
            loop {
                ensure_graph_execution(cancellation, deadline)?;
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccTrimSelect,
                    })
                    .map_err(candle_error)?;
                let trim_control = read_u32_words(&workspace, control_offset, 4)?;
                status_error(trim_control.get(1).copied().unwrap_or(1), "SCC")?;
                if trim_control.get(3).copied().unwrap_or(0) == 0 {
                    break;
                }
                for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                    ensure_graph_execution(cancellation, deadline)?;
                    let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                    workspace = workspace
                        .apply_op1_no_bwd(&GraphCommand {
                            graph: graph.clone(),
                            kind: CommandKind::SccTrimEdgeChunk {
                                start: u32::try_from(start).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC trim chunk start exceeds u32",
                                    )
                                })?,
                                span: u32::try_from(span).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC trim chunk length exceeds u32",
                                    )
                                })?,
                            },
                        })
                        .map_err(candle_error)?;
                    let control = read_u32_words(&workspace, control_offset, 2)?;
                    status_error(control.get(1).copied().unwrap_or(1), "SCC trim")?;
                }
            }
            let after_trim = read_u32_words(&workspace, control_offset, 4)?;
            let remaining = after_trim.get(2).copied().unwrap_or(visible_count);
            if remaining == 0 {
                break;
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccCycleProbeInitialize,
                })
                .map_err(candle_error)?;
            for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
                let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccCycleProbeEdgeChunk {
                            start: u32::try_from(start).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "SCC cycle chunk start exceeds u32",
                                )
                            })?,
                            span: u32::try_from(span).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "SCC cycle chunk length exceeds u32",
                                )
                            })?,
                        },
                    })
                    .map_err(candle_error)?;
                let control = read_u32_words(&workspace, control_offset, 2)?;
                status_error(control.get(1).copied().unwrap_or(1), "SCC cycle probe")?;
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccCycleProbeValidate,
                })
                .map_err(candle_error)?;
            let cycle_probe = read_u32_words(&workspace, control_offset, 4)?;
            status_error(cycle_probe.get(1).copied().unwrap_or(1), "SCC cycle probe")?;
            if cycle_probe.get(3).copied().unwrap_or(1) == 0 {
                let node_count = u32::try_from(self.node_count).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "Metal SCC node count exceeds u32",
                    )
                })?;
                let rounds = if node_count <= 1 {
                    0
                } else {
                    u32::BITS - (node_count - 1).leading_zeros()
                };
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccCycleFinish { rounds },
                    })
                    .map_err(candle_error)?;
                let cycle_finish = read_u32_words(&workspace, control_offset, 4)?;
                status_error(
                    cycle_finish.get(1).copied().unwrap_or(1),
                    "SCC cycle pointer jumping",
                )?;
                if cycle_finish.get(2).copied().unwrap_or(remaining) != 0 {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "Metal SCC cycle pointer jumping left a node unassigned",
                    ));
                }
                ensure_graph_execution(cancellation, deadline)?;
                break;
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccColorInitialize,
                })
                .map_err(candle_error)?;
            loop {
                ensure_graph_execution(cancellation, deadline)?;
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccColorPrepare,
                    })
                    .map_err(candle_error)?;
                for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                    ensure_graph_execution(cancellation, deadline)?;
                    let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                    workspace = workspace
                        .apply_op1_no_bwd(&GraphCommand {
                            graph: graph.clone(),
                            kind: CommandKind::SccColorEdgeChunk {
                                start: u32::try_from(start).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC color chunk start exceeds u32",
                                    )
                                })?,
                                span: u32::try_from(span).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC color chunk length exceeds u32",
                                    )
                                })?,
                            },
                        })
                        .map_err(candle_error)?;
                    let control = read_u32_words(&workspace, control_offset, 2)?;
                    status_error(
                        control.get(1).copied().unwrap_or(1),
                        "SCC color propagation",
                    )?;
                }
                let control = read_u32_words(&workspace, control_offset, 2)?;
                status_error(
                    control.get(1).copied().unwrap_or(1),
                    "SCC color propagation",
                )?;
                if control.first().copied().unwrap_or(1) == 0 {
                    break;
                }
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccBackwardSeed,
                })
                .map_err(candle_error)?;
            loop {
                ensure_graph_execution(cancellation, deadline)?;
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccBackwardPrepare,
                    })
                    .map_err(candle_error)?;
                for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                    ensure_graph_execution(cancellation, deadline)?;
                    let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                    workspace = workspace
                        .apply_op1_no_bwd(&GraphCommand {
                            graph: graph.clone(),
                            kind: CommandKind::SccBackwardEdgeChunk {
                                start: u32::try_from(start).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC backward chunk start exceeds u32",
                                    )
                                })?,
                                span: u32::try_from(span).map_err(|_| {
                                    Error::new(
                                        ErrorCode::ResultBudgetExceeded,
                                        "SCC backward chunk length exceeds u32",
                                    )
                                })?,
                            },
                        })
                        .map_err(candle_error)?;
                    let control = read_u32_words(&workspace, control_offset, 2)?;
                    status_error(
                        control.get(1).copied().unwrap_or(1),
                        "SCC backward extraction",
                    )?;
                }
                let control = read_u32_words(&workspace, control_offset, 2)?;
                status_error(
                    control.get(1).copied().unwrap_or(1),
                    "SCC backward extraction",
                )?;
                if control.first().copied().unwrap_or(1) == 0 {
                    break;
                }
            }
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::SccColorAssign,
                })
                .map_err(candle_error)?;
            let assigned = read_u32_words(&workspace, control_offset, 4)?;
            status_error(assigned.get(1).copied().unwrap_or(1), "SCC")?;
            if assigned.get(3).copied().unwrap_or(0) == 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal SCC color extraction made no forward progress",
                ));
            }
            for start in (0..graph.adjacency_count).step_by(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
                let span = (graph.adjacency_count - start).min(METRICS_EDGE_CHUNK);
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::SccTrimEdgeChunk {
                            start: u32::try_from(start).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "SCC trim chunk start exceeds u32",
                                )
                            })?,
                            span: u32::try_from(span).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "SCC trim chunk length exceeds u32",
                                )
                            })?,
                        },
                    })
                    .map_err(candle_error)?;
                let control = read_u32_words(&workspace, control_offset, 2)?;
                status_error(control.get(1).copied().unwrap_or(1), "SCC trim")?;
            }
        }
        let layout = ComponentCanonicalLayout::scc(self.node_count).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal SCC canonical workspace size overflow",
            )
        })?;
        let component = canonical_component_ids(
            &graph,
            selected_rows,
            workspace,
            layout,
            cancellation,
            deadline,
            "SCC",
        )?;
        Ok(ResidentGraphProcedureResult::Components {
            node_rows,
            component,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn metal_undirected_metrics_workspace(
        &self,
        visible_mask: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
        publish_clustering: bool,
    ) -> Result<Option<(Tensor, MetricsLayout)>> {
        ensure_graph_execution(cancellation, deadline)?;
        let Some(graph) = GraphImage::from_resident(self, visible_mask, layers)? else {
            return Ok(None);
        };
        let unique = super::graph_louvain::build_unique_undirected_csr(
            self,
            visible_mask,
            layers,
            cancellation,
            deadline,
        )?;
        let graph = graph.with_unique_undirected(unique);
        let layout = MetricsLayout::new(self.node_count).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal metrics workspace layout overflow",
            )
        })?;
        let words = layout.words;
        let partial_count = self.node_count.div_ceil(THREADS);
        ensure_graph_execution(cancellation, deadline)?;
        let mut workspace = Tensor::zeros(words, DType::U32, visible_mask.device())
            .map_err(candle_error)?
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::MetricsInitialize,
            })
            .map_err(candle_error)?;
        let synchronize_control = |workspace: &Tensor| -> Result<u32> {
            let value = read_u32_words(workspace, layout.status, 2)?;
            status_error(value.first().copied().unwrap_or(1), "undirected metrics")?;
            Ok(value.get(1).copied().unwrap_or(0))
        };
        synchronize_control(&workspace)?;
        for start in (0..graph.node_count).step_by(METRICS_EDGE_CHUNK) {
            ensure_graph_execution(cancellation, deadline)?;
            let span = (graph.node_count - start).min(METRICS_EDGE_CHUNK);
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::MetricsDegreeChunk {
                        start: u32::try_from(start).map_err(|_| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "metrics degree node chunk start exceeds u32",
                            )
                        })?,
                        span: u32::try_from(span).map_err(|_| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "metrics degree node chunk length exceeds u32",
                            )
                        })?,
                    },
                })
                .map_err(candle_error)?;
            synchronize_control(&workspace)?;
        }
        for start in (0..graph.undirected_adjacency_count).step_by(METRICS_EDGE_CHUNK) {
            let span = (graph.undirected_adjacency_count - start).min(METRICS_EDGE_CHUNK);
            let start = u32::try_from(start).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "metrics edge chunk start exceeds u32",
                )
            })?;
            let span = u32::try_from(span).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "metrics edge chunk length exceeds u32",
                )
            })?;
            let mut reset = true;
            loop {
                ensure_graph_execution(cancellation, deadline)?;
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::MetricsTriangleChunk { start, span, reset },
                    })
                    .map_err(candle_error)?;
                reset = false;
                if synchronize_control(&workspace)? == 0 {
                    break;
                }
            }
        }
        ensure_graph_execution(cancellation, deadline)?;
        workspace = workspace
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::MetricsTrianglePartials,
            })
            .map_err(candle_error)?;
        synchronize_control(&workspace)?;
        let mut reduction_count = partial_count;
        let mut input_second = false;
        while reduction_count > 1 {
            ensure_graph_execution(cancellation, deadline)?;
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::MetricsTriangleReduce {
                        input_count: u32::try_from(reduction_count).map_err(|_| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "Metal metrics reduction count exceeds u32",
                            )
                        })?,
                        input_second,
                    },
                })
                .map_err(candle_error)?;
            synchronize_control(&workspace)?;
            reduction_count = reduction_count.div_ceil(THREADS);
            input_second = !input_second;
        }
        ensure_graph_execution(cancellation, deadline)?;
        workspace = workspace
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::MetricsTriangleResult { input_second },
            })
            .map_err(candle_error)?;
        synchronize_control(&workspace)?;
        if publish_clustering {
            for start in (0..self.node_count).step_by(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
                let span = (self.node_count - start).min(METRICS_EDGE_CHUNK);
                workspace = workspace
                    .apply_op1_no_bwd(&GraphCommand {
                        graph: graph.clone(),
                        kind: CommandKind::MetricsClusteringChunk {
                            start: u32::try_from(start).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "Metal clustering node start exceeds u32",
                                )
                            })?,
                            span: u32::try_from(span).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "Metal clustering node span exceeds u32",
                                )
                            })?,
                        },
                    })
                    .map_err(candle_error)?;
                synchronize_control(&workspace)?;
            }
        }
        ensure_graph_execution(cancellation, deadline)?;
        Ok(Some((workspace, layout)))
    }

    pub fn metal_triangle_count(
        &self,
        visible_mask: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        let Some((workspace, layout)) = self.metal_undirected_metrics_workspace(
            visible_mask,
            layers,
            cancellation,
            deadline,
            false,
        )?
        else {
            return Ok(ResidentGraphProcedureResult::TriangleCount { count: 0 });
        };
        let words = read_u32_words(&workspace, layout.result, 2)?;
        let count = u64::from(words[0]) | (u64::from(words[1]) << 32);
        ensure_graph_execution(cancellation, deadline)?;
        Ok(ResidentGraphProcedureResult::TriangleCount { count })
    }

    pub fn metal_clustering_coefficient(
        &self,
        visible_mask: &Tensor,
        selected_rows: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        ensure_graph_execution(cancellation, deadline)?;
        let node_rows = visible_rows(selected_rows)?;
        let Some((workspace, _layout)) = self.metal_undirected_metrics_workspace(
            visible_mask,
            layers,
            cancellation,
            deadline,
            true,
        )?
        else {
            return Ok(ResidentGraphProcedureResult::ClusteringCoefficient {
                coefficient: vec![0.0; node_rows.len()],
                node_rows,
            });
        };
        let packet_words = self.node_count.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal clustering packet width overflow",
            )
        })?;
        let compact = workspace
            .narrow(0, 0, packet_words)
            .and_then(|values| values.reshape((self.node_count, 2)))
            .and_then(|values| values.index_select(selected_rows, 0))
            .and_then(|values| values.flatten_all())
            .and_then(|values| values.to_vec1::<u32>())
            .map_err(candle_error)?;
        let mut coefficient = Vec::with_capacity(node_rows.len());
        for (index, pair) in compact.chunks_exact(2).enumerate() {
            if index.is_multiple_of(METRICS_EDGE_CHUNK) {
                ensure_graph_execution(cancellation, deadline)?;
            }
            let bits = u64::from(pair[0]) | (u64::from(pair[1]) << 32);
            let value = f64::from_bits(bits);
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal clustering coefficient produced an invalid binary64 value",
                ));
            }
            coefficient.push(value);
        }
        Ok(ResidentGraphProcedureResult::ClusteringCoefficient {
            node_rows,
            coefficient,
        })
    }

    #[allow(clippy::too_many_lines)]
    pub fn metal_k_core(
        &self,
        visible_mask: &Tensor,
        selected_rows: &Tensor,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        ensure_graph_execution(cancellation, deadline)?;
        let node_rows = visible_rows(selected_rows)?;
        ensure_graph_execution(cancellation, deadline)?;
        if node_rows.is_empty() {
            return Ok(ResidentGraphProcedureResult::KCore {
                node_rows,
                core: Vec::new(),
            });
        }
        let Some(graph) = GraphImage::from_resident(self, visible_mask, layers)? else {
            return Ok(ResidentGraphProcedureResult::KCore {
                core: vec![0; node_rows.len()],
                node_rows,
            });
        };
        let unique = super::graph_louvain::build_unique_undirected_csr(
            self,
            visible_mask,
            layers,
            cancellation,
            deadline,
        )?;
        let graph = graph.with_unique_undirected(unique);
        let visible_count = u32::try_from(node_rows.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "visible node count exceeds u32",
            )
        })?;
        let words = CommandKind::KCoreInitialize { visible_count }
            .workspace_words(self.node_count)
            .map_err(candle_error)?;
        let mut workspace = Tensor::zeros(words, DType::U32, visible_mask.device())
            .map_err(candle_error)?
            .apply_op1_no_bwd(&GraphCommand {
                graph: graph.clone(),
                kind: CommandKind::KCoreInitialize { visible_count },
            })
            .map_err(candle_error)?;
        let control_offset = 5 * self.node_count;
        let status = read_u32_words(&workspace, control_offset + 3, 1)?;
        status_error(status.first().copied().unwrap_or(1), "k-core")?;
        let mut current = 0_u32;
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            workspace = workspace
                .apply_op1_no_bwd(&GraphCommand {
                    graph: graph.clone(),
                    kind: CommandKind::KCoreSeed { core: current },
                })
                .map_err(candle_error)?;
            let control = read_u32_words(&workspace, control_offset, 6)?;
            status_error(control.get(3).copied().unwrap_or(1), "k-core")?;
            let alive = control.first().copied().unwrap_or(visible_count);
            if alive == 0 {
                break;
            }
            let tail = control.get(5).copied().unwrap_or(0);
            if tail != 0 {
                loop {
                    ensure_graph_execution(cancellation, deadline)?;
                    workspace = workspace
                        .apply_op1_no_bwd(&GraphCommand {
                            graph: graph.clone(),
                            kind: CommandKind::KCoreDrainPrepare,
                        })
                        .map_err(candle_error)?;
                    let mut start = 0_u32;
                    loop {
                        let state = read_u32_words(&workspace, control_offset, 6)?;
                        status_error(state.get(3).copied().unwrap_or(1), "k-core")?;
                        let dynamic_tail = state.get(5).copied().unwrap_or(0);
                        if start >= dynamic_tail {
                            break;
                        }
                        ensure_graph_execution(cancellation, deadline)?;
                        let span = dynamic_tail
                            .saturating_sub(start)
                            .min(u32::try_from(KCORE_TICKET_CHUNK).unwrap_or(u32::MAX));
                        workspace = workspace
                            .apply_op1_no_bwd(&GraphCommand {
                                graph: graph.clone(),
                                kind: CommandKind::KCoreDrainChunk {
                                    core: current,
                                    start,
                                    span,
                                },
                            })
                            .map_err(candle_error)?;
                        start = start.checked_add(span).ok_or_else(|| {
                            Error::new(
                                ErrorCode::ResultBudgetExceeded,
                                "k-core ticket cursor overflow",
                            )
                        })?;
                    }
                    let drained = read_u32_words(&workspace, control_offset, 6)?;
                    status_error(drained.get(3).copied().unwrap_or(1), "k-core")?;
                    if drained.get(1).copied().unwrap_or(1) == 0 {
                        break;
                    }
                }
                continue;
            }
            let minimum = control.get(2).copied().unwrap_or(UNREACHED);
            if minimum == UNREACHED || minimum <= current {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "Metal k-core peeling made no forward progress: core={current}, minimum={minimum}, alive={alive}, tail={tail}"
                    ),
                ));
            }
            current = minimum;
        }
        let core = workspace
            .narrow(0, 2 * self.node_count, self.node_count)
            .and_then(|values| values.index_select(selected_rows, 0))
            .and_then(|values| values.to_vec1::<u32>())
            .map_err(candle_error)?;
        ensure_graph_execution(cancellation, deadline)?;
        Ok(ResidentGraphProcedureResult::KCore { node_rows, core })
    }
}

#[cfg(test)]
#[derive(Clone, Debug)]
struct MetalRatioProbe;

#[cfg(test)]
impl CustomOp1 for MetalRatioProbe {
    fn name(&self) -> &'static str {
        "irongraph-metal-clustering-ratio-probe"
    }

    fn cpu_fwd(
        &self,
        _pairs: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal clustering ratio probe cannot execute on CPU".to_owned(),
        ))
    }

    fn metal_fwd(
        &self,
        pairs: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        if pairs.dtype() != DType::U32
            || !layout.is_contiguous()
            || layout.dims().len() != 1
            || !layout.shape().elem_count().is_multiple_of(4)
        {
            return Err(candle_core::Error::Msg(
                "Metal clustering ratio probe contract is invalid".to_owned(),
            ));
        }
        let rows = layout.shape().elem_count() / 4;
        let output_words = rows.checked_mul(2).ok_or_else(|| {
            candle_core::Error::Msg("Metal clustering ratio output size overflow".to_owned())
        })?;
        let row_count = u32::try_from(rows).map_err(|_| {
            candle_core::Error::Msg("Metal clustering ratio row count exceeds u32".to_owned())
        })?;
        let device = pairs.device();
        let library = device
            .metal_device()
            .new_library_with_source(
                include_str!("../../../../kernels/metal/graph_components_metrics.metal"),
                None,
            )
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "compiling Metal clustering ratio probe failed: {error}"
                ))
            })?;
        let function = library
            .get_function("ig_cm_ratio_probe", None)
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "loading Metal clustering ratio probe failed: {error}"
                ))
            })?;
        let raw = device
            .metal_device()
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "creating Metal clustering ratio probe failed: {error:?}"
                ))
            })?;
        let pipeline = candle_metal_kernels::metal::ComputePipeline::new(raw);
        let output = device
            .new_buffer_builder()
            .with_size_for(output_words, DType::U32)
            .with_label("irongraph clustering ratio probe")
            .build()?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_input_buffer(
            0,
            Some(pairs.buffer()),
            layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, &row_count);
        encoder.dispatch_thread_groups(
            objc2_metal::MTLSize {
                width: rows.div_ceil(THREADS),
                height: 1,
                depth: 1,
            },
            objc2_metal::MTLSize {
                width: THREADS,
                height: 1,
                depth: 1,
            },
        );
        Ok((
            MetalStorage::new(output, device.clone(), output_words, DType::U32),
            Shape::from(output_words),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_sizes_are_linear_and_checked() -> crate::Result<()> {
        assert_eq!(
            CommandKind::WccCompress
                .workspace_words(7)
                .map_err(candle_error)?,
            23
        );
        assert_eq!(
            CommandKind::SccInitialize { visible_count: 7 }
                .workspace_words(7)
                .map_err(candle_error)?,
            39
        );
        assert_eq!(
            CommandKind::MetricsInitialize
                .workspace_words(7)
                .map_err(candle_error)?,
            32_797
        );
        assert_eq!(
            CommandKind::KCoreSeed { core: 0 }
                .workspace_words(7)
                .map_err(candle_error)?,
            41
        );
        assert_eq!(KCORE_ROW_SCAN_QUANTUM, 128);
        assert!(
            CommandKind::SccInitialize { visible_count: 1 }
                .workspace_words(usize::MAX)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn metrics_scratch_includes_the_fixed_cancellation_cursor_bank() -> crate::Result<()> {
        assert_eq!(METRICS_EDGE_CHUNK, 16_384);
        assert_eq!(METRICS_TRIANGLE_SCAN_QUANTUM, 128);
        assert_eq!(metrics_workspace_words(256), Some(33_544));
        assert!(metrics_scratch_bytes(256, 1_024)? >= 33_544 * 4);
        assert!(metrics_scratch_bytes(usize::MAX, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn pagerank_scratch_uses_the_procedure_peak_and_checks_overflow() -> crate::Result<()> {
        let layout = super::super::metal_graph_pagerank_layout(256).map_err(candle_error)?;
        // cursor V + rank_a 2V (fp64 double-word, aligned) + rank_b 2V + scalars 2V + control 4 +
        // publication 4 + published-rank 2V = 9V + 8.
        assert_eq!(layout.words, 9 * 256 + 8);
        // The fp64 rank banks push the workspace (2312 words -> 9248 bytes -> 16384 pool class) so
        // the procedure peak now dominates the shared-selection peak (12_544 for V=256):
        //   retained  = pool(256*1) + pool(256*4)       = 256 + 1_024        = 1_280
        //   workspace = next_pow2(9_248)                                     = 16_384
        //   full_rank = pool(256*8) counted twice        = 2_048 + 2_048     = 4_096
        //   procedure = 1_280 + 16_384 + 4_096                               = 21_760  (> 12_544)
        assert_eq!(
            super::super::metal_pagerank_scratch_bytes(256)?,
            1_280 + 16_384 + 2 * 2_048
        );
        assert!(super::super::metal_pagerank_scratch_bytes(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn dedicated_metal_source_declares_every_pipeline() {
        let source = include_str!("../../../../kernels/metal/graph_components_metrics.metal");
        for name in [
            "ig_cm_wcc_initialize",
            "ig_cm_wcc_relax_edges",
            "ig_cm_wcc_compress",
            "ig_cm_component_canonical_initialize",
            "ig_cm_component_canonical_mark",
            "ig_cm_component_canonical_flags",
            "ig_cm_scc_initialize",
            "ig_cm_scc_initial_degree_edges",
            "ig_cm_scc_trim_mark",
            "ig_cm_scc_assign_candidates",
            "ig_cm_scc_trim_decrement_edges",
            "ig_cm_scc_cycle_probe_initialize",
            "ig_cm_scc_cycle_probe_edges",
            "ig_cm_scc_cycle_probe_validate",
            "ig_cm_scc_cycle_jump",
            "ig_cm_scc_cycle_assign",
            "ig_cm_scc_color_initialize",
            "ig_cm_scc_color_edges",
            "ig_cm_scc_backward_seed",
            "ig_cm_scc_backward_edges",
            "ig_cm_scc_color_assign",
            "ig_cm_degree_prepare",
            "ig_cm_degree_csr",
            "ig_cm_degree_edges",
            "ig_cm_undirected_prepare",
            "ig_cm_undirected_degree_edges",
            "ig_cm_triangle_cursor_prepare",
            "ig_cm_triangle_oriented_edges",
            "ig_cm_triangle_tiles",
            "ig_cm_u64_reduce_tiles",
            "ig_cm_triangle_finalize",
            "ig_cm_clustering_publish",
            "ig_cm_ratio_probe",
            "ig_cm_kcore_initialize",
            "ig_cm_kcore_prepare",
            "ig_cm_kcore_seed",
            "ig_cm_kcore_drain_prepare",
            "ig_cm_kcore_drain",
        ] {
            assert!(
                source.contains(&format!("kernel void {name}")),
                "missing {name}"
            );
        }
        for obsolete in [
            "ig_cm_scc_find_pivot",
            "ig_cm_scc_seed",
            "ig_cm_scc_reach",
            "ig_cm_scc_assign",
            "ig_cm_scc_set_unreached",
            "ig_cm_group_min_edge",
            "ig_cm_has_undirected_edge",
            "ig_cm_undirected_representative",
            "ig_cm_kcore_initial_degree_edges",
        ] {
            assert!(
                !source.contains(&format!("kernel void {obsolete}(")),
                "obsolete serial SCC pipeline remains: {obsolete}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_clustering_ratio_matches_rust_binary64_for_boundaries_and_random_pairs()
    -> crate::Result<()> {
        let _guard = crate::metal_test_guard();
        let Some(device) = crate::metal_test_device() else {
            return Ok(());
        };
        let mut pairs = vec![
            (0_u64, 1_u64),
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 3),
            ((1_u64 << 52) - 1, 1_u64 << 52),
            (1_u64 << 52, (1_u64 << 52) + 1),
            ((1_u64 << 53) - 1, 1_u64 << 53),
            (1_u64 << 53, (1_u64 << 53) + 1),
            (i64::MAX as u64 - 1, i64::MAX as u64),
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        while pairs.len() < 20_000 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let denominator = (state & i64::MAX as u64).max(1);
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let numerator = state % denominator.saturating_add(1);
            pairs.push((numerator, denominator));
        }
        let mut words = Vec::with_capacity(pairs.len() * 4);
        for (numerator, denominator) in &pairs {
            words.extend([
                *numerator as u32,
                (*numerator >> 32) as u32,
                *denominator as u32,
                (*denominator >> 32) as u32,
            ]);
        }
        let output = Tensor::from_slice(&words, words.len(), &device)
            .map_err(candle_error)?
            .apply_op1_no_bwd(&MetalRatioProbe)
            .and_then(|values| values.to_vec1::<u32>())
            .map_err(candle_error)?;
        for (row, (numerator, denominator)) in pairs.into_iter().enumerate() {
            let actual = u64::from(output[row * 2]) | (u64::from(output[row * 2 + 1]) << 32);
            let expected = ((numerator as f64) / (denominator as f64)).to_bits();
            assert_eq!(
                actual, expected,
                "ratio mismatch for {numerator}/{denominator}"
            );
        }
        Ok(())
    }
}
