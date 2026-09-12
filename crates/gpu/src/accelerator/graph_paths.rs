//! Exact Metal-resident DFS and shortest-path procedures.
//!
//! This module is deliberately separate from the generic tensor operators: these algorithms own
//! persistent, procedure-specific workspaces and compile a small dedicated Metal library.

#![cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]

use std::{
    mem::size_of,
    sync::{Arc, OnceLock},
    time::Instant,
};

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Device, Layout, MetalStorage, Shape, Storage, Tensor,
    backend::BackendStorage,
};
use candle_metal_kernels::metal::ComputePipeline;
use objc2_metal::MTLDevice;
use tokio_util::sync::CancellationToken;

use crate::{
    Error, ErrorCode, Result,
    execution::{ResidentGraphProcedureResult, ensure_graph_execution},
    graph::LayerMask,
    types::PropertyId,
};

use super::{
    CandleResident, MetalGraphBfsChunk, MetalGraphBfsInitialize, candle_error, checked_u32,
};

const PATH_THREADS: usize = 256;
const PATH_CONTROL_WORDS: usize = 8;
pub const PATH_NODE_QUANTUM: usize = 262_144;
const DFS_QUANTUM: usize = 16_384;
const SHORTEST_PATH_QUANTUM: usize = 4_096;
// Bound one persistent dispatch to at most 64M worst-case adjacency visits. This remains a hard
// cancellation quantum while amortizing command submission and bounded control readback on regular
// low-degree graphs. Sparse degree-3..8 rows use eight lanes: enough to consume their narrow
// same-depth frontier in parallel without making every sequential level pay a 256-lane barrier.
const BFS_PERSISTENT_EDGE_QUANTUM: usize = 1 << 26;
const BFS_PERSISTENT_NARROW_THREADS: usize = 1;
const BFS_PERSISTENT_NARROW_MAX_EDGES_PER_NODE: usize = 2;
const BFS_PERSISTENT_SPARSE_THREADS: usize = 8;
const BFS_PERSISTENT_SPARSE_MAX_EDGES_PER_NODE: usize = 8;
pub const BFS_ADAPTIVE_PROBE_LEVELS: usize = 8;
const BFS_PERSISTENT_MAX_FRONTIER: usize = 32;
const BFS_PERSISTENT_MAX_EDGES_PER_NODE: usize = 8;
const DIJKSTRA_ADAPTIVE_PROBE_ROUNDS: usize = 8;
const DIJKSTRA_HEAP_TILE_QUANTUM: usize = 2_048;
const DIJKSTRA_HEAP_MAX_FRONTIER: usize = 32;
const DIJKSTRA_HEAP_MAX_EDGES_PER_NODE: usize = 8;

macro_rules! metal_input {
    ($owner:expr, $field:ident, $guard:ident, $layout:ident, $storage:ident) => {
        let ($guard, $layout) = $owner.$field.storage_and_layout();
        let Storage::Metal($storage) = &*$guard else {
            return Err(candle_core::Error::Msg(
                concat!("Metal path input moved off device: ", stringify!($field)).to_owned(),
            ));
        };
    };
}

#[derive(Clone)]
struct PathPipelines {
    #[cfg(test)]
    arithmetic_probe: ComputePipeline,
    dfs_initialize: ComputePipeline,
    dfs_chunk: ComputePipeline,
    shortest_initialize: ComputePipeline,
    shortest_chunk: ComputePipeline,
    bfs_persistent_initialize: ComputePipeline,
    bfs_persistent_chunk: ComputePipeline,
    dijkstra_initialize: ComputePipeline,
    dijkstra_heap_initialize: ComputePipeline,
    dijkstra_heap_chunk: ComputePipeline,
    dijkstra_prepare: ComputePipeline,
    dijkstra_relax: ComputePipeline,
    dijkstra_publish: ComputePipeline,
    dijkstra_finalize_prepare: ComputePipeline,
    dijkstra_finalize: ComputePipeline,
}

fn path_pipelines(device: &candle_core::MetalDevice) -> candle_core::Result<PathPipelines> {
    static PIPELINES: OnceLock<PathPipelines> = OnceLock::new();
    if let Some(pipelines) = PIPELINES.get() {
        return Ok(pipelines.clone());
    }
    let library = device
        .metal_device()
        .new_library_with_source(
            include_str!("../../../../kernels/metal/graph_paths.metal"),
            None,
        )
        .map_err(|error| {
            candle_core::Error::Msg(format!("compiling Metal path kernels failed: {error}"))
        })?;
    let pipeline = |name: &str| -> candle_core::Result<_> {
        let function = library.get_function(name, None).map_err(|error| {
            candle_core::Error::Msg(format!("loading Metal path kernel {name} failed: {error}"))
        })?;
        let raw = device
            .metal_device()
            .as_ref()
            .newComputePipelineStateWithFunction_error(function.as_ref())
            .map_err(|error| {
                candle_core::Error::Msg(format!(
                    "creating Metal path pipeline {name} failed: {error:?}"
                ))
            })?;
        Ok(ComputePipeline::new(raw))
    };
    let pipelines = PathPipelines {
        #[cfg(test)]
        arithmetic_probe: pipeline("ig_path_arithmetic_probe")?,
        dfs_initialize: pipeline("ig_path_dfs_initialize")?,
        dfs_chunk: pipeline("ig_path_dfs_chunk")?,
        shortest_initialize: pipeline("ig_path_shortest_initialize")?,
        shortest_chunk: pipeline("ig_path_shortest_chunk")?,
        bfs_persistent_initialize: pipeline("ig_path_bfs_persistent_initialize")?,
        bfs_persistent_chunk: pipeline("ig_path_bfs_persistent_chunk")?,
        dijkstra_initialize: pipeline("ig_path_dijkstra_initialize")?,
        dijkstra_heap_initialize: pipeline("ig_path_dijkstra_heap_initialize")?,
        dijkstra_heap_chunk: pipeline("ig_path_dijkstra_heap_chunk")?,
        dijkstra_prepare: pipeline("ig_path_dijkstra_prepare")?,
        dijkstra_relax: pipeline("ig_path_dijkstra_relax")?,
        dijkstra_publish: pipeline("ig_path_dijkstra_publish")?,
        dijkstra_finalize_prepare: pipeline("ig_path_dijkstra_finalize_prepare")?,
        dijkstra_finalize: pipeline("ig_path_dijkstra_finalize")?,
    };
    for candidate in [
        &pipelines.dfs_initialize,
        &pipelines.dfs_chunk,
        &pipelines.shortest_initialize,
        &pipelines.shortest_chunk,
        &pipelines.bfs_persistent_initialize,
        &pipelines.bfs_persistent_chunk,
        &pipelines.dijkstra_initialize,
        &pipelines.dijkstra_heap_initialize,
        &pipelines.dijkstra_heap_chunk,
        &pipelines.dijkstra_prepare,
        &pipelines.dijkstra_relax,
        &pipelines.dijkstra_publish,
        &pipelines.dijkstra_finalize_prepare,
        &pipelines.dijkstra_finalize,
    ] {
        if candidate.max_total_threads_per_threadgroup() < PATH_THREADS {
            return Err(candle_core::Error::Msg(
                "selected Metal device cannot run 256-thread path groups".to_owned(),
            ));
        }
    }
    #[cfg(test)]
    if pipelines
        .arithmetic_probe
        .max_total_threads_per_threadgroup()
        < PATH_THREADS
    {
        return Err(candle_core::Error::Msg(
            "selected Metal device cannot run the 256-thread path arithmetic probe".to_owned(),
        ));
    }
    let _ = PIPELINES.set(pipelines.clone());
    Ok(PIPELINES.get().cloned().unwrap_or(pipelines))
}

/// Compile and cache every exact path pipeline during Metal backend preparation. Traversal queries
/// therefore never pay shader compilation or pipeline-state construction on their latency path.
pub fn prepare(device: &Device) -> Result<()> {
    let Device::Metal(device) = device else {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "Metal path pipeline preparation requires a Metal device",
        ));
    };
    path_pipelines(device).map_err(candle_error)?;
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct PathWorkspaceAllocate {
    words: usize,
    label: &'static str,
}

impl CustomOp1 for PathWorkspaceAllocate {
    fn name(&self) -> &'static str {
        "irongraph-metal-path-workspace-allocate"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal path workspace allocation cannot execute on CPU".to_owned(),
        ))
    }

    fn metal_fwd(
        &self,
        input: &MetalStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        if self.words == 0 {
            return Err(candle_core::Error::Msg(
                "Metal path workspace allocation is empty".to_owned(),
            ));
        }
        let device = input.device();
        let workspace = device
            .new_buffer_builder()
            .with_size_for(self.words, DType::U32)
            .with_label(self.label)
            .build()?;
        Ok((
            MetalStorage::new(workspace, device.clone(), self.words, DType::U32),
            Shape::from(self.words),
        ))
    }
}

/// Reserve an uninitialized private Metal workspace. Every caller must fill all result and control
/// lanes through bounded initialization kernels before observing the tensor. Keeping allocation
/// separate from initialization lets the host synchronize and honor cancellation between tiles.
pub fn allocate_u32_workspace(input: &Tensor, words: usize, label: &'static str) -> Result<Tensor> {
    input
        .apply_op1_no_bwd(&PathWorkspaceAllocate { words, label })
        .map_err(candle_error)
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DfsArgs {
    node_count: u32,
    edge_count: u32,
    adjacency_count: u32,
    source: u32,
    layer_mask: u32,
    quantum: u32,
    threadgroup_width: u32,
    node_begin: u32,
    overlay_count: u32,
}

impl DfsArgs {
    fn new(
        node_count: usize,
        edge_count: usize,
        adjacency_count: usize,
        source: u32,
        layer_mask: u32,
        quantum: usize,
        node_begin: usize,
        overlay_count: usize,
    ) -> candle_core::Result<Self> {
        validate_u32_workspace_words(node_count, 4, PATH_CONTROL_WORDS, "DFS")?;
        Ok(Self {
            node_count: u32::try_from(node_count).map_err(|_| {
                candle_core::Error::Msg("Metal DFS node count exceeds u32".to_owned())
            })?,
            edge_count: u32::try_from(edge_count).map_err(|_| {
                candle_core::Error::Msg("Metal DFS edge count exceeds u32".to_owned())
            })?,
            adjacency_count: u32::try_from(adjacency_count).map_err(|_| {
                candle_core::Error::Msg("Metal DFS adjacency count exceeds u32".to_owned())
            })?,
            source,
            layer_mask,
            quantum: u32::try_from(quantum)
                .map_err(|_| candle_core::Error::Msg("Metal DFS quantum exceeds u32".to_owned()))?,
            threadgroup_width: checked_candle_u32(
                persistent_bfs_threadgroup_width(node_count, adjacency_count),
                "DFS threadgroup width",
            )?,
            node_begin: checked_candle_u32(node_begin, "DFS node tile start")?,
            overlay_count: checked_candle_u32(overlay_count, "DFS overlay row count")?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct DfsInitialize {
    node_count: usize,
    edge_count: usize,
    source: u32,
    layer_mask: u32,
    words: usize,
    node_begin: usize,
    node_end: usize,
}

impl CustomOp2 for DfsInitialize {
    fn name(&self) -> &'static str {
        "irongraph-metal-dfs-initialize"
    }

    fn cpu_fwd(
        &self,
        _visible: &CpuStorage,
        _visible_layout: &Layout,
        _workspace: &CpuStorage,
        _workspace_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal DFS initialization cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        visible: &MetalStorage,
        visible_layout: &Layout,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        if self.node_count == 0
            || self.source as usize >= self.node_count
            || self.layer_mask == 0
            || self.node_begin >= self.node_end
            || self.node_end > self.node_count
            || visible.dtype() != DType::U8
            || workspace.dtype() != DType::U32
            || !visible_layout.is_contiguous()
            || !workspace_layout.is_contiguous()
            || visible_layout.dims().len() != 1
            || workspace_layout.dims().len() != 1
            || visible_layout.shape().elem_count() != self.node_count
            || workspace_layout.shape().elem_count() != self.words
        {
            return Err(candle_core::Error::Msg(
                "Metal DFS initialization contract is invalid".to_owned(),
            ));
        }
        let args = DfsArgs::new(
            self.node_count,
            self.edge_count,
            0,
            self.source,
            self.layer_mask,
            0,
            self.node_begin,
            0,
        )?;
        let device = visible.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.dfs_initialize);
        encoder.set_input_buffer(
            0,
            Some(visible.buffer()),
            visible_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder.set_output_buffer(
            1,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(2, &args);
        dispatch_rows(encoder, self.node_end - self.node_begin);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct DfsChunk {
    outgoing_offsets: Tensor,
    outgoing_neighbors: Tensor,
    outgoing_edges: Tensor,
    outgoing_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    node_count: usize,
    edge_count: usize,
    adjacency_count: usize,
    outgoing_overlay_count: usize,
    source: u32,
    layer_mask: u32,
    quantum: usize,
}

impl CustomOp1 for DfsChunk {
    fn name(&self) -> &'static str {
        "irongraph-metal-dfs-chunk"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal DFS chunk cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            outgoing_offsets,
            offsets_guard,
            outgoing_offsets_layout,
            offsets
        );
        metal_input!(
            self,
            outgoing_neighbors,
            neighbors_guard,
            outgoing_neighbors_layout,
            neighbors
        );
        metal_input!(
            self,
            outgoing_edges,
            edges_guard,
            outgoing_edges_layout,
            edges
        );
        metal_input!(
            self,
            outgoing_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(
            self,
            visible_nodes,
            visible_guard,
            visible_nodes_layout,
            visible
        );
        metal_input!(self, edge_active, active_guard, edge_active_layout, active);
        metal_input!(self, edge_layers, layers_guard, edge_layers_layout, layers);
        let words = self.node_count * 4 + PATH_CONTROL_WORDS;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            words,
            "DFS workspace",
        )?;
        validate_bound_vector(
            &self.outgoing_offsets,
            outgoing_offsets_layout,
            DType::U32,
            self.node_count + 1,
            "DFS offsets",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.outgoing_neighbors,
                &outgoing_neighbors_layout,
                DType::U32,
                self.adjacency_count,
                "DFS neighbors",
            ),
            (
                &self.outgoing_edges,
                &outgoing_edges_layout,
                DType::U32,
                self.adjacency_count,
                "DFS edge rows",
            ),
            (
                &self.visible_nodes,
                &visible_nodes_layout,
                DType::U8,
                self.node_count,
                "DFS visible mask",
            ),
            (
                &self.edge_active,
                &edge_active_layout,
                DType::U8,
                self.edge_count,
                "DFS edge active",
            ),
            (
                &self.edge_layers,
                &edge_layers_layout,
                DType::U8,
                self.edge_count,
                "DFS edge layers",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.outgoing_overlay,
            overlay_layout,
            self.outgoing_overlay_count,
            "DFS overlay",
        )?;
        let args = DfsArgs::new(
            self.node_count,
            self.edge_count,
            self.adjacency_count,
            self.source,
            self.layer_mask,
            self.quantum,
            0,
            self.outgoing_overlay_count,
        )?;
        if !matches!(
            args.threadgroup_width as usize,
            BFS_PERSISTENT_NARROW_THREADS | BFS_PERSISTENT_SPARSE_THREADS | PATH_THREADS
        ) {
            return Err(candle_core::Error::Msg(
                "Metal DFS launch geometry is invalid".to_owned(),
            ));
        }
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.dfs_chunk);
        bind_tensor(encoder, 0, offsets, outgoing_offsets_layout, DType::U32);
        bind_tensor(encoder, 1, neighbors, outgoing_neighbors_layout, DType::U32);
        bind_tensor(encoder, 2, edges, outgoing_edges_layout, DType::U32);
        bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
        bind_tensor(encoder, 4, visible, visible_nodes_layout, DType::U8);
        bind_tensor(encoder, 5, active, edge_active_layout, DType::U8);
        bind_tensor(encoder, 6, layers, edge_layers_layout, DType::U8);
        encoder.set_output_buffer(
            7,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(8, &args);
        dispatch_group_width(encoder, args.threadgroup_width as usize);
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

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ShortestArgs {
    node_count: u32,
    edge_count: u32,
    adjacency_count: u32,
    source: u32,
    target: u32,
    layer_mask: u32,
    distance: u32,
    quantum: u32,
    threadgroup_width: u32,
    overlay_count: u32,
}

impl ShortestArgs {
    #[allow(clippy::too_many_arguments)]
    fn new(
        node_count: usize,
        edge_count: usize,
        adjacency_count: usize,
        source: u32,
        target: u32,
        layer_mask: u32,
        distance: u32,
        quantum: usize,
        overlay_count: usize,
    ) -> candle_core::Result<Self> {
        validate_u32_workspace_words(node_count, 3, PATH_CONTROL_WORDS, "shortest-path BFS")?;
        Ok(Self {
            node_count: checked_candle_u32(node_count, "shortest-path node count")?,
            edge_count: checked_candle_u32(edge_count, "shortest-path edge count")?,
            adjacency_count: checked_candle_u32(adjacency_count, "shortest-path adjacency count")?,
            source,
            target,
            layer_mask,
            distance,
            quantum: checked_candle_u32(quantum, "shortest-path quantum")?,
            threadgroup_width: checked_candle_u32(
                persistent_bfs_threadgroup_width(node_count, adjacency_count),
                "shortest-path threadgroup width",
            )?,
            overlay_count: checked_candle_u32(overlay_count, "shortest-path overlay row count")?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct ShortestInitialize {
    args: ShortestArgs,
    words: usize,
    node_begin: usize,
    node_end: usize,
}

impl CustomOp1 for ShortestInitialize {
    fn name(&self) -> &'static str {
        "irongraph-metal-shortest-path-initialize"
    }

    fn cpu_fwd(
        &self,
        _packet: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal shortest-path initialization cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        packet: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        validate_vector(
            packet,
            layout,
            DType::U32,
            self.words,
            "shortest-path workspace",
        )?;
        if self.node_begin >= self.node_end || self.node_end > self.args.node_count as usize {
            return Err(candle_core::Error::Msg(
                "Metal shortest-path initialization node tile is invalid".to_owned(),
            ));
        }
        let device = packet.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.shortest_initialize);
        encoder.set_output_buffer(
            0,
            Some(packet.buffer()),
            layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(1, &self.args);
        let node_begin = checked_candle_u32(self.node_begin, "shortest-path node tile start")?;
        encoder.set_bytes(2, &node_begin);
        dispatch_rows(encoder, self.node_end - self.node_begin);
        Ok((
            MetalStorage::new(
                Arc::new(packet.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct ShortestChunk {
    outgoing_offsets: Tensor,
    outgoing_neighbors: Tensor,
    outgoing_edges: Tensor,
    outgoing_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    forward_workspace: Tensor,
    reverse_workspace: Tensor,
    args: ShortestArgs,
    words: usize,
}

impl CustomOp1 for ShortestChunk {
    fn name(&self) -> &'static str {
        "irongraph-metal-shortest-path-chunk"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal shortest-path chunk cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            outgoing_offsets,
            offsets_guard,
            outgoing_offsets_layout,
            offsets
        );
        metal_input!(
            self,
            outgoing_neighbors,
            neighbors_guard,
            outgoing_neighbors_layout,
            neighbors
        );
        metal_input!(
            self,
            outgoing_edges,
            edges_guard,
            outgoing_edges_layout,
            edges
        );
        metal_input!(
            self,
            outgoing_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(
            self,
            visible_nodes,
            visible_guard,
            visible_nodes_layout,
            visible
        );
        metal_input!(self, edge_active, active_guard, edge_active_layout, active);
        metal_input!(self, edge_layers, layers_guard, edge_layers_layout, layers);
        metal_input!(
            self,
            forward_workspace,
            forward_guard,
            forward_workspace_layout,
            forward
        );
        metal_input!(
            self,
            reverse_workspace,
            reverse_guard,
            reverse_workspace_layout,
            reverse
        );
        let nodes = self.args.node_count as usize;
        let edges_count = self.args.edge_count as usize;
        let adjacency = self.args.adjacency_count as usize;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "shortest-path workspace",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.outgoing_offsets,
                &outgoing_offsets_layout,
                DType::U32,
                nodes + 1,
                "shortest offsets",
            ),
            (
                &self.outgoing_neighbors,
                &outgoing_neighbors_layout,
                DType::U32,
                adjacency,
                "shortest neighbors",
            ),
            (
                &self.outgoing_edges,
                &outgoing_edges_layout,
                DType::U32,
                adjacency,
                "shortest edges",
            ),
            (
                &self.visible_nodes,
                &visible_nodes_layout,
                DType::U8,
                nodes,
                "shortest visible",
            ),
            (
                &self.edge_active,
                &edge_active_layout,
                DType::U8,
                edges_count,
                "shortest active",
            ),
            (
                &self.edge_layers,
                &edge_layers_layout,
                DType::U8,
                edges_count,
                "shortest layers",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.outgoing_overlay,
            overlay_layout,
            self.args.overlay_count as usize,
            "shortest-path overlay",
        )?;
        if !matches!(
            self.args.threadgroup_width as usize,
            BFS_PERSISTENT_NARROW_THREADS | BFS_PERSISTENT_SPARSE_THREADS | PATH_THREADS
        ) {
            return Err(candle_core::Error::Msg(
                "Metal shortest-path launch geometry is invalid".to_owned(),
            ));
        }
        let bfs_words = nodes * 3 + super::METAL_GRAPH_CONTROL_WORDS;
        validate_bound_vector(
            &self.forward_workspace,
            forward_workspace_layout,
            DType::U32,
            bfs_words,
            "shortest forward distances",
        )?;
        validate_bound_vector(
            &self.reverse_workspace,
            reverse_workspace_layout,
            DType::U32,
            bfs_words,
            "shortest reverse distances",
        )?;
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.shortest_chunk);
        bind_tensor(encoder, 0, offsets, outgoing_offsets_layout, DType::U32);
        bind_tensor(encoder, 1, neighbors, outgoing_neighbors_layout, DType::U32);
        bind_tensor(encoder, 2, edges, outgoing_edges_layout, DType::U32);
        bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
        bind_tensor(encoder, 4, visible, visible_nodes_layout, DType::U8);
        bind_tensor(encoder, 5, active, edge_active_layout, DType::U8);
        bind_tensor(encoder, 6, layers, edge_layers_layout, DType::U8);
        bind_tensor(encoder, 7, forward, forward_workspace_layout, DType::U32);
        bind_tensor(encoder, 8, reverse, reverse_workspace_layout, DType::U32);
        encoder.set_output_buffer(
            9,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(10, &self.args);
        dispatch_group_width(encoder, self.args.threadgroup_width as usize);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PersistentBfsArgs {
    node_count: u32,
    edge_count: u32,
    adjacency_count: u32,
    source: u32,
    target: u32,
    maximum_distance: u32,
    layer_mask: u32,
    quantum: u32,
    threadgroup_width: u32,
    node_begin: u32,
    overlay_count: u32,
}

impl PersistentBfsArgs {
    fn new(
        node_count: usize,
        edge_count: usize,
        adjacency_count: usize,
        source: u32,
        target: u32,
        maximum_distance: u32,
        layer_mask: u32,
        overlay_count: usize,
    ) -> candle_core::Result<Self> {
        validate_u32_workspace_words(node_count, 3, PATH_CONTROL_WORDS, "persistent BFS")?;
        let threadgroup_width = persistent_bfs_threadgroup_width(node_count, adjacency_count);
        let edges_per_transition =
            threadgroup_width.checked_mul(PATH_THREADS).ok_or_else(|| {
                candle_core::Error::Msg(
                    "persistent BFS edge-quantum calculation overflowed".to_owned(),
                )
            })?;
        let quantum = BFS_PERSISTENT_EDGE_QUANTUM
            .checked_div(edges_per_transition)
            .unwrap_or(0)
            .max(1);
        Ok(Self {
            node_count: checked_candle_u32(node_count, "persistent BFS node count")?,
            edge_count: checked_candle_u32(edge_count, "persistent BFS edge count")?,
            adjacency_count: checked_candle_u32(adjacency_count, "persistent BFS adjacency count")?,
            source,
            target,
            maximum_distance,
            layer_mask,
            quantum: checked_candle_u32(quantum, "persistent BFS quantum")?,
            threadgroup_width: checked_candle_u32(
                threadgroup_width,
                "persistent BFS threadgroup width",
            )?,
            node_begin: 0,
            overlay_count: checked_candle_u32(overlay_count, "persistent BFS overlay row count")?,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct PersistentBfsInitialize {
    args: PersistentBfsArgs,
    words: usize,
    node_begin: usize,
    node_end: usize,
}

impl CustomOp2 for PersistentBfsInitialize {
    fn name(&self) -> &'static str {
        "irongraph-metal-persistent-bfs-initialize"
    }

    fn cpu_fwd(
        &self,
        _visible: &CpuStorage,
        _visible_layout: &Layout,
        _workspace: &CpuStorage,
        _workspace_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal persistent BFS initialization cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        visible: &MetalStorage,
        visible_layout: &Layout,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        validate_vector(
            visible,
            visible_layout,
            DType::U8,
            self.args.node_count as usize,
            "persistent BFS visible mask",
        )?;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "persistent BFS workspace",
        )?;
        if self.node_begin >= self.node_end || self.node_end > self.args.node_count as usize {
            return Err(candle_core::Error::Msg(
                "Metal persistent BFS initialization node tile is invalid".to_owned(),
            ));
        }
        let device = visible.device();
        let mut args = self.args;
        args.node_begin = checked_candle_u32(self.node_begin, "persistent BFS node tile start")?;
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.bfs_persistent_initialize);
        encoder.set_input_buffer(
            0,
            Some(visible.buffer()),
            visible_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder.set_output_buffer(
            1,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(2, &args);
        dispatch_rows(encoder, self.node_end - self.node_begin);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct PersistentBfsChunk {
    outgoing_offsets: Tensor,
    outgoing_neighbors: Tensor,
    outgoing_edges: Tensor,
    outgoing_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    args: PersistentBfsArgs,
    words: usize,
}

impl CustomOp1 for PersistentBfsChunk {
    fn name(&self) -> &'static str {
        "irongraph-metal-persistent-bfs-chunk"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal persistent BFS chunk cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            outgoing_offsets,
            offsets_guard,
            offsets_layout,
            offsets
        );
        metal_input!(
            self,
            outgoing_neighbors,
            neighbors_guard,
            neighbors_layout,
            neighbors
        );
        metal_input!(self, outgoing_edges, edges_guard, edges_layout, edges);
        metal_input!(
            self,
            outgoing_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(self, visible_nodes, visible_guard, visible_layout, visible);
        metal_input!(self, edge_active, active_guard, active_layout, active);
        metal_input!(self, edge_layers, layers_guard, layers_layout, layers);
        let nodes = self.args.node_count as usize;
        let edge_count = self.args.edge_count as usize;
        let adjacency = self.args.adjacency_count as usize;
        let adjacency_storage = adjacency.max(1);
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "persistent BFS workspace",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.outgoing_offsets,
                &offsets_layout,
                DType::U32,
                nodes + 1,
                "persistent BFS offsets",
            ),
            (
                &self.outgoing_neighbors,
                &neighbors_layout,
                DType::U32,
                adjacency_storage,
                "persistent BFS neighbors",
            ),
            (
                &self.outgoing_edges,
                &edges_layout,
                DType::U32,
                adjacency_storage,
                "persistent BFS edge rows",
            ),
            (
                &self.visible_nodes,
                &visible_layout,
                DType::U8,
                nodes,
                "persistent BFS visible mask",
            ),
            (
                &self.edge_active,
                &active_layout,
                DType::U8,
                edge_count,
                "persistent BFS active mask",
            ),
            (
                &self.edge_layers,
                &layers_layout,
                DType::U8,
                edge_count,
                "persistent BFS layer column",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.outgoing_overlay,
            overlay_layout,
            self.args.overlay_count as usize,
            "persistent BFS overlay",
        )?;
        if self.args.quantum == 0
            || !matches!(
                self.args.threadgroup_width as usize,
                BFS_PERSISTENT_NARROW_THREADS | BFS_PERSISTENT_SPARSE_THREADS | PATH_THREADS
            )
        {
            return Err(candle_core::Error::Msg(
                "Metal persistent BFS launch geometry is invalid".to_owned(),
            ));
        }
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.bfs_persistent_chunk);
        bind_tensor(encoder, 0, offsets, offsets_layout, DType::U32);
        bind_tensor(encoder, 1, neighbors, neighbors_layout, DType::U32);
        bind_tensor(encoder, 2, edges, edges_layout, DType::U32);
        bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
        bind_tensor(encoder, 4, visible, visible_layout, DType::U8);
        bind_tensor(encoder, 5, active, active_layout, DType::U8);
        bind_tensor(encoder, 6, layers, layers_layout, DType::U8);
        encoder.set_output_buffer(
            7,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(8, &self.args);
        dispatch_group_width(encoder, self.args.threadgroup_width as usize);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DijkstraArgs {
    node_count: u32,
    edge_count: u32,
    adjacency_count: u32,
    source: u32,
    layer_mask: u32,
    weight_kind: u32,
    weight_byte_count: u32,
    quantum: u32,
    outgoing_overlay_count: u32,
    incoming_overlay_count: u32,
}

impl DijkstraArgs {
    fn new(
        node_count: usize,
        edge_count: usize,
        adjacency_count: usize,
        source: u32,
        layer_mask: u32,
        weight: &PathWeight,
        outgoing_overlay_count: usize,
        incoming_overlay_count: usize,
    ) -> candle_core::Result<Self> {
        validate_u32_workspace_words(node_count, 7, PATH_CONTROL_WORDS, "weighted Dijkstra")?;
        Ok(Self {
            node_count: checked_candle_u32(node_count, "Dijkstra node count")?,
            edge_count: checked_candle_u32(edge_count, "Dijkstra edge count")?,
            adjacency_count: checked_candle_u32(adjacency_count, "Dijkstra adjacency count")?,
            source,
            layer_mask,
            weight_kind: weight.kind,
            weight_byte_count: checked_candle_u32(
                weight.mixed_bytes.elem_count(),
                "Dijkstra mixed weight byte count",
            )?,
            quantum: checked_candle_u32(DIJKSTRA_HEAP_TILE_QUANTUM, "Dijkstra heap tile quantum")?,
            outgoing_overlay_count: checked_candle_u32(
                outgoing_overlay_count,
                "Dijkstra outgoing overlay row count",
            )?,
            incoming_overlay_count: checked_candle_u32(
                incoming_overlay_count,
                "Dijkstra incoming overlay row count",
            )?,
        })
    }
}

#[derive(Clone, Debug)]
struct PathWeight {
    kind: u32,
    homogeneous_values: Tensor,
    validity: Tensor,
    mixed_offsets: Tensor,
    mixed_bytes: Tensor,
}

#[derive(Clone, Copy, Debug)]
struct DijkstraInitialize {
    args: DijkstraArgs,
    words: usize,
    heap: bool,
    node_begin: usize,
    node_end: usize,
}

impl CustomOp2 for DijkstraInitialize {
    fn name(&self) -> &'static str {
        if self.heap {
            "irongraph-metal-weighted-dijkstra-heap-initialize"
        } else {
            "irongraph-metal-weighted-dijkstra-initialize"
        }
    }

    fn cpu_fwd(
        &self,
        _visible: &CpuStorage,
        _visible_layout: &Layout,
        _workspace: &CpuStorage,
        _workspace_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal weighted Dijkstra initialization cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        visible: &MetalStorage,
        visible_layout: &Layout,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        validate_vector(
            visible,
            visible_layout,
            DType::U8,
            self.args.node_count as usize,
            "Dijkstra visible mask",
        )?;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "Dijkstra workspace",
        )?;
        if self.node_begin >= self.node_end || self.node_end > self.args.node_count as usize {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra initialization node tile is invalid".to_owned(),
            ));
        }
        let device = visible.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(if self.heap {
            &pipelines.dijkstra_heap_initialize
        } else {
            &pipelines.dijkstra_initialize
        });
        encoder.set_input_buffer(
            0,
            Some(visible.buffer()),
            visible_layout.start_offset() * DType::U8.size_in_bytes(),
        );
        encoder.set_output_buffer(
            1,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(2, &self.args);
        let node_begin = checked_candle_u32(self.node_begin, "Dijkstra node tile start")?;
        encoder.set_bytes(3, &node_begin);
        dispatch_rows(encoder, self.node_end - self.node_begin);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct DijkstraChunk {
    incoming_offsets: Tensor,
    incoming_neighbors: Tensor,
    incoming_edges: Tensor,
    incoming_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    weight: PathWeight,
    args: DijkstraArgs,
    words: usize,
    first_round: usize,
    round_count: usize,
    edge_begin: usize,
    edge_end: usize,
    node_begin: usize,
    node_end: usize,
    prepare: bool,
    publish: bool,
}

impl CustomOp1 for DijkstraChunk {
    fn name(&self) -> &'static str {
        "irongraph-metal-weighted-dijkstra-chunk"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal weighted Dijkstra chunk cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            incoming_offsets,
            incoming_offsets_guard,
            incoming_offsets_layout,
            incoming_offsets
        );
        metal_input!(
            self,
            incoming_neighbors,
            incoming_neighbors_guard,
            incoming_neighbors_layout,
            incoming_neighbors
        );
        metal_input!(
            self,
            incoming_edges,
            incoming_edges_guard,
            incoming_edges_layout,
            incoming_edges
        );
        metal_input!(
            self,
            incoming_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(
            self,
            visible_nodes,
            visible_guard,
            visible_nodes_layout,
            visible
        );
        metal_input!(self, edge_active, active_guard, edge_active_layout, active);
        metal_input!(self, edge_layers, layers_guard, edge_layers_layout, layers);
        let (values_guard, values_layout) = self.weight.homogeneous_values.storage_and_layout();
        let Storage::Metal(values) = &*values_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra weight values moved off device".to_owned(),
            ));
        };
        let (validity_guard, validity_layout) = self.weight.validity.storage_and_layout();
        let Storage::Metal(validity) = &*validity_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra weight validity moved off device".to_owned(),
            ));
        };
        let (mixed_offsets_guard, mixed_offsets_layout) =
            self.weight.mixed_offsets.storage_and_layout();
        let Storage::Metal(mixed_offsets) = &*mixed_offsets_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra mixed offsets moved off device".to_owned(),
            ));
        };
        let (mixed_bytes_guard, mixed_bytes_layout) = self.weight.mixed_bytes.storage_and_layout();
        let Storage::Metal(mixed_bytes) = &*mixed_bytes_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra mixed bytes moved off device".to_owned(),
            ));
        };
        let nodes = self.args.node_count as usize;
        let edge_count = self.args.edge_count as usize;
        let adjacency = self.args.adjacency_count as usize;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "Dijkstra workspace",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.incoming_offsets,
                &incoming_offsets_layout,
                DType::U32,
                nodes + 1,
                "Dijkstra incoming offsets",
            ),
            (
                &self.incoming_neighbors,
                &incoming_neighbors_layout,
                DType::U32,
                adjacency,
                "Dijkstra incoming neighbors",
            ),
            (
                &self.incoming_edges,
                &incoming_edges_layout,
                DType::U32,
                adjacency,
                "Dijkstra incoming edges",
            ),
            (
                &self.visible_nodes,
                &visible_nodes_layout,
                DType::U8,
                nodes,
                "Dijkstra visible mask",
            ),
            (
                &self.edge_active,
                &edge_active_layout,
                DType::U8,
                edge_count,
                "Dijkstra active mask",
            ),
            (
                &self.edge_layers,
                &edge_layers_layout,
                DType::U8,
                edge_count,
                "Dijkstra edge layers",
            ),
            (
                &self.weight.validity,
                &validity_layout,
                DType::U8,
                edge_count,
                "Dijkstra weight validity",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.incoming_overlay,
            overlay_layout,
            self.args.incoming_overlay_count as usize,
            "Dijkstra incoming overlay",
        )?;
        if self.weight.kind == 1 || self.weight.kind == 2 {
            validate_bound_vector(
                &self.weight.homogeneous_values,
                values_layout,
                DType::I64,
                edge_count,
                "Dijkstra homogeneous weights",
            )?;
        } else if self.weight.kind == 3 {
            validate_bound_vector(
                &self.weight.mixed_offsets,
                mixed_offsets_layout,
                DType::U32,
                edge_count + 1,
                "Dijkstra mixed offsets",
            )?;
            validate_bound_vector(
                &self.weight.mixed_bytes,
                mixed_bytes_layout,
                DType::U8,
                self.weight.mixed_bytes.elem_count(),
                "Dijkstra mixed bytes",
            )?;
        } else if self.weight.kind != 4 {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra weight kind is invalid".to_owned(),
            ));
        }
        if self.round_count != 1
            || self.edge_begin > self.edge_end
            || self.edge_end > adjacency.max(edge_count)
            || self.node_begin >= self.node_end
            || self.node_end > nodes
        {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra round edge tile is invalid".to_owned(),
            ));
        }
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        let workspace_offset = workspace_layout.start_offset() * DType::U32.size_in_bytes();
        for round in self.first_round..self.first_round + self.round_count {
            let round = checked_candle_u32(round, "Dijkstra round")?;
            let node_begin = checked_candle_u32(self.node_begin, "Dijkstra node tile start")?;
            if self.prepare {
                encoder.set_compute_pipeline_state(&pipelines.dijkstra_prepare);
                encoder.set_output_buffer(0, Some(workspace.buffer()), workspace_offset);
                encoder.set_bytes(1, &self.args);
                encoder.set_bytes(2, &round);
                encoder.set_bytes(3, &node_begin);
                dispatch_rows(encoder, self.node_end - self.node_begin);
                encoder.insert_memory_barrier();
            }

            encoder.set_compute_pipeline_state(&pipelines.dijkstra_relax);
            bind_tensor(
                encoder,
                0,
                incoming_offsets,
                incoming_offsets_layout,
                DType::U32,
            );
            bind_tensor(
                encoder,
                1,
                incoming_neighbors,
                incoming_neighbors_layout,
                DType::U32,
            );
            bind_tensor(
                encoder,
                2,
                incoming_edges,
                incoming_edges_layout,
                DType::U32,
            );
            bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
            bind_tensor(encoder, 4, visible, visible_nodes_layout, DType::U8);
            bind_tensor(encoder, 5, active, edge_active_layout, DType::U8);
            bind_tensor(encoder, 6, layers, edge_layers_layout, DType::U8);
            bind_tensor(encoder, 7, values, values_layout, DType::I64);
            bind_tensor(encoder, 8, validity, validity_layout, DType::U8);
            bind_tensor(encoder, 9, mixed_offsets, mixed_offsets_layout, DType::U32);
            bind_tensor(encoder, 10, mixed_bytes, mixed_bytes_layout, DType::U8);
            encoder.set_output_buffer(11, Some(workspace.buffer()), workspace_offset);
            encoder.set_bytes(12, &self.args);
            encoder.set_bytes(13, &round);
            let edge_begin = checked_candle_u32(self.edge_begin, "Dijkstra tile start")?;
            let edge_end = checked_candle_u32(self.edge_end, "Dijkstra tile end")?;
            encoder.set_bytes(14, &edge_begin);
            encoder.set_bytes(15, &edge_end);
            encoder.set_bytes(16, &node_begin);
            dispatch_rows(encoder, self.node_end - self.node_begin);
            encoder.insert_memory_barrier();

            if self.publish {
                encoder.set_compute_pipeline_state(&pipelines.dijkstra_publish);
                encoder.set_output_buffer(0, Some(workspace.buffer()), workspace_offset);
                encoder.set_bytes(1, &self.args);
                encoder.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                );
                encoder.insert_memory_barrier();
            }
        }
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct DijkstraHeapChunk {
    outgoing_offsets: Tensor,
    outgoing_neighbors: Tensor,
    outgoing_edges: Tensor,
    outgoing_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    weight: PathWeight,
    args: DijkstraArgs,
    words: usize,
}

impl CustomOp1 for DijkstraHeapChunk {
    fn name(&self) -> &'static str {
        "irongraph-metal-weighted-dijkstra-heap-chunk"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal weighted Dijkstra heap chunk cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            outgoing_offsets,
            offsets_guard,
            outgoing_offsets_layout,
            offsets
        );
        metal_input!(
            self,
            outgoing_neighbors,
            neighbors_guard,
            outgoing_neighbors_layout,
            neighbors
        );
        metal_input!(
            self,
            outgoing_edges,
            edges_guard,
            outgoing_edges_layout,
            edges
        );
        metal_input!(
            self,
            outgoing_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(
            self,
            visible_nodes,
            visible_guard,
            visible_nodes_layout,
            visible
        );
        metal_input!(self, edge_active, active_guard, edge_active_layout, active);
        metal_input!(self, edge_layers, layers_guard, edge_layers_layout, layers);
        let (values_guard, values_layout) = self.weight.homogeneous_values.storage_and_layout();
        let Storage::Metal(values) = &*values_guard else {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra weight values moved off device".to_owned(),
            ));
        };
        let (validity_guard, validity_layout) = self.weight.validity.storage_and_layout();
        let Storage::Metal(validity) = &*validity_guard else {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra weight validity moved off device".to_owned(),
            ));
        };
        let (mixed_offsets_guard, mixed_offsets_layout) =
            self.weight.mixed_offsets.storage_and_layout();
        let Storage::Metal(mixed_offsets) = &*mixed_offsets_guard else {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra mixed offsets moved off device".to_owned(),
            ));
        };
        let (mixed_bytes_guard, mixed_bytes_layout) = self.weight.mixed_bytes.storage_and_layout();
        let Storage::Metal(mixed_bytes) = &*mixed_bytes_guard else {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra mixed bytes moved off device".to_owned(),
            ));
        };
        let nodes = self.args.node_count as usize;
        let edge_count = self.args.edge_count as usize;
        let adjacency = self.args.adjacency_count as usize;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.words,
            "heap Dijkstra workspace",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.outgoing_offsets,
                &outgoing_offsets_layout,
                DType::U32,
                nodes + 1,
                "heap Dijkstra offsets",
            ),
            (
                &self.outgoing_neighbors,
                &outgoing_neighbors_layout,
                DType::U32,
                adjacency,
                "heap Dijkstra neighbors",
            ),
            (
                &self.outgoing_edges,
                &outgoing_edges_layout,
                DType::U32,
                adjacency,
                "heap Dijkstra edge rows",
            ),
            (
                &self.visible_nodes,
                &visible_nodes_layout,
                DType::U8,
                nodes,
                "heap Dijkstra visible mask",
            ),
            (
                &self.edge_active,
                &edge_active_layout,
                DType::U8,
                edge_count,
                "heap Dijkstra active mask",
            ),
            (
                &self.edge_layers,
                &edge_layers_layout,
                DType::U8,
                edge_count,
                "heap Dijkstra edge layers",
            ),
            (
                &self.weight.validity,
                &validity_layout,
                DType::U8,
                edge_count,
                "heap Dijkstra weight validity",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.outgoing_overlay,
            overlay_layout,
            self.args.outgoing_overlay_count as usize,
            "heap Dijkstra outgoing overlay",
        )?;
        if self.weight.kind == 1 || self.weight.kind == 2 {
            validate_bound_vector(
                &self.weight.homogeneous_values,
                values_layout,
                DType::I64,
                edge_count,
                "heap Dijkstra homogeneous weights",
            )?;
        } else if self.weight.kind == 3 {
            validate_bound_vector(
                &self.weight.mixed_offsets,
                mixed_offsets_layout,
                DType::U32,
                edge_count + 1,
                "heap Dijkstra mixed offsets",
            )?;
            validate_bound_vector(
                &self.weight.mixed_bytes,
                mixed_bytes_layout,
                DType::U8,
                self.weight.mixed_bytes.elem_count(),
                "heap Dijkstra mixed bytes",
            )?;
        } else if self.weight.kind != 4 {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra weight kind is invalid".to_owned(),
            ));
        }
        if self.args.quantum == 0 {
            return Err(candle_core::Error::Msg(
                "Metal heap Dijkstra quantum is zero".to_owned(),
            ));
        }
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.dijkstra_heap_chunk);
        bind_tensor(encoder, 0, offsets, outgoing_offsets_layout, DType::U32);
        bind_tensor(encoder, 1, neighbors, outgoing_neighbors_layout, DType::U32);
        bind_tensor(encoder, 2, edges, outgoing_edges_layout, DType::U32);
        bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
        bind_tensor(encoder, 4, visible, visible_nodes_layout, DType::U8);
        bind_tensor(encoder, 5, active, edge_active_layout, DType::U8);
        bind_tensor(encoder, 6, layers, edge_layers_layout, DType::U8);
        bind_tensor(encoder, 7, values, values_layout, DType::I64);
        bind_tensor(encoder, 8, validity, validity_layout, DType::U8);
        bind_tensor(encoder, 9, mixed_offsets, mixed_offsets_layout, DType::U32);
        bind_tensor(encoder, 10, mixed_bytes, mixed_bytes_layout, DType::U8);
        encoder.set_output_buffer(
            11,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(12, &self.args);
        dispatch_group(encoder);
        Ok((
            MetalStorage::new(
                Arc::new(workspace.buffer().clone()),
                device.clone(),
                self.words,
                DType::U32,
            ),
            Shape::from(self.words),
        ))
    }
}

#[derive(Clone, Debug)]
struct DijkstraFinalize {
    incoming_offsets: Tensor,
    incoming_neighbors: Tensor,
    incoming_edges: Tensor,
    incoming_overlay: Tensor,
    visible_nodes: Tensor,
    edge_active: Tensor,
    edge_layers: Tensor,
    weight: PathWeight,
    args: DijkstraArgs,
    workspace_words: usize,
    packet_words: usize,
    edge_begin: usize,
    edge_end: usize,
    initialize: bool,
    node_begin: usize,
    node_end: usize,
}

impl CustomOp2 for DijkstraFinalize {
    fn name(&self) -> &'static str {
        "irongraph-metal-weighted-dijkstra-finalize"
    }

    fn cpu_fwd(
        &self,
        _workspace: &CpuStorage,
        _workspace_layout: &Layout,
        _packet: &CpuStorage,
        _packet_layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal weighted Dijkstra finalization cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening, clippy::too_many_lines)]
    fn metal_fwd(
        &self,
        workspace: &MetalStorage,
        workspace_layout: &Layout,
        packet: &MetalStorage,
        packet_layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        metal_input!(
            self,
            incoming_offsets,
            offsets_guard,
            incoming_offsets_layout,
            offsets
        );
        metal_input!(
            self,
            incoming_neighbors,
            neighbors_guard,
            incoming_neighbors_layout,
            neighbors
        );
        metal_input!(
            self,
            incoming_edges,
            edges_guard,
            incoming_edges_layout,
            edges
        );
        metal_input!(
            self,
            incoming_overlay,
            overlay_guard,
            overlay_layout,
            overlay
        );
        metal_input!(
            self,
            visible_nodes,
            visible_guard,
            visible_nodes_layout,
            visible
        );
        metal_input!(self, edge_active, active_guard, edge_active_layout, active);
        metal_input!(self, edge_layers, layers_guard, edge_layers_layout, layers);
        let (values_guard, values_layout) = self.weight.homogeneous_values.storage_and_layout();
        let Storage::Metal(values) = &*values_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra weights moved off device".to_owned(),
            ));
        };
        let (validity_guard, validity_layout) = self.weight.validity.storage_and_layout();
        let Storage::Metal(validity) = &*validity_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra validity moved off device".to_owned(),
            ));
        };
        let (mixed_offsets_guard, mixed_offsets_layout) =
            self.weight.mixed_offsets.storage_and_layout();
        let Storage::Metal(mixed_offsets) = &*mixed_offsets_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra mixed offsets moved off device".to_owned(),
            ));
        };
        let (mixed_bytes_guard, mixed_bytes_layout) = self.weight.mixed_bytes.storage_and_layout();
        let Storage::Metal(mixed_bytes) = &*mixed_bytes_guard else {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra mixed bytes moved off device".to_owned(),
            ));
        };
        let nodes = self.args.node_count as usize;
        let edge_count = self.args.edge_count as usize;
        let adjacency = self.args.adjacency_count as usize;
        validate_vector(
            workspace,
            workspace_layout,
            DType::U32,
            self.workspace_words,
            "Dijkstra workspace",
        )?;
        validate_vector(
            packet,
            packet_layout,
            DType::U32,
            self.packet_words,
            "Dijkstra packet",
        )?;
        for (tensor, layout, dtype, count, name) in [
            (
                &self.incoming_offsets,
                &incoming_offsets_layout,
                DType::U32,
                nodes + 1,
                "Dijkstra incoming offsets",
            ),
            (
                &self.incoming_neighbors,
                &incoming_neighbors_layout,
                DType::U32,
                adjacency,
                "Dijkstra incoming neighbors",
            ),
            (
                &self.incoming_edges,
                &incoming_edges_layout,
                DType::U32,
                adjacency,
                "Dijkstra incoming edges",
            ),
            (
                &self.visible_nodes,
                &visible_nodes_layout,
                DType::U8,
                nodes,
                "Dijkstra visible mask",
            ),
            (
                &self.edge_active,
                &edge_active_layout,
                DType::U8,
                edge_count,
                "Dijkstra active mask",
            ),
            (
                &self.edge_layers,
                &edge_layers_layout,
                DType::U8,
                edge_count,
                "Dijkstra edge layers",
            ),
            (
                &self.weight.validity,
                &validity_layout,
                DType::U8,
                edge_count,
                "Dijkstra validity",
            ),
        ] {
            validate_bound_vector(tensor, layout, dtype, count, name)?;
        }
        validate_csr_overlay(
            &self.incoming_overlay,
            overlay_layout,
            self.args.incoming_overlay_count as usize,
            "Dijkstra finalizer incoming overlay",
        )?;
        if self.weight.kind == 1 || self.weight.kind == 2 {
            validate_bound_vector(
                &self.weight.homogeneous_values,
                values_layout,
                DType::I64,
                edge_count,
                "Dijkstra weights",
            )?;
        } else if self.weight.kind == 3 {
            validate_bound_vector(
                &self.weight.mixed_offsets,
                mixed_offsets_layout,
                DType::U32,
                edge_count + 1,
                "Dijkstra mixed offsets",
            )?;
        } else if self.weight.kind != 4 {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra weight kind is invalid".to_owned(),
            ));
        }
        if self.edge_begin > self.edge_end
            || self.edge_end > adjacency.max(edge_count)
            || self.node_begin >= self.node_end
            || self.node_end > nodes
        {
            return Err(candle_core::Error::Msg(
                "Metal Dijkstra finalizer edge tile is invalid".to_owned(),
            ));
        }
        let device = workspace.device();
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        if self.initialize && self.node_begin == 0 {
            encoder.set_compute_pipeline_state(&pipelines.dijkstra_finalize_prepare);
            encoder.set_output_buffer(
                0,
                Some(packet.buffer()),
                (packet_layout.start_offset() + nodes * 3) * DType::U32.size_in_bytes(),
            );
            encoder.dispatch_thread_groups(
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
                objc2_metal::MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            encoder.insert_memory_barrier();
        }
        encoder.set_compute_pipeline_state(&pipelines.dijkstra_finalize);
        bind_tensor(encoder, 0, offsets, incoming_offsets_layout, DType::U32);
        bind_tensor(encoder, 1, neighbors, incoming_neighbors_layout, DType::U32);
        bind_tensor(encoder, 2, edges, incoming_edges_layout, DType::U32);
        bind_tensor(encoder, 3, overlay, overlay_layout, DType::U32);
        bind_tensor(encoder, 4, visible, visible_nodes_layout, DType::U8);
        bind_tensor(encoder, 5, active, edge_active_layout, DType::U8);
        bind_tensor(encoder, 6, layers, edge_layers_layout, DType::U8);
        bind_tensor(encoder, 7, values, values_layout, DType::I64);
        bind_tensor(encoder, 8, validity, validity_layout, DType::U8);
        bind_tensor(encoder, 9, mixed_offsets, mixed_offsets_layout, DType::U32);
        bind_tensor(encoder, 10, mixed_bytes, mixed_bytes_layout, DType::U8);
        encoder.set_input_buffer(
            11,
            Some(workspace.buffer()),
            workspace_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_output_buffer(
            12,
            Some(packet.buffer()),
            packet_layout.start_offset() * DType::U32.size_in_bytes(),
        );
        encoder.set_bytes(13, &self.args);
        let edge_begin = checked_candle_u32(self.edge_begin, "Dijkstra finalizer tile start")?;
        let edge_end = checked_candle_u32(self.edge_end, "Dijkstra finalizer tile end")?;
        let initialize = u32::from(self.initialize);
        let node_begin = checked_candle_u32(self.node_begin, "Dijkstra finalizer node tile start")?;
        encoder.set_bytes(14, &edge_begin);
        encoder.set_bytes(15, &edge_end);
        encoder.set_bytes(16, &initialize);
        encoder.set_bytes(17, &node_begin);
        dispatch_rows(encoder, self.node_end - self.node_begin);
        Ok((
            MetalStorage::new(
                Arc::new(packet.buffer().clone()),
                device.clone(),
                self.packet_words,
                DType::U32,
            ),
            Shape::from(self.packet_words),
        ))
    }
}

impl CandleResident {
    pub fn metal_breadth_first_direct(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        max_output_rows: usize,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        let workspace =
            self.metal_bfs_workspace(visible_mask, source_dense, layers, cancellation, deadline)?;
        let distance_lane = workspace
            .narrow(0, 0, self.node_count)
            .map_err(candle_error)?;
        let reached = distance_lane.ne(u32::MAX).map_err(candle_error)?;
        let selected = super::selected_positions_with_deadline(
            &reached,
            self.node_count,
            workspace.device(),
            cancellation,
            deadline,
        )?;
        let result_count = selected.elem_count();
        if result_count > max_output_rows {
            return Err(path_budget("BFS"));
        }
        let compact_distance = distance_lane
            .index_select(&selected, 0)
            .map_err(candle_error)?;
        let node_rows = read_u32_bounded(&selected, 0, result_count, cancellation, deadline)?;
        let distance =
            read_u32_bounded(&compact_distance, 0, result_count, cancellation, deadline)?;
        ensure_graph_execution(cancellation, deadline)?;
        Ok(ResidentGraphProcedureResult::BreadthFirst {
            node_rows,
            distance,
        })
    }

    /// Restart a measured narrow frontier into a bounded persistent FIFO traversal. The workspace
    /// retains the ordinary BFS distance layout, so downstream BFS and unit-Dijkstra publication
    /// remains shared with the direction-optimizing grid implementation.
    pub fn metal_bfs_persistent_workspace(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Tensor> {
        self.metal_bfs_persistent_workspace_for_direction(
            visible_mask,
            source_dense,
            layers,
            false,
            cancellation,
            deadline,
        )
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn metal_bfs_persistent_workspace_for_direction(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        reverse: bool,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Tensor> {
        self.metal_bfs_persistent_workspace_bounded(
            visible_mask,
            source_dense,
            layers,
            reverse,
            u32::MAX,
            u32::MAX,
            cancellation,
            deadline,
        )
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn metal_bfs_persistent_workspace_bounded(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        reverse: bool,
        target_dense: u32,
        maximum_distance: u32,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Tensor> {
        self.validate_path_source(visible_mask, source_dense, "BFS source")?;
        let (offsets, neighbors, edges, overlay, overlay_count) = if reverse {
            (
                self.incoming_offsets.clone(),
                self.incoming_neighbors.as_ref(),
                self.incoming_edges.as_ref(),
                self.incoming_overlay
                    .packet
                    .clone()
                    .unwrap_or_else(|| self.incoming_offsets.clone()),
                self.incoming_overlay.rows.len(),
            )
        } else {
            (
                self.outgoing_offsets.clone(),
                self.outgoing_neighbors.as_ref(),
                self.outgoing_edges.as_ref(),
                self.outgoing_overlay
                    .packet
                    .clone()
                    .unwrap_or_else(|| self.outgoing_offsets.clone()),
                self.outgoing_overlay.rows.len(),
            )
        };
        let adjacency_count = neighbors.map_or(0, Tensor::elem_count);
        let args = PersistentBfsArgs::new(
            self.node_count,
            self.edge_count,
            adjacency_count,
            source_dense,
            target_dense,
            maximum_distance,
            u32::from(layers.bits()),
            overlay_count,
        )
        .map_err(candle_error)?;
        let words = self
            .node_count
            .checked_mul(3)
            .and_then(|words| words.checked_add(PATH_CONTROL_WORDS))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal persistent BFS workspace size overflow",
                )
            })?;
        let mut workspace = allocate_u32_workspace(
            visible_mask,
            words,
            "irongraph Metal persistent BFS workspace",
        )?;
        let control_offset = self.node_count * 3;
        for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
            ensure_graph_execution(cancellation, deadline)?;
            let node_end = node_begin
                .saturating_add(PATH_NODE_QUANTUM)
                .min(self.node_count);
            workspace = visible_mask
                .apply_op2_no_bwd(
                    &workspace,
                    &PersistentBfsInitialize {
                        args,
                        words,
                        node_begin,
                        node_end,
                    },
                )
                .map_err(candle_error)?;
            let status = read_u32_slice(&workspace, control_offset + 2, 1)?
                .first()
                .copied()
                .unwrap_or(1);
            dijkstra_status(status)?;
            ensure_graph_execution(cancellation, deadline)?;
        }
        if adjacency_count == 0 && overlay_count == 0 {
            return Ok(workspace);
        }
        let adjacency_dummy =
            Tensor::zeros(1, DType::U32, visible_mask.device()).map_err(candle_error)?;
        let neighbors = neighbors.unwrap_or(&adjacency_dummy);
        let edges = edges.unwrap_or(&adjacency_dummy);
        if adjacency_count != 0 && edges.elem_count() != adjacency_count {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident persistent BFS CSR cardinalities differ",
            ));
        }
        let (edge_active, edge_layers) = self.path_edge_visibility()?;
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            let control = read_u32_slice(&workspace, control_offset, PATH_CONTROL_WORDS)?;
            dijkstra_status(control.get(2).copied().unwrap_or(1))?;
            let head = control.first().copied().unwrap_or(u32::MAX) as usize;
            let tail = control.get(1).copied().unwrap_or(0) as usize;
            let processed = control.get(6).copied().unwrap_or(u32::MAX) as usize;
            if head > self.node_count || tail > self.node_count || processed > self.node_count {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal persistent BFS queue escaped the resident node bound",
                ));
            }
            if control.get(7).copied().unwrap_or(0) != 0 {
                return Ok(workspace);
            }
            workspace = workspace
                .apply_op1_no_bwd(&PersistentBfsChunk {
                    outgoing_offsets: offsets.clone(),
                    outgoing_neighbors: neighbors.clone(),
                    outgoing_edges: edges.clone(),
                    outgoing_overlay: overlay.clone(),
                    visible_nodes: visible_mask.clone(),
                    edge_active: edge_active.clone(),
                    edge_layers: edge_layers.clone(),
                    args,
                    words,
                })
                .map_err(candle_error)?;
        }
    }

    /// Exact lexicographic DFS preorder.  General directed DFS has a sequential dependency; the
    /// Metal kernel therefore uses one persistent cooperative threadgroup, parallelizing every
    /// adjacency scan while retaining O(E + width*V) work and exact CPU ordering.
    pub fn metal_depth_first(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        max_output_rows: usize,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        self.validate_path_source(visible_mask, source_dense, "DFS")?;
        ensure_graph_execution(cancellation, deadline)?;
        validate_u32_workspace_words(self.node_count, 4, PATH_CONTROL_WORDS, "DFS")
            .map_err(candle_error)?;
        let Some(neighbors) = self.outgoing_neighbors.as_ref() else {
            if max_output_rows == 0 {
                return Err(path_budget("DFS"));
            }
            return Ok(ResidentGraphProcedureResult::DepthFirst {
                node_rows: vec![source_dense],
                order: vec![0],
            });
        };
        let edges = self.path_outgoing_edges(neighbors.elem_count())?;
        let (edge_active, edge_layers) = self.path_edge_visibility()?;
        let control_offset = self.node_count * 4;
        let words = control_offset + PATH_CONTROL_WORDS;
        let mut workspace =
            allocate_u32_workspace(visible_mask, words, "irongraph Metal DFS workspace")?;
        for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
            ensure_graph_execution(cancellation, deadline)?;
            let node_end = node_begin
                .saturating_add(PATH_NODE_QUANTUM)
                .min(self.node_count);
            workspace = visible_mask
                .apply_op2_no_bwd(
                    &workspace,
                    &DfsInitialize {
                        node_count: self.node_count,
                        edge_count: self.edge_count,
                        source: source_dense,
                        layer_mask: u32::from(layers.bits()),
                        words,
                        node_begin,
                        node_end,
                    },
                )
                .map_err(candle_error)?;
            let status = read_u32_slice(&workspace, control_offset + 3, 1)?
                .first()
                .copied()
                .unwrap_or(1);
            if status != 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal DFS rejected corrupt resident graph metadata",
                ));
            }
            ensure_graph_execution(cancellation, deadline)?;
        }
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            let control = read_u32_slice(&workspace, control_offset, PATH_CONTROL_WORDS)?;
            if control.get(3).copied().unwrap_or(1) != 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal DFS rejected corrupt resident graph metadata",
                ));
            }
            if control.get(2).copied().unwrap_or(0) != 0 {
                let output_len = control.get(1).copied().unwrap_or(0) as usize;
                if output_len > max_output_rows {
                    return Err(path_budget("DFS"));
                }
                let node_rows = read_u32_bounded(
                    &workspace,
                    self.node_count * 3,
                    output_len,
                    cancellation,
                    deadline,
                )?;
                let mut order = Vec::with_capacity(output_len);
                for value in 0..output_len {
                    if value.is_multiple_of(PATH_NODE_QUANTUM) {
                        ensure_graph_execution(cancellation, deadline)?;
                    }
                    order.push(checked_u32(value, "DFS order")?);
                }
                ensure_graph_execution(cancellation, deadline)?;
                return Ok(ResidentGraphProcedureResult::DepthFirst { node_rows, order });
            }
            workspace = workspace
                .apply_op1_no_bwd(&DfsChunk {
                    outgoing_offsets: self.outgoing_offsets.clone(),
                    outgoing_neighbors: neighbors.clone(),
                    outgoing_edges: edges.clone(),
                    outgoing_overlay: self
                        .outgoing_overlay
                        .packet
                        .clone()
                        .unwrap_or_else(|| self.outgoing_offsets.clone()),
                    visible_nodes: visible_mask.clone(),
                    edge_active: edge_active.clone(),
                    edge_layers: edge_layers.clone(),
                    node_count: self.node_count,
                    edge_count: self.edge_count,
                    adjacency_count: neighbors.elem_count(),
                    outgoing_overlay_count: self.outgoing_overlay.rows.len(),
                    source: source_dense,
                    layer_mask: u32::from(layers.bits()),
                    quantum: DFS_QUANTUM,
                })
                .map_err(candle_error)?;
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn metal_shortest_path(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        target_dense: u32,
        layers: LayerMask,
        max_output_rows: usize,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        self.validate_path_source(visible_mask, source_dense, "shortest-path source")?;
        self.validate_path_source(visible_mask, target_dense, "shortest-path target")?;
        ensure_graph_execution(cancellation, deadline)?;
        let forward = if self.outgoing_neighbors.as_ref().is_some_and(|neighbors| {
            bfs_should_start_persistent(self.node_count, neighbors.elem_count())
        }) {
            self.metal_bfs_persistent_workspace_bounded(
                visible_mask,
                source_dense,
                layers,
                false,
                target_dense,
                u32::MAX,
                cancellation,
                deadline,
            )?
        } else {
            self.metal_bfs_workspace(visible_mask, source_dense, layers, cancellation, deadline)?
        };
        let distance = read_u32_slice(&forward, target_dense as usize, 1)?
            .first()
            .copied()
            .unwrap_or(u32::MAX);
        ensure_graph_execution(cancellation, deadline)?;
        if distance == u32::MAX {
            return Ok(ResidentGraphProcedureResult::ShortestPath {
                node_rows: Vec::new(),
                edge_rows: Vec::new(),
                cost: 0,
            });
        }
        if max_output_rows == 0 {
            return Err(path_budget("shortest path"));
        }
        if source_dense == target_dense {
            return Ok(ResidentGraphProcedureResult::ShortestPath {
                node_rows: vec![source_dense],
                edge_rows: Vec::new(),
                cost: 0,
            });
        }
        let reverse = self.metal_reverse_bfs_workspace(
            visible_mask,
            target_dense,
            layers,
            distance,
            cancellation,
            deadline,
        )?;
        let reverse_source = read_u32_slice(&reverse, source_dense as usize, 1)?
            .first()
            .copied()
            .unwrap_or(u32::MAX);
        if reverse_source != distance {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Metal shortest-path forward and reverse labels disagree",
            ));
        }
        let neighbors = self.outgoing_neighbors.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "reachable shortest path has no outgoing adjacency",
            )
        })?;
        let edges = self.path_outgoing_edges(neighbors.elem_count())?;
        let (edge_active, edge_layers) = self.path_edge_visibility()?;
        let words = self.node_count * 2 + PATH_CONTROL_WORDS;
        let args = ShortestArgs::new(
            self.node_count,
            self.edge_count,
            neighbors.elem_count(),
            source_dense,
            target_dense,
            u32::from(layers.bits()),
            distance,
            SHORTEST_PATH_QUANTUM,
            self.outgoing_overlay.rows.len(),
        )
        .map_err(candle_error)?;
        let control_offset = self.node_count * 2;
        let mut workspace = allocate_u32_workspace(
            visible_mask,
            words,
            "irongraph Metal shortest-path workspace",
        )?;
        for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
            ensure_graph_execution(cancellation, deadline)?;
            let node_end = node_begin
                .saturating_add(PATH_NODE_QUANTUM)
                .min(self.node_count);
            workspace = workspace
                .apply_op1_no_bwd(&ShortestInitialize {
                    args,
                    words,
                    node_begin,
                    node_end,
                })
                .map_err(candle_error)?;
            let status = read_u32_slice(&workspace, control_offset + 3, 1)?
                .first()
                .copied()
                .unwrap_or(1);
            if status != 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal shortest-path initialization failed",
                ));
            }
            ensure_graph_execution(cancellation, deadline)?;
        }
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            let control = read_u32_slice(&workspace, control_offset, PATH_CONTROL_WORDS)?;
            if control.get(3).copied().unwrap_or(1) != 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal shortest-path reconstruction is inconsistent with resident CSR",
                ));
            }
            if control.get(2).copied().unwrap_or(0) != 0 {
                let length = control.first().copied().unwrap_or(0) as usize;
                if length != distance as usize + 1 || length > self.node_count {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "Metal shortest-path reconstruction returned an invalid length",
                    ));
                }
                let node_rows = read_u32_bounded(&workspace, 0, length, cancellation, deadline)?;
                let edge_rows = read_u32_bounded(
                    &workspace,
                    self.node_count,
                    length - 1,
                    cancellation,
                    deadline,
                )?;
                ensure_graph_execution(cancellation, deadline)?;
                return Ok(ResidentGraphProcedureResult::ShortestPath {
                    node_rows,
                    edge_rows,
                    cost: distance,
                });
            }
            workspace = workspace
                .apply_op1_no_bwd(&ShortestChunk {
                    outgoing_offsets: self.outgoing_offsets.clone(),
                    outgoing_neighbors: neighbors.clone(),
                    outgoing_edges: edges.clone(),
                    outgoing_overlay: self
                        .outgoing_overlay
                        .packet
                        .clone()
                        .unwrap_or_else(|| self.outgoing_offsets.clone()),
                    visible_nodes: visible_mask.clone(),
                    edge_active: edge_active.clone(),
                    edge_layers: edge_layers.clone(),
                    forward_workspace: forward.clone(),
                    reverse_workspace: reverse.clone(),
                    args,
                    words,
                })
                .map_err(candle_error)?;
        }
    }

    /// Exact binary64 weighted shortest paths. Broad frontiers use a physically tiled,
    /// deterministic target-owned incoming pull; narrow sparse frontiers adapt to a persistent
    /// cooperative device heap. Neither path uses cross-thread distance locks. A separate incoming
    /// pass canonicalizes predecessor ties. Integer conversion and every path addition use software
    /// IEEE-754 round-to-nearest-even in the Metal kernel.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn metal_dijkstra_weighted(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        weight_property: PropertyId,
        layers: LayerMask,
        max_output_rows: usize,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<ResidentGraphProcedureResult> {
        self.validate_path_source(visible_mask, source_dense, "Dijkstra source")?;
        ensure_graph_execution(cancellation, deadline)?;
        if self.edge_count == 0 {
            if max_output_rows == 0 {
                return Err(path_budget("Dijkstra"));
            }
            return Ok(ResidentGraphProcedureResult::Dijkstra {
                node_rows: vec![source_dense],
                cost: vec![0.0],
                predecessor: vec![None],
            });
        }
        let neighbors = self.outgoing_neighbors.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident weighted graph has no outgoing-neighbor tensor",
            )
        })?;
        let outgoing_edges = self.path_outgoing_edges(neighbors.elem_count())?;
        let incoming_neighbors = self.incoming_neighbors.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident weighted graph has no incoming-neighbor tensor",
            )
        })?;
        let incoming_edges = self.incoming_edges.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident weighted graph has no incoming-edge tensor",
            )
        })?;
        if incoming_neighbors.elem_count() != neighbors.elem_count()
            || incoming_edges.elem_count() != neighbors.elem_count()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident weighted incoming and outgoing CSR cardinalities differ",
            ));
        }
        let (edge_active, edge_layers) = self.path_edge_visibility()?;
        let weight = self.path_weight(weight_property, visible_mask)?;
        let args = DijkstraArgs::new(
            self.node_count,
            self.edge_count,
            neighbors.elem_count(),
            source_dense,
            u32::from(layers.bits()),
            &weight,
            self.outgoing_overlay.rows.len(),
            self.incoming_overlay.rows.len(),
        )
        .map_err(candle_error)?;
        let workspace_words = self
            .node_count
            .checked_mul(7)
            .and_then(|words| words.checked_add(PATH_CONTROL_WORDS))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal weighted Dijkstra workspace size overflow",
                )
            })?;
        let control_offset = self.node_count * 7;
        let initialize_workspace = |heap: bool| -> Result<Tensor> {
            let mut workspace = allocate_u32_workspace(
                visible_mask,
                workspace_words,
                "irongraph Metal exact weighted Dijkstra workspace",
            )?;
            for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
                ensure_graph_execution(cancellation, deadline)?;
                let node_end = node_begin
                    .saturating_add(PATH_NODE_QUANTUM)
                    .min(self.node_count);
                workspace = visible_mask
                    .apply_op2_no_bwd(
                        &workspace,
                        &DijkstraInitialize {
                            args,
                            words: workspace_words,
                            heap,
                            node_begin,
                            node_end,
                        },
                    )
                    .map_err(candle_error)?;
                let status = read_u32_slice(&workspace, control_offset + 2, 1)?
                    .first()
                    .copied()
                    .unwrap_or(1);
                dijkstra_status(status)?;
                ensure_graph_execution(cancellation, deadline)?;
            }
            Ok(workspace)
        };
        let mut workspace = initialize_workspace(false)?;
        let mut rounds = 0_usize;
        let mut heap_mode = false;
        loop {
            ensure_graph_execution(cancellation, deadline)?;
            let control = read_u32_slice(&workspace, control_offset, PATH_CONTROL_WORDS)?;
            dijkstra_status(control.get(2).copied().unwrap_or(1))?;
            let active_count = control.first().copied().unwrap_or(0) as usize;
            if heap_mode {
                let settled = control.get(1).copied().unwrap_or(u32::MAX) as usize;
                if settled > self.node_count {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "exact heap Dijkstra settled more nodes than the resident graph contains",
                    ));
                }
                // A high-degree source is removed from the heap while its CSR row is resumed in
                // fixed edge tiles. The heap may therefore be empty before the active row is done.
                if control.get(3).copied().unwrap_or(0) != 0 {
                    break;
                }
                workspace = workspace
                    .apply_op1_no_bwd(&DijkstraHeapChunk {
                        outgoing_offsets: self.outgoing_offsets.clone(),
                        outgoing_neighbors: neighbors.clone(),
                        outgoing_edges: outgoing_edges.clone(),
                        outgoing_overlay: self
                            .outgoing_overlay
                            .packet
                            .clone()
                            .unwrap_or_else(|| self.outgoing_offsets.clone()),
                        visible_nodes: visible_mask.clone(),
                        edge_active: edge_active.clone(),
                        edge_layers: edge_layers.clone(),
                        weight: weight.clone(),
                        args,
                        words: workspace_words,
                    })
                    .map_err(candle_error)?;
                continue;
            }
            if active_count == 0 {
                break;
            }
            if weighted_dijkstra_should_switch_to_heap(
                self.node_count,
                neighbors.elem_count(),
                rounds,
                active_count,
            ) {
                // Restarting does not consume the probed labels. Release both the host readback
                // and private workspace first so the heap initializer reuses their pool buckets.
                drop(control);
                drop(workspace);
                workspace = initialize_workspace(true)?;
                heap_mode = true;
                continue;
            }
            if rounds >= self.node_count {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "exact weighted Dijkstra did not converge within the simple-path bound",
                ));
            }
            let adjacency_tile_domain = neighbors.elem_count().max(self.edge_count);
            let tile_count = adjacency_tile_domain
                .max(1)
                .div_ceil(super::METAL_GRAPH_EDGE_TILE);
            for tile in 0..tile_count {
                let edge_begin = tile
                    .saturating_mul(super::METAL_GRAPH_EDGE_TILE)
                    .min(adjacency_tile_domain);
                let edge_end = edge_begin
                    .saturating_add(super::METAL_GRAPH_EDGE_TILE)
                    .min(adjacency_tile_domain);
                for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
                    ensure_graph_execution(cancellation, deadline)?;
                    let node_end = node_begin
                        .saturating_add(PATH_NODE_QUANTUM)
                        .min(self.node_count);
                    workspace = workspace
                        .apply_op1_no_bwd(&DijkstraChunk {
                            incoming_offsets: self.incoming_offsets.clone(),
                            incoming_neighbors: incoming_neighbors.clone(),
                            incoming_edges: incoming_edges.clone(),
                            incoming_overlay: self
                                .incoming_overlay
                                .packet
                                .clone()
                                .unwrap_or_else(|| self.incoming_offsets.clone()),
                            visible_nodes: visible_mask.clone(),
                            edge_active: edge_active.clone(),
                            edge_layers: edge_layers.clone(),
                            weight: weight.clone(),
                            args,
                            words: workspace_words,
                            first_round: rounds,
                            round_count: 1,
                            edge_begin,
                            edge_end,
                            node_begin,
                            node_end,
                            prepare: tile == 0,
                            publish: tile + 1 == tile_count && node_end == self.node_count,
                        })
                        .map_err(candle_error)?;
                    let tile_control =
                        read_u32_slice(&workspace, control_offset, PATH_CONTROL_WORDS)?;
                    dijkstra_status(tile_control.get(2).copied().unwrap_or(1))?;
                    ensure_graph_execution(cancellation, deadline)?;
                }
            }
            rounds = rounds.saturating_add(1);
        }
        ensure_graph_execution(cancellation, deadline)?;
        let present_lane = workspace
            .narrow(0, self.node_count * 2, self.node_count)
            .map_err(candle_error)?;
        let reached = present_lane.ne(0_u32).map_err(candle_error)?;
        let selected = super::selected_positions_with_deadline(
            &reached,
            self.node_count,
            workspace.device(),
            cancellation,
            deadline,
        )?;
        let result_count = selected.elem_count();
        if result_count > max_output_rows {
            return Err(path_budget("Dijkstra"));
        }
        let packet_words = self
            .node_count
            .checked_mul(3)
            .and_then(|words| words.checked_add(1))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal weighted Dijkstra result size overflow",
                )
            })?;
        let mut packet = allocate_u32_workspace(
            visible_mask,
            packet_words,
            "irongraph Metal weighted Dijkstra result packet",
        )?;
        let adjacency_tile_domain = neighbors.elem_count().max(self.edge_count);
        let tile_count = adjacency_tile_domain
            .max(1)
            .div_ceil(super::METAL_GRAPH_EDGE_TILE);
        for tile in 0..tile_count {
            let edge_begin = tile
                .saturating_mul(super::METAL_GRAPH_EDGE_TILE)
                .min(adjacency_tile_domain);
            let edge_end = edge_begin
                .saturating_add(super::METAL_GRAPH_EDGE_TILE)
                .min(adjacency_tile_domain);
            for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
                ensure_graph_execution(cancellation, deadline)?;
                let node_end = node_begin
                    .saturating_add(PATH_NODE_QUANTUM)
                    .min(self.node_count);
                packet = workspace
                    .apply_op2_no_bwd(
                        &packet,
                        &DijkstraFinalize {
                            incoming_offsets: self.incoming_offsets.clone(),
                            incoming_neighbors: incoming_neighbors.clone(),
                            incoming_edges: incoming_edges.clone(),
                            incoming_overlay: self
                                .incoming_overlay
                                .packet
                                .clone()
                                .unwrap_or_else(|| self.incoming_offsets.clone()),
                            visible_nodes: visible_mask.clone(),
                            edge_active: edge_active.clone(),
                            edge_layers: edge_layers.clone(),
                            weight: weight.clone(),
                            args,
                            workspace_words,
                            packet_words,
                            edge_begin,
                            edge_end,
                            initialize: tile == 0,
                            node_begin,
                            node_end,
                        },
                    )
                    .map_err(candle_error)?;
                let status = read_u32_slice(&packet, self.node_count * 3, 1)?
                    .first()
                    .copied()
                    .unwrap_or(1);
                dijkstra_status(status)?;
                ensure_graph_execution(cancellation, deadline)?;
            }
        }
        let status = read_u32_slice(&packet, self.node_count * 3, 1)?
            .first()
            .copied()
            .unwrap_or(1);
        dijkstra_status(status)?;
        let compact_cost_words = packet
            .narrow(0, 0, self.node_count * 2)
            .and_then(|words| words.reshape((self.node_count, 2)))
            .and_then(|words| words.index_select(&selected, 0))
            .and_then(|words| words.flatten_all())
            .map_err(candle_error)?;
        let compact_predecessor = packet
            .narrow(0, self.node_count * 2, self.node_count)
            .and_then(|values| values.index_select(&selected, 0))
            .map_err(candle_error)?;
        let node_rows = read_u32_bounded(&selected, 0, result_count, cancellation, deadline)?;
        let cost_words = read_u32_bounded(
            &compact_cost_words,
            0,
            result_count * 2,
            cancellation,
            deadline,
        )?;
        let raw_predecessor = read_u32_bounded(
            &compact_predecessor,
            0,
            result_count,
            cancellation,
            deadline,
        )?;
        let mut cost = Vec::with_capacity(result_count);
        let mut predecessor = Vec::with_capacity(result_count);
        for (index, node) in node_rows.iter().copied().enumerate() {
            if index.is_multiple_of(PATH_NODE_QUANTUM) {
                ensure_graph_execution(cancellation, deadline)?;
            }
            let bits =
                u64::from(cost_words[index * 2]) | (u64::from(cost_words[index * 2 + 1]) << 32);
            let value = f64::from_bits(bits);
            if value.is_nan() || value.is_sign_negative() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal weighted Dijkstra returned an invalid exact cost",
                ));
            }
            let predecessor_row =
                (raw_predecessor[index] != u32::MAX).then_some(raw_predecessor[index]);
            if node == source_dense && predecessor_row.is_some() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal weighted Dijkstra returned a predecessor for its source",
                ));
            }
            if node != source_dense && predecessor_row.is_none() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal weighted Dijkstra returned a reached node without a predecessor",
                ));
            }
            cost.push(value);
            predecessor.push(predecessor_row);
        }
        ensure_graph_execution(cancellation, deadline)?;
        Ok(ResidentGraphProcedureResult::Dijkstra {
            node_rows,
            cost,
            predecessor,
        })
    }

    fn validate_path_source(&self, visible: &Tensor, source: u32, label: &str) -> Result<()> {
        if source as usize >= self.node_count {
            return Err(Error::new(
                ErrorCode::QueryType,
                format!("{label} node is out of bounds"),
            ));
        }
        let selected = read_u8_slice(visible, source as usize, 1)?
            .first()
            .copied()
            .unwrap_or(0);
        if selected != 1 {
            return Err(Error::new(
                ErrorCode::QueryType,
                "graph procedure node is not visible",
            ));
        }
        Ok(())
    }

    fn path_outgoing_edges(&self, adjacency_count: usize) -> Result<&Tensor> {
        let edges = self.outgoing_edges.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident outgoing adjacency has no edge-row tensor",
            )
        })?;
        if edges.elem_count() != adjacency_count {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident outgoing CSR cardinalities differ",
            ));
        }
        Ok(edges)
    }

    fn path_edge_visibility(&self) -> Result<(&Tensor, &Tensor)> {
        let active = self.edge_active.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident adjacency has no edge-active tensor",
            )
        })?;
        let layers = self.edge_layers.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident adjacency has no edge-layer tensor",
            )
        })?;
        Ok((active, layers))
    }

    fn path_weight(&self, property: PropertyId, dummy_bytes: &Tensor) -> Result<PathWeight> {
        if let Some(column) = self.integer_edges.get(&property) {
            let values = column.values.as_ref().ok_or_else(weight_type_error)?;
            let validity = column.validity.as_ref().ok_or_else(weight_type_error)?;
            if column.rows != self.edge_count {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident integer edge-weight column has the wrong row count",
                ));
            }
            return Ok(PathWeight {
                kind: 1,
                homogeneous_values: values.clone(),
                validity: validity.clone(),
                mixed_offsets: self.outgoing_offsets.clone(),
                mixed_bytes: dummy_bytes.clone(),
            });
        }
        if let Some(column) = self.float_edges.get(&property) {
            let values = column.values.as_ref().ok_or_else(weight_type_error)?;
            let validity = column.validity.as_ref().ok_or_else(weight_type_error)?;
            if column.rows != self.edge_count {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "resident float edge-weight column has the wrong row count",
                ));
            }
            return Ok(PathWeight {
                kind: 2,
                homogeneous_values: values.clone(),
                validity: validity.clone(),
                mixed_offsets: self.outgoing_offsets.clone(),
                mixed_bytes: dummy_bytes.clone(),
            });
        }
        if let Some(column) = self.mixed_edges.get(&property) {
            let validity = column.validity.as_ref().ok_or_else(weight_type_error)?;
            let bytes = column.bytes.as_ref().ok_or_else(weight_type_error)?;
            let homogeneous_values =
                Tensor::zeros(1, DType::I64, dummy_bytes.device()).map_err(candle_error)?;
            return Ok(PathWeight {
                kind: 3,
                homogeneous_values,
                validity: validity.clone(),
                mixed_offsets: column.offsets.clone(),
                mixed_bytes: bytes.clone(),
            });
        }
        let validity = self.edge_active.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident graph has no edge-active tensor for lazy weight validation",
            )
        })?;
        Ok(PathWeight {
            // Unsupported scalar shape. The kernel reports QueryType only when a reachable edge
            // is examined, matching the CPU reference's lazy weight callback semantics.
            kind: 4,
            homogeneous_values: Tensor::zeros(1, DType::I64, dummy_bytes.device())
                .map_err(candle_error)?,
            validity: validity.clone(),
            mixed_offsets: self.outgoing_offsets.clone(),
            mixed_bytes: dummy_bytes.clone(),
        })
    }

    #[allow(clippy::too_many_lines)]
    fn metal_reverse_bfs_workspace(
        &self,
        visible_mask: &Tensor,
        source_dense: u32,
        layers: LayerMask,
        maximum_distance: u32,
        cancellation: &CancellationToken,
        deadline: Option<Instant>,
    ) -> Result<Tensor> {
        self.validate_path_source(visible_mask, source_dense, "reverse BFS")?;
        if self.incoming_neighbors.as_ref().is_some_and(|neighbors| {
            bfs_should_start_persistent(self.node_count, neighbors.elem_count())
        }) {
            return self.metal_bfs_persistent_workspace_bounded(
                visible_mask,
                source_dense,
                layers,
                true,
                u32::MAX,
                maximum_distance,
                cancellation,
                deadline,
            );
        }
        let control_offset = self.node_count * 3;
        let workspace_words = control_offset
            .checked_add(PATH_CONTROL_WORDS)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "Metal reverse BFS workspace size overflow",
                )
            })?;
        let mut workspace = allocate_u32_workspace(
            visible_mask,
            workspace_words,
            "irongraph Metal reverse BFS workspace",
        )?;
        for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
            ensure_graph_execution(cancellation, deadline)?;
            let node_end = node_begin
                .saturating_add(PATH_NODE_QUANTUM)
                .min(self.node_count);
            workspace = visible_mask
                .apply_op2_no_bwd(
                    &workspace,
                    &MetalGraphBfsInitialize {
                        node_count: self.node_count,
                        edge_count: self.edge_count,
                        source_dense,
                        layer_mask: u32::from(layers.bits()),
                        workspace_words,
                        node_begin,
                        node_end,
                    },
                )
                .map_err(candle_error)?;
            let status = read_u32_slice(&workspace, control_offset + 1, 1)?
                .first()
                .copied()
                .unwrap_or(1);
            if status != 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "Metal reverse BFS rejected corrupt resident graph metadata",
                ));
            }
            ensure_graph_execution(cancellation, deadline)?;
        }
        let Some(reverse_neighbors) = self.incoming_neighbors.as_ref() else {
            return Ok(workspace);
        };
        let reverse_edges = self.incoming_edges.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident incoming adjacency has no edge-row tensor",
            )
        })?;
        let pull_neighbors = self.outgoing_neighbors.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident incoming adjacency has no outgoing counterpart",
            )
        })?;
        let pull_edges = self.outgoing_edges.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "resident outgoing adjacency has no edge-row tensor",
            )
        })?;
        let (edge_active, edge_layers) = self.path_edge_visibility()?;
        let adjacency_count = reverse_neighbors.elem_count();
        if adjacency_count == 0
            && self.outgoing_overlay.rows.is_empty()
            && self.incoming_overlay.rows.is_empty()
        {
            return Ok(workspace);
        }
        if reverse_edges.elem_count() != adjacency_count
            || pull_neighbors.elem_count() != adjacency_count
            || pull_edges.elem_count() != adjacency_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "resident incoming and outgoing CSR cardinalities differ",
            ));
        }
        let visible_count = visible_mask
            .to_dtype(DType::U32)
            .and_then(|mask| mask.sum_all())
            .and_then(|count| count.to_scalar::<u32>())
            .map_err(candle_error)? as usize;
        let mut control = read_u32_slice(&workspace, control_offset, 2)?;
        let mut frontier_count = control.first().copied().unwrap_or(0) as usize;
        let mut first_depth = 1_usize;
        while first_depth <= self.node_count
            && first_depth <= maximum_distance as usize
            && frontier_count != 0
        {
            let bottom_up = frontier_count.saturating_mul(20) > visible_count;
            let adjacency_tile_domain = adjacency_count.max(self.edge_count);
            for edge_begin in (0..adjacency_tile_domain).step_by(super::METAL_GRAPH_EDGE_TILE) {
                let edge_end = edge_begin
                    .saturating_add(super::METAL_GRAPH_EDGE_TILE)
                    .min(adjacency_tile_domain);
                for node_begin in (0..self.node_count).step_by(PATH_NODE_QUANTUM) {
                    ensure_graph_execution(cancellation, deadline)?;
                    let node_end = node_begin
                        .saturating_add(PATH_NODE_QUANTUM)
                        .min(self.node_count);
                    workspace = workspace
                        .apply_op1_no_bwd(&MetalGraphBfsChunk {
                            outgoing_offsets: self.incoming_offsets.clone(),
                            outgoing_neighbors: reverse_neighbors.clone(),
                            outgoing_edges: reverse_edges.clone(),
                            outgoing_overlay: self
                                .incoming_overlay
                                .packet
                                .clone()
                                .unwrap_or_else(|| self.incoming_offsets.clone()),
                            incoming_offsets: self.outgoing_offsets.clone(),
                            incoming_neighbors: pull_neighbors.clone(),
                            incoming_edges: pull_edges.clone(),
                            incoming_overlay: self
                                .outgoing_overlay
                                .packet
                                .clone()
                                .unwrap_or_else(|| self.outgoing_offsets.clone()),
                            visible_nodes: visible_mask.clone(),
                            edge_active: edge_active.clone(),
                            edge_layers: edge_layers.clone(),
                            node_count: self.node_count,
                            edge_count: self.edge_count,
                            adjacency_count,
                            outgoing_overlay_count: self.incoming_overlay.rows.len(),
                            incoming_overlay_count: self.outgoing_overlay.rows.len(),
                            source_dense,
                            layer_mask: u32::from(layers.bits()),
                            first_depth,
                            level_count: 1,
                            bottom_up,
                            edge_begin,
                            edge_end,
                            prepare: edge_begin == 0,
                            node_begin,
                            node_end,
                        })
                        .map_err(candle_error)?;
                    control = read_u32_slice(&workspace, control_offset, 2)?;
                    if control.get(1).copied().unwrap_or(1) != 0 {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "Metal reverse BFS rejected corrupt resident graph metadata",
                        ));
                    }
                    ensure_graph_execution(cancellation, deadline)?;
                }
            }
            frontier_count = control.first().copied().unwrap_or(0) as usize;
            first_depth = first_depth.saturating_add(1);
            if bfs_should_switch_to_persistent(
                self.node_count,
                adjacency_count,
                first_depth.saturating_sub(1),
                frontier_count,
            ) {
                // Reverse traversal also restarts from its target over incoming CSR. Drop the
                // eight-level probe first so the persistent workspace reuses its allocation.
                drop(control);
                drop(workspace);
                return self.metal_bfs_persistent_workspace_bounded(
                    visible_mask,
                    source_dense,
                    layers,
                    true,
                    u32::MAX,
                    maximum_distance,
                    cancellation,
                    deadline,
                );
            }
        }
        ensure_graph_execution(cancellation, deadline)?;
        Ok(workspace)
    }
}

/// Switch only after the parallel engine has exposed a genuinely narrow frontier. The density
/// guard keeps high-degree workloads on the grid-wide target-owned pull, while the progress guard
/// avoids restarting into the heap after most of the simple-path bound has already elapsed.
const fn weighted_dijkstra_should_switch_to_heap(
    node_count: usize,
    adjacency_count: usize,
    completed_rounds: usize,
    frontier_count: usize,
) -> bool {
    if node_count <= DIJKSTRA_ADAPTIVE_PROBE_ROUNDS
        || completed_rounds < DIJKSTRA_ADAPTIVE_PROBE_ROUNDS
        || frontier_count == 0
        || frontier_count > DIJKSTRA_HEAP_MAX_FRONTIER
        || completed_rounds > node_count / 4
    {
        return false;
    }
    adjacency_count <= node_count.saturating_mul(DIJKSTRA_HEAP_MAX_EDGES_PER_NODE)
}

/// Select the persistent engine before allocating a grid workspace when the graph already meets
/// the same size and density bounds used after the probe horizon. The initial source frontier has
/// exactly one row; denser shapes remain on the direction-optimizing grid path.
pub const fn bfs_should_start_persistent(node_count: usize, adjacency_count: usize) -> bool {
    adjacency_count != 0
        && bfs_should_switch_to_persistent(
            node_count,
            adjacency_count,
            BFS_ADAPTIVE_PROBE_LEVELS,
            1,
        )
}

pub const fn bfs_should_switch_to_persistent(
    node_count: usize,
    adjacency_count: usize,
    completed_levels: usize,
    frontier_count: usize,
) -> bool {
    if node_count <= BFS_ADAPTIVE_PROBE_LEVELS
        || completed_levels < BFS_ADAPTIVE_PROBE_LEVELS
        || frontier_count == 0
        || frontier_count > BFS_PERSISTENT_MAX_FRONTIER
        || completed_levels > node_count / 4
    {
        return false;
    }
    adjacency_count <= node_count.saturating_mul(BFS_PERSISTENT_MAX_EDGES_PER_NODE)
}

const fn persistent_bfs_threadgroup_width(node_count: usize, adjacency_count: usize) -> usize {
    if adjacency_count <= node_count.saturating_mul(BFS_PERSISTENT_NARROW_MAX_EDGES_PER_NODE) {
        BFS_PERSISTENT_NARROW_THREADS
    } else if adjacency_count <= node_count.saturating_mul(BFS_PERSISTENT_SPARSE_MAX_EDGES_PER_NODE)
    {
        BFS_PERSISTENT_SPARSE_THREADS
    } else {
        PATH_THREADS
    }
}

pub fn read_u32_slice(tensor: &Tensor, start: usize, length: usize) -> Result<Vec<u32>> {
    tensor
        .narrow(0, start, length)
        .and_then(|values| values.copy())
        .and_then(|values| values.to_vec1::<u32>())
        .map_err(candle_error)
}

pub fn read_u8_slice(tensor: &Tensor, start: usize, length: usize) -> Result<Vec<u8>> {
    tensor
        .narrow(0, start, length)
        .and_then(|values| values.copy())
        .and_then(|values| values.to_vec1::<u8>())
        .map_err(candle_error)
}

pub fn read_u32_bounded(
    tensor: &Tensor,
    start: usize,
    length: usize,
    cancellation: &CancellationToken,
    deadline: Option<Instant>,
) -> Result<Vec<u32>> {
    let mut output = Vec::with_capacity(length);
    for offset in (0..length).step_by(PATH_NODE_QUANTUM) {
        ensure_graph_execution(cancellation, deadline)?;
        let count = PATH_NODE_QUANTUM.min(length - offset);
        output.extend(read_u32_slice(tensor, start + offset, count)?);
        ensure_graph_execution(cancellation, deadline)?;
    }
    Ok(output)
}

fn path_budget(name: &str) -> Error {
    Error::new(
        ErrorCode::ResultBudgetExceeded,
        format!("{name} result exceeds the query row budget"),
    )
}

fn weight_type_error() -> Error {
    Error::new(
        ErrorCode::QueryType,
        "graph.dijkstra weight must be a present numeric relationship property",
    )
}

fn dijkstra_status(status: u32) -> Result<()> {
    match status {
        0 => Ok(()),
        1 | 2 | 5 | 6 => Err(Error::new(
            ErrorCode::CorruptStorage,
            "Metal weighted Dijkstra rejected corrupt resident graph metadata",
        )),
        3 => Err(weight_type_error()),
        4 => Err(Error::new(
            ErrorCode::QueryType,
            "Dijkstra weight must be finite and non-negative",
        )),
        _ => Err(Error::new(
            ErrorCode::CorruptStorage,
            "Metal weighted Dijkstra returned an unknown status",
        )),
    }
}

pub fn depth_first_scratch_bytes(node_count: usize) -> Result<usize> {
    max_path_phase(&depth_first_scratch_phases(node_count)?, "DFS")
}

pub fn breadth_first_scratch_bytes(node_count: usize) -> Result<usize> {
    max_path_phase(
        &breadth_first_scratch_phases(node_count)?,
        "breadth-first search",
    )
}

pub fn unit_dijkstra_scratch_bytes(node_count: usize) -> Result<usize> {
    max_path_phase(&unit_dijkstra_scratch_phases(node_count)?, "unit Dijkstra")
}

pub fn shortest_path_scratch_bytes(node_count: usize) -> Result<usize> {
    max_path_phase(&shortest_path_scratch_phases(node_count)?, "shortest path")
}

pub fn weighted_dijkstra_scratch_bytes(node_count: usize) -> Result<usize> {
    max_path_phase(
        &weighted_dijkstra_scratch_phases(node_count)?,
        "weighted Dijkstra",
    )
}

fn depth_first_scratch_phases(node_count: usize) -> Result<[usize; 4]> {
    if node_count == 0 {
        return Ok([0; 4]);
    }
    let visible = path_allocation_bytes(node_count, "DFS visibility")?;
    let workspace = path_word_allocation(node_count, 4, PATH_CONTROL_WORDS, "DFS workspace")?;
    let output = path_vector_allocation(node_count, size_of::<u32>(), "DFS output")?;
    let control = path_vector_allocation(PATH_CONTROL_WORDS, size_of::<u32>(), "DFS control")?;
    let read_temporary = bounded_u32_read_temporary(node_count, "DFS bounded read")?;
    Ok([
        source_readback_phase(visible, "DFS")?,
        sum_path_allocations(
            &[visible, workspace, control, control, control],
            "DFS control",
        )?,
        sum_path_allocations(
            &[visible, workspace, output, read_temporary],
            "DFS bounded result readback",
        )?,
        sum_path_allocations(&[visible, workspace, output, output], "DFS publication")?,
    ])
}

fn breadth_first_scratch_phases(node_count: usize) -> Result<[usize; 4]> {
    if node_count == 0 {
        return Ok([0; 4]);
    }
    let visible = path_allocation_bytes(node_count, "breadth-first visibility")?;
    let workspace =
        path_word_allocation(node_count, 3, PATH_CONTROL_WORDS, "breadth-first workspace")?;
    let output = path_vector_allocation(node_count, size_of::<u32>(), "breadth-first output")?;
    let mask = path_vector_allocation(node_count, size_of::<u8>(), "breadth-first reached mask")?;
    let control = path_vector_allocation(
        PATH_CONTROL_WORDS,
        size_of::<u32>(),
        "breadth-first control",
    )?;
    let selection = super::graph_components_metrics::shared_selection_scratch_bytes(
        node_count,
        "breadth-first reached selection",
    )?;
    let read_temporary = bounded_u32_read_temporary(node_count, "breadth-first bounded read")?;
    Ok([
        source_readback_phase(visible, "breadth-first search")?,
        sum_path_allocations(
            &[visible, workspace, control, control, control],
            "breadth-first control",
        )?,
        sum_path_allocations(
            &[visible, workspace, selection],
            "breadth-first reached selection",
        )?,
        sum_path_allocations(
            &[
                visible,
                workspace,
                mask,
                output,
                output,
                output,
                output,
                read_temporary,
            ],
            "breadth-first compact publication",
        )?,
    ])
}

fn unit_dijkstra_scratch_phases(node_count: usize) -> Result<[usize; 5]> {
    if node_count == 0 {
        return Ok([0; 5]);
    }
    let visible = path_allocation_bytes(node_count, "unit Dijkstra visibility")?;
    let workspace = path_word_allocation(
        node_count,
        3,
        PATH_CONTROL_WORDS,
        "unit Dijkstra BFS workspace",
    )?;
    let packet = path_word_allocation(node_count, 3, 1, "unit Dijkstra packet")?;
    let rows = path_vector_allocation(node_count, size_of::<u32>(), "unit Dijkstra rows")?;
    let costs = path_vector_allocation(node_count, size_of::<f64>(), "unit Dijkstra costs")?;
    let predecessors = path_vector_allocation(
        node_count,
        size_of::<Option<u32>>(),
        "unit Dijkstra predecessors",
    )?;
    let control = path_vector_allocation(
        PATH_CONTROL_WORDS,
        size_of::<u32>(),
        "unit Dijkstra control",
    )?;
    let scalar = path_vector_allocation(1, size_of::<u32>(), "unit Dijkstra status")?;
    let mask = path_vector_allocation(node_count, size_of::<u8>(), "unit Dijkstra reached mask")?;
    let selection = super::graph_components_metrics::shared_selection_scratch_bytes(
        node_count,
        "unit Dijkstra reached selection",
    )?;
    let row_read = bounded_u32_read_temporary(node_count, "unit Dijkstra row read")?;
    let cost_read = bounded_u32_read_temporary(
        node_count.checked_mul(2).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Metal unit Dijkstra compact cost length overflow",
            )
        })?,
        "unit Dijkstra cost read",
    )?;
    let gpu_costs = path_vector_allocation(
        node_count,
        size_of::<u32>() * 2,
        "unit Dijkstra compact cost words",
    )?;
    let publication_base = sum_path_allocations(
        &[visible, workspace, mask, rows, packet, gpu_costs, rows],
        "unit Dijkstra compact device publication",
    )?;
    let publication = max_path_phase(
        &[
            sum_path_allocations(
                &[publication_base, rows, row_read],
                "unit Dijkstra row read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, cost_read],
                "unit Dijkstra cost read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, rows, row_read],
                "unit Dijkstra predecessor read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, rows, costs, predecessors],
                "unit Dijkstra host publication",
            )?,
        ],
        "unit Dijkstra compact publication",
    )?;
    Ok([
        source_readback_phase(visible, "unit Dijkstra")?,
        sum_path_allocations(
            &[visible, workspace, control, control, control],
            "unit Dijkstra BFS control",
        )?,
        sum_path_allocations(
            &[visible, workspace, selection],
            "unit Dijkstra reached selection",
        )?,
        sum_path_allocations(
            &[
                visible, workspace, mask, rows, packet, scalar, scalar, scalar,
            ],
            "unit Dijkstra finalizer control",
        )?,
        publication,
    ])
}

fn shortest_path_scratch_phases(node_count: usize) -> Result<[usize; 5]> {
    if node_count == 0 {
        return Ok([0; 5]);
    }
    let visible = path_allocation_bytes(node_count, "shortest-path visibility")?;
    let bfs = path_word_allocation(
        node_count,
        3,
        PATH_CONTROL_WORDS,
        "shortest-path BFS workspace",
    )?;
    let reconstruction = path_word_allocation(
        node_count,
        2,
        PATH_CONTROL_WORDS,
        "shortest-path reconstruction",
    )?;
    let output = path_vector_allocation(node_count, size_of::<u32>(), "shortest-path output")?;
    let control = path_vector_allocation(
        PATH_CONTROL_WORDS,
        size_of::<u32>(),
        "shortest-path control",
    )?;
    let read_temporary = bounded_u32_read_temporary(node_count, "shortest-path bounded read")?;
    let reconstruction_base = sum_path_allocations(
        &[visible, bfs, bfs, reconstruction],
        "shortest-path reconstruction base",
    )?;
    Ok([
        source_readback_phase(visible, "shortest path")?,
        sum_path_allocations(
            &[visible, bfs, control, control, control],
            "shortest-path forward control",
        )?,
        sum_path_allocations(
            &[visible, bfs, bfs, control, control, control],
            "shortest-path reverse control",
        )?,
        max_path_phase(
            &[
                sum_path_allocations(
                    &[reconstruction_base, control, control, control],
                    "shortest-path reconstruction control",
                )?,
                sum_path_allocations(
                    &[reconstruction_base, output, read_temporary],
                    "shortest-path node read",
                )?,
            ],
            "shortest-path reconstruction",
        )?,
        sum_path_allocations(
            &[reconstruction_base, output, output, read_temporary],
            "shortest-path edge read and publication",
        )?,
    ])
}

fn weighted_dijkstra_scratch_phases(node_count: usize) -> Result<[usize; 5]> {
    if node_count == 0 {
        return Ok([0; 5]);
    }
    let visible = path_allocation_bytes(node_count, "weighted Dijkstra visibility")?;
    let workspace = path_word_allocation(
        node_count,
        7,
        PATH_CONTROL_WORDS,
        "weighted Dijkstra workspace",
    )?;
    let packet = path_word_allocation(node_count, 3, 1, "weighted Dijkstra packet")?;
    let rows = path_vector_allocation(node_count, size_of::<u32>(), "weighted Dijkstra rows")?;
    let costs = path_vector_allocation(node_count, size_of::<f64>(), "weighted Dijkstra costs")?;
    let predecessors = path_vector_allocation(
        node_count,
        size_of::<Option<u32>>(),
        "weighted Dijkstra predecessors",
    )?;
    let control = path_vector_allocation(
        PATH_CONTROL_WORDS,
        size_of::<u32>(),
        "weighted Dijkstra control",
    )?;
    let dummy = path_allocation_bytes(
        DType::I64.size_in_bytes(),
        "weighted Dijkstra dummy binding",
    )?;
    let scalar = path_vector_allocation(1, size_of::<u32>(), "weighted Dijkstra status")?;
    let mask = path_vector_allocation(
        node_count,
        size_of::<u8>(),
        "weighted Dijkstra reached mask",
    )?;
    let selection = super::graph_components_metrics::shared_selection_scratch_bytes(
        node_count,
        "weighted Dijkstra reached selection",
    )?;
    let row_read = bounded_u32_read_temporary(node_count, "weighted Dijkstra row read")?;
    let doubled_count = node_count.checked_mul(2).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Metal weighted Dijkstra compact cost length overflow",
        )
    })?;
    let cost_read = bounded_u32_read_temporary(doubled_count, "weighted Dijkstra cost read")?;
    let gpu_costs = path_vector_allocation(
        node_count,
        size_of::<u32>() * 2,
        "weighted Dijkstra compact cost words",
    )?;
    let publication_base = sum_path_allocations(
        &[
            visible, workspace, dummy, mask, rows, packet, gpu_costs, rows,
        ],
        "weighted Dijkstra compact device publication",
    )?;
    let publication = max_path_phase(
        &[
            sum_path_allocations(
                &[publication_base, rows, row_read],
                "weighted Dijkstra row read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, cost_read],
                "weighted Dijkstra cost read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, rows, row_read],
                "weighted Dijkstra predecessor read",
            )?,
            sum_path_allocations(
                &[publication_base, rows, gpu_costs, rows, costs, predecessors],
                "weighted Dijkstra host publication",
            )?,
        ],
        "weighted Dijkstra compact publication",
    )?;
    Ok([
        source_readback_phase(visible, "weighted Dijkstra")?,
        sum_path_allocations(
            &[visible, workspace, control, control, control, dummy],
            "weighted Dijkstra control",
        )?,
        sum_path_allocations(
            &[visible, workspace, dummy, selection],
            "weighted Dijkstra reached selection",
        )?,
        sum_path_allocations(
            &[
                visible, workspace, dummy, mask, rows, packet, scalar, scalar, scalar,
            ],
            "weighted Dijkstra finalizer control",
        )?,
        publication,
    ])
}

fn source_readback_phase(visible: usize, name: &str) -> Result<usize> {
    let scalar = path_allocation_bytes(1, name)?;
    sum_path_allocations(&[visible, scalar, scalar, scalar], name)
}

/// `read_u32_slice` owns one private contiguous copy, one shared Metal-to-CPU staging buffer, and
/// the temporary host vector returned by Candle. `read_u32_bounded` caps all three at this quantum
/// while its separately-accounted destination vector remains live across chunks.
fn bounded_u32_read_temporary(elements: usize, name: &str) -> Result<usize> {
    if elements == 0 {
        return Ok(0);
    }
    let chunk = elements.min(PATH_NODE_QUANTUM);
    let allocation = path_vector_allocation(chunk, size_of::<u32>(), name)?;
    sum_path_allocations(&[allocation, allocation, allocation], name)
}

fn path_vector_allocation(elements: usize, bytes: usize, name: &str) -> Result<usize> {
    let logical = elements.checked_mul(bytes).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("Metal {name} host vector overflow"),
        )
    })?;
    path_allocation_bytes(logical, name)
}

fn max_path_phase(phases: &[usize], name: &str) -> Result<usize> {
    phases.iter().copied().max().ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("Metal {name} scratch phase inventory is empty"),
        )
    })
}

fn path_word_allocation(
    node_count: usize,
    words_per_node: usize,
    fixed_words: usize,
    name: &str,
) -> Result<usize> {
    let logical = node_count
        .checked_mul(words_per_node)
        .and_then(|words| words.checked_add(fixed_words))
        .and_then(|words| words.checked_mul(DType::U32.size_in_bytes()))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                format!("Metal {name} logical allocation overflow"),
            )
        })?;
    path_allocation_bytes(logical, name)
}

fn path_allocation_bytes(logical: usize, name: &str) -> Result<usize> {
    if logical == 0 {
        return Ok(0);
    }
    // Candle 0.11's pooled Metal `buf_size` is exactly `next_power_of_two`; both private
    // workspaces and shared `to_cpu` staging use that policy. Unobservable driver/page overhead
    // belongs to the backend's configured safety reserve, not to a fabricated per-buffer floor.
    logical.checked_next_power_of_two().ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("Metal {name} allocator bucket overflow"),
        )
    })
}

fn sum_path_allocations(allocations: &[usize], name: &str) -> Result<usize> {
    allocations.iter().try_fold(0_usize, |total, bytes| {
        total.checked_add(*bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                format!("Metal {name} allocator-rounded scratch overflow"),
            )
        })
    })
}

fn checked_candle_u32(value: usize, label: &str) -> candle_core::Result<u32> {
    u32::try_from(value).map_err(|_| candle_core::Error::Msg(format!("Metal {label} exceeds u32")))
}

/// Reject a launch before any MSL `uint` workspace-offset arithmetic can wrap. `fixed_words`
/// includes the trailing control packet, so the final accessible word remains representable too.
pub fn validate_u32_workspace_words(
    node_count: usize,
    words_per_node: usize,
    fixed_words: usize,
    label: &str,
) -> candle_core::Result<()> {
    let words = node_count
        .checked_mul(words_per_node)
        .and_then(|value| value.checked_add(fixed_words))
        .ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "Metal {label} workspace offset calculation overflowed"
            ))
        })?;
    if words > u32::MAX as usize {
        return Err(candle_core::Error::Msg(format!(
            "Metal {label} workspace exceeds the exact u32 kernel-offset limit"
        )));
    }
    Ok(())
}

fn validate_vector(
    storage: &MetalStorage,
    layout: &Layout,
    dtype: DType,
    count: usize,
    name: &str,
) -> candle_core::Result<()> {
    if storage.dtype() != dtype
        || !layout.is_contiguous()
        || layout.dims().len() != 1
        || layout.shape().elem_count() != count
    {
        return Err(candle_core::Error::Msg(format!(
            "Metal {name} tensor contract is invalid"
        )));
    }
    Ok(())
}

fn validate_bound_vector(
    tensor: &Tensor,
    layout: &Layout,
    dtype: DType,
    count: usize,
    name: &str,
) -> candle_core::Result<()> {
    if tensor.dtype() != dtype
        || !layout.is_contiguous()
        || layout.dims().len() != 1
        || layout.shape().elem_count() != count
    {
        return Err(candle_core::Error::Msg(format!(
            "Metal {name} tensor contract is invalid"
        )));
    }
    Ok(())
}

fn validate_csr_overlay(
    tensor: &Tensor,
    layout: &Layout,
    row_count: usize,
    name: &str,
) -> candle_core::Result<()> {
    let minimum_words = row_count.saturating_mul(2).saturating_add(1);
    if tensor.dtype() != DType::U32
        || !layout.is_contiguous()
        || layout.dims().len() != 1
        || layout.shape().elem_count() < minimum_words
    {
        return Err(candle_core::Error::Msg(format!(
            "Metal {name} tensor contract is invalid"
        )));
    }
    Ok(())
}

fn bind_tensor(
    encoder: &candle_metal_kernels::metal::ComputeCommandEncoder,
    index: usize,
    storage: &MetalStorage,
    layout: &Layout,
    dtype: DType,
) {
    encoder.set_input_buffer(
        index,
        Some(storage.buffer()),
        layout.start_offset() * dtype.size_in_bytes(),
    );
}

fn dispatch_rows(encoder: &candle_metal_kernels::metal::ComputeCommandEncoder, rows: usize) {
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows.div_ceil(PATH_THREADS),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: PATH_THREADS,
            height: 1,
            depth: 1,
        },
    );
}

fn dispatch_group(encoder: &candle_metal_kernels::metal::ComputeCommandEncoder) {
    dispatch_group_width(encoder, PATH_THREADS);
}

fn dispatch_group_width(
    encoder: &candle_metal_kernels::metal::ComputeCommandEncoder,
    width: usize,
) {
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
struct ArithmeticProbe {
    mode: u32,
    count: usize,
}

#[cfg(test)]
impl CustomOp1 for ArithmeticProbe {
    fn name(&self) -> &'static str {
        "irongraph-metal-path-arithmetic-probe"
    }

    fn cpu_fwd(
        &self,
        _input: &CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        Err(candle_core::Error::Msg(
            "Metal path arithmetic probe cannot execute on CPU".to_owned(),
        ))
    }

    #[allow(clippy::significant_drop_tightening)]
    fn metal_fwd(
        &self,
        input: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let input_count = self.count * if self.mode == 1 { 2 } else { 1 };
        validate_vector(
            input,
            layout,
            DType::I64,
            input_count,
            "arithmetic probe input",
        )?;
        if self.count == 0 || !matches!(self.mode, 1 | 2) {
            return Err(candle_core::Error::Msg(
                "Metal path arithmetic probe arguments are invalid".to_owned(),
            ));
        }
        let count = checked_candle_u32(self.count, "arithmetic probe count")?;
        let device = input.device();
        let output = device
            .new_buffer_builder()
            .with_size_for(self.count, DType::I64)
            .with_label("irongraph exact binary64 arithmetic probe")
            .build()?;
        let pipelines = path_pipelines(device)?;
        let encoder = device.command_encoder()?;
        let encoder = encoder.as_ref();
        encoder.set_compute_pipeline_state(&pipelines.arithmetic_probe);
        encoder.set_input_buffer(
            0,
            Some(input.buffer()),
            layout.start_offset() * DType::I64.size_in_bytes(),
        );
        encoder.set_output_buffer(1, Some(&output), 0);
        encoder.set_bytes(2, &count);
        encoder.set_bytes(3, &self.mode);
        dispatch_rows(encoder, self.count);
        Ok((
            MetalStorage::new(output, device.clone(), self.count, DType::I64),
            Shape::from(self.count),
        ))
    }
}

#[cfg(test)]
const PATH_KERNEL_SOURCE: &str = include_str!("../../../../kernels/metal/graph_paths.metal");

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use candle_core::Tensor;

    use crate::Result;

    use super::{
        ArithmeticProbe, PATH_KERNEL_SOURCE, bfs_should_start_persistent,
        bfs_should_switch_to_persistent, breadth_first_scratch_bytes, candle_error,
        depth_first_scratch_bytes, shortest_path_scratch_bytes, unit_dijkstra_scratch_bytes,
        validate_u32_workspace_words, weighted_dijkstra_scratch_bytes,
        weighted_dijkstra_should_switch_to_heap,
    };

    #[test]
    fn path_library_contains_all_exact_procedure_kernels() {
        for name in [
            "ig_path_dfs_initialize",
            "ig_path_dfs_chunk",
            "ig_path_shortest_initialize",
            "ig_path_shortest_chunk",
            "ig_path_bfs_persistent_initialize",
            "ig_path_bfs_persistent_chunk",
            "ig_path_dijkstra_initialize",
            "ig_path_dijkstra_heap_initialize",
            "ig_path_dijkstra_heap_chunk",
            "ig_path_dijkstra_prepare",
            "ig_path_dijkstra_relax",
            "ig_path_dijkstra_publish",
            "ig_path_dijkstra_finalize_prepare",
            "ig_path_dijkstra_finalize",
        ] {
            assert!(PATH_KERNEL_SOURCE.contains(&format!("kernel void {name}")));
        }
        assert!(PATH_KERNEL_SOURCE.contains("ig_path_f64_add_nonnegative"));
        assert!(!PATH_KERNEL_SOURCE.contains("double"));
    }

    #[test]
    fn weighted_strategy_requires_a_measured_narrow_sparse_frontier() {
        assert!(!weighted_dijkstra_should_switch_to_heap(1_000, 1_998, 7, 1));
        assert!(weighted_dijkstra_should_switch_to_heap(1_000, 1_998, 8, 1));
        assert!(!weighted_dijkstra_should_switch_to_heap(
            1_000, 1_998, 8, 33
        ));
        assert!(!weighted_dijkstra_should_switch_to_heap(1_000, 8_001, 8, 1));
        assert!(!weighted_dijkstra_should_switch_to_heap(
            1_000, 1_998, 251, 1
        ));
    }

    #[test]
    fn bfs_strategy_admits_sparse_start_and_requires_narrow_restart_frontier() {
        assert!(bfs_should_start_persistent(20_000, 160_000));
        assert!(!bfs_should_start_persistent(20_000, 160_001));
        assert!(!bfs_should_start_persistent(20_000, 0));
        assert!(!bfs_should_switch_to_persistent(20_000, 160_000, 7, 8));
        assert!(bfs_should_switch_to_persistent(20_000, 160_000, 8, 8));
        assert!(!bfs_should_switch_to_persistent(20_000, 160_000, 8, 33));
        assert!(!bfs_should_switch_to_persistent(20_000, 160_001, 8, 8));
        assert!(!bfs_should_switch_to_persistent(20_000, 160_000, 5_001, 8));
        assert_eq!(super::persistent_bfs_threadgroup_width(5_000, 4_999), 1);
        assert_eq!(super::persistent_bfs_threadgroup_width(20_000, 160_000), 8);
    }

    #[test]
    fn kernel_workspace_offsets_reject_the_first_wrapping_node_count() {
        for (words_per_node, fixed_words) in [(2, 8), (3, 8), (4, 8), (7, 8)] {
            let maximum = (u32::MAX as usize - fixed_words) / words_per_node;
            validate_u32_workspace_words(maximum, words_per_node, fixed_words, "boundary")
                .expect("the exact safe u32 boundary must be admitted");
            assert!(
                validate_u32_workspace_words(maximum + 1, words_per_node, fixed_words, "boundary")
                    .is_err(),
                "the first wrapping node count must fail for {words_per_node}N+{fixed_words}"
            );
        }
        assert!(validate_u32_workspace_words(usize::MAX, 7, 8, "overflow").is_err());
    }

    #[test]
    fn scratch_estimates_cover_simultaneously_live_path_workspaces() -> Result<()> {
        // Candle 0.11 pools each private and shared Metal buffer in its exact next-power-of-two
        // request bucket. These values enumerate the simultaneously live buckets, including
        // readback staging and restart retention, rather than summing logical tensor widths.
        let dfs = super::depth_first_scratch_phases(10)?;
        let bfs = super::breadth_first_scratch_phases(10)?;
        let unit = super::unit_dijkstra_scratch_phases(10)?;
        let shortest = super::shortest_path_scratch_phases(10)?;
        let weighted = super::weighted_dijkstra_scratch_phases(10)?;
        assert_eq!(dfs, [19, 368, 528, 400]);
        assert_eq!(bfs, [19, 368, 1_056, 736]);
        assert_eq!(unit, [19, 368, 1_056, 492, 1_248]);
        assert_eq!(shortest, [19, 368, 624, 912, 976]);
        assert_eq!(weighted, [19, 632, 1_320, 756, 1_512]);
        assert_eq!(depth_first_scratch_bytes(10)?, 528);
        assert_eq!(breadth_first_scratch_bytes(10)?, 1_056);
        assert_eq!(unit_dijkstra_scratch_bytes(10)?, 1_248);
        assert_eq!(shortest_path_scratch_bytes(10)?, 976);
        assert_eq!(weighted_dijkstra_scratch_bytes(10)?, 1_512);
        assert_eq!(depth_first_scratch_bytes(0)?, 0);
        assert_eq!(breadth_first_scratch_bytes(0)?, 0);
        assert_eq!(unit_dijkstra_scratch_bytes(0)?, 0);
        assert_eq!(shortest_path_scratch_bytes(0)?, 0);
        assert_eq!(weighted_dijkstra_scratch_bytes(0)?, 0);
        assert!(depth_first_scratch_bytes(usize::MAX).is_err());
        assert!(breadth_first_scratch_bytes(usize::MAX).is_err());
        assert!(unit_dijkstra_scratch_bytes(usize::MAX).is_err());
        assert!(shortest_path_scratch_bytes(usize::MAX).is_err());
        assert!(weighted_dijkstra_scratch_bytes(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn bounded_publication_readback_stops_growing_at_the_node_quantum() -> Result<()> {
        let quantum_bytes = super::path_vector_allocation(
            super::PATH_NODE_QUANTUM,
            size_of::<u32>(),
            "publication boundary",
        )?;
        assert_eq!(quantum_bytes, 1 << 20);
        assert_eq!(
            super::bounded_u32_read_temporary(super::PATH_NODE_QUANTUM, "boundary")?,
            quantum_bytes * 3,
        );
        assert_eq!(
            super::bounded_u32_read_temporary(super::PATH_NODE_QUANTUM + 1, "boundary")?,
            quantum_bytes * 3,
            "the first element beyond the publication quantum must start a new host pass, not a larger GPU/staging allocation",
        );
        assert!(
            breadth_first_scratch_bytes(super::PATH_NODE_QUANTUM + 1)?
                >= breadth_first_scratch_bytes(super::PATH_NODE_QUANTUM)?
        );
        assert!(
            weighted_dijkstra_scratch_bytes(super::PATH_NODE_QUANTUM + 1)?
                >= weighted_dijkstra_scratch_bytes(super::PATH_NODE_QUANTUM)?
        );
        Ok(())
    }

    #[test]
    fn real_metal_software_binary64_matches_cpu_for_adversarial_and_random_inputs() -> Result<()> {
        let _guard = crate::metal_test_guard();
        let Some(device) = crate::metal_test_device() else {
            return Ok(());
        };
        let mut pairs = vec![
            (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
            (f64::from_bits(1).to_bits(), f64::from_bits(1).to_bits()),
            (
                f64::from_bits(0x000f_ffff_ffff_ffff).to_bits(),
                1.0_f64.to_bits(),
            ),
            (f64::MIN_POSITIVE.to_bits(), f64::from_bits(1).to_bits()),
            (1.0_f64.to_bits(), 2.0_f64.powi(-53).to_bits()),
            (1.0_f64.to_bits(), (3.0 * 2.0_f64.powi(-53)).to_bits()),
            (f64::MAX.to_bits(), f64::MAX.to_bits()),
            (f64::MAX.to_bits(), (-0.0_f64).to_bits()),
            (2.0_f64.powi(1023).to_bits(), 2.0_f64.powi(970).to_bits()),
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut left = state & 0x7fff_ffff_ffff_ffff;
            if (left >> 52) == 0x7ff {
                left &= !(0x7ff_u64 << 52);
            }
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let mut right = state & 0x7fff_ffff_ffff_ffff;
            if (right >> 52) == 0x7ff {
                right &= !(0x7ff_u64 << 52);
            }
            pairs.push((left, right));
        }
        let input = pairs
            .iter()
            .flat_map(|(left, right)| [*left as i64, *right as i64])
            .collect::<Vec<_>>();
        let actual = Tensor::from_slice(&input, input.len(), &device)
            .map_err(candle_error)?
            .apply_op1_no_bwd(&ArithmeticProbe {
                mode: 1,
                count: pairs.len(),
            })
            .and_then(|values| values.to_vec1::<i64>())
            .map_err(candle_error)?;
        for ((left, right), actual) in pairs.into_iter().zip(actual) {
            let expected = (f64::from_bits(left) + f64::from_bits(right)).to_bits();
            assert_eq!(
                actual as u64, expected,
                "software binary64 add differs: left={left:016x} right={right:016x}"
            );
        }

        let integers = [
            0_i64,
            1,
            2,
            (1_i64 << 52) - 1,
            1_i64 << 52,
            (1_i64 << 53) - 1,
            1_i64 << 53,
            (1_i64 << 53) + 1,
            (1_i64 << 53) + 3,
            (1_i64 << 62) - 1,
            i64::MAX,
        ];
        let actual = Tensor::from_slice(&integers, integers.len(), &device)
            .map_err(candle_error)?
            .apply_op1_no_bwd(&ArithmeticProbe {
                mode: 2,
                count: integers.len(),
            })
            .and_then(|values| values.to_vec1::<i64>())
            .map_err(candle_error)?;
        for (value, actual) in integers.into_iter().zip(actual) {
            assert_eq!(actual as u64, (value as f64).to_bits(), "integer={value}");
        }
        Ok(())
    }
}
