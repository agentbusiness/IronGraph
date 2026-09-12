#include <metal_stdlib>
using namespace metal;

/// Highest valid value of the resident layer byte: OBSERVED=0, KNOWLEDGE=1, WORKSPACE=2.
/// Integrity checks compare against this rather than a literal so a fourth layer is a one-line
/// change instead of a hunt through every kernel.
constant uint IG_MAX_LAYER = 2u;
constant ulong IG_MAX_LAYER_UL = 2ul;


// Native resident graph components and undirected metrics.  All kernels consume the canonical
// outgoing/incoming CSR plus visibility columns directly.  No edge list or adjacency is rebuilt
// on the host.  Every global dependency crosses a dispatch boundary; Rust encodes those phases
// into one ordered command buffer and reads back only bounded control words between chunks.

constant uint IG_CM_UNREACHED = 0xffffffffu;

struct IgCmArgs {
    uint node_count;
    uint edge_count;
    uint adjacency_count;
    uint layer_mask;
    uint scalar;
    uint partial_count;
    uint reserved_1;
    uint reserved_2;
};

inline bool ig_cm_layer_visible(uchar layer, uint mask) {
    return layer < 32u && (mask & (1u << uint(layer))) != 0u;
}

inline void ig_cm_status(device atomic_uint* status, uint value) {
    atomic_fetch_max_explicit(status, value, memory_order_relaxed);
}

kernel void ig_cm_control_clear(
    device atomic_uint* control [[buffer(0)]],
    constant uint4& values [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    atomic_store_explicit(control + 0, values.x, memory_order_relaxed);
    atomic_store_explicit(control + 1, values.y, memory_order_relaxed);
    atomic_store_explicit(control + 2, values.z, memory_order_relaxed);
    atomic_store_explicit(control + 3, values.w, memory_order_relaxed);
}

inline bool ig_cm_validate_metadata(
    uint edge,
    uint neighbor,
    device const uchar* visible_nodes,
    device const uchar* edge_active,
    device const uchar* edge_layers,
    constant IgCmArgs& args,
    device atomic_uint* status) {
    if (edge >= args.edge_count || neighbor >= args.node_count) {
        ig_cm_status(status, 2u);
        return false;
    }
    uchar visible = visible_nodes[neighbor];
    uchar active = edge_active[edge];
    uchar layer = edge_layers[edge];
    if (visible > 1u || active > 1u || layer > IG_MAX_LAYER) {
        ig_cm_status(status, 3u);
        return false;
    }
    return visible != 0u && active != 0u && ig_cm_layer_visible(layer, args.layer_mask);
}

kernel void ig_cm_wcc_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device atomic_uint* labels [[buffer(1)]],
    device atomic_uint* control [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint label = visible_nodes[node] == 1u ? node : IG_CM_UNREACHED;
    atomic_store_explicit(labels + node, label, memory_order_relaxed);
    if (visible_nodes[node] > 1u) ig_cm_status(control + 1, 3u);
}

kernel void ig_cm_clear_changed(
    device atomic_uint* changed [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0u) atomic_store_explicit(changed, 0u, memory_order_relaxed);
}

// Canonical component publication is entirely device-side. First compute the minimum dense row
// belonging to every opaque algorithm root, then place one flag at each component minimum. An
// inclusive GPU scan over those flags gives the exact CPU component numbering: components are
// ordered by their first visible dense row, independently of which representative an algorithm
// happened to choose.
kernel void ig_cm_component_canonical_initialize(
    device atomic_uint* root_minimum [[buffer(0)]],
    device uint* minimum_flags [[buffer(1)]],
    constant IgCmArgs& args [[buffer(2)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) return;
    size_t node = size_t(args.scalar) + size_t(local_node);
    atomic_store_explicit(root_minimum + node, IG_CM_UNREACHED, memory_order_relaxed);
    minimum_flags[node] = 0u;
}

kernel void ig_cm_component_canonical_mark(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device atomic_uint* root_minimum [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint node = args.scalar + local_node;
    uchar visible = visible_nodes[node];
    if (visible > 1u) {
        ig_cm_status(status, 3u);
        return;
    }
    if (visible == 0u) return;
    uint root = assignment[node];
    if (root >= args.node_count) {
        ig_cm_status(status, 2u);
        return;
    }
    atomic_fetch_min_explicit(root_minimum + root, node, memory_order_relaxed);
}

kernel void ig_cm_component_canonical_flags(
    device const atomic_uint* root_minimum [[buffer(0)]],
    device atomic_uint* minimum_flags [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint node = args.scalar + local_node;
    uint minimum = atomic_load_explicit(root_minimum + node, memory_order_relaxed);
    if (minimum < args.node_count) {
        atomic_store_explicit(minimum_flags + minimum, 1u, memory_order_relaxed);
    }
}

kernel void ig_cm_wcc_relax_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device atomic_uint* labels [[buffer(6)]],
    device atomic_uint* control [[buffer(7)]],
    constant IgCmArgs& args [[buffer(8)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(control + 1, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count
            || visible_nodes[source] != 1u
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, control + 1)) return;
    uint source_label = atomic_load_explicit(labels + source, memory_order_relaxed);
    uint target_label = atomic_load_explicit(labels + target, memory_order_relaxed);
    uint best = min(source_label, target_label);
    uint previous_source = atomic_fetch_min_explicit(labels + source, best, memory_order_relaxed);
    uint previous_target = atomic_fetch_min_explicit(labels + target, best, memory_order_relaxed);
    if (previous_source > best || previous_target > best) {
        atomic_store_explicit(control, 1u, memory_order_relaxed);
    }
}

kernel void ig_cm_wcc_compress(
    device const uchar* visible_nodes [[buffer(0)]],
    device atomic_uint* labels [[buffer(1)]],
    device atomic_uint* changed [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint previous = atomic_load_explicit(labels + node, memory_order_relaxed);
    uint label = previous;
    if (visible_nodes[node] == 1u && label < args.node_count) {
        label = min(label, atomic_load_explicit(labels + label, memory_order_relaxed));
    }
    atomic_store_explicit(labels + node, label, memory_order_relaxed);
    if (label != previous) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device uint* assignment [[buffer(1)]],
    device atomic_uint* in_degree [[buffer(2)]],
    device atomic_uint* out_degree [[buffer(3)]],
    device atomic_uint* control [[buffer(4)]],
    constant IgCmArgs& args [[buffer(5)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    assignment[node] = IG_CM_UNREACHED;
    atomic_store_explicit(in_degree + node, 0u, memory_order_relaxed);
    atomic_store_explicit(out_degree + node, 0u, memory_order_relaxed);
    if (visible_nodes[node] > 1u) ig_cm_status(control + 1, 3u);
}

// Batched SCC path. Directed zero-in/zero-out trimming maintains degrees with frontier
// decrements, so every relationship is revisited only when an endpoint leaves the active graph.
// The remaining cyclic core uses stable maximum-ancestor colors and simultaneously extracts one
// backward-reachable SCC for every independent color. Assignment stores an opaque root row;
// publication canonically ranks roots by their minimum dense node.

kernel void ig_cm_scc_initial_degree_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device const uint* assignment [[buffer(6)]],
    device atomic_uint* in_degree [[buffer(7)]],
    device atomic_uint* out_degree [[buffer(8)]],
    device atomic_uint* status [[buffer(9)]],
    constant IgCmArgs& args [[buffer(10)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    if (edge >= args.edge_count) {
        ig_cm_status(status, 2u);
        return;
    }
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count || visible_nodes[source] != 1u
            || assignment[source] != IG_CM_UNREACHED
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, status)
            || assignment[target] != IG_CM_UNREACHED) return;
    atomic_fetch_add_explicit(out_degree + source, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(in_degree + target, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_trim_mark(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device const atomic_uint* in_degree [[buffer(2)]],
    device const atomic_uint* out_degree [[buffer(3)]],
    device uint* candidate [[buffer(4)]],
    device atomic_uint* count [[buffer(5)]],
    constant IgCmArgs& args [[buffer(6)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    bool trim = visible_nodes[node] == 1u && assignment[node] == IG_CM_UNREACHED
        && (atomic_load_explicit(in_degree + node, memory_order_relaxed) == 0u
            || atomic_load_explicit(out_degree + node, memory_order_relaxed) == 0u);
    candidate[node] = trim ? 1u : 0u;
    if (trim) atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_assign_candidates(
    device const uint* candidate [[buffer(0)]],
    device uint* assignment [[buffer(1)]],
    device atomic_uint* remaining [[buffer(2)]],
    device atomic_uint* assigned [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count || candidate[node] == 0u
            || assignment[node] != IG_CM_UNREACHED) return;
    assignment[node] = node;
    atomic_fetch_sub_explicit(remaining, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(assigned, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_trim_decrement_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device const uint* assignment [[buffer(6)]],
    device const uint* candidate [[buffer(7)]],
    device atomic_uint* in_degree [[buffer(8)]],
    device atomic_uint* out_degree [[buffer(9)]],
    device atomic_uint* status [[buffer(10)]],
    constant IgCmArgs& args [[buffer(11)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    if (edge >= args.edge_count) {
        ig_cm_status(status, 2u);
        return;
    }
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count || visible_nodes[source] != 1u
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, status)) return;
    if (candidate[source] != 0u && assignment[target] == IG_CM_UNREACHED) {
        uint previous = atomic_fetch_sub_explicit(in_degree + target, 1u, memory_order_relaxed);
        if (previous == 0u) ig_cm_status(status, 6u);
    }
    if (candidate[target] != 0u && assignment[source] == IG_CM_UNREACHED) {
        uint previous = atomic_fetch_sub_explicit(out_degree + source, 1u, memory_order_relaxed);
        if (previous == 0u) ig_cm_status(status, 6u);
    }
}

// If every remaining vertex has exactly one visible incoming and outgoing relationship, the
// residual graph is a disjoint union of directed cycles. Probe without disturbing maintained
// degrees; the host selects the logarithmic pointer-jump path only when every residual row passes.
kernel void ig_cm_scc_cycle_probe_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device const atomic_uint* in_degree [[buffer(2)]],
    device const atomic_uint* out_degree [[buffer(3)]],
    device atomic_uint* successor [[buffer(4)]],
    device uint* label [[buffer(5)]],
    device atomic_uint* control [[buffer(6)]],
    constant IgCmArgs& args [[buffer(7)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    atomic_store_explicit(successor + node, IG_CM_UNREACHED, memory_order_relaxed);
    label[node] = IG_CM_UNREACHED;
    if (visible_nodes[node] != 1u || assignment[node] != IG_CM_UNREACHED) return;
    if (atomic_load_explicit(in_degree + node, memory_order_relaxed) != 1u
            || atomic_load_explicit(out_degree + node, memory_order_relaxed) != 1u) {
        atomic_fetch_add_explicit(control + 3, 1u, memory_order_relaxed);
        return;
    }
    label[node] = node;
}

kernel void ig_cm_scc_cycle_probe_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device const uint* assignment [[buffer(6)]],
    device atomic_uint* successor [[buffer(7)]],
    device atomic_uint* control [[buffer(8)]],
    constant IgCmArgs& args [[buffer(9)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(control + 1, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count || assignment[source] != IG_CM_UNREACHED
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, control + 1)
            || assignment[target] != IG_CM_UNREACHED) return;
    uint prior = atomic_exchange_explicit(successor + source, target, memory_order_relaxed);
    if (prior != IG_CM_UNREACHED && prior != target) {
        atomic_fetch_add_explicit(control + 3, 1u, memory_order_relaxed);
    }
}

kernel void ig_cm_scc_cycle_probe_validate(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device const atomic_uint* successor [[buffer(2)]],
    device atomic_uint* control [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count || visible_nodes[node] != 1u
            || assignment[node] != IG_CM_UNREACHED) return;
    if (atomic_load_explicit(successor + node, memory_order_relaxed) >= args.node_count) {
        atomic_fetch_add_explicit(control + 3, 1u, memory_order_relaxed);
    }
}

kernel void ig_cm_scc_cycle_jump(
    device const uint* current_successor [[buffer(0)]],
    device const uint* current_label [[buffer(1)]],
    device uint* next_successor [[buffer(2)]],
    device uint* next_label [[buffer(3)]],
    device const uint* assignment [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant IgCmArgs& args [[buffer(6)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    if (assignment[node] != IG_CM_UNREACHED) {
        next_successor[node] = IG_CM_UNREACHED;
        next_label[node] = IG_CM_UNREACHED;
        return;
    }
    uint first = current_successor[node];
    if (first >= args.node_count || current_label[node] >= args.node_count) {
        ig_cm_status(status, 2u);
        return;
    }
    uint second = current_successor[first];
    uint following_label = current_label[first];
    if (second >= args.node_count || following_label >= args.node_count) {
        ig_cm_status(status, 2u);
        return;
    }
    next_successor[node] = second;
    next_label[node] = max(current_label[node], following_label);
}

kernel void ig_cm_scc_cycle_assign(
    device const uint* label [[buffer(0)]],
    device uint* assignment [[buffer(1)]],
    device atomic_uint* remaining [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count || assignment[node] != IG_CM_UNREACHED) return;
    if (label[node] >= args.node_count) {
        ig_cm_status(status, 2u);
        return;
    }
    assignment[node] = label[node];
    atomic_fetch_sub_explicit(remaining, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_color_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device atomic_uint* color [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint initial_color = visible_nodes[node] == 1u && assignment[node] == IG_CM_UNREACHED
        ? node : IG_CM_UNREACHED;
    atomic_store_explicit(color + node, initial_color, memory_order_relaxed);
}

kernel void ig_cm_scc_color_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device const uint* assignment [[buffer(6)]],
    device atomic_uint* color [[buffer(7)]],
    device atomic_uint* control [[buffer(8)]],
    constant IgCmArgs& args [[buffer(9)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(control + 1, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count || assignment[source] != IG_CM_UNREACHED
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, control + 1)
            || assignment[target] != IG_CM_UNREACHED) return;
    uint source_color = atomic_load_explicit(color + source, memory_order_relaxed);
    uint previous = atomic_fetch_max_explicit(color + target, source_color, memory_order_relaxed);
    if (previous < source_color) atomic_store_explicit(control, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_backward_seed(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* assignment [[buffer(1)]],
    device const uint* color [[buffer(2)]],
    device atomic_uint* backward [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint root = visible_nodes[node] == 1u && assignment[node] == IG_CM_UNREACHED
        && color[node] == node ? node : IG_CM_UNREACHED;
    atomic_store_explicit(backward + node, root, memory_order_relaxed);
}

kernel void ig_cm_scc_backward_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device const uint* assignment [[buffer(6)]],
    device const uint* color [[buffer(7)]],
    device atomic_uint* backward [[buffer(8)]],
    device atomic_uint* control [[buffer(9)]],
    constant IgCmArgs& args [[buffer(10)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(control + 1, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count || assignment[source] != IG_CM_UNREACHED
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, control + 1)
            || assignment[target] != IG_CM_UNREACHED || color[source] != color[target]
            || atomic_load_explicit(backward + target, memory_order_relaxed) != color[source]
            || atomic_load_explicit(backward + source, memory_order_relaxed) == color[source]) return;
    atomic_store_explicit(backward + source, color[source], memory_order_relaxed);
    atomic_store_explicit(control, 1u, memory_order_relaxed);
}

kernel void ig_cm_scc_color_assign(
    device const uint* backward [[buffer(0)]],
    device uint* assignment [[buffer(1)]],
    device uint* candidate [[buffer(2)]],
    device atomic_uint* remaining [[buffer(3)]],
    device atomic_uint* assigned [[buffer(4)]],
    constant IgCmArgs& args [[buffer(5)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    bool selected = assignment[node] == IG_CM_UNREACHED
        && backward[node] != IG_CM_UNREACHED;
    candidate[node] = selected ? 1u : 0u;
    if (selected) {
        assignment[node] = backward[node];
        atomic_fetch_sub_explicit(remaining, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(assigned, 1u, memory_order_relaxed);
    }
}

kernel void ig_cm_degree_prepare(
    device const uchar* visible_nodes [[buffer(0)]],
    device atomic_uint* out_degree [[buffer(1)]],
    device atomic_uint* in_degree [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    atomic_store_explicit(out_degree + node, 0u, memory_order_relaxed);
    atomic_store_explicit(in_degree + node, 0u, memory_order_relaxed);
    if (visible_nodes[node] > 1u) ig_cm_status(status, 3u);
}

// Whole-resident-graph fast path. The host selects this only when every node and relationship is
// active, every layer is visible, and neither directed CSR has overlay rows. In that shape the two
// canonical CSR row lengths are the exact directed degrees, so work is O(V) without E atomics.
kernel void ig_cm_degree_csr(
    device const uint* outgoing_offsets [[buffer(0)]],
    device const uint* incoming_offsets [[buffer(1)]],
    device uint* out_degree [[buffer(2)]],
    device uint* in_degree [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant IgCmArgs& args [[buffer(5)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint out_begin = outgoing_offsets[node];
    uint out_end = outgoing_offsets[node + 1u];
    uint in_begin = incoming_offsets[node];
    uint in_end = incoming_offsets[node + 1u];
    if (out_end < out_begin || out_end > args.adjacency_count
            || in_end < in_begin || in_end > args.adjacency_count
            || (node == 0u && (out_begin != 0u || in_begin != 0u))
            || (node + 1u == args.node_count
                && (out_end != args.adjacency_count || in_end != args.adjacency_count))) {
        ig_cm_status(status, 1u);
        out_degree[node] = 0u;
        in_degree[node] = 0u;
        return;
    }
    out_degree[node] = out_end - out_begin;
    in_degree[node] = in_end - in_begin;
}

// Every canonical edge-column position represents one directed physical relationship. Parallel
// relationships remain distinct and a self-loop increments both directed degree columns.
kernel void ig_cm_degree_edges(
    device const uint* edge_targets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device const uchar* edge_active [[buffer(3)]],
    device const uchar* edge_layers [[buffer(4)]],
    device const uint* edge_sources [[buffer(5)]],
    device atomic_uint* out_degree [[buffer(6)]],
    device atomic_uint* in_degree [[buffer(7)]],
    device atomic_uint* status [[buffer(8)]],
    constant IgCmArgs& args [[buffer(9)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.edge_count
            || args.reserved_1 > args.edge_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint position = args.scalar + local_position;
    uint edge = position;
    if (edge >= args.edge_count) {
        ig_cm_status(status, 2u);
        return;
    }
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source >= args.node_count
            || !ig_cm_validate_metadata(
                edge, target, visible_nodes, edge_active, edge_layers, args, status)
            || visible_nodes[source] != 1u) return;
    atomic_fetch_add_explicit(out_degree + source, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(in_degree + target, 1u, memory_order_relaxed);
}

inline bool ig_cm_unique_row_bounds(
    uint node,
    device const uint* offsets,
    constant IgCmArgs& args,
    device atomic_uint* status,
    thread uint& begin,
    thread uint& end) {
    if (node >= args.node_count) {
        ig_cm_status(status, 2u);
        return false;
    }
    begin = offsets[node];
    end = offsets[node + 1u];
    if (end < begin || end > args.adjacency_count
            || (node == 0u && begin != 0u)
            || (node + 1u == args.node_count && end != args.adjacency_count)) {
        ig_cm_status(status, 1u);
        return false;
    }
    return true;
}

// Exact upper-bound lookup of the source row for one reciprocal CSR position. At most 32
// iterations are required because resident node ordinals are u32-bounded.
inline uint ig_cm_unique_source(
    uint position,
    device const uint* offsets,
    constant IgCmArgs& args,
    device atomic_uint* status) {
    if (position >= args.adjacency_count) {
        ig_cm_status(status, 2u);
        return IG_CM_UNREACHED;
    }
    uint low = 0u;
    uint high = args.node_count;
    while (low < high) {
        uint middle = low + (high - low) / 2u;
        if (offsets[middle + 1u] <= position) low = middle + 1u;
        else high = middle;
    }
    if (low >= args.node_count || offsets[low] > position || offsets[low + 1u] <= position) {
        ig_cm_status(status, 1u);
        return IG_CM_UNREACHED;
    }
    return low;
}

inline bool ig_cm_rank_less(
    uint left,
    uint right,
    device const atomic_uint* degree) {
    uint left_degree = atomic_load_explicit(degree + left, memory_order_relaxed);
    uint right_degree = atomic_load_explicit(degree + right, memory_order_relaxed);
    return left_degree < right_degree || (left_degree == right_degree && left < right);
}

inline void ig_cm_atomic_add_u64(
    device atomic_uint* low_word,
    uint amount,
    device atomic_uint* status) {
    if (amount == 0u) return;
    uint previous = atomic_fetch_add_explicit(low_word, amount, memory_order_relaxed);
    if (previous > ~0u - amount) {
        uint high = atomic_fetch_add_explicit(low_word + 1u, 1u, memory_order_relaxed);
        if (high == ~0u) ig_cm_status(status, 4u);
    }
}

kernel void ig_cm_undirected_prepare(
    device const uint* offsets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(1)]],
    device atomic_uint* packet [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint node = args.scalar + local_node;
    size_t link_base = size_t(node) * 2ul;
    size_t degree_base = size_t(args.node_count) * 2ul;
    atomic_store_explicit(packet + link_base, 0u, memory_order_relaxed);
    atomic_store_explicit(packet + link_base + 1ul, 0u, memory_order_relaxed);
    atomic_store_explicit(packet + degree_base + size_t(node), 0u, memory_order_relaxed);
    if (visible_nodes[node] > 1u) ig_cm_status(status, 3u);
    uint begin = 0u;
    uint end = 0u;
    ig_cm_unique_row_bounds(node, offsets, args, status, begin, end);
}

kernel void ig_cm_undirected_degree_edges(
    device const uint* offsets [[buffer(0)]],
    device const uint* neighbors [[buffer(1)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device atomic_uint* packet [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant IgCmArgs& args [[buffer(5)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint node = args.scalar + local_node;
    uint begin = 0u;
    uint end = 0u;
    if (!ig_cm_unique_row_bounds(node, offsets, args, status, begin, end)) return;
    if (visible_nodes[node] != 1u) {
        if (begin != end) ig_cm_status(status, 3u);
        return;
    }
    (void)neighbors;
    size_t degree_base = size_t(args.node_count) * 2ul;
    atomic_store_explicit(
        packet + degree_base + size_t(node), end - begin, memory_order_relaxed);
}

kernel void ig_cm_triangle_cursor_prepare(
    device uint* outgoing_cursor [[buffer(0)]],
    device uint* incoming_cursor [[buffer(1)]],
    constant IgCmArgs& args [[buffer(2)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    outgoing_cursor[local_position] = 0u;
    incoming_cursor[local_position] = 0u;
}

// One lane owns one reciprocal CSR row position. Only rank(source)<rank(target) positions merge
// their two sorted unique rows. At most args.reserved_2 merge comparisons occur per dispatch;
// the two cursor banks resume the exact intersection after cancellation polling.
kernel void ig_cm_triangle_oriented_edges(
    device const uint* offsets [[buffer(0)]],
    device const uint* neighbors [[buffer(1)]],
    device const uchar* visible_nodes [[buffer(2)]],
    device atomic_uint* packet [[buffer(3)]],
    device atomic_uint* status [[buffer(4)]],
    constant IgCmArgs& args [[buffer(5)]],
    device uint* source_cursor [[buffer(6)]],
    device uint* target_cursor [[buffer(7)]],
    device atomic_uint* unfinished [[buffer(8)]],
    uint local_position [[thread_position_in_grid]]) {
    if (local_position >= args.reserved_1) return;
    if (args.scalar > args.adjacency_count
            || args.reserved_1 > args.adjacency_count - args.scalar) {
        ig_cm_status(status, 2u);
        return;
    }
    uint source_progress = source_cursor[local_position];
    uint target_progress = target_cursor[local_position];
    if (source_progress == IG_CM_UNREACHED && target_progress == IG_CM_UNREACHED) return;
    uint position = args.scalar + local_position;
    uint source = ig_cm_unique_source(position, offsets, args, status);
    uint target = neighbors[position];
    if (source >= args.node_count || target >= args.node_count || target == source
            || visible_nodes[source] != 1u || visible_nodes[target] != 1u) {
        ig_cm_status(status, 2u);
        source_cursor[local_position] = IG_CM_UNREACHED;
        target_cursor[local_position] = IG_CM_UNREACHED;
        return;
    }
    device const atomic_uint* degree = packet + size_t(args.node_count) * 2ul;
    if (!ig_cm_rank_less(source, target, degree)) {
        source_cursor[local_position] = IG_CM_UNREACHED;
        target_cursor[local_position] = IG_CM_UNREACHED;
        return;
    }
    uint source_begin = 0u;
    uint source_end = 0u;
    uint target_begin = 0u;
    uint target_end = 0u;
    if (!ig_cm_unique_row_bounds(source, offsets, args, status, source_begin, source_end)
            || !ig_cm_unique_row_bounds(target, offsets, args, status, target_begin, target_end)) {
        source_cursor[local_position] = IG_CM_UNREACHED;
        target_cursor[local_position] = IG_CM_UNREACHED;
        return;
    }
    uint source_count = source_end - source_begin;
    uint target_count = target_end - target_begin;
    if (source_progress > source_count || target_progress > target_count) {
        ig_cm_status(status, 2u);
        source_cursor[local_position] = IG_CM_UNREACHED;
        target_cursor[local_position] = IG_CM_UNREACHED;
        return;
    }

    if (args.reserved_2 == 0u) {
        ig_cm_status(status, 2u);
        return;
    }
    uint budget = args.reserved_2;
    uint edge_triangles = 0u;
    while (source_progress < source_count && target_progress < target_count && budget != 0u) {
        uint source_neighbor = neighbors[source_begin + source_progress];
        uint target_neighbor = neighbors[target_begin + target_progress];
        --budget;
        if (source_neighbor < target_neighbor) {
            ++source_progress;
            continue;
        }
        if (target_neighbor < source_neighbor) {
            ++target_progress;
            continue;
        }
        ++source_progress;
        ++target_progress;
        uint candidate = source_neighbor;
        if (candidate >= args.node_count || visible_nodes[candidate] != 1u) {
            ig_cm_status(status, 2u);
            continue;
        }
        if (ig_cm_rank_less(target, candidate, degree)) {
            ++edge_triangles;
            ig_cm_atomic_add_u64(packet + size_t(candidate) * 2ul, 1u, status);
        }
    }
    ig_cm_atomic_add_u64(packet + size_t(source) * 2ul, edge_triangles, status);
    ig_cm_atomic_add_u64(packet + size_t(target) * 2ul, edge_triangles, status);
    if (source_progress < source_count && target_progress < target_count) {
        source_cursor[local_position] = source_progress;
        target_cursor[local_position] = target_progress;
        atomic_store_explicit(unfinished, 1u, memory_order_relaxed);
    } else {
        source_cursor[local_position] = IG_CM_UNREACHED;
        target_cursor[local_position] = IG_CM_UNREACHED;
    }
}

kernel void ig_cm_triangle_tiles(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uint* packet [[buffer(1)]],
    device uint* partials [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgCmArgs& args [[buffer(4)]],
    uint tile [[thread_position_in_grid]]) {
    if (tile >= args.partial_count) return;
    uint begin = tile * 256u;
    uint end = min(begin + 256u, args.node_count);
    ulong sum = 0ul;
    for (uint node = begin; node < end; ++node) {
        if (visible_nodes[node] == 0u) continue;
        size_t link_base = size_t(node) * 2ul;
        ulong links = ulong(packet[link_base]) | (ulong(packet[link_base + 1ul]) << 32u);
        if (~0ul - sum < links) {
            ig_cm_status(status, 4u);
            return;
        }
        sum += links;
    }
    partials[tile * 2u] = uint(sum);
    partials[tile * 2u + 1u] = uint(sum >> 32u);
}

kernel void ig_cm_triangle_finalize(
    device const uint* partials [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    ulong sum = ulong(partials[0]) | (ulong(partials[1]) << 32u);
    if (sum % 3ul != 0ul) {
        ig_cm_status(status, 5u);
        return;
    }
    ulong triangles = sum / 3ul;
    output[0] = uint(triangles);
    output[1] = uint(triangles >> 32u);
}

// Bounded hierarchical reduction: each lane consumes at most 256 binary64-width integer pairs.
// Rust submits logarithmic levels and checks cancellation between them; alternating banks avoid
// read/write overlap within one dispatch.
kernel void ig_cm_u64_reduce_tiles(
    device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant IgCmArgs& args [[buffer(3)]],
    uint tile [[thread_position_in_grid]]) {
    uint input_count = args.scalar;
    uint output_count = (input_count + 255u) / 256u;
    if (tile >= output_count) return;
    uint begin = tile * 256u;
    uint end = min(begin + 256u, input_count);
    ulong sum = 0ul;
    for (uint position = begin; position < end; ++position) {
        size_t base = size_t(position) * 2ul;
        ulong value = ulong(input[base]) | (ulong(input[base + 1ul]) << 32u);
        if (~0ul - sum < value) {
            ig_cm_status(status, 4u);
            return;
        }
        sum += value;
    }
    size_t output_base = size_t(tile) * 2ul;
    output[output_base] = uint(sum);
    output[output_base + 1ul] = uint(sum >> 32u);
}

// Minimal software binary64 publication used for the exact CPU-visible clustering ratio. Apple
// Metal has no native FP64 scalar arithmetic, so the numerator and denominator are converted and
// divided with round-to-nearest-even integer operations on-device.
inline bool ig_cm_f64_is_nan(ulong bits) {
    return (bits & 0x7ff0000000000000ul) == 0x7ff0000000000000ul
        && (bits & 0x000ffffffffffffful) != 0ul;
}

inline bool ig_cm_f64_is_infinite(ulong bits) {
    return (bits & 0x7ffffffffffffffful) == 0x7ff0000000000000ul;
}

inline ulong ig_cm_shift_right_jam(ulong value, uint distance) {
    if (distance == 0u) return value;
    if (distance < 64u) {
        ulong discarded = value << (64u - distance);
        return (value >> distance) | (discarded != 0ul ? 1ul : 0ul);
    }
    return value != 0ul ? 1ul : 0ul;
}

inline bool ig_cm_f64_decode_finite(
    ulong bits,
    thread int& exponent,
    thread ulong& significand) {
    uint exponent_bits = uint((bits >> 52u) & 0x7fful);
    significand = bits & 0x000ffffffffffffful;
    if (exponent_bits == 0x7ffu || (exponent_bits == 0u && significand == 0ul)) return false;
    if (exponent_bits == 0u) {
        exponent = -1022;
        while ((significand & 0x0010000000000000ul) == 0ul) {
            significand <<= 1u;
            --exponent;
        }
    } else {
        exponent = int(exponent_bits) - 1023;
        significand |= 0x0010000000000000ul;
    }
    return true;
}

inline ulong ig_cm_f64_round_pack(bool negative, int exponent, ulong significand) {
    if (significand == 0ul) return negative ? 0x8000000000000000ul : 0ul;
    while ((significand & 0x0080000000000000ul) == 0ul) {
        significand <<= 1u;
        --exponent;
    }
    if (exponent < -1022) {
        uint distance = uint(min(4096, -1022 - exponent));
        significand = ig_cm_shift_right_jam(significand, distance);
        exponent = -1022;
    }
    ulong round_bits = significand & 7ul;
    ulong rounded = significand >> 3u;
    if (round_bits > 4ul || (round_bits == 4ul && (rounded & 1ul) != 0ul)) ++rounded;
    if (rounded >= 0x0020000000000000ul) {
        rounded >>= 1u;
        ++exponent;
    }
    if (exponent > 1023) {
        return negative ? 0xfff0000000000000ul : 0x7ff0000000000000ul;
    }
    ulong sign = negative ? 0x8000000000000000ul : 0ul;
    if (exponent == -1022 && rounded < 0x0010000000000000ul) return sign | rounded;
    return sign | (ulong(exponent + 1023) << 52u)
        | (rounded & 0x000ffffffffffffful);
}

inline ulong ig_cm_u64_to_f64(ulong magnitude) {
    if (magnitude == 0ul) return 0ul;
    uint most_significant = 63u - clz(magnitude);
    ulong significand;
    uint exponent = most_significant + 1023u;
    if (most_significant <= 52u) {
        significand = magnitude << (52u - most_significant);
    } else {
        uint shift = most_significant - 52u;
        significand = magnitude >> shift;
        ulong mask = (1ul << shift) - 1ul;
        ulong remainder = magnitude & mask;
        ulong halfway = 1ul << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (significand & 1ul) != 0ul)) {
            ++significand;
            if (significand == 0x0020000000000000ul) {
                significand >>= 1u;
                ++exponent;
            }
        }
    }
    return (ulong(exponent) << 52u) | (significand & 0x000ffffffffffffful);
}

inline ulong ig_cm_f64_divide(ulong left, ulong right) {
    const ulong canonical_nan = 0x7ff8000000000000ul;
    if (ig_cm_f64_is_nan(left) || ig_cm_f64_is_nan(right)) return canonical_nan;
    bool negative = ((left ^ right) >> 63u) != 0ul;
    bool left_infinite = ig_cm_f64_is_infinite(left);
    bool right_infinite = ig_cm_f64_is_infinite(right);
    bool left_zero = (left & 0x7ffffffffffffffful) == 0ul;
    bool right_zero = (right & 0x7ffffffffffffffful) == 0ul;
    if ((left_infinite && right_infinite) || (left_zero && right_zero)) return canonical_nan;
    if (left_infinite || right_zero) {
        return negative ? 0xfff0000000000000ul : 0x7ff0000000000000ul;
    }
    if (right_infinite || left_zero) return negative ? 0x8000000000000000ul : 0ul;
    int left_exponent = 0;
    int right_exponent = 0;
    ulong left_significand = 0ul;
    ulong right_significand = 0ul;
    ig_cm_f64_decode_finite(left, left_exponent, left_significand);
    ig_cm_f64_decode_finite(right, right_exponent, right_significand);
    int exponent = left_exponent - right_exponent;
    ulong remainder = left_significand;
    if (remainder < right_significand) {
        remainder <<= 1u;
        --exponent;
    }
    ulong quotient = 0ul;
    for (int bit = 55; bit >= 0; --bit) {
        if (remainder >= right_significand) {
            remainder -= right_significand;
            quotient |= 1ul << uint(bit);
        }
        if (bit != 0) remainder <<= 1u;
    }
    if (remainder != 0ul) quotient |= 1ul;
    return ig_cm_f64_round_pack(negative, exponent, quotient);
}

kernel void ig_cm_clustering_publish(
    device uint* packet [[buffer(0)]],
    constant IgCmArgs& args [[buffer(1)]],
    uint local_node [[thread_position_in_grid]]) {
    if (local_node >= args.reserved_1) return;
    if (args.scalar > args.node_count || args.reserved_1 > args.node_count - args.scalar) return;
    uint node = args.scalar + local_node;
    size_t link_base = size_t(node) * 2ul;
    size_t degree_base = size_t(args.node_count) * 2ul;
    ulong links = ulong(packet[link_base]) | (ulong(packet[link_base + 1ul]) << 32u);
    ulong degree = ulong(packet[degree_base + size_t(node)]);
    ulong possible = degree * (degree > 0ul ? degree - 1ul : 0ul) / 2ul;
    ulong coefficient = possible == 0ul
        ? 0ul : ig_cm_f64_divide(ig_cm_u64_to_f64(links), ig_cm_u64_to_f64(possible));
    packet[link_base] = uint(coefficient);
    packet[link_base + 1ul] = uint(coefficient >> 32u);
}

kernel void ig_cm_ratio_probe(
    device const uint* pairs [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant uint& row_count [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= row_count) return;
    size_t pair_base = size_t(row) * 4ul;
    size_t output_base = size_t(row) * 2ul;
    ulong numerator = ulong(pairs[pair_base]) | (ulong(pairs[pair_base + 1ul]) << 32u);
    ulong denominator = ulong(pairs[pair_base + 2ul]) | (ulong(pairs[pair_base + 3ul]) << 32u);
    ulong bits = denominator == 0ul ? 0ul
        : ig_cm_f64_divide(ig_cm_u64_to_f64(numerator), ig_cm_u64_to_f64(denominator));
    output[output_base] = uint(bits);
    output[output_base + 1ul] = uint(bits >> 32u);
}

kernel void ig_cm_kcore_initialize(
    device const uint* offsets [[buffer(0)]],
    device const uchar* visible_nodes [[buffer(1)]],
    device atomic_uint* alive [[buffer(2)]],
    device atomic_uint* queue [[buffer(3)]],
    device uint* core [[buffer(4)]],
    device uint* degree [[buffer(5)]],
    device uint* cursor [[buffer(6)]],
    device atomic_uint* control [[buffer(7)]],
    constant IgCmArgs& args [[buffer(8)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    uint present = visible_nodes[node] == 1u ? 1u : 0u;
    atomic_store_explicit(alive + node, present, memory_order_relaxed);
    atomic_store_explicit(queue + node, IG_CM_UNREACHED, memory_order_relaxed);
    core[node] = 0u;
    cursor[node] = 0u;
    if (visible_nodes[node] > 1u) ig_cm_status(control + 3, 3u);
    uint begin = 0u;
    uint end = 0u;
    if (!ig_cm_unique_row_bounds(node, offsets, args, control + 3, begin, end)) return;
    if (present == 0u && begin != end) {
        ig_cm_status(control + 3, 3u);
        return;
    }
    degree[node] = end - begin;
}

kernel void ig_cm_kcore_prepare(
    device atomic_uint* queue [[buffer(0)]],
    device atomic_uint* control [[buffer(1)]],
    constant IgCmArgs& args [[buffer(2)]],
    uint node [[thread_position_in_grid]]) {
    if (node < args.node_count) {
        atomic_store_explicit(queue + node, IG_CM_UNREACHED, memory_order_relaxed);
    }
    if (node != 0u) return;
    atomic_store_explicit(control + 1, 0u, memory_order_relaxed); // processed this quantum
    atomic_store_explicit(control + 2, IG_CM_UNREACHED, memory_order_relaxed); // minimum
    atomic_store_explicit(control + 4, 0u, memory_order_relaxed); // queue head
    atomic_store_explicit(control + 5, 0u, memory_order_relaxed); // queue tail
}

inline bool ig_cm_kcore_enqueue(
    device atomic_uint* queue,
    device atomic_uint* control,
    uint node,
    constant IgCmArgs& args) {
    uint ticket = atomic_fetch_add_explicit(control + 5, 1u, memory_order_relaxed);
    if (ticket >= args.node_count || args.node_count == 0u) {
        ig_cm_status(control + 3, 2u);
        return false;
    }
    // Every node wins the alive CAS at most once, so a core cascade publishes at most V tickets.
    // Tickets are append-only until the host-observed cascade completes; no producer waits for a
    // consumer and no slot is reused within a dispatch.
    atomic_store_explicit(queue + ticket, node, memory_order_relaxed);
    return true;
}

kernel void ig_cm_kcore_seed(
    device atomic_uint* alive [[buffer(0)]],
    device const atomic_uint* degree [[buffer(1)]],
    device atomic_uint* queue [[buffer(2)]],
    device uint* core [[buffer(3)]],
    device uint* cursor [[buffer(4)]],
    device atomic_uint* control [[buffer(5)]],
    constant IgCmArgs& args [[buffer(6)]],
    uint node [[thread_position_in_grid]]) {
    if (node >= args.node_count) return;
    if (atomic_load_explicit(alive + node, memory_order_relaxed) == 0u) return;
    uint value = atomic_load_explicit(degree + node, memory_order_relaxed);
    atomic_fetch_min_explicit(control + 2, value, memory_order_relaxed);
    if (value > args.scalar) return;
    if (atomic_exchange_explicit(alive + node, 0u, memory_order_relaxed) != 1u) return;
    cursor[node] = 0u;
    if (!ig_cm_kcore_enqueue(queue, control, node, args)) return;
    core[node] = args.scalar;
    atomic_fetch_sub_explicit(control + 0, 1u, memory_order_relaxed);
}

kernel void ig_cm_kcore_drain_prepare(
    device atomic_uint* processed [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0u) atomic_store_explicit(processed, 0u, memory_order_relaxed);
}

// Append-only ticket chunks drain the degree-k cascade without producer/consumer spin waiting.
// Each ticket scans at most args.reserved_2 raw positions and keeps its slot until the row is done.
// Newly removed nodes append unique tickets and are observed by a later host-issued chunk.
kernel void ig_cm_kcore_drain(
    device const uint* offsets [[buffer(0)]],
    device const uint* neighbors [[buffer(1)]],
    device atomic_uint* alive [[buffer(2)]],
    device atomic_uint* queue [[buffer(3)]],
    device atomic_uint* degree [[buffer(4)]],
    device uint* core [[buffer(5)]],
    device uint* cursor [[buffer(6)]],
    device atomic_uint* control [[buffer(7)]],
    constant IgCmArgs& args [[buffer(8)]],
    uint local_ticket [[thread_position_in_grid]]) {
        if (local_ticket >= args.partial_count) return;
        if (args.reserved_1 > args.node_count
                || args.partial_count > args.node_count - args.reserved_1) {
            ig_cm_status(control + 3, 2u);
            return;
        }
        uint ticket = args.reserved_1 + local_ticket;
        uint tail = atomic_load_explicit(control + 5, memory_order_relaxed);
        if (ticket >= tail) return;
        uint node = atomic_load_explicit(queue + ticket, memory_order_relaxed);
        if (node == IG_CM_UNREACHED) return;
        if (node >= args.node_count) {
            ig_cm_status(control + 3, 2u);
            return;
        }
        uint begin = 0u;
        uint end = 0u;
        if (!ig_cm_unique_row_bounds(node, offsets, args, control + 3, begin, end)) return;
        uint count = end - begin;
        uint progress = cursor[node];
        if (args.reserved_2 == 0u || progress > count) {
            ig_cm_status(control + 3, 2u);
            return;
        }
        uint budget = args.reserved_2;
        while (budget != 0u && progress < count) {
            uint neighbor = neighbors[begin + progress++];
            --budget;
            if (neighbor >= args.node_count || neighbor == node) {
                ig_cm_status(control + 3, 2u);
                continue;
            }
            if (atomic_load_explicit(alive + neighbor, memory_order_relaxed) == 0u) continue;
            uint previous = atomic_fetch_sub_explicit(
                degree + neighbor, 1u, memory_order_relaxed);
            if (previous == 0u) {
                ig_cm_status(control + 3, 6u);
                continue;
            }
            if (previous - 1u > args.scalar) continue;
            if (atomic_exchange_explicit(
                    alive + neighbor, 0u, memory_order_relaxed) != 1u) continue;
            cursor[neighbor] = 0u;
            core[neighbor] = args.scalar;
            atomic_fetch_sub_explicit(control + 0, 1u, memory_order_relaxed);
            if (!ig_cm_kcore_enqueue(queue, control, neighbor, args)) return;
        }
        cursor[node] = progress;
        if (progress < count) {
            atomic_fetch_add_explicit(control + 1, 1u, memory_order_relaxed);
        } else {
            atomic_store_explicit(queue + ticket, IG_CM_UNREACHED, memory_order_relaxed);
        }
}
