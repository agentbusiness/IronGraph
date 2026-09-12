#include <metal_stdlib>

using namespace metal;

/// Highest valid value of the resident layer byte: OBSERVED=0, KNOWLEDGE=1, WORKSPACE=2.
/// Integrity checks compare against this rather than a literal so a fourth layer is a one-line
/// change instead of a hunt through every kernel.
constant uint IG_MAX_LAYER = 2u;
constant ulong IG_MAX_LAYER_UL = 2ul;


// Exact resident path procedures.  The host splits persistent algorithms into bounded quanta so
// cancellation can be observed without copying canonical graph columns off the selected device.

constant uint IG_PATH_THREADS = 256u;
constant uint IG_PATH_UNREACHED = 0xffffffffu;
constant ulong IG_PATH_F64_INF = 0x7ff0000000000000ul;
constant ulong IG_PATH_F64_MAG = 0x7ffffffffffffffful;
constant ulong IG_PATH_F64_FRAC = 0x000ffffffffffffful;
constant uint IG_PATH_DIJKSTRA_QUERY_TYPE = 3u;
constant uint IG_PATH_DIJKSTRA_INVALID_WEIGHT = 4u;
constant uint IG_PATH_DIJKSTRA_CORRUPT_CSR = 5u;
constant uint IG_PATH_DIJKSTRA_CORRUPT_VALUE = 6u;

inline bool ig_path_layer_visible(uchar layer, uint layer_mask) {
    return layer < 32u && (layer_mask & (1u << uint(layer))) != 0u;
}

struct IgPathCsrRow {
    uint begin;
    uint end;
    uint overlay;
};

// Sparse delta rows are complete replacements for cold CSR rows. Re-resolving a row is safe for
// persistent workspaces because their cursor remains an index within the selected row payload.
inline IgPathCsrRow ig_path_csr_row(
    device const uint* offsets,
    device const uint* overlay,
    uint overlay_count,
    uint row) {
    uint low = 0u;
    uint high = overlay_count;
    while (low < high) {
        uint middle = low + (high - low) / 2u;
        uint candidate = overlay[middle];
        if (candidate < row) low = middle + 1u;
        else high = middle;
    }
    if (low < overlay_count && overlay[low] == row) {
        return IgPathCsrRow{
            overlay[overlay_count + low],
            overlay[overlay_count + low + 1u],
            1u};
    }
    return IgPathCsrRow{offsets[row], offsets[row + 1u], 0u};
}

inline uint ig_path_csr_neighbor(
    device const uint* neighbors,
    device const uint* overlay,
    uint overlay_count,
    IgPathCsrRow row,
    uint position) {
    return row.overlay == 0u ? neighbors[position]
        : overlay[overlay_count * 2u + 1u + position];
}

inline uint ig_path_csr_edge(
    device const uint* edges,
    device const uint* overlay,
    uint overlay_count,
    IgPathCsrRow row,
    uint position) {
    if (row.overlay == 0u) return edges[position];
    uint payload_count = overlay[overlay_count * 2u];
    return overlay[overlay_count * 2u + 1u + payload_count + position];
}

inline ulong ig_path_words_load(device const uint* words, uint row) {
    uint base = row * 2u;
    return ulong(words[base]) | (ulong(words[base + 1u]) << 32u);
}

inline void ig_path_words_store(device uint* words, uint row, ulong bits) {
    uint base = row * 2u;
    words[base] = uint(bits);
    words[base + 1u] = uint(bits >> 32u);
}

inline ulong ig_path_shift_right_jam(ulong value, uint distance) {
    if (distance == 0u) return value;
    if (distance < 64u) {
        ulong discarded_mask = (1ul << distance) - 1ul;
        return (value >> distance) | ulong((value & discarded_mask) != 0ul);
    }
    return ulong(value != 0ul);
}

// Correctly-rounded binary64 addition for the non-negative domain used by Dijkstra.  Metal has no
// portable fp64 arithmetic; keeping the operation in integer space preserves the CPU contract,
// including subnormals, ties-to-even, and overflow to positive infinity.
inline ulong ig_path_f64_add_nonnegative(ulong left, ulong right) {
    left &= IG_PATH_F64_MAG;
    right &= IG_PATH_F64_MAG;
    if (left == IG_PATH_F64_INF || right == IG_PATH_F64_INF) return IG_PATH_F64_INF;
    if (left == 0ul) return right;
    if (right == 0ul) return left;

    uint left_field = uint((left >> 52u) & 0x7fful);
    uint right_field = uint((right >> 52u) & 0x7fful);
    long left_exponent = left_field == 0u ? -1022l : long(left_field) - 1023l;
    long right_exponent = right_field == 0u ? -1022l : long(right_field) - 1023l;
    ulong left_significand = left & IG_PATH_F64_FRAC;
    ulong right_significand = right & IG_PATH_F64_FRAC;
    if (left_field != 0u) left_significand |= 1ul << 52u;
    if (right_field != 0u) right_significand |= 1ul << 52u;

    if (right_exponent > left_exponent
            || (right_exponent == left_exponent && right_significand > left_significand)) {
        long swap_exponent = left_exponent;
        left_exponent = right_exponent;
        right_exponent = swap_exponent;
        ulong swap_significand = left_significand;
        left_significand = right_significand;
        right_significand = swap_significand;
    }

    uint separation = uint(left_exponent - right_exponent);
    ulong extended = (left_significand << 3u)
        + ig_path_shift_right_jam(right_significand << 3u, separation);
    if ((extended & (1ul << 56u)) != 0ul) {
        extended = ig_path_shift_right_jam(extended, 1u);
        left_exponent += 1l;
    }

    ulong significand = extended >> 3u;
    ulong rounding = extended & 7ul;
    if (rounding > 4ul || (rounding == 4ul && (significand & 1ul) != 0ul)) {
        significand += 1ul;
    }
    if (significand >= (1ul << 53u)) {
        significand >>= 1u;
        left_exponent += 1l;
    }
    if (left_exponent > 1023l) return IG_PATH_F64_INF;
    if (left_exponent == -1022l && significand < (1ul << 52u)) return significand;
    ulong exponent_field = ulong(left_exponent + 1023l);
    return (exponent_field << 52u) | (significand & IG_PATH_F64_FRAC);
}

inline ulong ig_path_nonnegative_i64_to_f64(ulong raw) {
    if (raw == 0ul) return 0ul;
    uint most_significant = 63u - clz(raw);
    if (most_significant <= 52u) {
        ulong exponent = ulong(most_significant + 1023u) << 52u;
        ulong normalized = raw << (52u - most_significant);
        return exponent | (normalized & IG_PATH_F64_FRAC);
    }
    uint shift = most_significant - 52u;
    ulong significand = raw >> shift;
    ulong discarded_mask = (1ul << shift) - 1ul;
    ulong discarded = raw & discarded_mask;
    ulong halfway = 1ul << (shift - 1u);
    if (discarded > halfway || (discarded == halfway && (significand & 1ul) != 0ul)) {
        significand += 1ul;
    }
    uint exponent = most_significant;
    if (significand == (1ul << 53u)) {
        significand >>= 1u;
        exponent += 1u;
    }
    return (ulong(exponent + 1023u) << 52u) | (significand & IG_PATH_F64_FRAC);
}

// Direct arithmetic probe used by real-Metal differential tests.  It is intentionally kept in
// the same runtime library as Dijkstra so tests exercise the exact production helper functions.
kernel void ig_path_arithmetic_probe(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant uint& mode [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    output[index] = mode == 1u
        ? ig_path_f64_add_nonnegative(input[index * 2u], input[index * 2u + 1u])
        : ig_path_nonnegative_i64_to_f64(input[index]);
}

struct IgPathDfsArgs {
    uint node_count;
    uint edge_count;
    uint adjacency_count;
    uint source;
    uint layer_mask;
    uint quantum;
    uint threadgroup_width;
    uint node_begin;
    uint overlay_count;
};

// DFS workspace: visited[N], stack nodes[N], stack cursors[N], preorder[N], control[8].
kernel void ig_path_dfs_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device uint* workspace [[buffer(1)]],
    constant IgPathDfsArgs& args [[buffer(2)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = args.node_begin + local_node;
    if (node >= args.node_count) return;
    uint stack_nodes = args.node_count;
    uint stack_cursors = args.node_count * 2u;
    uint preorder = args.node_count * 3u;
    uint control = args.node_count * 4u;
    workspace[node] = node == args.source ? 1u : 0u;
    workspace[stack_nodes + node] = IG_PATH_UNREACHED;
    workspace[stack_cursors + node] = IG_PATH_UNREACHED;
    workspace[preorder + node] = IG_PATH_UNREACHED;
    if (node == 0u) {
        bool source_visible = args.source < args.node_count && visible_nodes[args.source] == 1u;
        if (source_visible) {
            workspace[stack_nodes] = args.source;
            workspace[preorder] = args.source;
        }
        workspace[control + 0u] = source_visible ? 1u : 0u;
        workspace[control + 1u] = source_visible ? 1u : 0u;
        workspace[control + 2u] = source_visible ? 0u : 1u;
        workspace[control + 3u] = 0u;
        workspace[control + 4u] = 0u;
        workspace[control + 5u] = 0u;
        workspace[control + 6u] = 0u;
        workspace[control + 7u] = 0u;
    }
}

kernel void ig_path_dfs_chunk(
    device const uint* outgoing_offsets [[buffer(0)]],
    device const uint* outgoing_neighbors [[buffer(1)]],
    device const uint* outgoing_edges [[buffer(2)]],
    device const uint* outgoing_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device uint* workspace [[buffer(7)]],
    constant IgPathDfsArgs& args [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint candidates[IG_PATH_THREADS];
    threadgroup atomic_uint group_error;
    threadgroup uint shared_stack_length;
    threadgroup uint shared_cursor;
    threadgroup uint shared_end;
    threadgroup uint shared_node;

    uint stack_nodes = args.node_count;
    uint stack_cursors = args.node_count * 2u;
    uint preorder = args.node_count * 3u;
    uint control = args.node_count * 4u;
    for (uint transition = 0u; transition < args.quantum; ++transition) {
        if (lane == 0u) {
            atomic_store_explicit(&group_error, 0u, memory_order_relaxed);
            shared_stack_length = workspace[control + 0u];
            shared_node = shared_stack_length == 0u
                ? IG_PATH_UNREACHED : workspace[stack_nodes + shared_stack_length - 1u];
            shared_cursor = shared_stack_length == 0u
                ? 0u : workspace[stack_cursors + shared_stack_length - 1u];
            if (shared_node >= args.node_count) {
                shared_end = 0u;
                atomic_store_explicit(&group_error, 1u, memory_order_relaxed);
            } else {
                IgPathCsrRow row = ig_path_csr_row(
                    outgoing_offsets, outgoing_overlay, args.overlay_count, shared_node);
                uint bound = row.overlay == 0u ? args.adjacency_count
                    : outgoing_overlay[args.overlay_count * 2u];
                uint begin = row.begin;
                shared_end = row.end;
                if (shared_end < begin || shared_end > bound) {
                    atomic_store_explicit(&group_error, 2u, memory_order_relaxed);
                }
                if (shared_cursor == IG_PATH_UNREACHED) shared_cursor = begin;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (shared_stack_length == 0u) {
            if (lane == 0u) workspace[control + 2u] = 1u;
            return;
        }

        uint position = shared_cursor + lane;
        uint candidate = IG_PATH_UNREACHED;
        if (position < shared_end) {
            IgPathCsrRow row = ig_path_csr_row(
                outgoing_offsets, outgoing_overlay, args.overlay_count, shared_node);
            uint edge = ig_path_csr_edge(
                outgoing_edges, outgoing_overlay, args.overlay_count, row, position);
            uint target = ig_path_csr_neighbor(
                outgoing_neighbors, outgoing_overlay, args.overlay_count, row, position);
            if (edge >= args.edge_count || target >= args.node_count) {
                atomic_fetch_max_explicit(&group_error, 2u, memory_order_relaxed);
            } else {
                uchar active = edge_active[edge];
                uchar layer = edge_layers[edge];
                uchar target_visible = visible_nodes[target];
                if (active > 1u || layer > IG_MAX_LAYER || target_visible > 1u) {
                    atomic_fetch_max_explicit(&group_error, 3u, memory_order_relaxed);
                } else if (active != 0u && target_visible != 0u
                        && ig_path_layer_visible(layer, args.layer_mask)
                        && workspace[target] == 0u) {
                    candidate = position;
                }
            }
        }
        candidates[lane] = candidate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = args.threadgroup_width / 2u; stride != 0u; stride >>= 1u) {
            if (lane < stride) candidates[lane] = min(candidates[lane], candidates[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) {
            uint error = atomic_load_explicit(&group_error, memory_order_relaxed);
            if (error != 0u) {
                workspace[control + 3u] = error;
                workspace[control + 2u] = 1u;
            } else if (candidates[0] != IG_PATH_UNREACHED) {
                uint selected = candidates[0];
                IgPathCsrRow row = ig_path_csr_row(
                    outgoing_offsets, outgoing_overlay, args.overlay_count, shared_node);
                uint target = ig_path_csr_neighbor(
                    outgoing_neighbors, outgoing_overlay,
                    args.overlay_count, row, selected);
                workspace[stack_cursors + shared_stack_length - 1u] = selected + 1u;
                if (workspace[target] == 0u) {
                    uint output_length = workspace[control + 1u];
                    if (shared_stack_length >= args.node_count || output_length >= args.node_count) {
                        workspace[control + 3u] = 4u;
                        workspace[control + 2u] = 1u;
                    } else {
                        workspace[target] = 1u;
                        workspace[stack_nodes + shared_stack_length] = target;
                        workspace[stack_cursors + shared_stack_length] = IG_PATH_UNREACHED;
                        workspace[preorder + output_length] = target;
                        workspace[control + 0u] = shared_stack_length + 1u;
                        workspace[control + 1u] = output_length + 1u;
                    }
                }
            } else {
                uint next_cursor = min(shared_cursor + args.threadgroup_width, shared_end);
                if (next_cursor < shared_end) {
                    workspace[stack_cursors + shared_stack_length - 1u] = next_cursor;
                } else {
                    workspace[control + 0u] = shared_stack_length - 1u;
                    if (shared_stack_length == 1u) workspace[control + 2u] = 1u;
                }
            }
            workspace[control + 4u] += 1u;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (workspace[control + 2u] != 0u || workspace[control + 3u] != 0u) return;
    }
}

struct IgPathShortestArgs {
    uint node_count;
    uint edge_count;
    uint adjacency_count;
    uint source;
    uint target;
    uint layer_mask;
    uint distance;
    uint quantum;
    uint threadgroup_width;
    uint overlay_count;
};

// Shortest-path workspace: nodes[N], edges[N], control[8].
kernel void ig_path_shortest_initialize(
    device uint* workspace [[buffer(0)]],
    constant IgPathShortestArgs& args [[buffer(1)]],
    constant uint& node_begin [[buffer(2)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = node_begin + local_node;
    if (node >= args.node_count) return;
    workspace[node] = IG_PATH_UNREACHED;
    workspace[args.node_count + node] = IG_PATH_UNREACHED;
    if (node == 0u) {
        uint control = args.node_count * 2u;
        workspace[0] = args.source;
        workspace[control + 0u] = 1u;
        workspace[control + 1u] = args.source;
        workspace[control + 2u] = args.source == args.target ? 1u : 0u;
        workspace[control + 3u] = 0u;
        workspace[control + 4u] = args.distance;
        workspace[control + 5u] = 0u;
        workspace[control + 6u] = IG_PATH_UNREACHED; // current-row cursor
        workspace[control + 7u] = 0u;
    }
}

kernel void ig_path_shortest_chunk(
    device const uint* outgoing_offsets [[buffer(0)]],
    device const uint* outgoing_neighbors [[buffer(1)]],
    device const uint* outgoing_edges [[buffer(2)]],
    device const uint* outgoing_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device const uint* forward_workspace [[buffer(7)]],
    device const uint* reverse_workspace [[buffer(8)]],
    device uint* workspace [[buffer(9)]],
    constant IgPathShortestArgs& args [[buffer(10)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint candidates[IG_PATH_THREADS];
    threadgroup atomic_uint group_error;
    threadgroup uint shared_current;
    threadgroup uint shared_length;
    threadgroup uint shared_begin;
    threadgroup uint shared_end;
    threadgroup uint shared_selected;
    uint control = args.node_count * 2u;

    for (uint step = 0u; step < args.quantum; ++step) {
        if (lane == 0u) {
            atomic_store_explicit(&group_error, 0u, memory_order_relaxed);
            shared_current = workspace[control + 1u];
            shared_length = workspace[control + 0u];
            shared_selected = IG_PATH_UNREACHED;
            if (workspace[control + 2u] != 0u || workspace[control + 3u] != 0u) {
                shared_begin = 0u;
                shared_end = 0u;
            } else if (shared_current >= args.node_count || shared_length == 0u) {
                shared_begin = 0u;
                shared_end = 0u;
                atomic_store_explicit(&group_error, 1u, memory_order_relaxed);
            } else {
                IgPathCsrRow row = ig_path_csr_row(
                    outgoing_offsets, outgoing_overlay, args.overlay_count, shared_current);
                uint bound = row.overlay == 0u ? args.adjacency_count
                    : outgoing_overlay[args.overlay_count * 2u];
                uint row_begin = row.begin;
                shared_end = row.end;
                uint saved_cursor = workspace[control + 6u];
                shared_begin = saved_cursor == IG_PATH_UNREACHED ? row_begin : saved_cursor;
                if (shared_end < row_begin || shared_end > bound
                        || shared_begin < row_begin || shared_begin > shared_end) {
                    atomic_store_explicit(&group_error, 2u, memory_order_relaxed);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (workspace[control + 2u] != 0u || workspace[control + 3u] != 0u) return;

        // CSR rows are sorted by (target, edge). Scan exactly one physical 256-edge tile per
        // transition. The cursor survives host re-entry, retaining lexicographic selection while
        // placing a hard bound on cancellation latency even for a single giant row.
        uint position = shared_begin + lane;
        uint tile_end = shared_begin
            + min(args.threadgroup_width, shared_end - shared_begin);
        uint candidate = IG_PATH_UNREACHED;
        if (position < tile_end) {
            IgPathCsrRow row = ig_path_csr_row(
                outgoing_offsets, outgoing_overlay, args.overlay_count, shared_current);
            uint edge = ig_path_csr_edge(
                outgoing_edges, outgoing_overlay, args.overlay_count, row, position);
            uint target = ig_path_csr_neighbor(
                outgoing_neighbors, outgoing_overlay, args.overlay_count, row, position);
            if (edge >= args.edge_count || target >= args.node_count) {
                atomic_fetch_max_explicit(&group_error, 2u, memory_order_relaxed);
            } else {
                uchar active = edge_active[edge];
                uchar layer = edge_layers[edge];
                uchar target_visible = visible_nodes[target];
                if (active > 1u || layer > IG_MAX_LAYER || target_visible > 1u) {
                    atomic_fetch_max_explicit(&group_error, 3u, memory_order_relaxed);
                } else {
                    uint forward = forward_workspace[target];
                    uint reverse = reverse_workspace[target];
                    uint next_depth = shared_length;
                    if (active != 0u && target_visible != 0u
                            && ig_path_layer_visible(layer, args.layer_mask)
                            && forward == next_depth && reverse != IG_PATH_UNREACHED
                            && next_depth <= args.distance
                            && reverse == args.distance - next_depth) {
                        candidate = position;
                    }
                }
            }
        }
        candidates[lane] = candidate;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = args.threadgroup_width / 2u; stride != 0u; stride >>= 1u) {
            if (lane < stride) candidates[lane] = min(candidates[lane], candidates[lane + stride]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lane == 0u) shared_selected = candidates[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) {
            uint error = atomic_load_explicit(&group_error, memory_order_relaxed);
            if (error != 0u) {
                workspace[control + 3u] = error;
            } else if (shared_selected == IG_PATH_UNREACHED && tile_end < shared_end) {
                workspace[control + 6u] = tile_end;
                workspace[control + 5u] += tile_end - shared_begin;
            } else if (shared_selected == IG_PATH_UNREACHED || shared_length >= args.node_count) {
                workspace[control + 3u] = 4u;
            } else {
                IgPathCsrRow row = ig_path_csr_row(
                    outgoing_offsets, outgoing_overlay, args.overlay_count, shared_current);
                uint next = ig_path_csr_neighbor(
                    outgoing_neighbors, outgoing_overlay,
                    args.overlay_count, row, shared_selected);
                workspace[args.node_count + shared_length - 1u] = ig_path_csr_edge(
                    outgoing_edges, outgoing_overlay,
                    args.overlay_count, row, shared_selected);
                workspace[shared_length] = next;
                workspace[control + 0u] = shared_length + 1u;
                workspace[control + 1u] = next;
                workspace[control + 5u] += shared_selected - shared_begin + 1u;
                workspace[control + 6u] = IG_PATH_UNREACHED;
                if (next == args.target) workspace[control + 2u] = 1u;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (workspace[control + 2u] != 0u || workspace[control + 3u] != 0u) return;
    }
}

struct IgPathBfsArgs {
    uint node_count;
    uint edge_count;
    uint adjacency_count;
    uint source;
    uint target;
    uint maximum_distance;
    uint layer_mask;
    uint quantum;
    uint threadgroup_width;
    uint node_begin;
    uint overlay_count;
};

// Persistent narrow-frontier BFS reuses the ordinary 3N+8 BFS allocation as distances[N],
// FIFO queue[N], spare[N], control[8]. The result distance lane is therefore directly compatible
// with the existing BFS publisher and deterministic unit-Dijkstra predecessor finalizer.
kernel void ig_path_bfs_persistent_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device uint* workspace [[buffer(1)]],
    constant IgPathBfsArgs& args [[buffer(2)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = args.node_begin + local_node;
    if (node >= args.node_count) return;
    bool source = node == args.source && visible_nodes[node] == 1u;
    workspace[node] = source ? 0u : IG_PATH_UNREACHED;
    workspace[args.node_count + node] = node == 0u ? args.source : IG_PATH_UNREACHED;
    workspace[args.node_count * 2u + node] = IG_PATH_UNREACHED;
    if (node == 0u) {
        uint control = args.node_count * 3u;
        bool source_visible = args.source < args.node_count && visible_nodes[args.source] == 1u;
        workspace[control + 0u] = 0u;                       // FIFO head
        workspace[control + 1u] = source_visible ? 1u : 0u;// FIFO tail
        workspace[control + 2u] = 0u;                       // status
        workspace[control + 3u] = IG_PATH_UNREACHED;        // current source
        workspace[control + 4u] = 0u;                       // current cursor
        workspace[control + 5u] = 0u;                       // current row end
        workspace[control + 6u] = 0u;                       // processed nodes
        workspace[control + 7u] = source_visible ? 0u : 1u;// done
    }
}

kernel void ig_path_bfs_persistent_chunk(
    device const uint* outgoing_offsets [[buffer(0)]],
    device const uint* outgoing_neighbors [[buffer(1)]],
    device const uint* outgoing_edges [[buffer(2)]],
    device const uint* outgoing_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device uint* workspace [[buffer(7)]],
    constant IgPathBfsArgs& args [[buffer(8)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint shared_running;
    threadgroup uint shared_distance;
    threadgroup uint shared_head;
    threadgroup uint shared_batch_count;

    device atomic_uint* distances = (device atomic_uint*)workspace;
    device uint* queue = workspace + args.node_count;
    device uint* cursors = workspace + args.node_count * 2u;
    uint control = args.node_count * 3u;
    if (lane == 0u) {
        shared_running = workspace[control + 2u] == 0u
            && workspace[control + 7u] == 0u && args.quantum != 0u ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint transition = 0u; transition < args.quantum && shared_running != 0u; ++transition) {
        if (lane == 0u) {
            shared_head = workspace[control + 0u];
            uint tail = atomic_load_explicit(
                (device atomic_uint*)(workspace + control + 1u), memory_order_relaxed);
            if (shared_head > tail || tail > args.node_count) {
                workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                shared_running = 0u;
                shared_batch_count = 0u;
            } else if (shared_head == tail) {
                workspace[control + 7u] = 1u;
                shared_running = 0u;
                shared_batch_count = 0u;
            } else {
                uint first_source = queue[shared_head];
                if (first_source >= args.node_count) {
                    workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                    shared_running = 0u;
                    shared_batch_count = 0u;
                } else {
                    shared_distance = atomic_load_explicit(
                        distances + first_source, memory_order_relaxed);
                    if (shared_distance == IG_PATH_UNREACHED
                            || shared_distance >= IG_PATH_UNREACHED - 1u) {
                        workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                        shared_running = 0u;
                        shared_batch_count = 0u;
                    } else if (shared_distance >= args.maximum_distance
                            || (args.target < args.node_count
                                && atomic_load_explicit(
                                    distances + args.target, memory_order_relaxed)
                                    != IG_PATH_UNREACHED
                                && shared_distance >= atomic_load_explicit(
                                    distances + args.target, memory_order_relaxed))) {
                        // FIFO order is nondecreasing by distance. Once every source below the
                        // bound (or below the discovered target depth) has been consumed, all
                        // labels needed by shortest-path reconstruction are complete.
                        workspace[control + 7u] = 1u;
                        shared_running = 0u;
                        shared_batch_count = 0u;
                    } else {
                        uint limit = min(tail, shared_head + args.threadgroup_width);
                        uint cursor = shared_head;
                        while (cursor < limit) {
                            uint source = queue[cursor];
                            if (source >= args.node_count
                                    || atomic_load_explicit(distances + source,
                                        memory_order_relaxed) != shared_distance) break;
                            cursor += 1u;
                        }
                        shared_batch_count = cursor - shared_head;
                        if (shared_batch_count == 0u) {
                            workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                            shared_running = 0u;
                        }
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (shared_running == 0u) break;

        if (lane < shared_batch_count) {
            uint source = queue[shared_head + lane];
            IgPathCsrRow row = ig_path_csr_row(
                outgoing_offsets, outgoing_overlay, args.overlay_count, source);
            uint begin = row.begin;
            uint end = row.end;
            uint bound = row.overlay == 0u ? args.adjacency_count
                : outgoing_overlay[args.overlay_count * 2u];
            if (end < begin || end > bound) {
                atomic_fetch_max_explicit(
                    (device atomic_uint*)(workspace + control + 2u),
                    IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
            } else {
                uint cursor = cursors[source];
                if (cursor != IG_PATH_UNREACHED - 1u) {
                    if (cursor == IG_PATH_UNREACHED) cursor = begin;
                    if (cursor < begin || cursor > end) {
                        atomic_fetch_max_explicit(
                            (device atomic_uint*)(workspace + control + 2u),
                            IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
                    } else {
                        uint tile_end = cursor + min(IG_PATH_THREADS, end - cursor);
                        for (uint position = cursor; position < tile_end; ++position) {
                            uint edge = ig_path_csr_edge(
                                outgoing_edges, outgoing_overlay,
                                args.overlay_count, row, position);
                            uint target = ig_path_csr_neighbor(
                                outgoing_neighbors, outgoing_overlay,
                                args.overlay_count, row, position);
                            if (edge >= args.edge_count || target >= args.node_count) {
                                atomic_fetch_max_explicit(
                                    (device atomic_uint*)(workspace + control + 2u),
                                    IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
                                continue;
                            }
                            uchar active = edge_active[edge];
                            uchar layer = edge_layers[edge];
                            uchar target_visible = visible_nodes[target];
                            if (active > 1u || layer > IG_MAX_LAYER || target_visible > 1u) {
                                atomic_fetch_max_explicit(
                                    (device atomic_uint*)(workspace + control + 2u),
                                    IG_PATH_DIJKSTRA_CORRUPT_VALUE, memory_order_relaxed);
                                continue;
                            }
                            if (active == 0u || target_visible == 0u
                                    || !ig_path_layer_visible(layer, args.layer_mask)) continue;
                            uint previous = atomic_fetch_min_explicit(
                                distances + target, shared_distance + 1u,
                                memory_order_relaxed);
                            bool discovered = previous == IG_PATH_UNREACHED;
                            if (discovered) {
                                uint slot = atomic_fetch_add_explicit(
                                    (device atomic_uint*)(workspace + control + 1u),
                                    1u, memory_order_relaxed);
                                if (slot >= args.node_count) {
                                    atomic_fetch_max_explicit(
                                        (device atomic_uint*)(workspace + control + 2u),
                                        IG_PATH_DIJKSTRA_CORRUPT_VALUE, memory_order_relaxed);
                                } else {
                                    queue[slot] = target;
                                }
                            }
                        }
                        cursors[source] = tile_end == end ? IG_PATH_UNREACHED - 1u : tile_end;
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

        if (lane == 0u) {
            if (workspace[control + 2u] != 0u) {
                shared_running = 0u;
            } else {
                uint completed_head = shared_head;
                uint batch_end = shared_head + shared_batch_count;
                while (completed_head < batch_end
                        && cursors[queue[completed_head]] == IG_PATH_UNREACHED - 1u) {
                    completed_head += 1u;
                }
                workspace[control + 0u] = completed_head;
                workspace[control + 6u] += completed_head - shared_head;
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        uint head = workspace[control + 0u];
        uint tail = atomic_load_explicit(
            (device atomic_uint*)(workspace + control + 1u), memory_order_relaxed);
        if (workspace[control + 2u] != 0u || head >= tail) {
            workspace[control + 7u] = 1u;
        }
    }
}

struct IgPathDijkstraArgs {
    uint node_count;
    uint edge_count;
    uint adjacency_count;
    uint source;
    uint layer_mask;
    uint weight_kind; // 1=i64, 2=f64 bits, 3=mixed tagged bytes
    uint weight_byte_count;
    uint quantum;
    uint outgoing_overlay_count;
    uint incoming_overlay_count;
};

// Dijkstra workspace: distance bits[2N], present[N], frontier A[N], frontier B[N], snapshot
// bits[2N], control[8].
inline uint ig_path_dijkstra_present(uint n) { return 2u * n; }
inline uint ig_path_dijkstra_frontier_a(uint n) { return 3u * n; }
inline uint ig_path_dijkstra_frontier_b(uint n) { return 4u * n; }
inline uint ig_path_dijkstra_snapshot(uint n) { return 5u * n; }
inline uint ig_path_dijkstra_control(uint n) { return 7u * n; }

kernel void ig_path_dijkstra_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device uint* workspace [[buffer(1)]],
    constant IgPathDijkstraArgs& args [[buffer(2)]],
    constant uint& node_begin [[buffer(3)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = node_begin + local_node;
    if (node >= args.node_count) return;
    ig_path_words_store(workspace, node, 0ul);
    workspace[ig_path_dijkstra_present(args.node_count) + node] = node == args.source ? 1u : 0u;
    workspace[ig_path_dijkstra_frontier_a(args.node_count) + node] = node == args.source ? 1u : 0u;
    workspace[ig_path_dijkstra_frontier_b(args.node_count) + node] = 0u;
    ig_path_words_store(workspace + ig_path_dijkstra_snapshot(args.node_count), node, 0ul);
    if (node == 0u) {
        uint control = ig_path_dijkstra_control(args.node_count);
        bool source_visible = args.source < args.node_count && visible_nodes[args.source] == 1u;
        workspace[control + 0u] = source_visible ? 1u : 0u;
        workspace[control + 1u] = 0u;
        workspace[control + 2u] = 0u;
        workspace[control + 3u] = 0u;
        workspace[control + 4u] = 0u;
        workspace[control + 5u] = 0u;
        workspace[control + 6u] = 0u;
        workspace[control + 7u] = 0u;
    }
}

// Persistent-heap initialization reuses the frontier workspace lanes as heap nodes and inverse
// heap positions.  The distance and reachability lanes remain identical to the parallel engine,
// so both strategies share one exact finalizer and result contract.
kernel void ig_path_dijkstra_heap_initialize(
    device const uchar* visible_nodes [[buffer(0)]],
    device uint* workspace [[buffer(1)]],
    constant IgPathDijkstraArgs& args [[buffer(2)]],
    constant uint& node_begin [[buffer(3)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = node_begin + local_node;
    if (node >= args.node_count) return;
    ig_path_words_store(workspace, node, 0ul);
    workspace[ig_path_dijkstra_present(args.node_count) + node] =
        node == args.source ? 1u : 0u;
    workspace[ig_path_dijkstra_frontier_a(args.node_count) + node] =
        node == 0u ? args.source : IG_PATH_UNREACHED;
    workspace[ig_path_dijkstra_frontier_b(args.node_count) + node] =
        node == args.source ? 0u : IG_PATH_UNREACHED;
    ig_path_words_store(workspace + ig_path_dijkstra_snapshot(args.node_count), node, 0ul);
    if (node == 0u) {
        uint control = ig_path_dijkstra_control(args.node_count);
        bool source_visible = args.source < args.node_count && visible_nodes[args.source] == 1u;
        workspace[control + 0u] = source_visible ? 1u : 0u; // heap length
        workspace[control + 1u] = 0u;                       // settled nodes
        workspace[control + 2u] = 0u;                       // status
        workspace[control + 3u] = source_visible ? 0u : 1u;// done
        workspace[control + 4u] = 0u;                       // scanned edges
        workspace[control + 5u] = source_visible ? 1u : 0u;// maximum heap length
        workspace[control + 6u] = IG_PATH_UNREACHED;        // active source
        workspace[control + 7u] = 0u;
    }
}

inline bool ig_path_decode_weight(
    uint edge,
    uint weight_kind,
    device const ulong* homogeneous_values,
    device const uchar* validity,
    device const uint* mixed_offsets,
    device const uchar* mixed_bytes,
    uint mixed_byte_count,
    thread ulong& output,
    thread uint& status) {
    uchar valid = validity[edge];
    if (valid > 1u) {
        status = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
        return false;
    }
    if (valid == 0u) {
        status = IG_PATH_DIJKSTRA_QUERY_TYPE;
        return false;
    }
    if (weight_kind == 1u) {
        ulong raw = homogeneous_values[edge];
        if ((raw >> 63u) != 0ul) {
            status = IG_PATH_DIJKSTRA_INVALID_WEIGHT;
            return false;
        }
        output = ig_path_nonnegative_i64_to_f64(raw);
        return true;
    }
    if (weight_kind == 2u) {
        ulong raw = homogeneous_values[edge];
        ulong magnitude = raw & IG_PATH_F64_MAG;
        uint exponent = uint((magnitude >> 52u) & 0x7fful);
        if (exponent == 0x7ffu || ((raw >> 63u) != 0ul && magnitude != 0ul)) {
            status = IG_PATH_DIJKSTRA_INVALID_WEIGHT;
            return false;
        }
        output = magnitude;
        return true;
    }
    if (weight_kind == 3u) {
        uint begin = mixed_offsets[edge];
        uint end = mixed_offsets[edge + 1u];
        if (end < begin || end > mixed_byte_count) {
            status = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
            return false;
        }
        if (end - begin != 9u) {
            status = IG_PATH_DIJKSTRA_QUERY_TYPE;
            return false;
        }
        uchar tag = mixed_bytes[begin];
        ulong raw = 0ul;
        for (uint byte = 0u; byte < 8u; ++byte) raw |= ulong(mixed_bytes[begin + 1u + byte]) << (byte * 8u);
        if (tag == 2u) {
            if ((raw >> 63u) != 0ul) {
                status = IG_PATH_DIJKSTRA_INVALID_WEIGHT;
                return false;
            }
            output = ig_path_nonnegative_i64_to_f64(raw);
            return true;
        }
        if (tag == 3u) {
            ulong magnitude = raw & IG_PATH_F64_MAG;
            uint exponent = uint((magnitude >> 52u) & 0x7fful);
            if (exponent == 0x7ffu || ((raw >> 63u) != 0ul && magnitude != 0ul)) {
                status = IG_PATH_DIJKSTRA_INVALID_WEIGHT;
                return false;
            }
            output = magnitude;
            return true;
        }
        status = IG_PATH_DIJKSTRA_QUERY_TYPE;
        return false;
    }
    // The property exists but its resident scalar shape is not numeric. This remains a query type
    // error and is raised lazily only if CPU Dijkstra would actually examine the edge.
    status = IG_PATH_DIJKSTRA_QUERY_TYPE;
    return false;
}

inline bool ig_path_dijkstra_heap_less(
    uint left,
    uint right,
    device const uint* workspace) {
    ulong left_distance = ig_path_words_load(workspace, left);
    ulong right_distance = ig_path_words_load(workspace, right);
    return left_distance < right_distance
        || (left_distance == right_distance && left < right);
}

inline void ig_path_dijkstra_heap_swap(
    device uint* heap,
    device uint* positions,
    uint left,
    uint right) {
    uint left_node = heap[left];
    uint right_node = heap[right];
    heap[left] = right_node;
    heap[right] = left_node;
    positions[left_node] = right;
    positions[right_node] = left;
}

inline void ig_path_dijkstra_heap_sift_up(
    device uint* heap,
    device uint* positions,
    device const uint* workspace,
    uint position) {
    while (position != 0u) {
        uint parent = (position - 1u) >> 1u;
        if (!ig_path_dijkstra_heap_less(heap[position], heap[parent], workspace)) break;
        ig_path_dijkstra_heap_swap(heap, positions, position, parent);
        position = parent;
    }
}

inline uint ig_path_dijkstra_heap_pop(
    device uint* heap,
    device uint* positions,
    device const uint* workspace,
    thread uint& heap_length) {
    uint root = heap[0];
    heap_length -= 1u;
    positions[root] = IG_PATH_UNREACHED;
    if (heap_length == 0u) return root;
    uint replacement = heap[heap_length];
    heap[0] = replacement;
    positions[replacement] = 0u;
    uint position = 0u;
    while (true) {
        uint left = position * 2u + 1u;
        if (left >= heap_length) break;
        uint right = left + 1u;
        uint selected = right < heap_length
                && ig_path_dijkstra_heap_less(heap[right], heap[left], workspace)
            ? right : left;
        if (!ig_path_dijkstra_heap_less(heap[selected], heap[position], workspace)) break;
        ig_path_dijkstra_heap_swap(heap, positions, position, selected);
        position = selected;
    }
    return root;
}

// Sparse, low-frontier exact path. One persistent cooperative threadgroup owns a deterministic
// decrease-key heap. Lanes decode and add one CSR block in parallel; lane zero applies the block
// in CSR order, which removes all distance races while retaining parallel weight arithmetic. The
// host bounds each dispatch by args.quantum physical 256-edge tiles (or empty rows), so even one
// giant CSR row has a fixed cancellation quantum. control[6:7] persist the active source/cursor.
kernel void ig_path_dijkstra_heap_chunk(
    device const uint* outgoing_offsets [[buffer(0)]],
    device const uint* outgoing_neighbors [[buffer(1)]],
    device const uint* outgoing_edges [[buffer(2)]],
    device const uint* outgoing_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device const ulong* homogeneous_values [[buffer(7)]],
    device const uchar* weight_validity [[buffer(8)]],
    device const uint* mixed_offsets [[buffer(9)]],
    device const uchar* mixed_bytes [[buffer(10)]],
    device uint* workspace [[buffer(11)]],
    constant IgPathDijkstraArgs& args [[buffer(12)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup uint group_targets[IG_PATH_THREADS];
    threadgroup ulong group_candidates[IG_PATH_THREADS];
    threadgroup uint group_valid[IG_PATH_THREADS];
    threadgroup uint group_status[IG_PATH_THREADS];
    threadgroup uint shared_running;
    threadgroup uint shared_source;
    threadgroup ulong shared_source_distance;
    threadgroup uint shared_cursor;
    threadgroup uint shared_end;
    threadgroup uint shared_batch_count;

    uint control = ig_path_dijkstra_control(args.node_count);
    device uint* heap = workspace + ig_path_dijkstra_frontier_a(args.node_count);
    device uint* positions = workspace + ig_path_dijkstra_frontier_b(args.node_count);
    uint heap_length = 0u;
    uint tiles_this_dispatch = 0u;
    if (lane == 0u) {
        heap_length = workspace[control + 0u];
        uint active_source = workspace[control + 6u];
        if (heap_length > args.node_count
                || (active_source != IG_PATH_UNREACHED && active_source >= args.node_count)) {
            workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
        }
        shared_running = workspace[control + 2u] == 0u && args.quantum != 0u
                && (heap_length != 0u || active_source != IG_PATH_UNREACHED) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    while (shared_running != 0u) {
        if (lane == 0u) {
            if (tiles_this_dispatch >= args.quantum) {
                shared_running = 0u;
                shared_cursor = 0u;
                shared_end = 0u;
                shared_batch_count = 0u;
            } else {
                shared_source = workspace[control + 6u];
                if (shared_source == IG_PATH_UNREACHED && heap_length != 0u) {
                    shared_source = ig_path_dijkstra_heap_pop(
                        heap, positions, workspace, heap_length);
                    workspace[control + 0u] = heap_length;
                    workspace[control + 6u] = shared_source;
                    if (shared_source < args.node_count) {
                        IgPathCsrRow row = ig_path_csr_row(
                            outgoing_offsets, outgoing_overlay,
                            args.outgoing_overlay_count, shared_source);
                        workspace[control + 7u] = row.begin;
                    } else {
                        workspace[control + 7u] = 0u;
                    }
                }
                if (shared_source >= args.node_count) {
                    workspace[control + 2u] = shared_source == IG_PATH_UNREACHED
                        ? IG_PATH_DIJKSTRA_CORRUPT_VALUE : IG_PATH_DIJKSTRA_CORRUPT_CSR;
                    shared_cursor = 0u;
                    shared_end = 0u;
                    shared_batch_count = 0u;
                } else {
                    shared_source_distance = ig_path_words_load(workspace, shared_source);
                    IgPathCsrRow row = ig_path_csr_row(
                        outgoing_offsets, outgoing_overlay,
                        args.outgoing_overlay_count, shared_source);
                    uint begin = row.begin;
                    shared_cursor = workspace[control + 7u];
                    shared_end = row.end;
                    uint bound = row.overlay == 0u ? args.adjacency_count
                        : outgoing_overlay[args.outgoing_overlay_count * 2u];
                    if (shared_end < begin || shared_end > bound
                            || shared_cursor < begin || shared_cursor > shared_end) {
                        workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_CSR;
                        shared_cursor = 0u;
                        shared_end = 0u;
                        shared_batch_count = 0u;
                    } else {
                        shared_batch_count = min(IG_PATH_THREADS, shared_end - shared_cursor);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (shared_running == 0u) break;

        if (shared_batch_count != 0u) {
            group_valid[lane] = 0u;
            group_status[lane] = 0u;
            group_targets[lane] = IG_PATH_UNREACHED;
            group_candidates[lane] = IG_PATH_F64_INF;
            if (lane < shared_batch_count) {
                uint position = shared_cursor + lane;
                IgPathCsrRow row = ig_path_csr_row(
                    outgoing_offsets, outgoing_overlay,
                    args.outgoing_overlay_count, shared_source);
                uint edge = ig_path_csr_edge(
                    outgoing_edges, outgoing_overlay,
                    args.outgoing_overlay_count, row, position);
                uint target = ig_path_csr_neighbor(
                    outgoing_neighbors, outgoing_overlay,
                    args.outgoing_overlay_count, row, position);
                if (edge >= args.edge_count || target >= args.node_count) {
                    group_status[lane] = IG_PATH_DIJKSTRA_CORRUPT_CSR;
                } else {
                    uchar active = edge_active[edge];
                    uchar layer = edge_layers[edge];
                    uchar target_visible = visible_nodes[target];
                    if (active > 1u || layer > IG_MAX_LAYER || target_visible > 1u) {
                        group_status[lane] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                    } else if (active != 0u && target_visible != 0u
                            && ig_path_layer_visible(layer, args.layer_mask)) {
                        ulong weight = 0ul;
                        uint status = 0u;
                        if (ig_path_decode_weight(edge, args.weight_kind, homogeneous_values,
                                weight_validity, mixed_offsets, mixed_bytes,
                                args.weight_byte_count, weight, status)) {
                            group_targets[lane] = target;
                            group_candidates[lane] = ig_path_f64_add_nonnegative(
                                shared_source_distance, weight);
                            group_valid[lane] = 1u;
                        } else {
                            group_status[lane] = status;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lane == 0u) {
                uint status = workspace[control + 2u];
                for (uint index = 0u; index < shared_batch_count; ++index) {
                    status = max(status, group_status[index]);
                }
                workspace[control + 2u] = status;
                if (status == 0u) {
                    uint present_offset = ig_path_dijkstra_present(args.node_count);
                    for (uint index = 0u; index < shared_batch_count; ++index) {
                        if (group_valid[index] == 0u) continue;
                        uint target = group_targets[index];
                        ulong candidate = group_candidates[index];
                        uint present = workspace[present_offset + target];
                        if (present != 0u && present != 1u) {
                            workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                            break;
                        }
                        if (present == 0u || candidate < ig_path_words_load(workspace, target)) {
                            ig_path_words_store(workspace, target, candidate);
                            workspace[present_offset + target] = 1u;
                            uint position = positions[target];
                            if (position == IG_PATH_UNREACHED) {
                                if (heap_length >= args.node_count) {
                                    workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                                    break;
                                }
                                position = heap_length;
                                heap[heap_length] = target;
                                positions[target] = position;
                                heap_length += 1u;
                            } else if (position >= heap_length || heap[position] != target) {
                                workspace[control + 2u] = IG_PATH_DIJKSTRA_CORRUPT_VALUE;
                                break;
                            }
                            ig_path_dijkstra_heap_sift_up(
                                heap, positions, workspace, position);
                        }
                    }
                }
                shared_cursor += shared_batch_count;
                workspace[control + 0u] = heap_length;
                workspace[control + 4u] += shared_batch_count;
                workspace[control + 5u] = max(workspace[control + 5u], heap_length);
                workspace[control + 7u] = shared_cursor;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }

        if (lane == 0u) {
            tiles_this_dispatch += 1u;
            if (workspace[control + 2u] == 0u && shared_cursor >= shared_end) {
                workspace[control + 6u] = IG_PATH_UNREACHED;
                workspace[control + 7u] = 0u;
                workspace[control + 1u] += 1u;
            }
            workspace[control + 0u] = heap_length;
            shared_running = workspace[control + 2u] == 0u
                    && tiles_this_dispatch < args.quantum
                    && (heap_length != 0u
                        || workspace[control + 6u] != IG_PATH_UNREACHED) ? 1u : 0u;
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        workspace[control + 0u] = heap_length;
        workspace[control + 3u] = workspace[control + 2u] != 0u
                || (heap_length == 0u && workspace[control + 6u] == IG_PATH_UNREACHED) ? 1u : 0u;
    }
}

kernel void ig_path_dijkstra_prepare(
    device uint* workspace [[buffer(0)]],
    constant IgPathDijkstraArgs& args [[buffer(1)]],
    constant uint& round [[buffer(2)]],
    constant uint& node_begin [[buffer(3)]],
    uint local_node [[thread_position_in_grid]]) {
    uint node = node_begin + local_node;
    if (node >= args.node_count) return;
    uint snapshot = ig_path_dijkstra_snapshot(args.node_count);
    ig_path_words_store(workspace + snapshot, node, ig_path_words_load(workspace, node));
    uint next_frontier = ((round + 1u) & 1u) == 0u
        ? ig_path_dijkstra_frontier_a(args.node_count)
        : ig_path_dijkstra_frontier_b(args.node_count);
    workspace[next_frontier + node] = 0u;
    if (node == 0u) workspace[ig_path_dijkstra_control(args.node_count) + 1u] = 0u;
}

// Deterministic pull phase. One lane exclusively owns each target and reduces the intersection of
// its incoming row with one fixed global physical-edge tile. Repeated tile dispatches accumulate
// the exact minimum without cross-thread distance writes.
kernel void ig_path_dijkstra_relax(
    device const uint* incoming_offsets [[buffer(0)]],
    device const uint* incoming_neighbors [[buffer(1)]],
    device const uint* incoming_edges [[buffer(2)]],
    device const uint* incoming_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device const ulong* homogeneous_values [[buffer(7)]],
    device const uchar* weight_validity [[buffer(8)]],
    device const uint* mixed_offsets [[buffer(9)]],
    device const uchar* mixed_bytes [[buffer(10)]],
    device uint* workspace [[buffer(11)]],
    constant IgPathDijkstraArgs& args [[buffer(12)]],
    constant uint& round [[buffer(13)]],
    constant uint& edge_begin [[buffer(14)]],
    constant uint& edge_end [[buffer(15)]],
    constant uint& node_begin [[buffer(16)]],
    uint local_target [[thread_position_in_grid]]) {
    uint target = node_begin + local_target;
    if (target >= args.node_count) return;
    uint control = ig_path_dijkstra_control(args.node_count);
    if (workspace[control + 2u] != 0u) return;
    uchar target_visible = visible_nodes[target];
    if (target_visible > 1u || edge_begin > edge_end
            || edge_end > max(args.adjacency_count, args.edge_count)) {
        atomic_fetch_max_explicit(
            (device atomic_uint*)(workspace + control + 2u),
            target_visible > 1u ? IG_PATH_DIJKSTRA_CORRUPT_VALUE
                : IG_PATH_DIJKSTRA_CORRUPT_CSR,
            memory_order_relaxed);
        return;
    }
    if (target_visible == 0u) return;
    IgPathCsrRow row = ig_path_csr_row(
        incoming_offsets, incoming_overlay, args.incoming_overlay_count, target);
    uint begin = row.begin;
    uint end = row.end;
    uint bound = row.overlay == 0u ? args.adjacency_count
        : incoming_overlay[args.incoming_overlay_count * 2u];
    if (end < begin || end > bound) {
        atomic_fetch_max_explicit(
            (device atomic_uint*)(workspace + control + 2u),
            IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
        return;
    }
    begin = max(begin, edge_begin);
    end = min(end, edge_end);
    uint frontier = (round & 1u) == 0u
        ? ig_path_dijkstra_frontier_a(args.node_count)
        : ig_path_dijkstra_frontier_b(args.node_count);
    uint snapshot = ig_path_dijkstra_snapshot(args.node_count);
    bool candidate_present = false;
    ulong best_candidate = IG_PATH_F64_INF;
    for (uint position = begin; position < end; ++position) {
        uint edge = ig_path_csr_edge(
            incoming_edges, incoming_overlay,
            args.incoming_overlay_count, row, position);
        uint source = ig_path_csr_neighbor(
            incoming_neighbors, incoming_overlay,
            args.incoming_overlay_count, row, position);
        if (edge >= args.edge_count || source >= args.node_count) {
            atomic_fetch_max_explicit(
                (device atomic_uint*)(workspace + control + 2u),
                IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
            continue;
        }
        uchar active = edge_active[edge];
        uchar layer = edge_layers[edge];
        uchar source_visible = visible_nodes[source];
        if (active > 1u || layer > IG_MAX_LAYER || source_visible > 1u) {
            atomic_fetch_max_explicit(
                (device atomic_uint*)(workspace + control + 2u),
                IG_PATH_DIJKSTRA_CORRUPT_VALUE, memory_order_relaxed);
            continue;
        }
        if (active == 0u || source_visible == 0u
                || workspace[frontier + source] == 0u
                || !ig_path_layer_visible(layer, args.layer_mask)) continue;
        ulong weight = 0ul;
        uint status = 0u;
        if (!ig_path_decode_weight(edge, args.weight_kind, homogeneous_values, weight_validity,
                mixed_offsets, mixed_bytes, args.weight_byte_count, weight, status)) {
            atomic_fetch_max_explicit(
                (device atomic_uint*)(workspace + control + 2u), status, memory_order_relaxed);
            continue;
        }
        ulong candidate = ig_path_f64_add_nonnegative(
            ig_path_words_load(workspace + snapshot, source), weight);
        if (!candidate_present || candidate < best_candidate) {
            candidate_present = true;
            best_candidate = candidate;
        }
    }
    uint present = ig_path_dijkstra_present(args.node_count);
    if (candidate_present && (workspace[present + target] == 0u
            || best_candidate < ig_path_words_load(workspace, target))) {
        ig_path_words_store(workspace, target, best_candidate);
        workspace[present + target] = 1u;
        uint next_frontier = ((round + 1u) & 1u) == 0u
            ? ig_path_dijkstra_frontier_a(args.node_count)
            : ig_path_dijkstra_frontier_b(args.node_count);
        if (workspace[next_frontier + target] == 0u) {
            workspace[next_frontier + target] = 1u;
            atomic_fetch_add_explicit(
                (device atomic_uint*)(workspace + control + 1u), 1u, memory_order_relaxed);
        }
    }
}

kernel void ig_path_dijkstra_publish(
    device uint* workspace [[buffer(0)]],
    constant IgPathDijkstraArgs& args [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index != 0u) return;
    uint control = ig_path_dijkstra_control(args.node_count);
    workspace[control + 0u] = workspace[control + 1u];
    workspace[control + 3u] += 1u;
}

kernel void ig_path_dijkstra_finalize_prepare(
    device atomic_uint* packet_control [[buffer(0)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0u) atomic_store_explicit(packet_control, 0u, memory_order_relaxed);
}

// Finalization independently derives the smallest predecessor satisfying the exact optimality
// equality.  This makes output independent of parallel relaxation scheduling.
kernel void ig_path_dijkstra_finalize(
    device const uint* incoming_offsets [[buffer(0)]],
    device const uint* incoming_neighbors [[buffer(1)]],
    device const uint* incoming_edges [[buffer(2)]],
    device const uint* incoming_overlay [[buffer(3)]],
    device const uchar* visible_nodes [[buffer(4)]],
    device const uchar* edge_active [[buffer(5)]],
    device const uchar* edge_layers [[buffer(6)]],
    device const ulong* homogeneous_values [[buffer(7)]],
    device const uchar* weight_validity [[buffer(8)]],
    device const uint* mixed_offsets [[buffer(9)]],
    device const uchar* mixed_bytes [[buffer(10)]],
    device uint* workspace [[buffer(11)]],
    device uint* packet [[buffer(12)]],
    constant IgPathDijkstraArgs& args [[buffer(13)]],
    constant uint& edge_begin [[buffer(14)]],
    constant uint& edge_end [[buffer(15)]],
    constant uint& initialize [[buffer(16)]],
    constant uint& node_begin [[buffer(17)]],
    uint local_target [[thread_position_in_grid]]) {
    uint target = node_begin + local_target;
    if (target >= args.node_count) return;
    uint packet_predecessor = args.node_count * 2u;
    uint packet_control = args.node_count * 3u;
    ulong target_distance = ig_path_words_load(workspace, target);
    uint present = workspace[ig_path_dijkstra_present(args.node_count) + target];
    if (initialize != 0u) {
        ig_path_words_store(packet, target, target_distance);
        packet[packet_predecessor + target] = IG_PATH_UNREACHED;
    }
    if (initialize > 1u || edge_begin > edge_end
            || edge_end > max(args.adjacency_count, args.edge_count)) {
        atomic_fetch_max_explicit((device atomic_uint*)(packet + packet_control),
            IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
        return;
    }
    if (present == 0u) return;
    if (target == args.source) return;

    IgPathCsrRow row = ig_path_csr_row(
        incoming_offsets, incoming_overlay, args.incoming_overlay_count, target);
    uint begin = row.begin;
    uint end = row.end;
    uint bound = row.overlay == 0u ? args.adjacency_count
        : incoming_overlay[args.incoming_overlay_count * 2u];
    if (end < begin || end > bound) {
        atomic_fetch_max_explicit((device atomic_uint*)(packet + packet_control),
            IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
        return;
    }
    begin = max(begin, edge_begin);
    end = min(end, edge_end);
    uint best = packet[packet_predecessor + target];
    for (uint position = begin; position < end; ++position) {
        uint edge = ig_path_csr_edge(
            incoming_edges, incoming_overlay,
            args.incoming_overlay_count, row, position);
        uint source = ig_path_csr_neighbor(
            incoming_neighbors, incoming_overlay,
            args.incoming_overlay_count, row, position);
        if (edge >= args.edge_count || source >= args.node_count) {
            atomic_fetch_max_explicit((device atomic_uint*)(packet + packet_control),
                IG_PATH_DIJKSTRA_CORRUPT_CSR, memory_order_relaxed);
            continue;
        }
        uchar active = edge_active[edge];
        uchar layer = edge_layers[edge];
        uchar source_visible = visible_nodes[source];
        if (active > 1u || layer > IG_MAX_LAYER || source_visible > 1u) {
            atomic_fetch_max_explicit((device atomic_uint*)(packet + packet_control),
                IG_PATH_DIJKSTRA_CORRUPT_VALUE, memory_order_relaxed);
            continue;
        }
        if (active == 0u || source_visible == 0u || !ig_path_layer_visible(layer, args.layer_mask)
                || workspace[ig_path_dijkstra_present(args.node_count) + source] == 0u) continue;
        ulong weight = 0ul;
        uint status = 0u;
        if (!ig_path_decode_weight(edge, args.weight_kind, homogeneous_values, weight_validity,
                mixed_offsets, mixed_bytes, args.weight_byte_count, weight, status)) {
            atomic_fetch_max_explicit((device atomic_uint*)(packet + packet_control), status, memory_order_relaxed);
            continue;
        }
        ulong candidate = ig_path_f64_add_nonnegative(ig_path_words_load(workspace, source), weight);
        if (candidate == target_distance && source < best) best = source;
    }
    packet[packet_predecessor + target] = best;
}
