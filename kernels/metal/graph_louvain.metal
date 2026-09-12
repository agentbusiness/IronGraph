#include <metal_stdlib>

using namespace metal;

// IronGraph's deterministic resident Louvain implementation stores an unsigned composite key
// as a sign-flipped `long`.  The repository's stable signed-I64 radix sorter therefore orders the
// encoded value in exactly the same order as the original `(high, low)` unsigned pair.
// The highest valid layer index, matching `Layer::Workspace = 2`. graph_paths.metal,
// graph_components_metrics.metal and operators.metal each already carry this constant; this
// file did not, hand-rolled the bound as a literal, and drifted when a third layer was added.
// A new layer is one edit per file here, not a hunt for literals.
constant uint IG_MAX_LAYER = 2u;
constant ulong IG_LV_SIGN = 0x8000000000000000ul;
constant ulong IG_LV_INVALID_RAW_KEY = 0xfffffffffffffffful;
constant uint IG_LV_INVALID_NODE = 0xffffffffu;
// This is a dispatch-latency bound, not merely a performance heuristic. The host may choose the
// direct candidate path only while every adjacency row fits below this ceiling.
constant ulong IG_LV_DIRECT_NEIGHBOR_HARD_LIMIT = 256ul;
constant ulong IG_LV_SORTED_DECISION_CHUNK_ROWS = 1024ul;
constant uint IG_LV_RADIX_THREADS = 256u;
constant uint IG_LV_RADIX_BUCKETS = 256u;
constant uint IG_LV_RADIX_TILE_ROWS = 1024u;

struct IgLouvainGraphArgs {
    ulong edge_capacity;
    ulong pair_capacity;
    ulong oriented_capacity;
    ulong total_weight;
    ulong work_offset;
    ulong work_count;
    uint node_count;
    uint edge_count;
    uint layer_mask;
    uint reduce_mode;
};

struct IgLouvainRadixArgs {
    ulong row_count;
    ulong block_count;
    ulong work_offset;
    ulong work_count;
    uint digit_pass;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

inline void ig_lv_atomic_add_u64(device atomic_uint* words, ulong value);

kernel void ig_louvain_arange_i64(
    device long* positions [[buffer(0)]],
    constant IgLouvainRadixArgs& args [[buffer(1)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) < args.work_count && row < args.row_count) positions[row] = long(row);
}

kernel void ig_louvain_radix_clear(
    device ulong* totals [[buffer(0)]],
    device ulong* running [[buffer(1)]],
    uint digit [[thread_position_in_grid]]) {
    if (digit < IG_LV_RADIX_BUCKETS) {
        totals[digit] = 0ul;
        running[digit] = 0ul;
    }
}

inline ushort ig_lv_radix_digit(
    device const long* values,
    device const long* positions,
    ulong position_row,
    constant IgLouvainRadixArgs& args) {
    ulong source = as_type<ulong>(positions[position_row]);
    ulong key = as_type<ulong>(values[source]) ^ IG_LV_SIGN;
    return ushort((key >> (args.digit_pass * 8u)) & 0xfful);
}

kernel void ig_louvain_radix_histogram(
    device const long* values [[buffer(0)]],
    device const long* positions [[buffer(1)]],
    device uint* histograms [[buffer(2)]],
    constant IgLouvainRadixArgs& args [[buffer(3)]],
    uint local_block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong block = args.work_offset + ulong(local_block);
    if (ulong(local_block) >= args.work_count || block >= args.block_count) return;
    threadgroup atomic_uint counts[IG_LV_RADIX_BUCKETS];
    atomic_store_explicit(&counts[lane], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong base = block * ulong(IG_LV_RADIX_TILE_ROWS);
    for (uint local = lane; local < IG_LV_RADIX_TILE_ROWS; local += IG_LV_RADIX_THREADS) {
        ulong row = base + ulong(local);
        if (row < args.row_count) {
            ushort digit = ig_lv_radix_digit(values, positions, row, args);
            atomic_fetch_add_explicit(&counts[digit], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    histograms[block * ulong(IG_LV_RADIX_BUCKETS) + ulong(lane)] =
        atomic_load_explicit(&counts[lane], memory_order_relaxed);
}

kernel void ig_louvain_radix_accumulate_totals(
    device const uint* histograms [[buffer(0)]],
    device ulong* totals [[buffer(1)]],
    constant IgLouvainRadixArgs& args [[buffer(2)]],
    uint digit [[thread_index_in_threadgroup]]) {
    ulong total = totals[digit];
    ulong end = min(args.work_offset + args.work_count, args.block_count);
    for (ulong block = args.work_offset; block < end; ++block) {
        total += ulong(histograms[block * ulong(IG_LV_RADIX_BUCKETS) + ulong(digit)]);
    }
    totals[digit] = total;
}

kernel void ig_louvain_radix_initialize_bases(
    device const ulong* totals [[buffer(0)]],
    device ulong* running [[buffer(1)]],
    uint digit [[thread_index_in_threadgroup]]) {
    threadgroup ulong inclusive[IG_LV_RADIX_BUCKETS];
    inclusive[digit] = totals[digit];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint distance = 1u; distance < IG_LV_RADIX_BUCKETS; distance <<= 1u) {
        ulong addend = digit >= distance ? inclusive[digit - distance] : 0ul;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inclusive[digit] += addend;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    running[digit] = digit == 0u ? 0ul : inclusive[digit - 1u];
}

kernel void ig_louvain_radix_offsets(
    device const uint* histograms [[buffer(0)]],
    device ulong* offsets [[buffer(1)]],
    device ulong* running [[buffer(2)]],
    constant IgLouvainRadixArgs& args [[buffer(3)]],
    uint digit [[thread_index_in_threadgroup]]) {
    ulong cursor = running[digit];
    ulong end = min(args.work_offset + args.work_count, args.block_count);
    for (ulong block = args.work_offset; block < end; ++block) {
        ulong index = block * ulong(IG_LV_RADIX_BUCKETS) + ulong(digit);
        offsets[index] = cursor;
        cursor += ulong(histograms[index]);
    }
    running[digit] = cursor;
}

inline bool ig_lv_radix_pair_greater(
    ushort left_digit,
    ushort left_local,
    ushort right_digit,
    ushort right_local) {
    return left_digit > right_digit
        || (left_digit == right_digit && left_local > right_local);
}

kernel void ig_louvain_radix_scatter(
    device const long* values [[buffer(0)]],
    device const long* input_positions [[buffer(1)]],
    device const ulong* offsets [[buffer(2)]],
    device long* output_positions [[buffer(3)]],
    constant IgLouvainRadixArgs& args [[buffer(4)]],
    uint local_block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    ulong block = args.work_offset + ulong(local_block);
    if (ulong(local_block) >= args.work_count || block >= args.block_count) return;
    threadgroup ushort digits[IG_LV_RADIX_TILE_ROWS];
    threadgroup ushort locals[IG_LV_RADIX_TILE_ROWS];
    ulong base = block * ulong(IG_LV_RADIX_TILE_ROWS);
    for (uint local = lane; local < IG_LV_RADIX_TILE_ROWS; local += IG_LV_RADIX_THREADS) {
        ulong row = base + ulong(local);
        digits[local] = row < args.row_count
            ? ig_lv_radix_digit(values, input_positions, row, args)
            : ushort(IG_LV_RADIX_BUCKETS);
        locals[local] = ushort(local);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 2u; width <= IG_LV_RADIX_TILE_ROWS; width <<= 1u) {
        for (uint stride = width >> 1u; stride != 0u; stride >>= 1u) {
            for (uint local = lane; local < IG_LV_RADIX_TILE_ROWS;
                    local += IG_LV_RADIX_THREADS) {
                uint partner = local ^ stride;
                if (partner > local) {
                    ushort left_digit = digits[local];
                    ushort left_local = locals[local];
                    ushort right_digit = digits[partner];
                    ushort right_local = locals[partner];
                    bool greater = ig_lv_radix_pair_greater(
                        left_digit, left_local, right_digit, right_local);
                    bool less = ig_lv_radix_pair_greater(
                        right_digit, right_local, left_digit, left_local);
                    bool ascending = (local & width) == 0u;
                    if ((ascending && greater) || (!ascending && less)) {
                        digits[local] = right_digit;
                        locals[local] = right_local;
                        digits[partner] = left_digit;
                        locals[partner] = left_local;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (uint sorted = lane; sorted < IG_LV_RADIX_TILE_ROWS;
            sorted += IG_LV_RADIX_THREADS) {
        ushort digit = digits[sorted];
        if (digit < IG_LV_RADIX_BUCKETS) {
            uint low = 0u;
            uint high = sorted;
            while (low < high) {
                uint middle = low + ((high - low) >> 1u);
                if (digits[middle] < digit) low = middle + 1u;
                else high = middle;
            }
            ulong destination = offsets[block * ulong(IG_LV_RADIX_BUCKETS) + ulong(digit)]
                + ulong(sorted - low);
            output_positions[destination] = input_positions[base + ulong(locals[sorted])];
        }
    }
}

kernel void ig_louvain_gather_i64(
    device const long* values [[buffer(0)]],
    device const long* positions [[buffer(1)]],
    device long* output [[buffer(2)]],
    constant IgLouvainRadixArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.row_count) return;
    output[row] = values[as_type<ulong>(positions[row])];
}

kernel void ig_louvain_gather_u32(
    device const uint* values [[buffer(0)]],
    device const uint* positions [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant IgLouvainRadixArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.row_count) return;
    output[row] = values[positions[row]];
}

kernel void ig_louvain_read_u32(
    device const uint* values [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant IgLouvainRadixArgs& args [[buffer(2)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u && args.work_offset < args.row_count) output[0] = values[args.work_offset];
}

kernel void ig_louvain_sum_u32_clear(
    device atomic_uint* output [[buffer(0)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u) atomic_store_explicit(output, 0u, memory_order_relaxed);
}

kernel void ig_louvain_sum_u32(
    device const uint* values [[buffer(0)]],
    device atomic_uint* output [[buffer(1)]],
    constant IgLouvainRadixArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.row_count) return;
    atomic_fetch_add_explicit(output, values[row], memory_order_relaxed);
}

kernel void ig_louvain_sum_i64_clear(
    device atomic_uint* output_words [[buffer(0)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u) {
        atomic_store_explicit(output_words, 0u, memory_order_relaxed);
        atomic_store_explicit(output_words + 1, 0u, memory_order_relaxed);
    }
}

kernel void ig_louvain_sum_i64(
    device const long* values [[buffer(0)]],
    device atomic_uint* output_words [[buffer(1)]],
    constant IgLouvainRadixArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.row_count) return;
    long value = values[row];
    if (value > 0l) ig_lv_atomic_add_u64(output_words, ulong(value));
}

kernel void ig_louvain_zero_u32(
    device uint* output [[buffer(0)]],
    constant IgLouvainRadixArgs& args [[buffer(1)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) < args.work_count && row < args.row_count) output[row] = 0u;
}

kernel void ig_louvain_u8_to_u32(
    device const uchar* input [[buffer(0)]],
    device uint* output [[buffer(1)]],
    constant IgLouvainRadixArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) < args.work_count && row < args.row_count) output[row] = uint(input[row]);
}

inline long ig_lv_encode_key(ulong raw) {
    return as_type<long>(raw ^ IG_LV_SIGN);
}

inline ulong ig_lv_decode_key(long encoded) {
    return as_type<ulong>(encoded) ^ IG_LV_SIGN;
}

inline ulong ig_lv_pair_key(uint high, uint low) {
    return (ulong(high) << 32u) | ulong(low);
}

inline bool ig_lv_layer_visible(uchar layer, uint mask) {
    // The bound is the mask's width, not the number of layers that existed when this was written.
    // `Layer::Workspace = 2` made `layer <= 1u` silently drop every Workspace relationship, and
    // the sibling helpers in graph_paths/graph_components_metrics/operators already use 32.
    return layer < 32u && (mask & (1u << uint(layer))) != 0u;
}

// Metal's portable integer atomics are 32-bit. Store every exact non-negative u64 accumulator as
// little-endian low/high atomic words. Each low-word fetch-add has a unique predecessor, so its
// carry is exact; the independent high-word fetch-add then makes the final two-word sum exact and
// scheduler independent after the dispatch boundary.
inline void ig_lv_atomic_add_u64(device atomic_uint* words, ulong value) {
    uint low = uint(value);
    uint previous = atomic_fetch_add_explicit(words, low, memory_order_relaxed);
    uint carry = uint(previous + low < previous);
    atomic_fetch_add_explicit(
        words + 1, uint(value >> 32u) + carry, memory_order_relaxed);
}

inline ulong ig_lv_lower_bound_key(
    device const long* sorted_keys,
    ulong length,
    long needle) {
    ulong low = 0ul;
    ulong upper = length;
    while (low < upper) {
        ulong middle = low + ((upper - low) >> 1u);
        if (sorted_keys[middle] < needle) low = middle + 1ul;
        else upper = middle;
    }
    return low;
}

kernel void ig_louvain_validate_clear(
    device atomic_uint* status [[buffer(0)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u) atomic_store_explicit(status, 0u, memory_order_relaxed);
}

// Status: 1=non-canonical node visibility, 2=non-canonical edge metadata, 3=endpoint bounds.
kernel void ig_louvain_validate_graph(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uchar* edge_active [[buffer(1)]],
    device const uchar* edge_layers [[buffer(2)]],
    device const uint* edge_sources [[buffer(3)]],
    device const uint* edge_targets [[buffer(4)]],
    device atomic_uint* status [[buffer(5)]],
    constant IgLouvainGraphArgs& args [[buffer(6)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position < ulong(args.node_count) && visible_nodes[position] > 1u) {
        atomic_fetch_max_explicit(status, 1u, memory_order_relaxed);
    }
    if (position < ulong(args.edge_count)) {
        // `edge_active` is a boolean. `edge_layers` is a layer index: treating index 2
        // (Workspace) as corrupt failed every projection over a graph holding Workspace
        // relationships, which is what `CALL graph.louvain()` reported as status 2.
        if (edge_active[position] > 1u || edge_layers[position] > IG_MAX_LAYER) {
            atomic_fetch_max_explicit(status, 2u, memory_order_relaxed);
        }
        if (edge_sources[position] >= args.node_count
                || edge_targets[position] >= args.node_count) {
            atomic_fetch_max_explicit(status, 3u, memory_order_relaxed);
        }
    }
}

// One canonical undirected key per resident relationship.  Invalid, hidden, and self-loop rows
// receive the terminal sentinel.  Sorting followed by `reduce_mode=0` collapses both relationship
// direction and parallel relationships to one unweighted undirected edge.
kernel void ig_louvain_base_pairs(
    device const uchar* visible_nodes [[buffer(0)]],
    device const uchar* edge_active [[buffer(1)]],
    device const uchar* edge_layers [[buffer(2)]],
    device const uint* edge_sources [[buffer(3)]],
    device const uint* edge_targets [[buffer(4)]],
    device long* pair_keys [[buffer(5)]],
    device long* pair_weights [[buffer(6)]],
    constant IgLouvainGraphArgs& args [[buffer(7)]],
    uint local [[thread_position_in_grid]]) {
    ulong edge = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (edge >= args.edge_capacity) return;
    pair_keys[edge] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
    pair_weights[edge] = 0;
    if (edge >= ulong(args.edge_count) || edge_active[edge] == 0u
            || !ig_lv_layer_visible(edge_layers[edge], args.layer_mask)) return;
    uint source = edge_sources[edge];
    uint target = edge_targets[edge];
    if (source == target || visible_nodes[source] == 0u || visible_nodes[target] == 0u) return;
    uint low = min(source, target);
    uint high = max(source, target);
    pair_keys[edge] = ig_lv_encode_key(ig_lv_pair_key(low, high));
    pair_weights[edge] = 1;
}

// The output retains the fixed pair capacity: only a run's first row is live and every other row
// is a zero-weight sentinel. Clearing and accumulation are separate ordered dispatches so every
// input row performs fixed O(log E) work; no lane scans an unbounded duplicate run. Base mode
// deduplicates to weight one. Coarse mode accumulates exact integer weights through two-word atomics.
kernel void ig_louvain_reduce_pairs_clear(
    device long* reduced_keys [[buffer(0)]],
    device atomic_uint* reduced_weight_words [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.pair_capacity) return;
    reduced_keys[position] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
    ulong word = ulong(position) * 2ul;
    atomic_store_explicit(reduced_weight_words + word, 0u, memory_order_relaxed);
    atomic_store_explicit(reduced_weight_words + word + 1ul, 0u, memory_order_relaxed);
}

kernel void ig_louvain_reduce_pairs(
    device const long* sorted_keys [[buffer(0)]],
    device const long* sorted_weights [[buffer(1)]],
    device long* reduced_keys [[buffer(2)]],
    device atomic_uint* reduced_weight_words [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.pair_capacity) return;
    ulong raw = ig_lv_decode_key(sorted_keys[position]);
    long signed_weight = sorted_weights[position];
    if (raw == IG_LV_INVALID_RAW_KEY || signed_weight <= 0l) return;
    ulong start = ig_lv_lower_bound_key(
        sorted_keys, args.pair_capacity, sorted_keys[position]);
    if (position == start) {
        reduced_keys[start] = sorted_keys[position];
        if (args.reduce_mode == 0u) {
            ulong word = start * 2ul;
            atomic_store_explicit(reduced_weight_words + word, 1u, memory_order_relaxed);
            atomic_store_explicit(reduced_weight_words + word + 1ul, 0u, memory_order_relaxed);
            return;
        }
    }
    if (args.reduce_mode != 0u) {
        ig_lv_atomic_add_u64(
            reduced_weight_words + start * 2ul, ulong(signed_weight));
    }
}

// Expand each live canonical pair into sorted-able directed adjacency entries.  A coarse self-loop
// already carries both directed halves in its weight and therefore emits exactly one row.
kernel void ig_louvain_orient_pairs(
    device const long* pair_keys [[buffer(0)]],
    device const long* pair_weights [[buffer(1)]],
    device long* oriented_keys [[buffer(2)]],
    device long* oriented_weights [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint local [[thread_position_in_grid]]) {
    ulong pair = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (pair >= args.pair_capacity) return;
    ulong first = pair * 2ul;
    oriented_keys[first] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
    oriented_keys[first + 1ul] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
    oriented_weights[first] = 0;
    oriented_weights[first + 1ul] = 0;
    long signed_weight = pair_weights[pair];
    if (signed_weight <= 0) return;
    ulong raw = ig_lv_decode_key(pair_keys[pair]);
    if (raw == IG_LV_INVALID_RAW_KEY) return;
    uint left = uint(raw >> 32u);
    uint right = uint(raw);
    oriented_keys[first] = ig_lv_encode_key(ig_lv_pair_key(left, right));
    oriented_weights[first] = signed_weight;
    if (left != right) {
        oriented_keys[first + 1ul] = ig_lv_encode_key(ig_lv_pair_key(right, left));
        oriented_weights[first + 1ul] = signed_weight;
    }
}

inline ulong ig_lv_lower_bound_high(
    device const long* sorted_keys,
    ulong length,
    uint high) {
    ulong needle = ulong(high) << 32u;
    ulong low = 0ul;
    ulong upper = length;
    while (low < upper) {
        ulong middle = low + ((upper - low) >> 1u);
        if (ig_lv_decode_key(sorted_keys[middle]) < needle) low = middle + 1ul;
        else upper = middle;
    }
    return low;
}

inline ulong ig_lv_upper_bound_high(
    device const long* sorted_keys,
    ulong length,
    uint high) {
    ulong needle = ulong(high + 1u) << 32u;
    ulong low = 0ul;
    ulong upper = length;
    while (low < upper) {
        ulong middle = low + ((upper - low) >> 1u);
        if (ig_lv_decode_key(sorted_keys[middle]) < needle) low = middle + 1ul;
        else upper = middle;
    }
    return low;
}

kernel void ig_louvain_degree_clear(
    device atomic_uint* degree_words [[buffer(0)]],
    constant IgLouvainGraphArgs& args [[buffer(1)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    ulong word = ulong(node) * 2ul;
    atomic_store_explicit(degree_words + word, 0u, memory_order_relaxed);
    atomic_store_explicit(degree_words + word + 1ul, 0u, memory_order_relaxed);
}

kernel void ig_louvain_degree(
    device const long* sorted_oriented_keys [[buffer(0)]],
    device const long* sorted_oriented_weights [[buffer(1)]],
    device const uint* active_nodes [[buffer(2)]],
    device atomic_uint* degree_words [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.oriented_capacity) return;
    ulong raw = ig_lv_decode_key(sorted_oriented_keys[position]);
    long signed_weight = sorted_oriented_weights[position];
    if (raw == IG_LV_INVALID_RAW_KEY || signed_weight <= 0l) return;
    uint node = uint(raw >> 32u);
    if (node >= args.node_count || active_nodes[node] == 0u) return;
    ig_lv_atomic_add_u64(degree_words + ulong(node) * 2ul, ulong(signed_weight));
}

kernel void ig_louvain_high_degree_clear(
    device atomic_uint* high_degree [[buffer(0)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u) atomic_store_explicit(high_degree, 0u, memory_order_relaxed);
}

kernel void ig_louvain_high_degree(
    device const long* sorted_oriented_keys [[buffer(0)]],
    device const uint* active_nodes [[buffer(1)]],
    device atomic_uint* maximum_degree [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || active_nodes[node] == 0u) return;
    ulong begin = ig_lv_lower_bound_high(sorted_oriented_keys, args.oriented_capacity, node);
    ulong end = ig_lv_upper_bound_high(sorted_oriented_keys, args.oriented_capacity, node);
    atomic_fetch_max_explicit(maximum_degree, uint(end - begin), memory_order_relaxed);
}

kernel void ig_louvain_initialize_level(
    device const uint* active_nodes [[buffer(0)]],
    device uint* membership [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) membership[node] = active_nodes[node] != 0u
        ? node : IG_LV_INVALID_NODE;
}

kernel void ig_louvain_candidate_keys(
    device const long* oriented_keys [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device long* candidate_keys [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.oriented_capacity) return;
    ulong raw = ig_lv_decode_key(oriented_keys[position]);
    if (raw == IG_LV_INVALID_RAW_KEY) {
        candidate_keys[position] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
        return;
    }
    uint source = uint(raw >> 32u);
    uint neighbor = uint(raw);
    // Coarse self-loops encode edges internal to the represented aggregate. They remain internal
    // under every move, so they contribute to degree but cancel from the move link term.
    if (source == neighbor) {
        candidate_keys[position] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
        return;
    }
    uint community = membership[neighbor];
    candidate_keys[position] = community == IG_LV_INVALID_NODE
        ? ig_lv_encode_key(IG_LV_INVALID_RAW_KEY)
        : ig_lv_encode_key(ig_lv_pair_key(source, community));
}

kernel void ig_louvain_candidate_aggregate_clear(
    device atomic_uint* aggregate_words [[buffer(0)]],
    constant IgLouvainGraphArgs& args [[buffer(1)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.oriented_capacity) return;
    ulong word = ulong(position) * 2ul;
    atomic_store_explicit(aggregate_words + word, 0u, memory_order_relaxed);
    atomic_store_explicit(aggregate_words + word + 1ul, 0u, memory_order_relaxed);
}

// Aggregate each `(node, community)` run at its first row. Every physical candidate row performs
// one bounded binary search and one two-word atomic addition; no lane owns a high-degree run.
kernel void ig_louvain_candidate_aggregate(
    device const long* sorted_candidate_keys [[buffer(0)]],
    device const long* sorted_candidate_weights [[buffer(1)]],
    device atomic_uint* aggregate_words [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position >= args.oriented_capacity) return;
    long key = sorted_candidate_keys[position];
    long signed_weight = sorted_candidate_weights[position];
    if (ig_lv_decode_key(key) == IG_LV_INVALID_RAW_KEY || signed_weight <= 0l) return;
    ulong start = ig_lv_lower_bound_key(sorted_candidate_keys, args.oriented_capacity, key);
    ig_lv_atomic_add_u64(aggregate_words + start * 2ul, ulong(signed_weight));
}

struct IgLvWide {
    ulong high;
    ulong low;
};

inline IgLvWide ig_lv_multiply_wide(ulong left, ulong right) {
    ulong left_low = left & 0xfffffffful;
    ulong left_high = left >> 32u;
    ulong right_low = right & 0xfffffffful;
    ulong right_high = right >> 32u;
    ulong product_0 = left_low * right_low;
    ulong product_1 = left_low * right_high;
    ulong product_2 = left_high * right_low;
    ulong product_3 = left_high * right_high;
    ulong middle = (product_0 >> 32u)
        + (product_1 & 0xfffffffful) + (product_2 & 0xfffffffful);
    IgLvWide result;
    result.low = (product_0 & 0xfffffffful) | (middle << 32u);
    result.high = product_3 + (product_1 >> 32u) + (product_2 >> 32u) + (middle >> 32u);
    return result;
}

inline IgLvWide ig_lv_add_wide(IgLvWide left, IgLvWide right) {
    IgLvWide result;
    result.low = left.low + right.low;
    result.high = left.high + right.high + ulong(result.low < left.low);
    return result;
}

inline int ig_lv_compare_wide(IgLvWide left, IgLvWide right) {
    if (left.high != right.high) return left.high < right.high ? -1 : 1;
    if (left.low != right.low) return left.low < right.low ? -1 : 1;
    return 0;
}

// Compare `link_a*T - degree*community_a` with the corresponding B score without signed
// subtraction: A > B iff `link_a*T + degree*community_b` is greater than the opposite sum.
inline int ig_lv_compare_gain(
    ulong link_a,
    ulong community_a,
    ulong link_b,
    ulong community_b,
    ulong degree,
    ulong total_weight) {
    IgLvWide left = ig_lv_add_wide(
        ig_lv_multiply_wide(link_a, total_weight),
        ig_lv_multiply_wide(degree, community_b));
    IgLvWide right = ig_lv_add_wide(
        ig_lv_multiply_wide(link_b, total_weight),
        ig_lv_multiply_wide(degree, community_a));
    return ig_lv_compare_wide(left, right);
}

kernel void ig_louvain_sorted_initialize(
    device const uint* active_nodes [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device long* state [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    ulong nodes = ulong(args.node_count);
    state[node] = active_nodes[node] != 0u ? long(membership[node]) : long(IG_LV_INVALID_NODE);
    state[nodes + node] = 0l;
    state[nodes * 2ul + node] = 0l;
}

// Resume an exact sorted-candidate decision over at most 1024 physical rows per node. The host
// synchronizes and checks cancellation between chunks, so even a single adversarial hub cannot
// monopolize one uninterruptible dispatch.
kernel void ig_louvain_sorted_chunk(
    device const long* prior_state [[buffer(0)]],
    device const uint* active_nodes [[buffer(1)]],
    device const uint* membership [[buffer(2)]],
    device const long* degree_values [[buffer(3)]],
    device const long* community_weights [[buffer(4)]],
    device const long* sorted_candidate_keys [[buffer(5)]],
    device const long* aggregate_weights [[buffer(6)]],
    device long* next_state [[buffer(7)]],
    constant IgLouvainGraphArgs& args [[buffer(8)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    ulong nodes = ulong(args.node_count);
    ulong link_base = nodes;
    ulong current_link_base = nodes * 2ul;
    if (active_nodes[node] == 0u) {
        next_state[node] = long(IG_LV_INVALID_NODE);
        next_state[link_base + node] = 0l;
        next_state[current_link_base + node] = 0l;
        return;
    }
    uint current = membership[node];
    ulong degree = ulong(max(degree_values[node], 0l));
    uint best = uint(max(prior_state[node], 0l));
    ulong best_link = ulong(max(prior_state[link_base + node], 0l));
    ulong current_link = ulong(max(prior_state[current_link_base + node], 0l));
    ulong current_weight = ulong(max(community_weights[current], 0l));
    ulong current_adjusted = current_weight >= degree ? current_weight - degree : 0ul;
    ulong best_weight = ulong(max(community_weights[best], 0l));
    if (best == current) best_weight = best_weight >= degree ? best_weight - degree : 0ul;

    ulong begin = ig_lv_lower_bound_high(sorted_candidate_keys, args.oriented_capacity, node);
    ulong end = ig_lv_upper_bound_high(sorted_candidate_keys, args.oriented_capacity, node);
    ulong cursor = min(begin + ulong(args.reduce_mode), end);
    ulong finish = min(cursor + IG_LV_SORTED_DECISION_CHUNK_ROWS, end);
    for (; cursor < finish; ++cursor) {
        long signed_link = aggregate_weights[cursor];
        if (signed_link <= 0l) continue;
        ulong raw = ig_lv_decode_key(sorted_candidate_keys[cursor]);
        uint candidate = uint(raw);
        ulong link = ulong(signed_link);
        ulong candidate_weight = ulong(max(community_weights[candidate], 0l));
        if (candidate == current) {
            candidate_weight = candidate_weight >= degree ? candidate_weight - degree : 0ul;
            current_link = link;
        }
        int ordering = ig_lv_compare_gain(
            link, candidate_weight, best_link, best_weight,
            degree, args.total_weight);
        if (ordering > 0 || (ordering == 0 && candidate < best)) {
            best = candidate;
            best_link = link;
            best_weight = candidate_weight;
        }
    }
    next_state[node] = long(best);
    next_state[link_base + node] = long(best_link);
    next_state[current_link_base + node] = long(current_link);
}

kernel void ig_louvain_decide_clear(
    device atomic_uint* changed [[buffer(0)]],
    uint lane [[thread_position_in_grid]]) {
    if (lane == 0u) atomic_store_explicit(changed, 0u, memory_order_relaxed);
}

// Deterministic proposal phase. Every node considers every neighboring community and emits a move
// only when the best candidate is strictly better than remaining in its current community. A
// separate deterministic community-disjoint matching phase below selects proposals that can be
// applied together with an additive, therefore non-negative, modularity change.
kernel void ig_louvain_decide(
    device const uint* active_nodes [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device const long* degree_values [[buffer(2)]],
    device const long* community_weights [[buffer(3)]],
    device const long* sorted_oriented_keys [[buffer(4)]],
    device const long* sorted_oriented_weights [[buffer(5)]],
    device uint* next_membership [[buffer(6)]],
    device atomic_uint* changed [[buffer(7)]],
    constant IgLouvainGraphArgs& args [[buffer(8)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    if (active_nodes[node] == 0u) {
        next_membership[node] = IG_LV_INVALID_NODE;
        return;
    }
    uint current = membership[node];
    ulong degree = ulong(max(degree_values[node], 0l));
    if (degree == 0ul || args.total_weight == 0ul) {
        next_membership[node] = current;
        return;
    }
    ulong current_weight = ulong(max(community_weights[current], 0l));
    ulong current_adjusted = current_weight >= degree ? current_weight - degree : 0ul;
    uint best = current;
    ulong best_link = 0ul;
    ulong best_community = current_adjusted;
    ulong current_link = 0ul;

    // The resident adjacency is already sorted by source. The host's cost router uses this exact
    // path only below IG_LV_DIRECT_NEIGHBOR_HARD_LIMIT and otherwise globally radix-sorts
    // `(node, community)`. Keep a device-side guard so a host/kernel contract drift fails closed
    // instead of launching an unbounded quadratic row scan.
    ulong begin = ig_lv_lower_bound_high(sorted_oriented_keys, args.oriented_capacity, node);
    ulong end = ig_lv_upper_bound_high(sorted_oriented_keys, args.oriented_capacity, node);
    if (end - begin > IG_LV_DIRECT_NEIGHBOR_HARD_LIMIT) {
        next_membership[node] = current;
        atomic_fetch_max_explicit(changed, 2u, memory_order_relaxed);
        return;
    }
    for (ulong cursor = begin; cursor < end; ++cursor) {
        ulong raw = ig_lv_decode_key(sorted_oriented_keys[cursor]);
        uint neighbor = uint(raw);
        if (neighbor == node) continue;
        uint candidate = membership[neighbor];
        if (candidate == IG_LV_INVALID_NODE) continue;
        bool first_for_community = true;
        for (ulong prior = begin; prior < cursor; ++prior) {
            ulong prior_raw = ig_lv_decode_key(sorted_oriented_keys[prior]);
            uint prior_neighbor = uint(prior_raw);
            if (prior_neighbor != node && membership[prior_neighbor] == candidate) {
                first_for_community = false;
                break;
            }
        }
        if (!first_for_community) continue;
        ulong link = 0ul;
        for (ulong scan = cursor; scan < end; ++scan) {
            ulong scan_raw = ig_lv_decode_key(sorted_oriented_keys[scan]);
            uint scan_neighbor = uint(scan_raw);
            if (scan_neighbor != node && membership[scan_neighbor] == candidate) {
                link += ulong(max(sorted_oriented_weights[scan], 0l));
            }
        }
        ulong candidate_weight = ulong(max(community_weights[candidate], 0l));
        if (candidate == current) {
            candidate_weight = candidate_weight >= degree ? candidate_weight - degree : 0ul;
            current_link = link;
        }
        int ordering = ig_lv_compare_gain(
            link, candidate_weight, best_link, best_community,
            degree, args.total_weight);
        if (ordering > 0 || (ordering == 0 && candidate < best)) {
            best = candidate;
            best_link = link;
            best_community = candidate_weight;
        }
    }
    int improvement = ig_lv_compare_gain(
        best_link, best_community, current_link, current_adjusted,
        degree, args.total_weight);
    uint proposal = best != current && improvement > 0 ? best : current;
    next_membership[node] = proposal;
    if (proposal != current) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

kernel void ig_louvain_decide_sorted(
    device const long* state [[buffer(0)]],
    device const uint* active_nodes [[buffer(1)]],
    device const uint* membership [[buffer(2)]],
    device const long* degree_values [[buffer(3)]],
    device const long* community_weights [[buffer(4)]],
    device uint* next_membership [[buffer(5)]],
    device atomic_uint* changed [[buffer(6)]],
    constant IgLouvainGraphArgs& args [[buffer(7)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    if (active_nodes[node] == 0u) {
        next_membership[node] = IG_LV_INVALID_NODE;
        return;
    }
    uint current = membership[node];
    ulong degree = ulong(max(degree_values[node], 0l));
    if (degree == 0ul || args.total_weight == 0ul) {
        next_membership[node] = current;
        return;
    }
    ulong current_weight = ulong(max(community_weights[current], 0l));
    ulong current_adjusted = current_weight >= degree ? current_weight - degree : 0ul;
    ulong nodes = ulong(args.node_count);
    long signed_best = state[node];
    if (signed_best < 0l || ulong(signed_best) >= nodes) {
        next_membership[node] = current;
        atomic_fetch_max_explicit(changed, 2u, memory_order_relaxed);
        return;
    }
    uint best = uint(signed_best);
    ulong best_link = ulong(max(state[nodes + node], 0l));
    ulong current_link = ulong(max(state[nodes * 2ul + node], 0l));
    ulong best_community = ulong(max(community_weights[best], 0l));
    if (best == current) {
        best_community = best_community >= degree ? best_community - degree : 0ul;
    }
    int improvement = ig_lv_compare_gain(
        best_link, best_community, current_link, current_adjusted,
        degree, args.total_weight);
    uint proposal = best != current && improvement > 0 ? best : current;
    next_membership[node] = proposal;
    if (proposal != current) atomic_store_explicit(changed, 1u, memory_order_relaxed);
}

inline uint ig_lv_priority(uint node) {
    uint value = node + 0x9e3779b9u;
    value = (value ^ (value >> 16u)) * 0x85ebca6bu;
    value = (value ^ (value >> 13u)) * 0xc2b2ae35u;
    return value ^ (value >> 16u);
}

inline bool ig_lv_priority_before(uint left, uint right) {
    uint left_priority = ig_lv_priority(left);
    uint right_priority = ig_lv_priority(right);
    return left_priority < right_priority
        || (left_priority == right_priority && left < right);
}

kernel void ig_louvain_matching_initialize(
    device const uint* active_nodes [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device const uint* proposals [[buffer(2)]],
    device uint* active_proposals [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) active_proposals[node] = active_nodes[node] != 0u
        && proposals[node] != membership[node] ? 1u : 0u;
}

kernel void ig_louvain_clear_u32_max(
    device uint* values [[buffer(0)]],
    constant IgLouvainGraphArgs& args [[buffer(1)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) {
        values[node] = IG_LV_INVALID_NODE;
        values[ulong(args.node_count) + node] = IG_LV_INVALID_NODE;
    }
}

// Exact deterministic `(hash priority, node)` endpoint minima without sorting 2N endpoint rows.
// The second phase resolves the theoretically possible 32-bit hash collision by node ordinal.
kernel void ig_louvain_matching_priority(
    device const uint* active_proposals [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device const uint* proposals [[buffer(2)]],
    device atomic_uint* minimum_priority [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || active_proposals[node] == 0u) return;
    uint priority = ig_lv_priority(node);
    atomic_fetch_min_explicit(
        minimum_priority + membership[node], priority, memory_order_relaxed);
    atomic_fetch_min_explicit(
        minimum_priority + proposals[node], priority, memory_order_relaxed);
}

kernel void ig_louvain_matching_minimum(
    device const uint* active_proposals [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device const uint* proposals [[buffer(2)]],
    device const uint* minimum_priority [[buffer(3)]],
    device atomic_uint* minimum_proposer [[buffer(4)]],
    constant IgLouvainGraphArgs& args [[buffer(5)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || active_proposals[node] == 0u) return;
    uint priority = ig_lv_priority(node);
    uint current = membership[node];
    uint target = proposals[node];
    if (minimum_priority[current] == priority) {
        atomic_fetch_min_explicit(minimum_proposer + current, node, memory_order_relaxed);
    }
    if (minimum_priority[target] == priority) {
        atomic_fetch_min_explicit(minimum_proposer + target, node, memory_order_relaxed);
    }
}

kernel void ig_louvain_matching_clear(
    device atomic_uint* accepted [[buffer(0)]],
    device atomic_uint* locked_communities [[buffer(1)]],
    device atomic_uint* accepted_count [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) {
        atomic_store_explicit(accepted + node, 0u, memory_order_relaxed);
        atomic_store_explicit(locked_communities + node, 0u, memory_order_relaxed);
    }
    if (node == 0u) atomic_store_explicit(accepted_count, 0u, memory_order_relaxed);
}

kernel void ig_louvain_matching_winners(
    device const uint* active_proposals [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device const uint* proposals [[buffer(2)]],
    device const uint* minimum_proposer [[buffer(3)]],
    device atomic_uint* accepted [[buffer(4)]],
    device atomic_uint* locked_communities [[buffer(5)]],
    device atomic_uint* accepted_count [[buffer(6)]],
    constant IgLouvainGraphArgs& args [[buffer(7)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || active_proposals[node] == 0u) return;
    uint current = membership[node];
    uint target = proposals[node];
    ulong proposer_base = ulong(args.node_count);
    if (minimum_proposer[proposer_base + current] != node
            || minimum_proposer[proposer_base + target] != node) return;
    atomic_store_explicit(accepted + node, 1u, memory_order_relaxed);
    atomic_store_explicit(locked_communities + current, 1u, memory_order_relaxed);
    atomic_store_explicit(locked_communities + target, 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(accepted_count, 1u, memory_order_relaxed);
}

kernel void ig_louvain_matching_advance(
    device const uint* active_proposals [[buffer(0)]],
    device const uint* accumulated_accepted [[buffer(1)]],
    device const uint* membership [[buffer(2)]],
    device const uint* proposals [[buffer(3)]],
    device const uint* round_accepted [[buffer(4)]],
    device const uint* locked_communities [[buffer(5)]],
    device uint* next_active [[buffer(6)]],
    device uint* next_accumulated [[buffer(7)]],
    constant IgLouvainGraphArgs& args [[buffer(8)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    if (active_proposals[node] == 0u) {
        next_active[node] = 0u;
        next_accumulated[node] = accumulated_accepted[node];
        return;
    }
    bool won = round_accepted[node] != 0u;
    next_accumulated[node] = accumulated_accepted[node] | uint(won);
    bool conflicts = locked_communities[membership[node]] != 0u
        || locked_communities[proposals[node]] != 0u;
    next_active[node] = active_proposals[node] != 0u && !won && !conflicts ? 1u : 0u;
}

kernel void ig_louvain_apply_matching(
    device const uint* membership [[buffer(0)]],
    device const uint* proposals [[buffer(1)]],
    device const uint* accepted [[buffer(2)]],
    device uint* next_membership [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) next_membership[node] = accepted[node] != 0u
        ? proposals[node] : membership[node];
}

kernel void ig_louvain_copy_community_weights(
    device const long* community_weights [[buffer(0)]],
    device long* next_community_weights [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node < args.node_count) next_community_weights[node] = community_weights[node];
}

// Accepted moves have disjoint source and target communities, so these exact I64 updates never
// target the same row and require neither floating point nor scheduler-dependent atomics.
kernel void ig_louvain_update_community_weights(
    device const uint* membership [[buffer(0)]],
    device const uint* proposals [[buffer(1)]],
    device const uint* accepted [[buffer(2)]],
    device const long* degree [[buffer(3)]],
    device long* next_community_weights [[buffer(4)]],
    constant IgLouvainGraphArgs& args [[buffer(5)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || accepted[node] == 0u) return;
    uint current = membership[node];
    uint target = proposals[node];
    ulong node_degree = ulong(max(degree[node], 0l));
    ulong current_weight = ulong(max(next_community_weights[current], 0l));
    ulong target_weight = ulong(max(next_community_weights[target], 0l));
    next_community_weights[current] = long(current_weight - node_degree);
    next_community_weights[target] = long(target_weight + node_degree);
}

kernel void ig_louvain_active_clear(
    device atomic_uint* next_active [[buffer(0)]],
    device atomic_uint* count [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (position < args.node_count) atomic_store_explicit(
        next_active + position, 0u, memory_order_relaxed);
    if (position == 0u) atomic_store_explicit(count, 0u, memory_order_relaxed);
}

kernel void ig_louvain_rebuild_active(
    device const uint* active_nodes [[buffer(0)]],
    device const uint* membership [[buffer(1)]],
    device atomic_uint* next_active [[buffer(2)]],
    device atomic_uint* count [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count || active_nodes[node] == 0u) return;
    uint community = membership[node];
    uint previous = atomic_exchange_explicit(
        next_active + community, 1u, memory_order_relaxed);
    if (previous == 0u) atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}

kernel void ig_louvain_map_original(
    device const uint* original_map [[buffer(0)]],
    device const uint* level_membership [[buffer(1)]],
    device uint* next_original_map [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint node [[thread_position_in_grid]]) {
    node += uint(args.work_offset);
    if (ulong(node) >= args.work_offset + args.work_count) return;
    if (node >= args.node_count) return;
    uint current = original_map[node];
    next_original_map[node] = current == IG_LV_INVALID_NODE
        ? IG_LV_INVALID_NODE : level_membership[current];
}

// Map a level's canonical pair through the stable membership.  When a non-self pair collapses,
// both directed halves become one coarse self-loop and its row weight therefore doubles exactly.
kernel void ig_louvain_coarsen_pairs(
    device const long* pair_keys [[buffer(0)]],
    device const long* pair_weights [[buffer(1)]],
    device const uint* membership [[buffer(2)]],
    device long* mapped_keys [[buffer(3)]],
    device long* mapped_weights [[buffer(4)]],
    constant IgLouvainGraphArgs& args [[buffer(5)]],
    uint local [[thread_position_in_grid]]) {
    ulong pair = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count) return;
    if (pair >= args.pair_capacity) return;
    mapped_keys[pair] = ig_lv_encode_key(IG_LV_INVALID_RAW_KEY);
    mapped_weights[pair] = 0;
    long signed_weight = pair_weights[pair];
    if (signed_weight <= 0) return;
    ulong raw = ig_lv_decode_key(pair_keys[pair]);
    if (raw == IG_LV_INVALID_RAW_KEY) return;
    uint left = uint(raw >> 32u);
    uint right = uint(raw);
    uint mapped_left = membership[left];
    uint mapped_right = membership[right];
    if (mapped_left == IG_LV_INVALID_NODE || mapped_right == IG_LV_INVALID_NODE) return;
    uint low = min(mapped_left, mapped_right);
    uint high = max(mapped_left, mapped_right);
    ulong weight = ulong(signed_weight);
    if (left != right && mapped_left == mapped_right) weight *= 2ul;
    mapped_keys[pair] = ig_lv_encode_key(ig_lv_pair_key(low, high));
    mapped_weights[pair] = long(weight);
}

kernel void ig_louvain_canonical_first_clear(
    device atomic_uint* first_position [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong node_index = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || node_index >= ulong(args.node_count)) return;
    uint node = uint(node_index);
    atomic_store_explicit(first_position + node, IG_LV_INVALID_NODE, memory_order_relaxed);
    if (node == 0u) atomic_store_explicit(status, 0u, memory_order_relaxed);
}

kernel void ig_louvain_canonical_first_scatter(
    device const uint* labels [[buffer(0)]],
    device const uint* visible_rows [[buffer(1)]],
    device atomic_uint* first_position [[buffer(2)]],
    device atomic_uint* status [[buffer(3)]],
    constant IgLouvainGraphArgs& args [[buffer(4)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.pair_capacity) return;
    uint visible = visible_rows[row];
    if (visible >= args.node_count) {
        atomic_fetch_max_explicit(status, 1u, memory_order_relaxed);
        return;
    }
    uint label = labels[visible];
    if (label >= args.node_count || row > ulong(IG_LV_INVALID_NODE - 1u)) {
        atomic_fetch_max_explicit(status, 2u, memory_order_relaxed);
        return;
    }
    atomic_fetch_min_explicit(first_position + label, uint(row), memory_order_relaxed);
}

kernel void ig_louvain_canonical_keys(
    device const uint* first_position [[buffer(0)]],
    device long* keys [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong node_index = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || node_index >= ulong(args.node_count)) return;
    uint node = uint(node_index);
    uint first = first_position[node];
    keys[node] = first == IG_LV_INVALID_NODE
        ? ig_lv_encode_key(IG_LV_INVALID_RAW_KEY)
        : ig_lv_encode_key(ig_lv_pair_key(first, node));
}

kernel void ig_louvain_canonical_publish_clear(
    device uint* canonical_by_label [[buffer(0)]],
    constant IgLouvainGraphArgs& args [[buffer(1)]],
    uint local [[thread_position_in_grid]]) {
    ulong node_index = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || node_index >= ulong(args.node_count)) return;
    canonical_by_label[node_index] = IG_LV_INVALID_NODE;
}

kernel void ig_louvain_canonical_publish(
    device const long* sorted_keys [[buffer(0)]],
    device uint* canonical_by_label [[buffer(1)]],
    constant IgLouvainGraphArgs& args [[buffer(2)]],
    uint local [[thread_position_in_grid]]) {
    ulong position = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || position >= ulong(args.node_count)) return;
    ulong raw = ig_lv_decode_key(sorted_keys[position]);
    if (raw == IG_LV_INVALID_RAW_KEY) return;
    uint label = uint(raw);
    canonical_by_label[label] = uint(position);
}

kernel void ig_louvain_csr_offsets(
    device const long* sorted_oriented_keys [[buffer(0)]],
    device const long* sorted_oriented_weights [[buffer(1)]],
    device uint* offsets [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong node = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || node > ulong(args.node_count)) return;
    if (node == ulong(args.node_count)) {
        offsets[node] = uint(args.total_weight);
        return;
    }
    ulong begin = ig_lv_lower_bound_high(
        sorted_oriented_keys, args.oriented_capacity, uint(node));
    // The unique base adjacency is packed before terminal sentinels by the stable radix sort.
    // Clamp insertion points for trailing isolated nodes to the exact live reciprocal-row count.
    offsets[node] = uint(min(begin, args.total_weight));
    (void)sorted_oriented_weights;
}

kernel void ig_louvain_csr_neighbors(
    device const long* sorted_oriented_keys [[buffer(0)]],
    device const long* sorted_oriented_weights [[buffer(1)]],
    device uint* neighbors [[buffer(2)]],
    constant IgLouvainGraphArgs& args [[buffer(3)]],
    uint local [[thread_position_in_grid]]) {
    ulong row = args.work_offset + ulong(local);
    if (ulong(local) >= args.work_count || row >= args.total_weight) return;
    ulong raw = ig_lv_decode_key(sorted_oriented_keys[row]);
    neighbors[row] = raw == IG_LV_INVALID_RAW_KEY || sorted_oriented_weights[row] <= 0l
        ? IG_LV_INVALID_NODE : uint(raw);
}
