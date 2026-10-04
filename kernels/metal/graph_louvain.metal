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
    // Small exact weights need no wide multiplication: both signed score terms
    // and their sum fit in i64 under this checked bound. Larger graphs keep u128.
    if (total_weight <= 0x7ffffffful && link_a <= total_weight && link_b <= total_weight
        && community_a <= total_weight && community_b <= total_weight && degree <= total_weight) {
        long score = (long(link_a) - long(link_b)) * long(total_weight)
            + (long(community_b) - long(community_a)) * long(degree);
        return score < 0l ? -1 : (score > 0l ? 1 : 0);
    }
    IgLvWide left = ig_lv_add_wide(
        ig_lv_multiply_wide(link_a, total_weight),
        ig_lv_multiply_wide(degree, community_b));
    IgLvWide right = ig_lv_add_wide(
        ig_lv_multiply_wide(link_b, total_weight),
        ig_lv_multiply_wide(degree, community_a));
    return ig_lv_compare_wide(left, right);
}

// One node commits at a time in the same dense order as the CPU oracle. Each dispatch consumes
// only a fixed number of node/edge steps; even a single high-degree row can yield for cancellation.
// State: community weights[N], temporary candidate weights[N], control[16].
kernel void ig_louvain_sequential_chunk(
    device ulong* state [[buffer(0)]],
    device uint* membership [[buffer(1)]],
    device const uint* active [[buffer(2)]],
    device const long* degrees [[buffer(3)]],
    device const long* keys [[buffer(4)]],
    device const long* weights [[buffer(5)]],
    constant IgLouvainGraphArgs& args [[buffer(6)]],
    device ulong* feedback [[buffer(7)]],
    device const ulong* offsets [[buffer(8)]],
    device const uint* active_rows [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    device ulong* links = state + args.node_count;
    device ulong* saved_control = links + args.node_count;
    ulong ctl[16];
    // One lane owns this bounded chunk. Keep progress local between dependent moves,
    // then publish it once for the host checkpoint and the next chunk.
    for (uint word = 0u; word < 16u; ++word) {
        ctl[word] = args.reduce_mode != 0u ? 0ul : saved_control[word];
    }
    for (ulong step = 0ul; step < args.work_count && ctl[10] == 0ul && ctl[11] == 0ul; ++step) {
        if (ctl[0] >= args.work_offset) { ctl[10] = 1ul; break; }
        uint node = active_rows[ctl[0]];
        if (node >= args.node_count) { ctl[11] = 6ul; break; }
        ulong degree = ulong(max(degrees[node], 0l));
        if (ctl[1] == 0ul) {
            if (active[node] == 0u || degree == 0ul) { ++ctl[0]; continue; }
            uint current = membership[node];
            if (current >= args.node_count || state[current] < degree) { ctl[11] = 1ul; break; }
            ctl[2] = offsets[node];
            ctl[12] = ctl[2];
            ctl[3] = offsets[node + 1u];
            ctl[4] = current;
            ctl[5] = current;
            ctl[6] = 0ul;
            ctl[7] = state[current] - degree;
            ctl[1] = 1ul;
            continue;
        }
        if (ctl[2] < ctl[3]) {
            ulong cursor = ctl[2]++;
            uint neighbor = uint(ig_lv_decode_key(keys[cursor]));
            if (neighbor >= args.node_count) { ctl[11] = 2ul; break; }
            if (neighbor == node) continue;
            uint candidate = membership[neighbor];
            if (candidate >= args.node_count) { ctl[11] = 3ul; break; }
            if (ctl[1] == 1ul) {
                ulong weight = ulong(max(weights[cursor], 0l));
                if (links[candidate] > ~0ul - weight) { ctl[11] = 4ul; break; }
                links[candidate] += weight;
            } else if (ctl[1] == 2ul) {
                ulong candidate_weight = state[candidate];
                if (candidate == uint(ctl[4])) candidate_weight -= degree;
                int ordering = ig_lv_compare_gain(links[candidate], candidate_weight,
                    ctl[6], ctl[7], degree, args.total_weight);
                if (ordering > 0 || (ordering == 0 && candidate < uint(ctl[5]))) {
                    ctl[5] = candidate;
                    ctl[6] = links[candidate];
                    ctl[7] = candidate_weight;
                }
            } else {
                links[candidate] = 0ul;
            }
            continue;
        }
        if (ctl[1] < 3ul) {
            if (ctl[1] == 1ul) ctl[8] = links[uint(ctl[4])];
            ++ctl[1];
            ctl[2] = ctl[12];
            continue;
        }
        uint current = uint(ctl[4]);
        uint best = uint(ctl[5]);
        if (best != current && ig_lv_compare_gain(ctl[6], ctl[7], ctl[8],
                state[current] - degree, degree, args.total_weight) > 0) {
            if (state[best] > ~0ul - degree) { ctl[11] = 5ul; break; }
            state[current] -= degree;
            state[best] += degree;
            membership[node] = best;
            ++ctl[9];
        }
        ctl[1] = 0ul;
        ++ctl[0];
    }
    for (uint word = 0u; word < 16u; ++word) {
        saved_control[word] = ctl[word];
        feedback[word] = ctl[word];
    }
}

inline ulong ig_lv_shuffle_u64(ulong value, uint lane) {
    return ulong(simd_shuffle(uint(value), lane))
        | (ulong(simd_shuffle(uint(value >> 32u), lane)) << 32u);
}

// Small-row levels retain the same dependent node order. One SIMD group only
// parallelizes the current node's neighborhood reads and exact candidate scores.
kernel void ig_louvain_simd_chunk(
    device ulong* state [[buffer(0)]],
    device uint* membership [[buffer(1)]],
    device const uint* active [[buffer(2)]],
    device const long* degrees [[buffer(3)]],
    device const long* keys [[buffer(4)]],
    device const long* weights [[buffer(5)]],
    constant IgLouvainGraphArgs& args [[buffer(6)]],
    device ulong* feedback [[buffer(7)]],
    device const ulong* offsets [[buffer(8)]],
    device const uint* active_rows [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    device ulong* saved = state + args.node_count * 2ul;
    ulong position = (args.reduce_mode & 1u) != 0u ? 0ul : saved[0];
    ulong accepted = (args.reduce_mode & 1u) != 0u ? 0ul : saved[9];
    uint status = width == 32u ? 0u : 6u;
    for (ulong work = 0ul; work < args.work_count && position < args.work_offset && status == 0u;) {
        uint node = active_rows[position];
        if (node >= args.node_count) { status = 6u; break; }
        ulong begin = offsets[node];
        ulong end = offsets[node + 1ul];
        ulong degree = ulong(max(degrees[node], 0l));
        if (active[node] == 0u || degree == 0ul) { ++position; ++work; continue; }
        if (end - begin > 32ul) { status = 7u; break; }
        uint current = membership[node];
        if (current >= args.node_count || state[current] < degree) { status = 1u; break; }
        uint candidate = args.node_count;
        ulong weight = 0ul;
        if (begin + lane < end) {
            uint neighbor = uint(ig_lv_decode_key(keys[begin + lane]));
            if (neighbor >= args.node_count) status = 2u;
            else if (neighbor != uint(node)) {
                candidate = membership[neighbor];
                weight = ulong(max(weights[begin + lane], 0l));
                if (candidate >= args.node_count) status = 3u;
            }
        }
        status = simd_max(status);
        if (status != 0u) break;
        ulong link = 0ul;
        ulong current_link = 0ul;
        uint neighbor_count = uint(end - begin);
        for (uint neighbor_lane = 0u; neighbor_lane < neighbor_count; ++neighbor_lane) {
            uint neighbor_community = simd_shuffle(candidate, neighbor_lane);
            ulong neighbor_weight = ig_lv_shuffle_u64(weight, neighbor_lane);
            if (neighbor_community == current) current_link += neighbor_weight;
            if (neighbor_community == candidate) link += neighbor_weight;
        }
        uint best = current;
        ulong best_link = 0ul;
        ulong best_weight = state[current] - degree;
        if (candidate < args.node_count) {
            ulong candidate_weight = state[candidate] - (candidate == current ? degree : 0ul);
            int ordering = ig_lv_compare_gain(link, candidate_weight,
                best_link, best_weight, degree, args.total_weight);
            if (ordering > 0 || (ordering == 0 && candidate < best)) {
                best = candidate; best_link = link; best_weight = candidate_weight;
            }
        }
        uint first_stride = 1u;
        while (first_stride < neighbor_count) first_stride <<= 1u;
        for (uint stride = first_stride >> 1u; stride != 0u; stride >>= 1u) {
            uint other_lane = min(lane + stride, 31u);
            uint other = simd_shuffle(best, other_lane);
            ulong other_link = ig_lv_shuffle_u64(best_link, other_lane);
            ulong other_weight = ig_lv_shuffle_u64(best_weight, other_lane);
            int ordering = ig_lv_compare_gain(other_link, other_weight,
                best_link, best_weight, degree, args.total_weight);
            if (lane + stride < neighbor_count && (ordering > 0 || (ordering == 0 && other < best))) {
                best = other; best_link = other_link; best_weight = other_weight;
            }
        }
        best = simd_broadcast_first(best);
        best_link = ig_lv_shuffle_u64(best_link, 0u);
        best_weight = ig_lv_shuffle_u64(best_weight, 0u);
        if (best != current && ig_lv_compare_gain(best_link, best_weight,
                current_link, state[current] - degree, degree, args.total_weight) > 0) {
            if (state[best] > ~0ul - degree) { status = 5u; break; }
            if (lane == 0u) {
                state[current] -= degree;
                state[best] += degree;
                membership[node] = best;
            }
            ++accepted;
        }
        simdgroup_barrier(mem_flags::mem_device);
        ++position;
        work += (end - begin) * 3ul + 5ul;
    }
    if (lane == 0u) {
        for (uint word = 0u; word < 16u; ++word) {
            ulong value = word == 0u ? position : (word == 9u ? accepted
                : (word == 10u ? ulong(position >= args.work_offset) : (word == 11u ? status : 0ul)));
            saved[word] = value;
            feedback[word] = value;
        }
    }
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
    if (args.reduce_mode == 1u) {
        reinterpret_cast<device ulong*>(offsets)[node] = ig_lv_lower_bound_high(
            sorted_oriented_keys, args.oriented_capacity, uint(node));
        return;
    }
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
