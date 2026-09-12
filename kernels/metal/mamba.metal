#include <metal_stdlib>
using namespace metal;

kernel void ig_mamba_causal_depthwise_conv_f32(
    device const float* packed [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong* args [[buffer(2)]],
    uint channel [[thread_position_in_grid]]) {
    const ulong sequence = args[0];
    const ulong width = args[1];
    const ulong kernel_width = args[2];
    if (channel >= width) return;

    const ulong x_offset = args[3];
    const ulong state_offset = args[4];
    const ulong weight_offset = args[5];
    const ulong bias_offset = args[6];
    const ulong next_state_elements = width * kernel_width;

    for (ulong position = 0; position < sequence; ++position) {
        float sum = packed[bias_offset + channel];
        for (ulong tap = 0; tap < kernel_width; ++tap) {
            const ulong stream_position = position + tap;
            const float value = stream_position < kernel_width - 1
                ? packed[state_offset + channel * kernel_width + stream_position + 1]
                : packed[x_offset + (stream_position - (kernel_width - 1)) * width + channel];
            sum = fma(
                value,
                packed[weight_offset + channel * kernel_width + tap],
                sum);
        }
        output[next_state_elements + position * width + channel] =
            sum / (1.0f + exp(-sum));
    }

    // The recurrent convolution state is exactly the last kernel_width items from the previous
    // state followed by this bounded input chunk.
    for (ulong item = 0; item < kernel_width; ++item) {
        const ulong stream_position = sequence + item;
        output[channel * kernel_width + item] = stream_position < kernel_width
            ? packed[state_offset + channel * kernel_width + stream_position]
            : packed[x_offset + (stream_position - kernel_width) * width + channel];
    }
}

kernel void ig_mamba_selective_scan_f32(
    device const float* packed [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ulong* args [[buffer(2)]],
    uint coordinate [[thread_position_in_grid]]) {
    const ulong sequence = args[0];
    const ulong heads = args[1];
    const ulong head_dim = args[2];
    const ulong state_size = args[3];
    const ulong groups = args[4];
    const ulong inner = heads * head_dim;
    if (coordinate >= inner) return;

    const ulong x_offset = args[5];
    const ulong b_offset = args[6];
    const ulong c_offset = args[7];
    const ulong dt_offset = args[8];
    const ulong gate_offset = args[9];
    const ulong state_offset = args[10];
    const ulong a_offset = args[11];
    const ulong d_offset = args[12];
    const ulong head = coordinate / head_dim;
    const ulong group = head / (heads / groups);
    const ulong state_base = ulong(coordinate) * state_size;
    const ulong gated_offset = inner * state_size;
    const float a = packed[a_offset + head];
    const float skip = packed[d_offset + head];

    for (ulong position = 0; position < sequence; ++position) {
        const float step = packed[dt_offset + position * heads + head];
        const float decay = exp(step * a);
        const float value = packed[x_offset + position * inner + coordinate];
        float sum = 0.0f;
        const ulong group_state = (position * groups + group) * state_size;
        for (ulong item = 0; item < state_size; ++item) {
            const ulong index = state_base + item;
            const float prior = position == 0
                ? packed[state_offset + index]
                : output[index];
            const float next = fma(
                step * value,
                packed[b_offset + group_state + item],
                prior * decay);
            output[index] = next;
            sum = fma(next, packed[c_offset + group_state + item], sum);
        }
        const float gate = packed[gate_offset + position * inner + coordinate];
        const float silu_gate = gate / (1.0f + exp(-gate));
        output[gated_offset + position * inner + coordinate] =
            fma(skip, value, sum) * silu_gate;
    }
}
