#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdint.h>

#include "blaze/nvfp4.h"

namespace {

__global__ void controller_up(const __nv_bfloat16* hidden, const __nv_bfloat16* w0,
                              const __nv_bfloat16* w1, const __nv_bfloat16* w2, float* out, int rows) {
    const int group = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / 8) + threadIdx.x / 8;
    const int lane = threadIdx.x % 8;
    if (row >= rows) return;
    const auto* w = (group == 0 ? w0 : group == 1 ? w1 : w2) + row * 64 + lane;
    const int projection = blockIdx.z * gridDim.y + group;
    const auto* x = hidden + projection * 64 + lane;
    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int k = i * 8;
        acc[i] = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), 0.0f);
        acc[i] = fmaf(__bfloat162float(w[k+32]), __bfloat162float(x[k+32]), acc[i]);
    }
    // Match the four AVX accumulators and their ordered horizontal sum.
    const float partial = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    const unsigned mask = __activemask();
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += __shfl_sync(mask, partial, i, 8);
    if (lane == 0) out[projection * rows + row] = sum;
}

__device__ float ordered_sum(float acc) {
    acc += __shfl_down_sync(0xffffffff, acc, 8);
    acc += __shfl_down_sync(0xffffffff, acc, 16);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) sum += __shfl_sync(0xffffffff, acc, i);
    return sum;
}

__global__ void router_logits(const __nv_bfloat16* w, const __nv_bfloat16* x, float* out, int cols) {
    const int lane = threadIdx.x;
    w += size_t(blockIdx.x) * cols;
    x += size_t(blockIdx.y) * cols;
    float acc = 0.0f;
    for (int k = lane; k < cols; k += 32) {
        acc = fmaf(__bfloat162float(w[k]), __bfloat162float(x[k]), acc);
    }
    const float sum = ordered_sum(acc);
    if (lane == 0) out[size_t(blockIdx.y) * gridDim.x + blockIdx.x] = sum;
}

__device__ float controller_tanh(float value) {
    const float x = fmaxf(-80.0f, -2.0f * fabsf(value));
    const int n = __float2int_rn(x * 1.4426950408889634f);
    const float r = fmaf(-float(n), 0.6931471805599453f, x);
    float p = fmaf(1.0f / 720.0f, r, 1.0f / 120.0f);
    p = fmaf(p, r, 1.0f / 24.0f);
    p = fmaf(p, r, 1.0f / 6.0f);
    p = fmaf(p, r, 0.5f);
    p = fmaf(p, r, 1.0f);
    p = fmaf(p, r, 1.0f);
    const float e = p * __int_as_float((n + 127) << 23);
    return copysignf(__fdiv_rn(1.0f - e, 1.0f + e), value);
}

__global__ void controller_norm(const float* gates, const float* x, const float* msg,
                                const float* norm, __nv_bfloat16* output, int width, float eps, int groups,
                                int projection_stride) {
    extern __shared__ float mixed[];
    const int offset = blockIdx.x * width;
    gates += 2 * offset;
    x += offset;
    msg += offset;
    const int group = blockIdx.x % groups;
    output += projection_stride ? (size_t(group) * projection_stride + blockIdx.x / groups) * width : offset;
    norm += group * width;
    __shared__ float scale;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float keep = __fmul_rn(x[i], 1.0f - controller_tanh(gates[i]));
        const float add = __fmul_rn(msg[i], controller_tanh(gates[width+i]));
        mixed[i] = keep + add;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < width; i += 32) acc = fmaf(mixed[i], mixed[i], acc);
        const float sum = ordered_sum(acc);
        if (threadIdx.x == 0) scale = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(sum, float(width)) + eps));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) output[i] = __float2bfloat16(mixed[i] * scale * norm[i]);
}

__global__ void controller_inputs(const float* x, const float* msg, __nv_bfloat16* out, int width, float eps) {
    const int group = blockIdx.x;
    const int operand = blockIdx.y;
    const float* src = (operand ? msg : x) + group * width;
    __shared__ float rms;
    if (threadIdx.x < 32) {
        float acc = 0.0f;
        for (int i = threadIdx.x; i < width; i += 32) acc = fmaf(src[i], src[i], acc);
        const float sum = ordered_sum(acc);
        if (threadIdx.x == 0) rms = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(sum, float(width)) + eps));
    }
    __syncthreads();
    auto* dst = out + (2 * group + operand) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = __float2bfloat16(src[i] * rms);
}

__global__ void controller_message_mix(const float* bank, const float* mix_weights, float* output, int width) {
    const int row = blockIdx.y, group = blockIdx.x;
    const auto* past = bank + size_t(row) * 2 * width;
    const size_t at = size_t(row) * gridDim.x + group;
    const float w0 = mix_weights[2 * at], w1 = mix_weights[2 * at + 1];
    auto* dst = output + at * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float first = __fadd_rn(0.0f, __fmul_rn(past[i], w0));
        dst[i] = __fadd_rn(first, __fmul_rn(past[width + i], w1));
    }
}

__global__ void norm_rows(const float* input, const float* weight, float* full, __nv_bfloat16* rounded,
                          int width, float eps) {
    input += size_t(blockIdx.x) * width;
    __shared__ float rms;
    if (threadIdx.x < 32) {
        float dot = 0.0f;
        for (int i = threadIdx.x; i < width; i += 32) dot = fmaf(input[i], input[i], dot);
        dot = ordered_sum(dot);
        if (threadIdx.x == 0) rms = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(dot, float(width)) + eps));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float value = input[i] * rms;
        if (weight) value *= weight[i];
        const size_t at = size_t(blockIdx.x) * width + i;
        if (full) full[at] = value;
        else rounded[at] = __float2bfloat16(value);
    }
}

__global__ void expand_bf16(const __nv_bfloat16* input, float* output, size_t values) {
    const size_t at = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at < values) output[at] = __bfloat162float(input[at]);
}

__global__ void pack_features(const float4* tap0, const float4* tap1, const float4* tap2, const float4* tap3,
                              __nv_bfloat162* output, int rows, int vectors_per_row) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * vectors_per_row) return;
    const float4* taps[] = {tap0, tap1, tap2, tap3};
    const int tap = blockIdx.y;
    const int row = i / vectors_per_row;
    const int col = i % vectors_per_row;
    const float4 value = taps[tap][i];
    const int dst = 2 * ((4 * row + tap) * vectors_per_row + col);
    output[dst] = __floats2bfloat162_rn(value.x, value.y);
    output[dst + 1] = __floats2bfloat162_rn(value.z, value.w);
}

__global__ void embed_rows(const __nv_bfloat16* input, const uint8_t* image, const float* weight,
                           float* output, int width, float eps) {
    const size_t row = size_t(blockIdx.x) * width;
    const bool text = image[blockIdx.x] == 0;
    __shared__ float rms;
    if (text) {
        if (threadIdx.x < 32) {
            float dot = 0.0f;
            for (int i = threadIdx.x; i < width; i += 32) {
                const float x = __bfloat162float(input[row + i]);
                dot = fmaf(x, x, dot);
            }
            dot = ordered_sum(dot);
            if (threadIdx.x == 0) rms = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(dot, float(width)) + eps));
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float value = __bfloat162float(input[row + i]);
        output[row + i] = text ? __bfloat162float(__float2bfloat16(value * rms * weight[i])) : value;
    }
}

__global__ void mix_normalized(const float* bank, const float* current, const float* query, float* output,
                               int depth, int rows, int width) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const size_t row = size_t(blockIdx.x) * width, plane = size_t(rows) * width;
    const int count = depth + (current != nullptr);
    const float factor = __frcp_rn(__fsqrt_rn(float(width)));
    __shared__ float probabilities[32];
    for (int source = warp; source < count; source += blockDim.x / 32) {
        const float* src = (source < depth ? bank + size_t(source) * plane : current) + row;
        float dot = 0.0f;
        for (int i = lane; i < width; i += 32) dot = fmaf(src[i], query[i], dot);
        dot = ordered_sum(dot);
        if (lane == 0) probabilities[source] = dot * factor;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float max_logit = -CUDART_INF_F;
        for (int i = 0; i < count; ++i) max_logit = fmaxf(max_logit, probabilities[i]);
        float sum = 0.0f;
        for (int i = 0; i < count; ++i) {
            probabilities[i] = expf(probabilities[i] - max_logit);
            sum += probabilities[i];
        }
        const float inv = __frcp_rn(sum);
        for (int i = 0; i < count; ++i) probabilities[i] *= inv;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float value = 0.0f;
        for (int source = 0; source < count; ++source) {
            const float* src = (source < depth ? bank + size_t(source) * plane : current) + row;
            value = __fadd_rn(value, __fmul_rn(src[i], probabilities[source]));
        }
        output[row + i] = value;
    }
}

__global__ void kv_slots(int* slots, int start, int capacity, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) slots[row] = (start + row) % capacity;
}

} // namespace

extern "C" cudaError_t blz_inkling_norm_rows(const float* input, const float* weight, float* full,
        uint16_t* rounded, int rows, int width, float eps, cudaStream_t stream) {
    norm_rows<<<rows, 128, 0, stream>>>(input, weight, full, reinterpret_cast<__nv_bfloat16*>(rounded), width, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_expand_bf16(const uint16_t* input, float* output, size_t values, cudaStream_t stream) {
    expand_bf16<<<(values + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(input), output, values);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_pack_spec_features(const float* tap0, const float* tap1, const float* tap2,
        const float* tap3, uint16_t* output, int rows, int width, cudaStream_t stream) {
    if (rows <= 0 || width <= 0 || width % 4 != 0) return cudaErrorInvalidValue;
    const int vectors_per_row = width / 4;
    const dim3 grid((rows * vectors_per_row + 255) / 256, 4);
    pack_features<<<grid, 256, 0, stream>>>(
        reinterpret_cast<const float4*>(tap0), reinterpret_cast<const float4*>(tap1),
        reinterpret_cast<const float4*>(tap2), reinterpret_cast<const float4*>(tap3),
        reinterpret_cast<__nv_bfloat162*>(output), rows, vectors_per_row);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_embed_rows(const uint16_t* input, const uint8_t* image, const float* weight,
        float* output, int rows, int width, float eps, cudaStream_t stream) {
    embed_rows<<<rows, 128, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(input), image, weight, output, width, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_mix_normalized_rows(const float* bank, const float* current, const float* query,
        float* output, int depth, int rows, int width, cudaStream_t stream) {
    if (depth < 0 || depth + (current != nullptr) < 1 || depth + (current != nullptr) > 32 || rows < 1 ||
        width < 32 || width % 32) return cudaErrorInvalidValue;
    mix_normalized<<<rows, 128, 0, stream>>>(bank, current, query, output, depth, rows, width);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_kv_slots(int* slots, int start, int capacity, int rows, cudaStream_t stream) {
    kv_slots<<<(rows + 127) / 128, 128, 0, stream>>>(slots, start, capacity, rows);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_controller_norm(const float* gates, const float* x, const float* msg,
        const float* norm, uint16_t* out, int width, float eps, int tokens, cudaStream_t stream) {
    if (width <= 0 || width > 4096 || width % 32) return cudaErrorInvalidValue;
    controller_norm<<<tokens, 256, width * sizeof(float), stream>>>(gates, x, msg, norm,
        reinterpret_cast<__nv_bfloat16*>(out), width, eps, 1, 0);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_controller_qkv(const float* gates, const float* x, const float* msg,
        const float* norms, uint16_t* out, int width, int tokens, int projection_stride, float eps, cudaStream_t stream) {
    if (width <= 0 || width > 4096 || width % 32 || tokens <= 0 || tokens > 65535) return cudaErrorInvalidValue;
    if (projection_stride && projection_stride < tokens) return cudaErrorInvalidValue;
    controller_norm<<<3 * tokens, 256, width * sizeof(float), stream>>>(gates, x, msg, norms,
        reinterpret_cast<__nv_bfloat16*>(out), width, eps, 3, projection_stride);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_controller_inputs(const float* x, const float* msg,
        uint16_t* out, int width, int rows, float eps, cudaStream_t stream) {
    if (width <= 0 || width > 4096 || width % 32 || rows <= 0) return cudaErrorInvalidValue;
    controller_inputs<<<dim3(rows, 2), 128, 0, stream>>>(x, msg, reinterpret_cast<__nv_bfloat16*>(out), width, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_controller_message_mix(const float* bank, const float* queries,
        float* output, int width, int rows, cudaStream_t stream) {
    if (width <= 0 || width > 4096 || width % 32 || rows <= 0 || rows > 65535) return cudaErrorInvalidValue;
    controller_message_mix<<<dim3(3, rows), 128, 0, stream>>>(bank, queries, output, width);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_router_logits(const uint16_t* w, const uint16_t* x, float* out,
        int rows, int cols, int tokens, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || cols % 32 || tokens <= 0 || tokens > 65535) return cudaErrorInvalidValue;
    router_logits<<<dim3(rows, tokens), 32, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<const __nv_bfloat16*>(x), out, cols);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_controller_up(const uint16_t* hidden, const uint16_t* w0,
        const uint16_t* w1, const uint16_t* w2, float* out, int rows, int groups, int tokens, cudaStream_t stream) {
    if (rows <= 0 || (groups != 1 && groups != 3)) return cudaErrorInvalidValue;
    controller_up<<<dim3((rows + 15) / 16, groups, tokens), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden), reinterpret_cast<const __nv_bfloat16*>(w0),
        reinterpret_cast<const __nv_bfloat16*>(w1), reinterpret_cast<const __nv_bfloat16*>(w2), out, rows);
    return cudaGetLastError();
}

namespace {

constexpr int HEAD_DIM = 128;
constexpr int WARPS = 4;

__device__ inline float warp_sum(float x) {
    for (int shift = 16; shift; shift /= 2) x += __shfl_xor_sync(0xffffffff, x, shift);
    return x;
}

template <bool Packed, bool Causal = false>
__global__ void attention_tiles(const __nv_bfloat16* q, const uint8_t* keys, const uint8_t* values,
                                const uint8_t* key_scales, const uint8_t* value_scales,
                                const __nv_bfloat16* relative, const __nv_bfloat16* sinks,
                                const __nv_bfloat16* gate, __nv_bfloat16* output,
                                float* scratch, int kv_heads, int capacity, int window, int position,
                                int extent, int tile, float global_scale, int query_stride) {
    q += size_t(blockIdx.z) * query_stride;
    output += size_t(blockIdx.z) * gridDim.x * HEAD_DIM;
    scratch += size_t(blockIdx.z) * gridDim.x * gridDim.y * (HEAD_DIM + 2);
    int blocks = gridDim.y;
    if constexpr (Causal) {
        position += blockIdx.z;
        relative += size_t(blockIdx.z) * gridDim.x * extent;
        sinks += size_t(blockIdx.z) * gridDim.x;
        gate += size_t(blockIdx.z) * query_stride;
        const int valid = min(window, position + 1);
        const int tiles = min(256, (valid + 63) / 64);
        tile = (valid + tiles - 1) / tiles;
        blocks = (valid + tile - 1) / tile;
        if (blockIdx.y >= blocks) return;
    }
    const int head = blockIdx.x;
    const int kv_head = head / (gridDim.x / kv_heads);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int first = max(0, position + 1 - window) + blockIdx.y * tile;
    const int end = min(first + tile, position + 1);
    float query[4], accum[4] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) query[j] = __bfloat162float(q[head * HEAD_DIM + lane * 4 + j]);

    float max_logit = -CUDART_INF_F;
    float sum = 0.0f;
    int slot = (first + warp) % capacity;
    for (int t = first + warp; t < end; t += WARPS) {
        const size_t row = size_t(slot) * kv_heads + kv_head;
        slot += WARPS;
        if (slot >= capacity) slot -= capacity;
        float dot = 0.0f;
        if constexpr (Packed) {
            constexpr int codes = HEAD_DIM / 2;
            constexpr int scales = HEAD_DIM / blaze::kNvfp4Block;
            const int group = lane / (blaze::kNvfp4Block / 4);
            const float ks = blaze::e4m3_to_float(key_scales[row * scales + group]);
            const uint16_t kb = reinterpret_cast<const uint16_t*>(keys + row * codes)[lane];
#pragma unroll
            for (int j = 0; j < 4; ++j) dot = fmaf(query[j], blaze::e2m1_value((kb >> (j * 4)) & 15), dot);
            dot *= ks;
        } else {
            const auto* k = reinterpret_cast<const __nv_bfloat162*>(keys) + row * (HEAD_DIM / 2) + lane * 2;
            const float2 lo = __bfloat1622float2(k[0]), hi = __bfloat1622float2(k[1]);
            dot = fmaf(query[0], lo.x, dot);
            dot = fmaf(query[1], lo.y, dot);
            dot = fmaf(query[2], hi.x, dot);
            dot = fmaf(query[3], hi.y, dot);
        }
        dot = warp_sum(dot) * ((Packed ? global_scale : 1.0f) * rsqrtf(float(HEAD_DIM)));
        const int distance = position - t;
        if (distance < extent) dot += __bfloat162float(relative[head * extent + distance]);
        const float next_max = fmaxf(max_logit, dot);
        const float keep = __expf(max_logit - next_max);
        const float add = __expf(dot - next_max);
        sum = fmaf(sum, keep, add);
        max_logit = next_max;
        if constexpr (Packed) {
            constexpr int codes = HEAD_DIM / 2;
            constexpr int scales = HEAD_DIM / blaze::kNvfp4Block;
            const int group = lane / (blaze::kNvfp4Block / 4);
            const float vs = blaze::e4m3_to_float(value_scales[row * scales + group]) * (global_scale * add);
            const uint16_t vb = reinterpret_cast<const uint16_t*>(values + row * codes)[lane];
#pragma unroll
            for (int j = 0; j < 4; ++j) accum[j] = fmaf(vs, blaze::e2m1_value((vb >> (j * 4)) & 15), keep * accum[j]);
        } else {
            const auto* v = reinterpret_cast<const __nv_bfloat162*>(values) + row * (HEAD_DIM / 2) + lane * 2;
            const float2 lo = __bfloat1622float2(v[0]), hi = __bfloat1622float2(v[1]);
            accum[0] = fmaf(add, lo.x, keep * accum[0]);
            accum[1] = fmaf(add, lo.y, keep * accum[1]);
            accum[2] = fmaf(add, hi.x, keep * accum[2]);
            accum[3] = fmaf(add, hi.y, keep * accum[3]);
        }
    }

    __shared__ float maxima[WARPS];
    __shared__ float sums[WARPS];
    __shared__ float partials[WARPS][HEAD_DIM];
    if (lane == 0) {
        maxima[warp] = max_logit;
        sums[warp] = sum;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) partials[warp][lane * 4 + j] = accum[j];
    __syncthreads();

    const float maximum = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float out = 0.0f, denom = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float factor = maximum == -CUDART_INF_F ? 0.0f : expf(maxima[w] - maximum);
        out = fmaf(factor, partials[w][threadIdx.x], out);
        denom = fmaf(factor, sums[w], denom);
    }
    if (blocks == 1) {
        const float sink = __bfloat162float(sinks[head]);
        const float m = fmaxf(maximum, sink);
        const float factor = __expf(maximum - m);
        const int i = head * HEAD_DIM + threadIdx.x;
        output[i] = __float2bfloat16(out * factor / (fmaf(denom, factor, __expf(sink - m))
                                      * (1.0f + __expf(-__bfloat162float(gate[i])))));
        return;
    }
    float* partial = scratch + (head * gridDim.y + blockIdx.y) * (HEAD_DIM + 2);
    partial[threadIdx.x] = out;
    if (threadIdx.x == 0) {
        partial[HEAD_DIM] = maximum;
        partial[HEAD_DIM + 1] = denom;
    }
}

template <bool Causal = false, bool PerQuery = Causal>
__global__ void attention_merge(const float* scratch, const __nv_bfloat16* sinks,
                                const __nv_bfloat16* gate, __nv_bfloat16* out, int stride,
                                int position = 0, int window = 0) {
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    scratch += size_t(blockIdx.z) * gridDim.x * stride * (HEAD_DIM + 2);
    out += size_t(blockIdx.z) * gridDim.x * HEAD_DIM;
    int tiles = stride;
    if constexpr (Causal) {
        const int valid = min(window, position + int(blockIdx.z) + 1);
        const int n = min(256, (valid + 63) / 64);
        const int tile = (valid + n - 1) / n;
        tiles = (valid + tile - 1) / tile;
        if (tiles == 1) return;
    }
    if constexpr (PerQuery) {
        sinks += size_t(blockIdx.z) * gridDim.x;
        gate += size_t(blockIdx.z) * gridDim.x * HEAD_DIM;
    }
    const float* row = scratch + head * stride * (HEAD_DIM + 2);
    const float sink = __bfloat162float(sinks[head]);
    float maximum = sink;
    for (int t = 0; t < tiles; ++t) maximum = fmaxf(maximum, row[t * (HEAD_DIM + 2) + HEAD_DIM]);
    float denom = expf(sink - maximum);
    float value = 0.0f;
    for (int t = 0; t < tiles; ++t) {
        const float* slot = row + t * (HEAD_DIM + 2);
        const float factor = expf(slot[HEAD_DIM] - maximum);
        denom = fmaf(factor, slot[HEAD_DIM + 1], denom);
        value = fmaf(factor, slot[col], value);
    }
    const int i = head * HEAD_DIM + col;
    out[i] = __float2bfloat16(value / (denom * (1.0f + expf(-__bfloat162float(gate[i])))));
}

template<bool Packed>
__device__ inline float prefill_kv_element(const uint8_t* data, const uint8_t* scales, size_t row, int column, float scale) {
    if constexpr (Packed) {
        const auto code = data[row * (HEAD_DIM / 2) + column / 2];
        const auto step = scales[row * (HEAD_DIM / 16) + column / 16];
        return blaze::e2m1_value((code >> (4 * (column % 2))) & 15) * blaze::e4m3_to_float(step) * scale;
    }
    return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(data)[row * HEAD_DIM + column]);
}

constexpr int kPromptQueries = 16;
constexpr int kPromptKeys = 32;

template<bool Packed>
__global__ void inkling_tensor_attention(const __nv_bfloat16* q, const uint8_t* keys, const uint8_t* values,
        const uint8_t* ks, const uint8_t* vs, const __nv_bfloat16* relative, float* scratch, int kv_heads,
        int capacity, int window, int start, int extent, int span, float scale) {
    namespace wmma = nvcuda::wmma;
    const int head = blockIdx.x;
    const int kv_head = head / (gridDim.x / kv_heads);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int first = max(0, start + 1 - window) + blockIdx.y * span;
    const int end = min(first + span, start + kPromptQueries);
    __align__(32) __shared__ __half q_tile[kPromptQueries][HEAD_DIM];
    __align__(32) __shared__ __half k_tile[kPromptKeys][HEAD_DIM];
    __align__(32) __shared__ __half v_tile[kPromptKeys][HEAD_DIM];
    __align__(32) __shared__ float score[kPromptQueries][kPromptKeys];
    __align__(32) __shared__ __half probability[kPromptQueries][kPromptKeys];
    __align__(32) __shared__ float numerator[kPromptQueries][HEAD_DIM];
    __align__(32) __shared__ float product[kPromptQueries][HEAD_DIM];
    __shared__ float maximum[kPromptQueries], mass[kPromptQueries], keep[kPromptQueries];

    for (int i = threadIdx.x; i < kPromptQueries * HEAD_DIM; i += blockDim.x) {
        const int row = i / HEAD_DIM;
        const int col = i % HEAD_DIM;
        q_tile[row][col] = __float2half(__bfloat162float(q[(size_t(row) * gridDim.x + head) * HEAD_DIM + col]));
        numerator[row][col] = 0.0f;
    }
    if (threadIdx.x < kPromptQueries) {
        maximum[threadIdx.x] = -CUDART_INF_F;
        mass[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int base = first; base < end; base += kPromptKeys) {
        for (int i = threadIdx.x; i < kPromptKeys * HEAD_DIM; i += blockDim.x) {
            const int pos = base + i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            const size_t kv_row = size_t(pos % capacity) * kv_heads + kv_head;
            k_tile[i / HEAD_DIM][col] = __float2half(pos < end ? prefill_kv_element<Packed>(keys, ks, kv_row, col, scale) : 0.0f);
            v_tile[i / HEAD_DIM][col] = __float2half(pos < end ? prefill_kv_element<Packed>(values, vs, kv_row, col, scale) : 0.0f);
        }
        __syncthreads();
        if (warp < kPromptKeys / 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot;
            wmma::fill_fragment(dot, 0.0f);
            for (int k = 0; k < HEAD_DIM; k += 16) {
                wmma::load_matrix_sync(q_fragment, &q_tile[0][k], HEAD_DIM);
                wmma::load_matrix_sync(k_fragment, &k_tile[warp * 16][k], HEAD_DIM);
                wmma::mma_sync(dot, q_fragment, k_fragment, dot);
            }
            wmma::store_matrix_sync(&score[0][warp * 16], dot, kPromptKeys, wmma::mem_row_major);
        }
        __syncthreads();
        for (int row = warp; row < kPromptQueries; row += WARPS) {
            const int distance = start + row - (base + lane);
            const bool visible = base + lane < end && distance >= 0 && distance < window;
            const float bias = visible && distance < extent
                ? __bfloat162float(relative[(size_t(row) * gridDim.x + head) * extent + distance]) : 0.0f;
            const float logit = visible ? score[row][lane] * rsqrtf(float(HEAD_DIM)) + bias : -CUDART_INF_F;
            float peak = logit;
            for (int shift = 16; shift; shift /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, shift));
            const float next = fmaxf(maximum[row], peak);
            const float carry = next == -CUDART_INF_F ? 0.0f : __expf(maximum[row] - next);
            const float p = visible ? __expf(logit - next) : 0.0f;
            const float sum = warp_sum(p);
            probability[row][lane] = __float2half(p);
            if (lane == 0) {
                mass[row] = fmaf(mass[row], carry, sum);
                maximum[row] = next;
                keep[row] = carry;
            }
        }
        __syncthreads();
        for (int col = warp * 16; col < HEAD_DIM; col += WARPS * 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
            wmma::fill_fragment(accum, 0.0f);
            for (int k = 0; k < kPromptKeys; k += 16) {
                wmma::load_matrix_sync(p_fragment, &probability[0][k], kPromptKeys);
                wmma::load_matrix_sync(v_fragment, &v_tile[k][col], HEAD_DIM);
                wmma::mma_sync(accum, p_fragment, v_fragment, accum);
            }
            wmma::store_matrix_sync(&product[0][col], accum, HEAD_DIM, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kPromptQueries * HEAD_DIM; i += blockDim.x) {
            const int row = i / HEAD_DIM;
            const int col = i % HEAD_DIM;
            numerator[row][col] = fmaf(numerator[row][col], keep[row], product[row][col]);
        }
        __syncthreads();
    }
    for (int row = 0; row < kPromptQueries; ++row) {
        auto* part = scratch + ((size_t(row) * gridDim.x + head) * gridDim.y + blockIdx.y) * (HEAD_DIM + 2);
        part[threadIdx.x] = numerator[row][threadIdx.x];
        if (threadIdx.x == 0) {
            part[HEAD_DIM] = maximum[row];
            part[HEAD_DIM + 1] = mass[row];
        }
    }
}

} // namespace

static cudaError_t launch_inkling_attention(const uint16_t* q, const uint8_t* keys, const uint8_t* values,
                                            const uint8_t* key_scales, const uint8_t* value_scales,
                                            const uint16_t* relative, const uint16_t* sinks, const uint16_t* gate,
                                            uint16_t* out, float* scratch, int heads, int kv_heads, int capacity,
                                            int window, int position, int extent, int tiles, float global_scale,
                                            int queries, int query_stride, cudaStream_t stream) {
    if (kv_heads < 1 || heads < 1 || heads % kv_heads != 0 || window < 1 || window > capacity || position < 0 || tiles < 1)
        return cudaErrorInvalidValue;
    const int valid = min(window, position + 1);
    const int tile = (valid + tiles - 1) / tiles;
    const int blocks = (valid + tile - 1) / tile;
    if (queries < 1 || (queries > 1 && query_stride < heads * HEAD_DIM)) return cudaErrorInvalidValue;
    const dim3 grid(heads, blocks, queries);
    auto query = reinterpret_cast<const __nv_bfloat16*>(q);
    auto biases = reinterpret_cast<const __nv_bfloat16*>(relative);
    auto sink = reinterpret_cast<const __nv_bfloat16*>(sinks);
    auto gated = reinterpret_cast<const __nv_bfloat16*>(gate);
    auto result = reinterpret_cast<__nv_bfloat16*>(out);
    if (key_scales) {
        attention_tiles<true><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, key_scales, value_scales,
            biases, sink, gated, result, scratch, kv_heads, capacity, window, position, extent, tile, global_scale, query_stride);
    } else {
        attention_tiles<false><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, nullptr, nullptr,
            biases, sink, gated, result, scratch, kv_heads, capacity, window, position, extent, tile, global_scale, query_stride);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess || blocks == 1) return err;
    attention_merge<><<<dim3(heads, 1, queries), HEAD_DIM, 0, stream>>>(scratch, sink, gated, result, blocks);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_attention(const uint16_t* q, const uint8_t* keys, const uint8_t* values,
        const uint8_t* ks, const uint8_t* vs, const uint16_t* relative, const uint16_t* sinks, const uint16_t* gate,
        uint16_t* out, float* scratch, int heads, int kv_heads, int capacity, int window, int position, int extent,
        int tiles, float scale, cudaStream_t stream) {
    return launch_inkling_attention(q, keys, values, ks, vs, relative, sinks, gate, out, scratch, heads, kv_heads,
                                    capacity, window, position, extent, tiles, scale, 1, 0, stream);
}

extern "C" cudaError_t blz_dspark_attention(const uint16_t* q, int stride, int rows, const uint8_t* keys,
        const uint8_t* values, const uint8_t* ks, const uint8_t* vs, const uint16_t* sinks, const uint16_t* gate,
        uint16_t* out, float* scratch, int heads, int kv_heads, int capacity, int position, int tiles, float scale,
        cudaStream_t stream) {
    return launch_inkling_attention(q, keys, values, ks, vs, nullptr, sinks, gate, out, scratch, heads, kv_heads,
                                    capacity, capacity, position, 0, tiles, scale, rows, stride, stream);
}

extern "C" cudaError_t blz_inkling_prefill_attention(
        const uint16_t* q, const uint8_t* keys, const uint8_t* values, const uint8_t* ks, const uint8_t* vs,
        const uint16_t* relative, const uint16_t* sinks, const uint16_t* gate, uint16_t* out, float* scratch,
        int heads, int kv_heads, int capacity, int window, int start, int extent, int rows, float scale, int tensor,
        cudaStream_t stream) {
    if (heads < 1 || kv_heads < 1 || heads % kv_heads || window < 1 || window > capacity || start < 0 || rows < 1)
        return cudaErrorInvalidValue;
    const int tiles = min(256, (min(window, start + rows) + 63) / 64);
    const dim3 grid(heads, tiles, rows);
    const auto* query = reinterpret_cast<const __nv_bfloat16*>(q);
    const auto* biases = reinterpret_cast<const __nv_bfloat16*>(relative);
    const auto* sink = reinterpret_cast<const __nv_bfloat16*>(sinks);
    const auto* gated = reinterpret_cast<const __nv_bfloat16*>(gate);
    auto* result = reinterpret_cast<__nv_bfloat16*>(out);
    const int valid = start + rows - max(0, start + 1 - window);
    if (tensor && rows == kPromptQueries && valid >= 128) {
        const int span = max(128, ((valid + 256 * kPromptKeys - 1) / (256 * kPromptKeys)) * kPromptKeys);
        const int splits = (valid + span - 1) / span;
        const dim3 tensor_grid(heads, splits);
        if (ks) {
            inkling_tensor_attention<true><<<tensor_grid, HEAD_DIM, 0, stream>>>(query, keys, values, ks, vs,
                biases, scratch, kv_heads, capacity, window, start, extent, span, scale);
        } else {
            inkling_tensor_attention<false><<<tensor_grid, HEAD_DIM, 0, stream>>>(query, keys, values, nullptr,
                nullptr, biases, scratch, kv_heads, capacity, window, start, extent, span, scale);
        }
        const auto err = cudaGetLastError();
        if (err != cudaSuccess) return err;
        attention_merge<false, true><<<dim3(heads, 1, rows), HEAD_DIM, 0, stream>>>(scratch, sink, gated, result, splits);
        return cudaGetLastError();
    }
    if (ks) {
        attention_tiles<true, true><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, ks, vs, biases, sink,
            gated, result, scratch, kv_heads, capacity, window, start, extent, 0, scale, heads * HEAD_DIM);
    } else {
        attention_tiles<false, true><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, nullptr, nullptr, biases,
            sink, gated, result, scratch, kv_heads, capacity, window, start, extent, 0, scale, heads * HEAD_DIM);
    }
    const auto err = cudaGetLastError();
    if (err != cudaSuccess || tiles == 1) return err;
    attention_merge<true><<<dim3(heads, 1, rows), HEAD_DIM, 0, stream>>>(scratch, sink, gated, result, tiles, start, window);
    return cudaGetLastError();
}

namespace {

__device__ void append_history(__nv_bfloat16* history, int rows, int taps, int width, int col, __nv_bfloat16 value) {
    if (taps == 1) return;
    const int first = rows == taps - 1;
    for (int i = first; i < rows; ++i) history[(i - first) * width + col] = history[i * width + col];
    history[(rows - first) * width + col] = value;
}

template<bool Key, bool Decode = false>
__global__ void inkling_prefill_conv(const __nv_bfloat16* raw, const __nv_bfloat16* dynamic,
        const float* weights, __nv_bfloat16* out, int heads, int taps, int history,
        float scale, float clip, float eps, __nv_bfloat16* past = nullptr) {
    const int lane = threadIdx.x;
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    const int width = heads * HEAD_DIM;
    const int offset = head * HEAD_DIM;
    const auto* current = raw + (Decode ? 0 : (history + row) * width) + offset;
    float values[HEAD_DIM / 8];
    float partial[4] = {};
    for (int i = 0; i < HEAD_DIM / 8; ++i) {
        const int col = i * 8 + lane;
        float v = __bfloat162float(current[col]);
        const int last = min(taps, history + row + 1);
        for (int lag = 0; lag < last; ++lag) {
            float w = weights[lag * width + offset + col];
            if (dynamic) w += __bfloat162float(dynamic[(row * heads + head) * taps + lag]);
            const auto x = Decode && lag > 0 ? past[(history - lag) * width + offset + col] : current[col - lag * width];
            v = __fadd_rn(v, __fmul_rn(__bfloat162float(x), w));
        }
        v = __bfloat162float(__float2bfloat16(v * scale));
        values[i] = v;
        partial[i % 4] = fmaf(v, v, partial[i % 4]);
    }
    // Match the eight-lane FP32 CPU reduction before rounding the KV inputs.
    const unsigned mask = __activemask();
    const float accum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += __shfl_sync(mask, accum, i, 8);
    const float norm = Key ? fminf(1.0f, __fdiv_rn(clip, __fsqrt_rn(sum) + 1e-6f))
                           : __fdiv_rn(1.0f, __fsqrt_rn(sum / HEAD_DIM + eps));
    auto* dest = out + row * width + offset;
    for (int i = 0; i < HEAD_DIM / 8; ++i) {
        const int col = i * 8 + lane;
        dest[col] = __float2bfloat16(values[i] * norm);
        if constexpr (Decode) append_history(past, history, taps, width, offset + col, current[col]);
    }
}

} // namespace

namespace {

template<bool Relative>
__global__ void split_clipped_projection(
    const __nv_bfloat16* projected, const float* weights, __nv_bfloat16* clipped,
    __nv_bfloat16* gates, __nv_bfloat16* sinks, int relative, int start,
    float threshold, float log_floor, float log_alpha) {
    const int heads = gridDim.x;
    const int head = blockIdx.x, row = blockIdx.y, lane = threadIdx.x;
    const int q_width = heads * HEAD_DIM;
    const int stride = 2 * q_width + heads * (relative + 1);
    const int width = Relative ? relative : HEAD_DIM;
    const auto* src = projected + size_t(row) * stride;
    const auto* values = src + (Relative ? 2 * q_width : 0) + head * width;
    const auto* scale = weights + head * width;
    auto* out = clipped + (size_t(row) * heads + head) * width;
    const float log_scale = 1.0f + __fmul_rn(log_alpha,
        logf(fmaxf(1.0f, __fdiv_rn(float(start + row + 1), log_floor))));
    if constexpr (Relative) {
        if (lane == 0) {
            float sum = 0.0f;
            for (int i = 0; i < width; ++i) {
                const float v = __bfloat162float(values[i]) * scale[i];
                sum = __fadd_rn(sum, __fmul_rn(v, v));
            }
            const float norm = fminf(1.0f, __fdiv_rn(threshold, __fsqrt_rn(sum) + 1e-6f));
            for (int i = 0; i < width; ++i) {
                const float v = (__bfloat162float(values[i]) * scale[i]) * norm;
                out[i] = __float2bfloat16(__bfloat162float(__float2bfloat16(v)) * log_scale);
            }
            sinks[size_t(row) * heads + head] = src[2 * q_width + heads * relative + head];
        }
    } else {
        float partial[4] = {};
        float channels[HEAD_DIM / 8];
        for (int i = 0; i < HEAD_DIM / 8; ++i) {
            const int col = i * 8 + lane;
            const float v = __bfloat162float(values[col]) * scale[col];
            channels[i] = v;
            partial[i % 4] = fmaf(v, v, partial[i % 4]);
            gates[size_t(row) * q_width + head * HEAD_DIM + col] = src[q_width + head * HEAD_DIM + col];
        }
        const float accum = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        float sum = 0.0f;
        const unsigned mask = __activemask();
        for (int i = 0; i < 8; ++i) sum += __shfl_sync(mask, accum, i, 8);
        const float norm = fminf(1.0f, __fdiv_rn(threshold, __fsqrt_rn(sum) + 1e-6f));
        for (int i = 0; i < HEAD_DIM / 8; ++i)
            out[i * 8 + lane] = __float2bfloat16(__bfloat162float(__float2bfloat16(channels[i] * norm)) * log_scale);
    }
}

} // namespace

extern "C" cudaError_t blz_inkling_split_projection(const uint16_t* projected, const float* weights,
        uint16_t* query, uint16_t* gate, uint16_t* relative, uint16_t* sink, int rows, int heads,
        int rel_width, int start, float clip, float rel_clip, float log_floor, float log_alpha, cudaStream_t stream) {
    if (rows < 1 || rows > 65535 || heads < 1 || rel_width < 1 || rel_width >= 32) return cudaErrorInvalidValue;
    const dim3 grid(heads, rows);
    const auto* source = reinterpret_cast<const __nv_bfloat16*>(projected);
    auto* sinks = reinterpret_cast<__nv_bfloat16*>(sink);
    split_clipped_projection<false><<<grid, 8, 0, stream>>>(source, weights,
        reinterpret_cast<__nv_bfloat16*>(query), reinterpret_cast<__nv_bfloat16*>(gate), sinks,
        rel_width, start, clip, log_floor, log_alpha);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    split_clipped_projection<true><<<grid, 1, 0, stream>>>(source, weights + heads * HEAD_DIM,
        reinterpret_cast<__nv_bfloat16*>(relative), nullptr, sinks,
        rel_width, start, rel_clip, log_floor, log_alpha);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_prefill_conv(const uint16_t* key, const uint16_t* value, const uint16_t* key_dynamic,
        const uint16_t* value_dynamic, const float* weights, uint16_t* keys, uint16_t* values, int heads, int taps,
        int history, int rows, float scale, float clip, float eps, cudaStream_t stream) {
    if (heads < 1 || taps < 1 || rows < 1 || history < 0 || history >= taps) return cudaErrorInvalidValue;
    const dim3 grid(heads, rows);
    inkling_prefill_conv<true><<<grid, 8, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(key),
        reinterpret_cast<const __nv_bfloat16*>(key_dynamic), weights, reinterpret_cast<__nv_bfloat16*>(keys),
        heads, taps, history, scale, clip, eps);
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    inkling_prefill_conv<false><<<grid, 8, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(value),
        reinterpret_cast<const __nv_bfloat16*>(value_dynamic), weights + size_t(heads) * HEAD_DIM * taps,
        reinterpret_cast<__nv_bfloat16*>(values), heads, taps, history, scale, clip, eps);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_decode_conv(const uint16_t* key, const uint16_t* value, const uint16_t* key_dynamic,
        const uint16_t* value_dynamic, const float* weights, uint16_t* keys, uint16_t* values, int heads, int taps,
        int history, uint16_t* past_keys, uint16_t* past_values, float scale, float clip, float eps, cudaStream_t stream) {
    if (heads < 1 || taps < 1 || history < 0 || history >= taps) return cudaErrorInvalidValue;
    inkling_prefill_conv<true, true><<<heads, 8, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(key),
        reinterpret_cast<const __nv_bfloat16*>(key_dynamic), weights, reinterpret_cast<__nv_bfloat16*>(keys),
        heads, taps, history, scale, clip, eps, reinterpret_cast<__nv_bfloat16*>(past_keys));
    const auto err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    inkling_prefill_conv<false, true><<<heads, 8, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(value),
        reinterpret_cast<const __nv_bfloat16*>(value_dynamic), weights + size_t(heads) * HEAD_DIM * taps,
        reinterpret_cast<__nv_bfloat16*>(values), heads, taps, history, scale, clip, eps,
        reinterpret_cast<__nv_bfloat16*>(past_values));
    return cudaGetLastError();
}

namespace {

template<bool Decode = false, class Residual = __nv_bfloat16>
__global__ void residual_conv(const __nv_bfloat16* raw, const Residual* previous,
        const float* weights, float* out, int width, int taps, int history, float scale,
        __nv_bfloat16* past = nullptr) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= width) return;
    const auto* current = raw + (Decode ? 0 : size_t(history + row) * width) + col;
    float value = __bfloat162float(*current);
    const int count = min(taps, history + row + 1);
    for (int lag = 0; lag < count; ++lag) {
        const auto x = Decode && lag > 0 ? past[(history - lag) * width + col] : current[-lag * width];
        value = __fadd_rn(value, __fmul_rn(__bfloat162float(x), weights[lag * width + col]));
    }
    value = __bfloat162float(__float2bfloat16(value * scale));
    const size_t index = size_t(row) * width + col;
    out[index] = previous ? value + float(previous[index]) : value;
    if constexpr (Decode) append_history(past, history, taps, width, col, *current);
}

template<bool Decode = false>
__global__ void mlp_residual_conv(const __nv_bfloat16* routed, const float* residual, const __nv_bfloat16* history,
        const float* weights, __nv_bfloat16* output, __nv_bfloat16* tail, int width, int taps,
        int history_rows, int tail_rows, float scale) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col >= width) return;
    const size_t index = size_t(row) * width + col;
    const auto current = routed[index];
    const int lag_count = min(taps, history_rows + row + 1);
    float value = __bfloat162float(current);
    for (int lag = 0; lag < lag_count; ++lag) {
        const int source = row - lag;
        const auto raw = source >= 0 ? routed[size_t(source) * width + col]
                                     : history[size_t(history_rows + source) * width + col];
        value = __fadd_rn(value, __fmul_rn(__bfloat162float(raw), weights[lag * width + col]));
    }
    output[index] = __float2bfloat16(residual[index] + __bfloat162float(__float2bfloat16(value * scale)));
    if constexpr (Decode) append_history(tail, history_rows, taps, width, col, current);
    else {
        const int tail_row = row - (int(gridDim.y) - tail_rows);
        if (tail_row >= 0) tail[size_t(tail_row) * width + col] = current;
    }
}

} // namespace

extern "C" cudaError_t blz_inkling_mlp_residual_conv(const uint16_t* routed, const float* residual,
        const uint16_t* history, const float* weights, uint16_t* output, uint16_t* tail,
        int width, int taps, int history_rows, int rows, int tail_rows, float scale, cudaStream_t stream) {
    if (width < 1 || taps < 1 || history_rows < 0 || history_rows >= taps || rows < 1 || rows > 65535
        || tail_rows < 0 || tail_rows > rows) return cudaErrorInvalidValue;
    mlp_residual_conv<><<<dim3((width + 127) / 128, rows), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(routed), residual,
        reinterpret_cast<const __nv_bfloat16*>(history), weights, reinterpret_cast<__nv_bfloat16*>(output),
        reinterpret_cast<__nv_bfloat16*>(tail), width, taps, history_rows, tail_rows, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_residual_conv(const uint16_t* raw, const uint16_t* previous,
        const float* weights, float* out, int width, int taps, int history, int rows, float scale,
        cudaStream_t stream) {
    if (width < 1 || taps < 1 || history < 0 || history >= taps || rows < 1 || rows > 65535)
        return cudaErrorInvalidValue;
    residual_conv<><<<dim3((width + 127) / 128, rows), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw), reinterpret_cast<const __nv_bfloat16*>(previous),
        weights, out, width, taps, history, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_decode_attention_residual(const uint16_t* raw, const float* previous,
        const float* weights, float* out, uint16_t* history, int width, int taps, int past, float scale, cudaStream_t stream) {
    residual_conv<true><<<(width + 127) / 128, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw), previous, weights, out, width, taps, past, scale,
        reinterpret_cast<__nv_bfloat16*>(history));
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_prefill_attention_residual(const uint16_t* raw, const float* previous,
        const float* weights, float* out, int width, int taps, int past, int rows, float scale, cudaStream_t stream) {
    residual_conv<false><<<dim3((width + 127) / 128, rows), 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(raw), previous, weights, out, width, taps, past, scale);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_decode_mlp_residual(const uint16_t* raw, const float* residual,
        const float* weights, uint16_t* out, uint16_t* history, int width, int taps, int past, float scale, cudaStream_t stream) {
    auto* state = reinterpret_cast<__nv_bfloat16*>(history);
    mlp_residual_conv<true><<<(width + 127) / 128, 128, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(raw),
        residual, state, weights, reinterpret_cast<__nv_bfloat16*>(out), state, width, taps, past, 0, scale);
    return cudaGetLastError();
}

namespace {

__global__ void dspark_norm_rope(__nv_bfloat16* x, int stride, const __nv_bfloat16* norm,
                                  const float* freq, int start, int* slots, int capacity) {
    const int row = blockIdx.y;
    const int lane = threadIdx.x;
    x += row * stride + blockIdx.x * HEAD_DIM;
    float v[4], sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = __bfloat162float(x[lane + i * 32]);
        sum = fmaf(v[i], v[i], sum);
    }
    const float inv = rsqrtf(warp_sum(sum) / HEAD_DIM + 1e-6f);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int j = lane + i * 32;
        float s, c;
        sincosf(float(start + row) * freq[j], &s, &c);
        const float a = v[i] * inv * __bfloat162float(norm[j]);
        const float b = v[i + 2] * inv * __bfloat162float(norm[j + 64]);
        x[j] = __float2bfloat16(a * c - b * s);
        x[j + 64] = __float2bfloat16(a * s + b * c);
    }
    if (slots && blockIdx.x == 0 && lane == 0) slots[row] = (start + row) % capacity;
}

} // namespace

extern "C" cudaError_t blz_dspark_norm_rope(uint16_t* x, int stride, int rows, int heads, const uint16_t* norm,
        const float* freq, int start, int* slots, int capacity, cudaStream_t stream) {
    if (stride < heads * HEAD_DIM || heads <= 0 || rows <= 0 || capacity <= 0) return cudaErrorInvalidValue;
    dspark_norm_rope<<<dim3(heads, rows), 32, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(x), stride,
            reinterpret_cast<const __nv_bfloat16*>(norm), freq, start, slots, capacity);
    return cudaGetLastError();
}

namespace {

__device__ inline int patch_axis(int block, int sub, int extent, int merge) {
    return min(block * merge + sub, extent - 1);
}

__global__ void vision_pos(__nv_bfloat16* x, const __nv_bfloat16* table, int height, int width,
                           int merge, int side, int hidden, int patches) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= patches * hidden) return;
    const int p = i / hidden;
    const int c = i % hidden;
    const int block = p / (merge * merge);
    const int sub = p % (merge * merge);
    const int row = patch_axis(block / (width / merge), sub / merge, height, merge);
    const int col = patch_axis(block % (width / merge), sub % merge, width, merge);
    const float y = fminf(side - 1.0f, fmaxf(0.0f, (row + 0.5f) * side / height - 0.5f));
    const float z = fminf(side - 1.0f, fmaxf(0.0f, (col + 0.5f) * side / width - 0.5f));
    const int y0 = int(y), y1 = min(y0 + 1, side - 1);
    const int x0 = int(z), x1 = min(x0 + 1, side - 1);
    const float fy = y - y0, fx = z - x0;
    const float v00 = __bfloat162float(table[(y0 * side + x0) * hidden + c]);
    const float v01 = __bfloat162float(table[(y0 * side + x1) * hidden + c]);
    const float v10 = __bfloat162float(table[(y1 * side + x0) * hidden + c]);
    const float v11 = __bfloat162float(table[(y1 * side + x1) * hidden + c]);
    const float pos = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) + fy * ((1.0f - fx) * v10 + fx * v11);
    x[i] = __float2bfloat16(__bfloat162float(x[i]) + __bfloat162float(__float2bfloat16(pos)));
}

__global__ void vision_qkv(const __nv_bfloat162* x, const __nv_bfloat162* bias,
                           __nv_bfloat162* q, __nv_bfloat162* k, __nv_bfloat162* v,
                           int width, int merge, int heads, int rows, int dim) {
    const int pairs = heads * dim / 2;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * pairs) return;
    const int r = i / pairs;
    const int ch = i % pairs;
    const int f = ch % (dim / 2);
    const int half = dim / 4;
    int pos = 0;
    if (r > 0) {
        const int group = (r - 1) / (merge * merge), sub = (r - 1) % (merge * merge);
        const int y = group / (width / merge) * merge + sub / merge;
        const int z = group % (width / merge) * merge + sub % merge;
        pos = 1 + (f < half ? z : y);
    }
    const float inv = powf(10000.0f, -float(f % half) / half);
    float s, c;
    sincosf(pos * inv, &s, &c);
    const int off = r * 3 * pairs + ch;
    for (int part = 0; part < 3; ++part) {
        const float2 p = __bfloat1622float2(__hadd2(x[off + part * pairs], bias[ch + part * pairs]));
        if (part == 0) q[i] = __floats2bfloat162_rn(c * p.x - s * p.y, s * p.x + c * p.y);
        if (part == 1) k[i] = __floats2bfloat162_rn(c * p.x - s * p.y, s * p.x + c * p.y);
        if (part == 2) v[i] = __floats2bfloat162_rn(p.x, p.y);
    }
}

__global__ void vision_residual(__nv_bfloat162* state, const __nv_bfloat162* x,
                                const __nv_bfloat162* bias, const __nv_bfloat162* scale,
                                int n, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = i % dim;
    state[i] = __hadd2(state[i], __hmul2(scale[c], __hadd2(x[i], bias[c])));
}

} // namespace

extern "C" cudaError_t blz_inkling_vision_pos(uint16_t* x, const uint16_t* table, int h, int w,
                                             int merge, int side, int hidden, cudaStream_t stream) {
    const int n = h * w * hidden;
    vision_pos<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(table), h, w, merge, side, hidden, h * w);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_vision_qkv(const uint16_t* x, const uint16_t* bias, uint16_t* q,
                                             uint16_t* k, uint16_t* v, int w, int merge, int heads,
                                             int rows, int dim, cudaStream_t stream) {
    const int n = rows * heads * dim / 2;
    vision_qkv<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<const __nv_bfloat162*>(x),
        reinterpret_cast<const __nv_bfloat162*>(bias), reinterpret_cast<__nv_bfloat162*>(q),
        reinterpret_cast<__nv_bfloat162*>(k), reinterpret_cast<__nv_bfloat162*>(v), w, merge, heads, rows, dim);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_vision_residual(uint16_t* state, const uint16_t* x, const uint16_t* bias,
                                                  const uint16_t* scale, int rows, int hidden, cudaStream_t stream) {
    const int n = rows * hidden / 2;
    vision_residual<<<(n + 255) / 256, 256, 0, stream>>>(reinterpret_cast<__nv_bfloat162*>(state),
        reinterpret_cast<const __nv_bfloat162*>(x), reinterpret_cast<const __nv_bfloat162*>(bias),
        reinterpret_cast<const __nv_bfloat162*>(scale), n, hidden / 2);
    return cudaGetLastError();
}
