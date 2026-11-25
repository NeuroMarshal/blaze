#include <math_constants.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__global__ void softcapped_swiglu(const __nv_bfloat162* gate_up, __nv_bfloat16* out,
                                  int n, float gate_cap, float up_cap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float2 x = __bfloat1622float2(gate_up[i]);
    const float gate = gate_cap * tanhf(x.x / gate_cap);
    const float up = up_cap * tanhf(x.y / up_cap);
    out[i] = __float2bfloat16(gate * up / (1.0f + expf(-x.x)));
}

__global__ void add_expert(const __nv_bfloat16* in, float* out, int n, float weight) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmaf(__bfloat162float(in[i]), weight, out[i]);
}

} // namespace

extern "C" cudaError_t blz_inkling_swiglu(const uint16_t* gate_up, uint16_t* out, int n,
                                          float gate_cap, float up_cap, cudaStream_t stream) {
    softcapped_swiglu<<<(n + 127) / 128, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat162*>(gate_up), reinterpret_cast<__nv_bfloat16*>(out), n, gate_cap, up_cap);
    return cudaGetLastError();
}

extern "C" cudaError_t blz_inkling_expert_accumulate(const uint16_t* in, float* out, int n,
                                                     float weight, cudaStream_t stream) {
    add_expert<<<(n + 127) / 128, 128, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(in), out, n, weight);
    return cudaGetLastError();
}

namespace {

constexpr int HEAD_DIM = 128;
constexpr int WARPS = 4;

__device__ inline float warp_sum(float x) {
    for (int shift = 16; shift; shift /= 2) x += __shfl_xor_sync(0xffffffff, x, shift);
    return x;
}

template <bool Quantized>
__global__ void attention_tiles(const __nv_bfloat16* q, const void* keys,
                                const void* values, const __nv_bfloat16* relative,
                                float* scratch, int kv_heads, int capacity, int position,
                                int extent, int tile) {
    const int head = blockIdx.x;
    const int kv_head = head / (gridDim.x / kv_heads);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int first = max(0, position + 1 - capacity) + blockIdx.y * tile;
    const int end = min(first + tile, position + 1);
    float query[4], accum[4] = {};
#pragma unroll
    for (int j = 0; j < 4; ++j) query[j] = __bfloat162float(q[head * HEAD_DIM + j * 32 + lane]);

    float max_logit = -CUDART_INF_F;
    float sum = 0.0f;
    for (int t = first + warp; t < end; t += WARPS) {
        const size_t row = size_t(t % capacity) * kv_heads + kv_head;
        float key_scale = 1.0f, value_scale = 1.0f;
        const int8_t* kb = nullptr;
        const int8_t* vb = nullptr;
        if constexpr (Quantized) {
            kb = static_cast<const int8_t*>(keys) + row * (HEAD_DIM + 4);
            vb = static_cast<const int8_t*>(values) + row * (HEAD_DIM + 4);
            key_scale = *reinterpret_cast<const float*>(kb + HEAD_DIM);
            value_scale = *reinterpret_cast<const float*>(vb + HEAD_DIM);
        }
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = j * 32 + lane;
            const float k = Quantized ? float(kb[c]) : __bfloat162float(static_cast<const __nv_bfloat16*>(keys)[row * HEAD_DIM + c]);
            dot = fmaf(query[j], k, dot);
        }
        dot *= key_scale;
        dot = warp_sum(dot) * rsqrtf(float(HEAD_DIM));
        const int distance = position - t;
        if (distance < extent) dot += __bfloat162float(relative[head * extent + distance]);
        const float next_max = fmaxf(max_logit, dot);
        const float keep = expf(max_logit - next_max);
        const float add = expf(dot - next_max);
        sum = sum * keep + add;
        max_logit = next_max;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = j * 32 + lane;
            const float v = Quantized ? float(vb[c]) : __bfloat162float(static_cast<const __nv_bfloat16*>(values)[row * HEAD_DIM + c]);
            accum[j] = fmaf(add * value_scale, v, keep * accum[j]);
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
    for (int j = 0; j < 4; ++j) partials[warp][j * 32 + lane] = accum[j];
    __syncthreads();

    const float maximum = fmaxf(fmaxf(maxima[0], maxima[1]), fmaxf(maxima[2], maxima[3]));
    float out = 0.0f, denom = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
        const float factor = maximum == -CUDART_INF_F ? 0.0f : expf(maxima[w] - maximum);
        out = fmaf(factor, partials[w][threadIdx.x], out);
        denom = fmaf(factor, sums[w], denom);
    }
    float* slot = scratch + (head * gridDim.y + blockIdx.y) * (HEAD_DIM + 2);
    slot[threadIdx.x] = out;
    if (threadIdx.x == 0) {
        slot[HEAD_DIM] = maximum;
        slot[HEAD_DIM + 1] = denom;
    }
}

__global__ void attention_merge(const float* scratch, const __nv_bfloat16* sinks,
                                const __nv_bfloat16* gate, __nv_bfloat16* out, int tiles) {
    const int head = blockIdx.x;
    const int col = threadIdx.x;
    const float* row = scratch + head * tiles * (HEAD_DIM + 2);
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

} // namespace

extern "C" cudaError_t blz_inkling_attention(const uint16_t* q, const void* keys, const void* values,
                                            const uint16_t* relative, const uint16_t* sinks, const uint16_t* gate,
                                            uint16_t* out, float* scratch, int heads, int kv_heads, int capacity,
                                            int position, int extent, int tiles, int q8, cudaStream_t stream) {
    const int valid = min(capacity, position + 1);
    const int tile = (valid + tiles - 1) / tiles;
    const int blocks = (valid + tile - 1) / tile;
    const dim3 grid(heads, blocks);
    auto query = reinterpret_cast<const __nv_bfloat16*>(q);
    auto biases = reinterpret_cast<const __nv_bfloat16*>(relative);
    if (q8) {
        attention_tiles<true><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, biases, scratch,
            kv_heads, capacity, position, extent, tile);
    } else {
        attention_tiles<false><<<grid, HEAD_DIM, 0, stream>>>(query, keys, values, biases, scratch,
            kv_heads, capacity, position, extent, tile);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    attention_merge<<<heads, HEAD_DIM, 0, stream>>>(scratch, reinterpret_cast<const __nv_bfloat16*>(sinks),
        reinterpret_cast<const __nv_bfloat16*>(gate), reinterpret_cast<__nv_bfloat16*>(out), blocks);
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

namespace {
__global__ void scatter_expert(const __nv_bfloat16* input, float* output, const int32_t* ids,
                               const float* weights, int hidden) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (col < hidden) {
        float& dst = output[ids[row] * hidden + col];
        dst = fmaf(__bfloat162float(input[row * hidden + col]), weights[row], dst);
    }
}
} // namespace

extern "C" cudaError_t blz_inkling_expert_scatter(const uint16_t* input, float* output, const int32_t* ids,
                                                 const float* weights, int rows, int hidden, cudaStream_t stream) {
    scatter_expert<<<dim3((hidden + 255) / 256, rows), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input), output, ids, weights, hidden);
    return cudaGetLastError();
}
