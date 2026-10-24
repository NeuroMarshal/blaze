// Normalization, attention input preparation, elementwise, routing and argmax kernels (blaze/ops.h).
// Arithmetic in fp32, one bf16 rounding per output, as the reference implementations do.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cfloat>

#include <algorithm>

#include "blaze/ops.h"

namespace blaze {
namespace {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ uint16_t to_bf(float f) {
  const __nv_bfloat16 h = __float2bfloat16_rn(f);
  return *reinterpret_cast<const uint16_t*>(&h);
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + __expf(-x)); }

template <int kThreads>
__device__ __forceinline__ float block_sum(float v, float* red) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  constexpr int kWarps = kThreads / 32;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) red[warp] = v;
  __syncthreads();
  float t = lane < kWarps ? red[lane] : 0.0f;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) t += __shfl_xor_sync(0xffffffffu, t, o);
  __syncthreads();
  return t;
}

// One block per row.
constexpr int kNormThreads = 256;

__global__ void __launch_bounds__(kNormThreads) rmsnorm_kernel(const uint16_t* __restrict__ x,
                                                               uint16_t* __restrict__ residual,
                                                               const uint16_t* __restrict__ w, float eps, int dim,
                                                               int zero_centered, uint16_t* __restrict__ out) {
  __shared__ float red[kNormThreads / 32];
  extern __shared__ float h[];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * dim;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += kNormThreads) {
    float v = bf(xr[i]);
    if (residual) {
      // The residual stream is bf16: round the sum before normalizing, as the reference adds in bf16.
      const uint16_t sum = to_bf(v + bf(residual[row * dim + i]));
      residual[row * dim + i] = sum;
      v = bf(sum);
    }
    h[i] = v;
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum<kNormThreads>(ss, red) / float(dim) + eps);
  for (int i = threadIdx.x; i < dim; i += kNormThreads) {
    const float wi = zero_centered ? 1.0f + bf(w[i]) : bf(w[i]);
    out[row * dim + i] = to_bf(h[i] * inv * wi);
  }
}

// One warp per (token, head) row.
__global__ void gated_rmsnorm_kernel(const uint16_t* __restrict__ x, int64_t ld_x, const uint16_t* __restrict__ z,
                                     int64_t ld_z, const uint16_t* __restrict__ w, float eps, int tokens, int heads,
                                     int dim, int act, uint16_t* __restrict__ out) {
  const int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  if (row >= int64_t(tokens) * heads) return;
  const int64_t t = row / heads;
  const int h = int(row % heads);
  const uint16_t* xr = x + t * ld_x + int64_t(h) * dim;
  const uint16_t* zr = z + t * ld_z + int64_t(h) * dim;
  float ss = 0.0f;
  for (int i = lane; i < dim; i += 32) {
    const float v = bf(xr[i]);
    ss += v * v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / float(dim) + eps);
  for (int i = lane; i < dim; i += 32) {
    const float zv = bf(zr[i]);
    const float sg = 1.0f / (1.0f + __expf(-zv));
    out[row * dim + i] = to_bf(bf(xr[i]) * inv * bf(w[i]) * (act == 1 ? sg : zv * sg));
  }
}

// One warp per (token, head) row of q or k. Rows are staged in shared memory (head_dim <= 512).
constexpr int kPrepWarps = 4;

// MROPE: multimodal rotary positions. Every rotary frequency j takes its angle from one of three position streams
// (temporal = `positions`, height, width), picked by freq_comp[j] in {0, 1, 2}; the text variant (MROPE false) is the
// same code with one stream.
template <bool MROPE>
__global__ void __launch_bounds__(32 * kPrepWarps) attn_prep_kernel(
    const uint16_t* __restrict__ qg, int64_t ld_qg, uint16_t* __restrict__ k, int64_t ld_k,
    const int64_t* __restrict__ positions, const int64_t* __restrict__ pos_h, const int64_t* __restrict__ pos_w,
    const uint8_t* __restrict__ freq_comp,
    const uint16_t* __restrict__ q_norm, const uint16_t* __restrict__ k_norm, float eps, int zero_centered, int tokens,
    int heads, int kv_heads, int head_dim, int rotary_dim, const float* __restrict__ inv_freq, float rope_scale,
    uint16_t* __restrict__ q_out, uint16_t* __restrict__ gate) {
  extern __shared__ float rows[];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float* r = rows + warp * head_dim;
  const int64_t item = int64_t(blockIdx.x) * kPrepWarps + warp;
  const int64_t per_token = heads + kv_heads;
  if (item >= int64_t(tokens) * per_token) return;
  const int64_t t = item / per_token;
  const int h = int(item % per_token);
  const bool is_q = h < heads;
  const uint16_t* src = is_q ? qg + t * ld_qg + int64_t(h) * 2 * head_dim : k + t * ld_k + int64_t(h - heads) * head_dim;
  const uint16_t* nw = is_q ? q_norm : k_norm;
  float ss = 0.0f;
  for (int i = lane; i < head_dim; i += 32) {
    const float v = bf(src[i]);
    r[i] = v;
    ss += v * v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / float(head_dim) + eps);
  // The reference rounds the normed head to bf16 before the rotation.
  for (int i = lane; i < head_dim; i += 32) {
    const float wi = zero_centered ? 1.0f + bf(nw[i]) : bf(nw[i]);
    r[i] = bf(to_bf(r[i] * inv * wi));
  }
  __syncwarp();
  const float pos = float(positions[t]);
  const int half = rotary_dim / 2;
  uint16_t* dst = is_q ? q_out + (t * heads + h) * head_dim : k + t * ld_k + int64_t(h - heads) * head_dim;
  for (int i = lane; i < head_dim; i += 32) {
    float v = r[i];
    if (i < rotary_dim) {
      const int j = i < half ? i : i - half;
      float s, c;
      if constexpr (MROPE) {
        const int comp = freq_comp[j];
        const float p = comp == 0 ? pos : float(comp == 1 ? pos_h[t] : pos_w[t]);
        sincosf(p * inv_freq[j], &s, &c);
      } else {
        sincosf(pos * inv_freq[j], &s, &c);
      }
      c *= rope_scale, s *= rope_scale;
      v = i < half ? r[i] * c - r[i + half] * s : r[i] * c + r[i - half] * s;
    }
    dst[i] = to_bf(v);
  }
  if (is_q) {
    const uint16_t* g = src + head_dim;
    for (int i = lane; i < head_dim; i += 32) gate[(t * heads + h) * head_dim + i] = g[i];
  }
}

__global__ void mul_sigmoid_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ gate, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    x[i] = to_bf(bf(x[i]) * sigmoidf(bf(gate[i])));
}

__global__ void silu_mul_kernel(const uint16_t* __restrict__ gate, int64_t ld_gate, const uint16_t* __restrict__ up,
                                int64_t ld_up, int rows, int n, uint16_t* __restrict__ out) {
  const int64_t total = int64_t(rows) * n;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / n, c = i % n;
    out[i] = to_bf(siluf(bf(gate[r * ld_gate + c])) * bf(up[r * ld_up + c]));
  }
}

__global__ void add_sigmoid_scaled_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b,
                                          const uint16_t* __restrict__ gate_logit, int64_t ld_gate, int rows, int dim,
                                          uint16_t* __restrict__ out) {
  const int64_t total = int64_t(rows) * dim;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / dim;
    // Reference: sigmoid(gate) * shared in bf16, then the bf16 sum.
    const float shared = bf(to_bf(sigmoidf(bf(gate_logit[r * ld_gate])) * bf(b[i])));
    out[i] = to_bf(bf(a[i]) + shared);
  }
}

__global__ void add_kernel(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, int64_t n,
                           uint16_t* __restrict__ out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += int64_t(gridDim.x) * blockDim.x)
    out[i] = to_bf(bf(a[i]) + bf(b[i]));
}

// One warp per token; experts <= 32 * kMaxPerLane, k <= 32.
constexpr int kMaxPerLane = 16;

__global__ void router_topk_kernel(const uint16_t* __restrict__ logits, int64_t ld, int tokens, int experts, int k,
                                   int renorm, int32_t* __restrict__ ids, float* __restrict__ w) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t t = int64_t(blockIdx.x) * (blockDim.x / 32) + warp;
  if (t >= tokens) return;
  float p[kMaxPerLane];
  float mx = -FLT_MAX;
#pragma unroll
  for (int j = 0; j < kMaxPerLane; ++j) {
    const int e = lane + 32 * j;
    p[j] = e < experts ? bf(logits[t * ld + e]) : -FLT_MAX;
    mx = fmaxf(mx, p[j]);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxPerLane; ++j) {
    const int e = lane + 32 * j;
    p[j] = e < experts ? __expf(p[j] - mx) : -1.0f;
    if (e < experts) sum += p[j];
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  float picked_sum = 0.0f, my_w = 0.0f;
  int my_id = 0;
  for (int r = 0; r < k; ++r) {
    float best = -1.0f;
    int best_e = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < kMaxPerLane; ++j) {
      const int e = lane + 32 * j;
      if (p[j] > best || (p[j] == best && e < best_e)) best = p[j], best_e = e;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, best_e, o);
      if (ob > best || (ob == best && oe < best_e)) best = ob, best_e = oe;
    }
    const float prob = best / sum;
    picked_sum += prob;
    if (lane == r) my_w = prob, my_id = best_e;
    if ((best_e & 31) == lane) p[best_e / 32] = -1.0f;
  }
  if (lane < k) {
    ids[t * k + lane] = my_id;
    w[t * k + lane] = renorm ? my_w / picked_sum : my_w;
  }
}

constexpr int kArgmaxThreads = 512;

__global__ void __launch_bounds__(kArgmaxThreads) argmax_kernel(const uint16_t* __restrict__ logits, int vocab,
                                                                int64_t ld, int32_t* __restrict__ out,
                                                                float* __restrict__ pair, int col0) {
  __shared__ float sv[kArgmaxThreads / 32];
  __shared__ int si[kArgmaxThreads / 32];
  const uint16_t* row = logits + int64_t(blockIdx.x) * ld;
  float best = -FLT_MAX;
  int best_i = 0x7fffffff;
  for (int i = threadIdx.x; i < vocab; i += kArgmaxThreads) {
    const float v = bf(row[i]);
    if (v > best) best = v, best_i = i;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
    if (ov > best || (ov == best && oi < best_i)) best = ov, best_i = oi;
  }
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  if (lane == 0) sv[warp] = best, si[warp] = best_i;
  __syncthreads();
  if (warp == 0) {
    best = lane < kArgmaxThreads / 32 ? sv[lane] : -FLT_MAX;
    best_i = lane < kArgmaxThreads / 32 ? si[lane] : 0x7fffffff;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
      if (ov > best || (ov == best && oi < best_i)) best = ov, best_i = oi;
    }
    if (lane == 0) {
      if (out) out[blockIdx.x] = best_i;
      if (pair) pair[2 * blockIdx.x] = best, pair[2 * blockIdx.x + 1] = float(best_i + col0);
    }
  }
}

__global__ void fp8_block_dequant_kernel(const uint8_t* __restrict__ w, const void* __restrict__ scale,
                                         int scale_is_f32, int n, int k, int block_n, int block_k,
                                         uint16_t* __restrict__ out) {
  const int64_t total = int64_t(n) * k;
  const int scale_cols = (k + block_k - 1) / block_k;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int r = int(i / k), c = int(i % k);
    const int64_t si = int64_t(r / block_n) * scale_cols + c / block_k;
    const float s = scale_is_f32 ? static_cast<const float*>(scale)[si] : bf(static_cast<const uint16_t*>(scale)[si]);
    __nv_fp8_e4m3 q;
    q.__x = w[i];
    out[i] = to_bf(float(q) * s);
  }
}

__global__ void gdn_gate_kernel(const uint16_t* __restrict__ a, int64_t ld_a, const uint16_t* __restrict__ b,
                                int64_t ld_b, const uint16_t* __restrict__ a_log, const uint16_t* __restrict__ dt_bias,
                                int tokens, int v_heads, float* __restrict__ g, float* __restrict__ beta) {
  const int64_t total = int64_t(tokens) * v_heads;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t t = i / v_heads;
    const int h = int(i % v_heads);
    const float x = bf(a[t * ld_a + h]) + bf(dt_bias[h]);
    const float softplus = x > 20.0f ? x : log1pf(__expf(x));
    g[i] = -__expf(bf(a_log[h])) * softplus;
    // The reference takes the sigmoid in bf16.
    beta[i] = bf(to_bf(sigmoidf(bf(b[t * ld_b + h]))));
  }
}

// One thread per (sequence, channel): walks the sequence with the last kernel - 1 inputs in registers.
constexpr int kMaxKernel = 8;

__global__ void gdn_conv_kernel(const uint16_t* __restrict__ x, int64_t ldx, const int32_t* __restrict__ seq_start,
                                const int32_t* __restrict__ slots, int channels, int kernel,
                                const uint16_t* __restrict__ weight, uint16_t* __restrict__ state,
                                uint16_t* __restrict__ out, int write_state, uint16_t* __restrict__ record) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int s = blockIdx.y;
  if (c >= channels) return;
  const int t0 = seq_start[s], t1 = seq_start[s + 1];
  uint16_t* st = state + (int64_t(slots[s]) * channels + c) * (kernel - 1);
  float hist[kMaxKernel], wt[kMaxKernel];
  for (int j = 0; j < kernel - 1; ++j) hist[j] = bf(st[j]);
  for (int j = 0; j < kernel; ++j) wt[j] = bf(weight[int64_t(c) * kernel + j]);
  for (int t = t0; t < t1; ++t) {
    const uint16_t xb = x[int64_t(t) * ldx + c];
    if (record) record[int64_t(t) * channels + c] = xb;
    const float xt = bf(xb);
    float acc = wt[kernel - 1] * xt;
    for (int j = 0; j < kernel - 1; ++j) acc += wt[j] * hist[j];
    // The reference convolves in bf16 (fp32 accumulation) and applies SiLU to the bf16 result.
    out[int64_t(t) * channels + c] = to_bf(siluf(bf(to_bf(acc))));
    for (int j = 0; j < kernel - 2; ++j) hist[j] = hist[j + 1];
    hist[kernel - 2] = xt;
  }
  if (write_state)
    for (int j = 0; j < kernel - 1; ++j) st[j] = to_bf(hist[j]);
}

__global__ void gather_rows_kernel(const uint16_t* __restrict__ src, int64_t ld_src, const int32_t* __restrict__ rows,
                                   int n, int dim, uint16_t* __restrict__ dst) {
  const int64_t total = int64_t(n) * dim;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t r = i / dim, c = i % dim;
    dst[i] = src[int64_t(rows[r]) * ld_src + c];
  }
}

__global__ void set_req_to_token_kernel(int32_t* __restrict__ rt, const int64_t* __restrict__ index,
                                        const int64_t* __restrict__ slot, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) rt[index[i]] = int32_t(slot[i]);
}

__global__ void l2_prefetch_kernel(const char* __restrict__ p, size_t bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr size_t kChunk = 64 << 10;
  const size_t off = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * kChunk;
  if (off < bytes) {
    const uint32_t n = uint32_t(min(kChunk, bytes - off)) & ~15u;
    if (n) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p + off), "r"(n) : "memory");
  }
#else
  for (size_t off = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128; off < bytes;
       off += size_t(gridDim.x) * blockDim.x * 128)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
#endif
}

// pairs [world][rows][2] (value, index as float; exact below 2^24) -> the best per row, ties to the lower index.
__global__ void argmax_merge_kernel(const float* __restrict__ pairs, int world, int rows, int32_t* __restrict__ out) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float best = -FLT_MAX, best_i = 3.4e38f;
  for (int w = 0; w < world; ++w) {
    const float v = pairs[(int64_t(w) * rows + r) * 2], i = pairs[(int64_t(w) * rows + r) * 2 + 1];
    if (v > best || (v == best && i < best_i)) best = v, best_i = i;
  }
  out[r] = int32_t(best_i);
}

int grid_for(int64_t n, int threads) {
  const int64_t blocks = (n + threads - 1) / threads;
  return int(blocks < 65535 * 16 ? blocks : 65535 * 16);
}

}  // namespace
}  // namespace blaze

using namespace blaze;

extern "C" cudaError_t blz_rmsnorm(const uint16_t* x, uint16_t* residual, const uint16_t* w, float eps, int rows,
                                   int dim, int zero_centered, uint16_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  rmsnorm_kernel<<<rows, kNormThreads, size_t(dim) * sizeof(float), stream>>>(x, residual, w, eps, dim, zero_centered,
                                                                             out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_gated_rmsnorm(const uint16_t* x, int64_t ld_x, const uint16_t* z, int64_t ld_z,
                                         const uint16_t* w, float eps, int tokens, int heads, int dim, int act,
                                         uint16_t* out, cudaStream_t stream) {
  const int64_t rows = int64_t(tokens) * heads;
  if (rows == 0) return cudaSuccess;
  gated_rmsnorm_kernel<<<unsigned((rows + 3) / 4), 128, 0, stream>>>(x, ld_x, z, ld_z, w, eps, tokens, heads, dim, act,
                                                                     out);
  return cudaGetLastError();
}

static cudaError_t attn_prep_launch(const uint16_t* qg, int64_t ld_qg, uint16_t* k, int64_t ld_k, const int64_t* positions,
                                    const int64_t* pos_h, const int64_t* pos_w, const uint8_t* freq_comp,
                                    const uint16_t* q_norm, const uint16_t* k_norm, float eps, int zero_centered, int tokens,
                                    int heads, int kv_heads, int head_dim, int rotary_dim, const float* inv_freq,
                                    float rope_scale, uint16_t* q_out, uint16_t* gate, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  if (head_dim > 512 || rotary_dim > head_dim || rotary_dim % 2) return cudaErrorInvalidValue;
  const int64_t items = int64_t(tokens) * (heads + kv_heads);
  const unsigned grid = unsigned((items + kPrepWarps - 1) / kPrepWarps);
  const size_t smem = size_t(kPrepWarps) * head_dim * sizeof(float);
  if (pos_h) {
    attn_prep_kernel<true><<<grid, 32 * kPrepWarps, smem, stream>>>(
        qg, ld_qg, k, ld_k, positions, pos_h, pos_w, freq_comp, q_norm, k_norm, eps, zero_centered, tokens, heads, kv_heads,
        head_dim, rotary_dim, inv_freq, rope_scale, q_out, gate);
  } else {
    attn_prep_kernel<false><<<grid, 32 * kPrepWarps, smem, stream>>>(
        qg, ld_qg, k, ld_k, positions, nullptr, nullptr, nullptr, q_norm, k_norm, eps, zero_centered, tokens, heads, kv_heads,
        head_dim, rotary_dim, inv_freq, rope_scale, q_out, gate);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t blz_attn_prep(const uint16_t* qg, int64_t ld_qg, uint16_t* k, int64_t ld_k,
                                     const int64_t* positions, const uint16_t* q_norm,
                                     const uint16_t* k_norm, float eps, int zero_centered, int tokens, int heads,
                                     int kv_heads, int head_dim, int rotary_dim, const float* inv_freq,
                                     float rope_scale, uint16_t* q_out, uint16_t* gate, cudaStream_t stream) {
  return attn_prep_launch(qg, ld_qg, k, ld_k, positions, nullptr, nullptr, nullptr, q_norm, k_norm, eps, zero_centered, tokens,
                          heads, kv_heads, head_dim, rotary_dim, inv_freq, rope_scale, q_out, gate, stream);
}

extern "C" cudaError_t blz_attn_prep_mrope(const uint16_t* qg, int64_t ld_qg, uint16_t* k, int64_t ld_k,
                                           const int64_t* positions, const int64_t* pos_h, const int64_t* pos_w,
                                           const uint8_t* freq_comp, const uint16_t* q_norm, const uint16_t* k_norm,
                                           float eps, int zero_centered, int tokens, int heads, int kv_heads, int head_dim,
                                           int rotary_dim, const float* inv_freq, float rope_scale, uint16_t* q_out,
                                           uint16_t* gate, cudaStream_t stream) {
  return attn_prep_launch(qg, ld_qg, k, ld_k, positions, pos_h, pos_w, freq_comp, q_norm, k_norm, eps, zero_centered, tokens,
                          heads, kv_heads, head_dim, rotary_dim, inv_freq, rope_scale, q_out, gate, stream);
}

extern "C" cudaError_t blz_mul_sigmoid(uint16_t* x, const uint16_t* gate, int64_t n, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  mul_sigmoid_kernel<<<grid_for(n, 256), 256, 0, stream>>>(x, gate, n);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_silu_mul(const uint16_t* gate, int64_t ld_gate, const uint16_t* up, int64_t ld_up, int rows,
                                    int n, uint16_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  silu_mul_kernel<<<grid_for(int64_t(rows) * n, 256), 256, 0, stream>>>(gate, ld_gate, up, ld_up, rows, n, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_add_sigmoid_scaled(const uint16_t* a, const uint16_t* b, const uint16_t* gate_logit,
                                              int64_t ld_gate, int rows, int dim, uint16_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  add_sigmoid_scaled_kernel<<<grid_for(int64_t(rows) * dim, 256), 256, 0, stream>>>(a, b, gate_logit, ld_gate, rows,
                                                                                   dim, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_add(const uint16_t* a, const uint16_t* b, int64_t n, uint16_t* out, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  add_kernel<<<grid_for(n, 256), 256, 0, stream>>>(a, b, n, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_router_topk(const uint16_t* logits, int64_t ld, int tokens, int experts, int k,
                                       int renorm, int32_t* ids, float* w, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  if (experts > 32 * kMaxPerLane || k > 32 || k > experts) return cudaErrorInvalidValue;
  router_topk_kernel<<<(tokens + 3) / 4, 128, 0, stream>>>(logits, ld, tokens, experts, k, renorm, ids, w);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_argmax_bf16(const uint16_t* logits, int rows, int vocab, int64_t ld, int32_t* out,
                                       cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  argmax_kernel<<<rows, kArgmaxThreads, 0, stream>>>(logits, vocab, ld, out, nullptr, 0);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_fp8_block_dequant(const uint8_t* w, const void* scale, int scale_is_f32, int n, int k,
                                             int block_n, int block_k, uint16_t* out, cudaStream_t stream) {
  if (int64_t(n) * k == 0) return cudaSuccess;
  fp8_block_dequant_kernel<<<grid_for(int64_t(n) * k, 256), 256, 0, stream>>>(w, scale, scale_is_f32, n, k, block_n,
                                                                             block_k, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_gdn_gate(const uint16_t* a, int64_t ld_a, const uint16_t* b, int64_t ld_b,
                                    const uint16_t* a_log, const uint16_t* dt_bias, int tokens, int v_heads, float* g,
                                    float* beta, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  gdn_gate_kernel<<<grid_for(int64_t(tokens) * v_heads, 256), 256, 0, stream>>>(a, ld_a, b, ld_b, a_log, dt_bias,
                                                                               tokens, v_heads, g, beta);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_gdn_conv_rec(const uint16_t* x, int64_t ldx, const int32_t* seq_start,
                                        const int32_t* slots, int seqs, int channels, int kernel,
                                        const uint16_t* weight, uint16_t* state, uint16_t* out, int write_state,
                                        uint16_t* record, cudaStream_t stream) {
  if (seqs == 0) return cudaSuccess;
  if (kernel < 2 || kernel > kMaxKernel) return cudaErrorInvalidValue;
  const dim3 grid(unsigned((channels + 127) / 128), unsigned(seqs));
  gdn_conv_kernel<<<grid, 128, 0, stream>>>(x, ldx, seq_start, slots, channels, kernel, weight, state, out, write_state,
                                            record);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_gdn_conv_ex(const uint16_t* x, int64_t ldx, const int32_t* seq_start,
                                       const int32_t* slots, int seqs, int channels, int kernel, const uint16_t* weight,
                                       uint16_t* state, uint16_t* out, int write_state, cudaStream_t stream) {
  return blz_gdn_conv_rec(x, ldx, seq_start, slots, seqs, channels, kernel, weight, state, out, write_state, nullptr,
                          stream);
}

extern "C" cudaError_t blz_gather_rows(const uint16_t* src, int64_t ld_src, const int32_t* rows, int n, int dim,
                                       uint16_t* dst, cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  gather_rows_kernel<<<grid_for(int64_t(n) * dim, 256), 256, 0, stream>>>(src, ld_src, rows, n, dim, dst);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_set_req_to_token(int32_t* req_to_token, const int64_t* index, const int64_t* slot, int n,
                                            cudaStream_t stream) {
  if (n == 0) return cudaSuccess;
  set_req_to_token_kernel<<<(n + 255) / 256, 256, 0, stream>>>(req_to_token, index, slot, n);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_l2_prefetch(const void* p, size_t bytes, cudaStream_t stream) {
  if (bytes < 128) return cudaSuccess;
  static int major_of[16] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  int& major = major_of[dev & 15];
  if (!major) cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  const int threads = 128;
  const size_t units = major >= 9 ? (bytes + (64 << 10) - 1) >> 16 : (bytes + 127) / 128;
  const int blocks = int(std::min<size_t>((units + threads - 1) / threads, 1024));
  l2_prefetch_kernel<<<blocks, threads, 0, stream>>>(static_cast<const char*>(p), bytes);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_argmax_part(const uint16_t* logits, int rows, int vocab, int64_t ld, int col0, float* pairs,
                                       cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  argmax_kernel<<<rows, kArgmaxThreads, 0, stream>>>(logits, vocab, ld, nullptr, pairs, col0);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_argmax_merge(const float* pairs, int world, int rows, int32_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  argmax_merge_kernel<<<(rows + 127) / 128, 128, 0, stream>>>(pairs, world, rows, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_gdn_conv(const uint16_t* x, int64_t ldx, const int32_t* seq_start, const int32_t* slots,
                                    int seqs, int channels, int kernel, const uint16_t* weight, uint16_t* state,
                                    uint16_t* out, cudaStream_t stream) {
  return blz_gdn_conv_ex(x, ldx, seq_start, slots, seqs, channels, kernel, weight, state, out, 1, stream);
}
