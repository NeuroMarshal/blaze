// EAGLE-4 draft kernels (blaze/eagle.h).
#include <cuda_bf16.h>

#include <cfloat>

#include "blaze/eagle.h"

namespace {

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t to_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16(v)); }

constexpr int kThreads = 256;

__device__ float block_sum(float v, float* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float t = 0.0f;
  for (int i = 0; i < int(blockDim.x >> 5); ++i) t += red[i];
  __syncthreads();
  return t;
}

// one block per row
__global__ void __launch_bounds__(kThreads) rmsnorm_ld_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                              const uint16_t* __restrict__ w, float eps, int dim,
                                                              uint16_t* __restrict__ out, int64_t ldo) {
  __shared__ float red[kThreads / 32];
  const uint16_t* xr = x + int64_t(blockIdx.x) * ldx;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += kThreads) {
    const float v = bf(xr[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / float(dim) + eps);
  uint16_t* o = out + int64_t(blockIdx.x) * ldo;
  for (int i = threadIdx.x; i < dim; i += kThreads) o[i] = to_bf(bf(xr[i]) * inv * bf(w[i]));
}

// the same with the zero-centered weight (1 + w) of the Qwen3.5 norms
__global__ void __launch_bounds__(kThreads) rmsnorm_ld_zc_kernel(const uint16_t* __restrict__ x, int64_t ldx,
                                                                 const uint16_t* __restrict__ w, float eps, int dim,
                                                                 uint16_t* __restrict__ out, int64_t ldo) {
  __shared__ float red[kThreads / 32];
  const uint16_t* xr = x + int64_t(blockIdx.x) * ldx;
  float ss = 0.0f;
  for (int i = threadIdx.x; i < dim; i += kThreads) {
    const float v = bf(xr[i]);
    ss += v * v;
  }
  const float inv = rsqrtf(block_sum(ss, red) / float(dim) + eps);
  uint16_t* o = out + int64_t(blockIdx.x) * ldo;
  for (int i = threadIdx.x; i < dim; i += kThreads) o[i] = to_bf(bf(xr[i]) * inv * (1.0f + bf(w[i])));
}

// one thread per (token, head, pair)
__global__ void rope_neox_kernel(uint16_t* __restrict__ x, int64_t ld, int tokens, int heads, int head_dim,
                                 const int64_t* __restrict__ positions, const float* __restrict__ inv_freq) {
  const int half = head_dim / 2;
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(tokens) * heads * half) return;
  const int j = int(i % half);
  const int64_t th = i / half;
  const int h = int(th % heads);
  const int64_t t = th / heads;
  uint16_t* p = x + t * ld + int64_t(h) * head_dim;
  float s, c;
  sincosf(float(positions[t]) * inv_freq[j], &s, &c);
  const float a = bf(p[j]), b = bf(p[j + half]);
  p[j] = to_bf(a * c - b * s);
  p[j + half] = to_bf(b * c + a * s);
}

// one warp per (row, candidate): 16-byte loads of the table row
__global__ void candidate_logits_kernel(const uint16_t* __restrict__ hidden, int64_t ld_hidden, int rows, int dim,
                                        const uint16_t* __restrict__ table, const int32_t* __restrict__ ids, int c,
                                        float* __restrict__ out) {
  const int64_t w = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (w >= int64_t(rows) * c) return;
  const int64_t r = w / c;
  const int id = ids[w];
  if (id < 0) {
    if (lane == 0) out[w] = -FLT_MAX;
    return;
  }
  const uint4* t = reinterpret_cast<const uint4*>(table + int64_t(id) * dim);
  const uint4* h = reinterpret_cast<const uint4*>(hidden + r * ld_hidden);
  float acc = 0.0f;
  for (int i = lane; i < dim / 8; i += 32) {
    const uint4 a = t[i], b = h[i];
    const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      acc += bf(uint16_t(aw[k])) * bf(uint16_t(bw[k]));
      acc += bf(uint16_t(aw[k] >> 16)) * bf(uint16_t(bw[k] >> 16));
    }
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) out[w] = acc;
}

// one warp per row
__global__ void candidate_pick_kernel(const float* __restrict__ logits, const int32_t* __restrict__ ids, int rows,
                                      int c, int64_t* __restrict__ token, int64_t* __restrict__ record) {
  const int r = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (r >= rows) return;
  float best = -FLT_MAX;
  int bid = 0x7fffffff;
  for (int j = lane; j < c; j += 32) {
    const float v = logits[int64_t(r) * c + j];
    const int id = ids[int64_t(r) * c + j];
    if (id >= 0 && (v > best || (v == best && id < bid))) best = v, bid = id;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, best, o);
    const int oi = __shfl_xor_sync(0xffffffffu, bid, o);
    if (ov > best || (ov == best && oi < bid)) best = ov, bid = oi;
  }
  if (lane == 0) {
    token[r] = bid;
    if (record) record[r] = bid;
  }
}

// one warp per (token, head)
__global__ void head_rmsnorm_kernel(uint16_t* __restrict__ x, int64_t ld, int tokens, int heads, int head_dim,
                                    const uint16_t* __restrict__ w, float eps) {
  const int64_t row = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (row >= int64_t(tokens) * heads) return;
  uint16_t* p = x + (row / heads) * ld + (row % heads) * head_dim;
  float ss = 0.0f;
  for (int i = lane; i < head_dim; i += 32) {
    const float v = bf(p[i]);
    ss += v * v;
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / float(head_dim) + eps);
  for (int i = lane; i < head_dim; i += 32) p[i] = to_bf(bf(p[i]) * inv * bf(w[i]));
}

// one thread per (row, channel)
__global__ void dflash_conv_kernel(const uint16_t* __restrict__ x, int64_t ldx, const uint16_t* __restrict__ delta,
                                   int64_t delta_row, int64_t delta_tap, const uint16_t* __restrict__ base,
                                   uint16_t* __restrict__ out, int64_t ldo, int rows, int channels, int block_size,
                                   int group_size, int taps) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= int64_t(rows) * channels) return;
  const int r = int(i / channels), c = int(i % channels), g = c / group_size, pos = r % block_size;
  float acc = 0.0f;
  for (int t = 0; t < taps && t <= pos; ++t) {
    const float coef = bf(base[int64_t(t) * channels + c]) + bf(delta[int64_t(r) * delta_row + int64_t(t) * delta_tap + g]);
    acc += coef * bf(x[int64_t(r - t) * ldx + c]);
  }
  out[int64_t(r) * ldo + c] = to_bf(acc);
}

// one block per item: steps in order, one warp per candidate's dot product
constexpr int kWalkMaxK = 32, kWalkMaxRank = 512;
__global__ void dflash_walk_kernel(const int32_t* __restrict__ cand, const float* __restrict__ unary,
                                   const uint16_t* __restrict__ hproj, const int64_t* __restrict__ anchor,
                                   const uint16_t* __restrict__ pred, const uint16_t* __restrict__ succ, int steps,
                                   int k, int rank, int64_t* __restrict__ out) {
  __shared__ float v[kWalkMaxRank];
  __shared__ float score[kWalkMaxK];
  __shared__ int prev_idx;
  const int item = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32, warps = blockDim.x / 32;
  if (threadIdx.x == 0) prev_idx = 0;
  __syncthreads();
  for (int s = 0; s < steps; ++s) {
    const int64_t base = (int64_t(item) * steps + s) * k;
    const int64_t prev_tok = s == 0 ? anchor[item] : int64_t(cand[base - k + prev_idx]);
    // v = pred[prev] * hproj[s]
    for (int r = threadIdx.x; r < rank; r += blockDim.x)
      v[r] = bf(pred[prev_tok * rank + r]) * bf(hproj[(int64_t(item) * steps + s) * rank + r]);
    __syncthreads();
    for (int c = warp; c < k; c += warps) {
      const uint16_t* sv = succ + int64_t(cand[base + c]) * rank;
      float d = 0.0f;
      for (int r = lane; r < rank; r += 32) d += v[r] * bf(sv[r]);
      for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
      if (lane == 0) score[c] = unary[base + c] + d;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int best = 0;
      for (int c = 1; c < k; ++c)
        if (score[c] > score[best]) best = c;
      out[int64_t(item) * steps + s] = cand[base + best];
      prev_idx = best;
    }
    __syncthreads();
  }
}

// one warp per row: up to 64 entries, each lane ranks its two against all (value desc, id asc)
__global__ void cand_merge_kernel(const float* __restrict__ pairs, int world, int rows, int k, int32_t* __restrict__ ids,
                                  float* __restrict__ vals) {
  const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x % 32;
  if (row >= rows) return;
  const int n = world * k;
  float v[2];
  int id[2];
  for (int j = 0; j < 2; ++j) {
    const int e = lane + 32 * j;
    v[j] = -FLT_MAX, id[j] = 0x7fffffff;
    if (e < n) {
      const int w = e / k, c = e % k;
      const float* p = pairs + ((int64_t(w) * rows + row) * (k + 1) + c) * 2;
      const int pid = int(p[1]);
      if (pid >= 0) v[j] = p[0], id[j] = pid;
    }
  }
  for (int j = 0; j < 2; ++j) {
    int rank = 0;
    for (int o = 0; o < 64; ++o) {
      const float ov = __shfl_sync(0xffffffffu, v[o / 32], o % 32);
      const int oi = __shfl_sync(0xffffffffu, id[o / 32], o % 32);
      rank += ov > v[j] || (ov == v[j] && oi < id[j]);
    }
    if (rank < k && lane + 32 * j < n) {
      ids[int64_t(row) * k + rank] = id[j] == 0x7fffffff ? 0 : id[j];
      vals[int64_t(row) * k + rank] = v[j];
    }
  }
}

}  // namespace

extern "C" cudaError_t blz_cand_merge(const float* pairs, int world, int rows, int k, int32_t* ids, float* vals,
                                      cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  if (world * k > 64 || k < 1) return cudaErrorInvalidValue;
  cand_merge_kernel<<<unsigned((rows * 32 + 255) / 256), 256, 0, stream>>>(pairs, world, rows, k, ids, vals);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_head_rmsnorm(uint16_t* x, int64_t ld, int tokens, int heads, int head_dim,
                                        const uint16_t* w, float eps, cudaStream_t stream) {
  const int64_t rows = int64_t(tokens) * heads;
  if (rows == 0) return cudaSuccess;
  head_rmsnorm_kernel<<<unsigned((rows * 32 + 255) / 256), 256, 0, stream>>>(x, ld, tokens, heads, head_dim, w, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_dflash_grouped_conv(const uint16_t* x, int64_t ldx, const uint16_t* delta, int64_t delta_row,
                                               int64_t delta_tap, const uint16_t* base, uint16_t* out, int64_t ldo,
                                               int rows, int channels, int block_size, int group_size, int taps,
                                               cudaStream_t stream) {
  const int64_t n = int64_t(rows) * channels;
  if (n == 0) return cudaSuccess;
  if (channels % group_size || block_size < 1) return cudaErrorInvalidValue;
  dflash_conv_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, ldx, delta, delta_row, delta_tap, base, out, ldo,
                                                                    rows, channels, block_size, group_size, taps);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_dflash_walk(const int32_t* cand, const float* unary, const uint16_t* hproj,
                                       const int64_t* anchor, const uint16_t* pred, const uint16_t* succ, int items,
                                       int steps, int k, int rank, int64_t* out, cudaStream_t stream) {
  if (items == 0 || steps == 0) return cudaSuccess;
  if (k < 1 || k > kWalkMaxK || rank > kWalkMaxRank) return cudaErrorInvalidValue;
  dflash_walk_kernel<<<items, 512, 0, stream>>>(cand, unary, hproj, anchor, pred, succ, steps, k, rank, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_rmsnorm_ld(const uint16_t* x, int64_t ldx, const uint16_t* w, float eps, int rows, int dim,
                                      uint16_t* out, int64_t ldo, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  rmsnorm_ld_kernel<<<rows, kThreads, 0, stream>>>(x, ldx, w, eps, dim, out, ldo);
  return cudaGetLastError();
}

__global__ void widen_i32_kernel(const int32_t* __restrict__ src, int64_t* __restrict__ a, int64_t* __restrict__ b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] = b[i] = int64_t(src[i]);
}

extern "C" cudaError_t blz_widen_i32(const int32_t* src, int64_t* a, int64_t* b, int n, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  widen_i32_kernel<<<(n + 127) / 128, 128, 0, stream>>>(src, a, b, n);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_rmsnorm_ld_zc(const uint16_t* x, int64_t ldx, const uint16_t* w, float eps, int rows, int dim,
                                         uint16_t* out, int64_t ldo, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  rmsnorm_ld_zc_kernel<<<rows, kThreads, 0, stream>>>(x, ldx, w, eps, dim, out, ldo);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_rope_neox(uint16_t* x, int64_t ld, int tokens, int heads, int head_dim,
                                     const int64_t* positions, const float* inv_freq, cudaStream_t stream) {
  const int64_t n = int64_t(tokens) * heads * (head_dim / 2);
  if (n == 0) return cudaSuccess;
  if (head_dim % 2) return cudaErrorInvalidValue;
  rope_neox_kernel<<<unsigned((n + 255) / 256), 256, 0, stream>>>(x, ld, tokens, heads, head_dim, positions, inv_freq);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_candidate_logits(const uint16_t* hidden, int64_t ld_hidden, int rows, int dim,
                                            const uint16_t* table, const int32_t* ids, int c, float* out,
                                            cudaStream_t stream) {
  const int64_t warps = int64_t(rows) * c;
  if (warps == 0) return cudaSuccess;
  if (dim % 8 || ld_hidden % 8) return cudaErrorInvalidValue;
  candidate_logits_kernel<<<unsigned((warps * 32 + 255) / 256), 256, 0, stream>>>(hidden, ld_hidden, rows, dim, table,
                                                                                   ids, c, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_candidate_pick(const float* logits, const int32_t* ids, int rows, int c, int64_t* token,
                                          int64_t* record, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  candidate_pick_kernel<<<unsigned((rows * 32 + 255) / 256), 256, 0, stream>>>(logits, ids, rows, c, token, record);
  return cudaGetLastError();
}
