// QSA indexer kernels (blaze/qsa_index.h).

#include <cuda_bf16.h>

#include <cfloat>

#include "blaze/qsa_index.h"

namespace {

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t to_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16(v)); }
__device__ __forceinline__ float rbf(float v) { return bf(to_bf(v)); }

// Gemma RMSNorm + neox rope of one dim-128 row held by a warp, 4 values per lane (lane l: dims 4l .. 4l + 3).
__device__ void norm_rope_128(float (&v)[4], const uint16_t* __restrict__ w, float eps, int64_t pos, int rotary_dim,
                              const float* __restrict__ inv_freq, float rope_scale) {
  const int lane = threadIdx.x & 31;
  float ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) ss += v[i] * v[i];
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const float inv = rsqrtf(ss / 128.0f + eps);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = rbf(v[i] * inv * (1.0f + bf(w[4 * lane + i])));
  // neox rope over the first rotary_dim values: pairs (i, i + rotary_dim / 2)
  const int half = rotary_dim / 2, lanes_half = half / 4;  // lanes holding the first half
  float other[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float up = __shfl_down_sync(0xffffffffu, v[i], lanes_half);
    const float dn = __shfl_up_sync(0xffffffffu, v[i], lanes_half);
    other[i] = lane < lanes_half ? up : dn;
  }
  if (lane < 2 * lanes_half) {
    const bool first = lane < lanes_half;
    const int base = 4 * (first ? lane : lane - lanes_half);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float s, c;
      sincosf(float(pos) * inv_freq[base + i], &s, &c);
      s *= rope_scale;
      c *= rope_scale;
      v[i] = first ? v[i] * c - other[i] * s : v[i] * c + other[i] * s;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = rbf(v[i]);
}

__global__ void index_q_kernel(const uint16_t* __restrict__ proj, int64_t ld, int col,
                               const int64_t* __restrict__ positions, const uint16_t* __restrict__ w, float eps,
                               int tokens, int heads, int rotary_dim, const float* __restrict__ inv_freq,
                               float rope_scale, uint16_t* __restrict__ out) {
  const int64_t warp = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  if (warp >= int64_t(tokens) * heads) return;
  const int64_t t = warp / heads;
  const int h = int(warp % heads), lane = threadIdx.x & 31;
  const uint16_t* src = proj + t * ld + col + h * 128 + 4 * lane;
  float v[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = bf(src[i]);
  norm_rope_128(v, w, eps, positions[t], rotary_dim, inv_freq, rope_scale);
  uint16_t* dst = out + (t * heads + h) * 128 + 4 * lane;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = to_bf(v[i]);
}

// One warp per token completing a block: mean of the block's raw keys (rows of this step, or the ring), norm, rope.
__global__ void compress_kernel(const uint16_t* __restrict__ proj, int64_t ld, int kcol,
                                const int64_t* __restrict__ positions, const int32_t* __restrict__ tok_row, int tokens,
                                int ratio, const uint16_t* __restrict__ w, float eps, int rotary_dim,
                                const float* __restrict__ inv_freq, float rope_scale,
                                const int32_t* __restrict__ rt, int64_t rt_stride, const uint16_t* __restrict__ ring,
                                uint16_t* __restrict__ comp) {
  const int j = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  if (j >= tokens) return;
  const int64_t p = positions[j];
  if ((p + 1) % ratio != 0) return;
  const int row = tok_row[j], lane = threadIdx.x & 31;
  float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int m = 0; m < ratio; ++m) {
    const int64_t pm = p - (ratio - 1) + m;
    const int jm = j - int(p - pm);
    const uint16_t* src = (jm >= 0 && tok_row[jm] == row && positions[jm] == pm)
                              ? proj + int64_t(jm) * ld + kcol
                              : ring + (int64_t(row) * ratio + pm % ratio) * 128;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] += bf(src[4 * lane + i]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = rbf(v[i] / float(ratio));
  const int64_t first = p - (ratio - 1);
  norm_rope_128(v, w, eps, first, rotary_dim, inv_freq, rope_scale);
  const int64_t page = rt[int64_t(row) * rt_stride + first] / ratio;
  uint16_t* dst = comp + page * 128 + 4 * lane;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = to_bf(v[i]);
}

// Raw keys of each sequence's trailing incomplete block into its ring.
__global__ void ring_kernel(const uint16_t* __restrict__ proj, int64_t ld, int kcol,
                            const int64_t* __restrict__ positions, const int32_t* __restrict__ tok_row,
                            const int32_t* __restrict__ tok_kv_len, int tokens, int ratio, uint16_t* __restrict__ ring) {
  const int j = int((int64_t(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  if (j >= tokens) return;
  const int64_t p = positions[j];
  if (p < int64_t(tok_kv_len[j] / ratio) * ratio) return;
  const int lane = threadIdx.x & 31;
  const uint16_t* src = proj + int64_t(j) * ld + kcol + 4 * lane;
  uint16_t* dst = ring + (int64_t(tok_row[j]) * ratio + p % ratio) * 128 + 4 * lane;
#pragma unroll
  for (int i = 0; i < 4; ++i) dst[i] = src[i];
}

constexpr int kScoreThreads = 256;
constexpr int kMaxGroupRows = 16;

// grid: (ceil(max_blocks / 256), ceil(rows / group)); thread t scores block tile * 256 + t for the group's rows.
// group > 1 only for rows of one sequence (a prefill span): the key is loaded once for all of them.
__global__ void __launch_bounds__(kScoreThreads) scores_kernel(const uint16_t* __restrict__ q, int heads,
                                                               const int64_t* __restrict__ positions,
                                                               const int32_t* __restrict__ tok_row, int row0, int rows,
                                                               int group, int ratio, const int32_t* __restrict__ rt,
                                                               int64_t rt_stride, const uint16_t* __restrict__ comp,
                                                               int max_blocks, float* __restrict__ scores,
                                                               int64_t ld) {
  __shared__ __align__(16) uint16_t sq[kMaxGroupRows * 8 * 128];
  const int g0 = blockIdx.y * group;
  const int n = min(group, rows - g0);
  for (int i = threadIdx.x; i < n * heads * 128; i += kScoreThreads)
    sq[i] = q[(int64_t(row0 + g0) * heads) * 128 + i];
  __syncthreads();
  const int b = blockIdx.x * kScoreThreads + threadIdx.x;
  if (b >= max_blocks) return;
  const int seq = tok_row[row0 + g0];
  int64_t vis_max = 0;
  for (int r = 0; r < n; ++r) vis_max = max(vis_max, (positions[row0 + g0 + r] + 1) / ratio);
  const float scale = rsqrtf(128.0f);
  if (b >= vis_max) {
    for (int r = 0; r < n; ++r) scores[int64_t(g0 + r) * ld + b] = -FLT_MAX;
    return;
  }
  // key of block b (sequence `seq`; the rows of a group share it)
  float k[128];
  const uint4* kp = reinterpret_cast<const uint4*>(comp + int64_t(rt[int64_t(seq) * rt_stride + int64_t(b) * ratio] / ratio) * 128);
#pragma unroll
  for (int c = 0; c < 16; ++c) {
    const uint4 u = kp[c];
    const uint32_t wds[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      k[8 * c + 2 * e] = __uint_as_float(wds[e] << 16);
      k[8 * c + 2 * e + 1] = __uint_as_float(wds[e] & 0xffff0000u);
    }
  }
  for (int r = 0; r < n; ++r) {
    const int row = row0 + g0 + r;
    float s = -FLT_MAX;
    if (b < (positions[row] + 1) / ratio) {
      s = 0.0f;
      for (int h = 0; h < heads; ++h) {
        const uint16_t* qh = sq + (r * heads + h) * 128;
        float d = 0.0f;
#pragma unroll
        for (int i = 0; i < 128; ++i) d = fmaf(bf(qh[i]), k[i], d);
        s += fmaxf(d, 0.0f);
      }
      s *= scale;
    }
    scores[int64_t(g0 + r) * ld + b] = s;
  }
}

constexpr int kSelThreads = 1024;

__device__ __forceinline__ uint32_t fkey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive scan of one int per thread over the block; returns the thread's offset and sets *total.
__device__ int block_scan(int v, int* total) {
  __shared__ int warp_sums[kSelThreads / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int s = lane < kSelThreads / 32 ? warp_sums[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += y;
    }
    if (lane < kSelThreads / 32) warp_sums[lane] = s;
  }
  __syncthreads();
  const int before = (warp > 0 ? warp_sums[warp - 1] : 0) + x - v;
  *total = warp_sums[kSelThreads / 32 - 1];
  __syncthreads();
  return before;
}

// One block per row: radix select (11 + 11 + 10 bits of the ordered float key) of the best block_topk visible blocks,
// collected in block order (above the threshold, then the lowest ties), expanded to KV slots, plus the tail.
__global__ void __launch_bounds__(kSelThreads) select_kernel(const float* __restrict__ scores, int64_t ld,
                                                             const int64_t* __restrict__ positions,
                                                             const int32_t* __restrict__ tok_row, int row0, int ratio,
                                                             int block_topk, const int32_t* __restrict__ rt,
                                                             int64_t rt_stride, int32_t* __restrict__ table,
                                                             int64_t table_stride, int32_t* __restrict__ counts) {
  __shared__ int hist[2048];
  __shared__ uint32_t s_prefix, s_mask;
  __shared__ int s_need;
  __shared__ int sel[2048];
  const int r = blockIdx.x, row = row0 + r, tid = threadIdx.x;
  const int64_t pos = positions[row];
  const int nvis = int((pos + 1) / ratio);
  const int seq = tok_row[row];
  const float* sc = scores ? scores + int64_t(r) * ld : nullptr;
  int nsel;
  if (sc == nullptr || nvis <= block_topk) {
    for (int i = tid; i < nvis; i += kSelThreads) sel[i] = i;
    nsel = nvis;
  } else {
    // radix select of the block_topk-th largest key
    if (tid == 0) s_prefix = 0, s_mask = 0, s_need = block_topk;
    const int shifts[3] = {21, 10, 0}, widths[3] = {11, 11, 10};
    for (int pass = 0; pass < 3; ++pass) {
      for (int i = tid; i < 2048; i += kSelThreads) hist[i] = 0;
      __syncthreads();
      const uint32_t prefix = s_prefix, mask = s_mask;
      const int sh = shifts[pass], bins = 1 << widths[pass];
      for (int i = tid; i < nvis; i += kSelThreads) {
        const uint32_t k = fkey(sc[i]);
        if ((k & mask) == prefix) atomicAdd(&hist[(k >> sh) & (bins - 1)], 1);
      }
      __syncthreads();
      if (tid == 0) {
        int need = s_need, bin = bins - 1;
        while (bin > 0 && hist[bin] < need) need -= hist[bin--];
        s_need = need;
        s_prefix = prefix | (uint32_t(bin) << sh);
        s_mask = mask | (uint32_t(bins - 1) << sh);
      }
      __syncthreads();
    }
    const uint32_t thr = s_prefix;
    const int ties_needed = s_need;  // elements equal to thr to take
    // contiguous ranges per thread keep block order
    const int per = (nvis + kSelThreads - 1) / kSelThreads;
    const int a = min(nvis, tid * per), b = min(nvis, a + per);
    int gt = 0, eq = 0;
    for (int i = a; i < b; ++i) {
      const uint32_t k = fkey(sc[i]);
      gt += k > thr;
      eq += k == thr;
    }
    int n_gt = 0, n_eq = 0;
    const int off_gt = block_scan(gt, &n_gt);
    const int off_eq = block_scan(eq, &n_eq);
    int og = off_gt, oe = off_eq;
    for (int i = a; i < b; ++i) {
      const uint32_t k = fkey(sc[i]);
      if (k > thr) sel[og++] = i;
      else if (k == thr) {
        if (oe < ties_needed) sel[n_gt + oe] = i;
        ++oe;
      }
    }
    nsel = n_gt + min(n_eq, ties_needed);
    __syncthreads();
  }
  __syncthreads();
  int32_t* out = table + int64_t(r) * table_stride;
  const int32_t* rtr = rt + int64_t(seq) * rt_stride;
  for (int i = tid; i < nsel * ratio; i += kSelThreads) out[i] = rtr[int64_t(sel[i / ratio]) * ratio + i % ratio];
  const int64_t tail0 = int64_t(nvis) * ratio;
  const int tail = int(pos + 1 - tail0);
  for (int i = tid; i < tail; i += kSelThreads) out[nsel * ratio + i] = rtr[tail0 + i];
  if (tid == 0) counts[r] = nsel * ratio + tail;
}

}  // namespace

extern "C" cudaError_t blz_qsa_index_q(const uint16_t* proj, int64_t ld, int col, const int64_t* positions,
                                       const uint16_t* norm_w, float eps, int tokens, int heads, int dim,
                                       int rotary_dim, const float* inv_freq, float rope_scale, uint16_t* out,
                                       cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  if (dim != 128 || rotary_dim % 8 != 0 || rotary_dim > 128) return cudaErrorInvalidValue;
  const int64_t threads = int64_t(tokens) * heads * 32;
  index_q_kernel<<<unsigned((threads + 255) / 256), 256, 0, stream>>>(proj, ld, col, positions, norm_w, eps, tokens,
                                                                      heads, rotary_dim, inv_freq, rope_scale, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_qsa_compress(const uint16_t* proj, int64_t ld, int kcol, const int64_t* positions,
                                        const int32_t* tok_row, const int32_t* tok_kv_len, int tokens, int dim,
                                        int ratio, const uint16_t* norm_w, float eps, int rotary_dim,
                                        const float* inv_freq, float rope_scale, const int32_t* req_to_token,
                                        int64_t rt_stride, uint16_t* ring, uint16_t* comp, cudaStream_t stream) {
  if (tokens == 0) return cudaSuccess;
  if (dim != 128) return cudaErrorInvalidValue;
  const unsigned blocks = unsigned((int64_t(tokens) * 32 + 255) / 256);
  compress_kernel<<<blocks, 256, 0, stream>>>(proj, ld, kcol, positions, tok_row, tokens, ratio, norm_w, eps,
                                              rotary_dim, inv_freq, rope_scale, req_to_token, rt_stride, ring, comp);
  ring_kernel<<<blocks, 256, 0, stream>>>(proj, ld, kcol, positions, tok_row, tok_kv_len, tokens, ratio, ring);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_qsa_scores(const uint16_t* q, int heads, int dim, const int64_t* positions,
                                      const int32_t* tok_row, int row0, int rows, int same_seq, int ratio,
                                      const int32_t* req_to_token, int64_t rt_stride, const uint16_t* comp,
                                      int max_blocks, float* scores, int64_t ld_scores, cudaStream_t stream) {
  if (rows == 0 || max_blocks == 0) return cudaSuccess;
  if (dim != 128 || heads > 8) return cudaErrorInvalidValue;
  // rows of one sequence (a prefill span) share key loads in groups of 16; decode rows (different sequences) go one by one
  const int group = same_seq ? kMaxGroupRows : 1;
  const dim3 grid(unsigned((max_blocks + kScoreThreads - 1) / kScoreThreads), unsigned((rows + group - 1) / group));
  scores_kernel<<<grid, kScoreThreads, 0, stream>>>(q, heads, positions, tok_row, row0, rows, group, ratio,
                                                    req_to_token, rt_stride, comp, max_blocks, scores, ld_scores);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_qsa_select(const float* scores, int64_t ld_scores, const int64_t* positions,
                                      const int32_t* tok_row, int row0, int rows, int ratio, int block_topk,
                                      const int32_t* req_to_token, int64_t rt_stride, int32_t* table,
                                      int64_t table_stride, int32_t* counts, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  if (block_topk > 2048) return cudaErrorInvalidValue;
  select_kernel<<<rows, kSelThreads, 0, stream>>>(scores, ld_scores, positions, tok_row, row0, ratio, block_topk,
                                                  req_to_token, rt_stride, table + int64_t(row0) * table_stride,
                                                  table_stride, counts + row0);
  return cudaGetLastError();
}
