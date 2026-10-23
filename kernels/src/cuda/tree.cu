// Draft trees (blaze/tree.h): GDN through tree rows, row moves for the commit, the draft tree's bookkeeping.

#include <cuda_bf16.h>

#include <cfloat>

#include "blaze/tree.h"

namespace blaze {
namespace {

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ uint16_t to_bf(float f) {
  const __nv_bfloat16 h = __float2bfloat16_rn(f);
  return *reinterpret_cast<const uint16_t*>(&h);
}
__device__ __forceinline__ float siluf(float x) { return x / (1.0f + __expf(-x)); }

constexpr int kMaxKernel = 8;

// One thread per (sequence, channel), as gdn_conv_kernel; the window walks the row's ancestors.
__global__ void gdn_conv_tree_kernel(const uint16_t* __restrict__ x, int64_t ldx, const int32_t* __restrict__ parent,
                                     const int32_t* __restrict__ seq_start, const int32_t* __restrict__ slots,
                                     int channels, int kernel, const uint16_t* __restrict__ weight,
                                     const uint16_t* __restrict__ state, uint16_t* __restrict__ out) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int s = blockIdx.y;
  if (c >= channels) return;
  const int t0 = seq_start[s], t1 = seq_start[s + 1];
  const uint16_t* st = state + (int64_t(slots[s]) * channels + c) * (kernel - 1);
  float wt[kMaxKernel];
  for (int j = 0; j < kernel; ++j) wt[j] = bf(weight[int64_t(c) * kernel + j]);
  for (int t = t0; t < t1; ++t) {
    // hist[kernel - 2] is the parent's input, hist[0] the oldest
    float hist[kMaxKernel];
    int cur = t, beyond = 0;
    for (int j = kernel - 2; j >= 0; --j) {
      const int p = cur >= 0 ? parent[cur] : -1;
      if (p >= 0) {
        hist[j] = bf(x[int64_t(p) * ldx + c]);
        cur = p;
      } else {
        hist[j] = bf(st[kernel - 2 - beyond]);
        ++beyond;
        cur = -1;
      }
    }
    const float xt = bf(x[int64_t(t) * ldx + c]);
    float acc = wt[kernel - 1] * xt;
    for (int j = 0; j < kernel - 1; ++j) acc += wt[j] * hist[j];
    out[int64_t(t) * channels + c] = to_bf(siluf(bf(to_bf(acc))));
  }
}

constexpr int kDk = 128;
constexpr int kSlice = 32;
constexpr int kRows = kDk / 4;
constexpr int kPad = kRows + 1;

// gdn_recurrent_kernel (gdn_recurrent.cu) over a depth-first program of tree rows; saved states per level in scratch
// ([cta][level][kRows][128 threads], each thread its own column of registers).
__global__ void __launch_bounds__(128) gdn_recurrent_tree_kernel(
    const uint16_t* __restrict__ qkv, const float* __restrict__ g, const float* __restrict__ beta,
    const int32_t* __restrict__ prog, const int32_t* __restrict__ seq_start, const int32_t* __restrict__ slots,
    int k_heads, int v_heads, int dv, const float* __restrict__ state, float* __restrict__ scratch, int levels,
    uint16_t* __restrict__ out) {
  __shared__ float sq[4 * kPad], sk[4 * kPad], sv[kSlice], sgb[2];
  const int s = blockIdx.x, vh = blockIdx.y, slice = blockIdx.z;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int col = warp * 8 + (lane & 7), rg = lane >> 3;
  const int j = slice * kSlice + col;
  const int kh = vh / (v_heads / k_heads);
  const int64_t ld = int64_t(2) * k_heads * kDk + int64_t(v_heads) * dv;
  const float* st = state + (int64_t(slots[s]) * v_heads + vh) * kDk * dv;
  float* sc = scratch + ((int64_t(s) * v_heads + vh) * gridDim.z + slice) * levels * kRows * 128;

  float S[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) S[r] = st[int64_t(rg * kRows + r) * dv + j];

  const float q_scale = rsqrtf(float(kDk));
  const int i0 = seq_start[s], i1 = seq_start[s + 1];
  for (int i = i0; i < i1; ++i) {
    const int t = prog[3 * i], load = prog[3 * i + 1], save = prog[3 * i + 2];
    if (load >= 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) S[r] = sc[(int64_t(load) * kRows + r) * 128 + tid];
    }
    const uint16_t* row = qkv + int64_t(t) * ld;
    __syncthreads();
    sq[(tid / kRows) * kPad + tid % kRows] = bf(row[kh * kDk + tid]);
    sk[(tid / kRows) * kPad + tid % kRows] = bf(row[k_heads * kDk + kh * kDk + tid]);
    if (tid < kSlice) sv[tid] = bf(row[2 * k_heads * kDk + vh * dv + slice * kSlice + tid]);
    if (tid == 0) {
      sgb[0] = __expf(g[int64_t(t) * v_heads + vh]);
      sgb[1] = beta[int64_t(t) * v_heads + vh];
    }
    __syncthreads();

    const float* q = sq + rg * kPad;
    const float* k = sk + rg * kPad;
    float qq = 0.0f, kk = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) qq += q[r] * q[r], kk += k[r] * k[r];
    qq += __shfl_xor_sync(0xffffffffu, qq, 8);
    qq += __shfl_xor_sync(0xffffffffu, qq, 16);
    kk += __shfl_xor_sync(0xffffffffu, kk, 8);
    kk += __shfl_xor_sync(0xffffffffu, kk, 16);
    const float inv_q = rsqrtf(qq + 1e-6f) * q_scale, inv_k = rsqrtf(kk + 1e-6f);
    const float decay = sgb[0], b = sgb[1];

    float kv = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      S[r] *= decay;
      kv += S[r] * k[r];
    }
    kv += __shfl_xor_sync(0xffffffffu, kv, 8);
    kv += __shfl_xor_sync(0xffffffffu, kv, 16);
    const float delta = (sv[col] - kv * inv_k) * b;
    const float dk_scaled = delta * inv_k;
    float o = 0.0f;
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      S[r] += k[r] * dk_scaled;
      o += S[r] * q[r];
    }
    o += __shfl_xor_sync(0xffffffffu, o, 8);
    o += __shfl_xor_sync(0xffffffffu, o, 16);
    if (rg == 0) out[(int64_t(t) * v_heads + vh) * dv + j] = to_bf(o * inv_q);
    if (save >= 0) {
#pragma unroll
      for (int r = 0; r < kRows; ++r) sc[(int64_t(save) * kRows + r) * 128 + tid] = S[r];
    }
  }
}

// grid: buffers; each thread copies 4-byte words of every pair in order.
__global__ void move_rows_kernel(const blz_row_buf* __restrict__ bufs, const int32_t* __restrict__ pairs, int n) {
  const blz_row_buf b = bufs[blockIdx.x];
  const int64_t words = b.row_bytes / 4;
  uint32_t* p = static_cast<uint32_t*>(b.ptr);
  for (int64_t w = threadIdx.x; w < words; w += blockDim.x)
    for (int i = 0; i < n; ++i) p[int64_t(pairs[2 * i + 1]) * words + w] = p[int64_t(pairs[2 * i]) * words + w];
}

constexpr int kTopkWarps = 4;
constexpr int kMaxCand = 1024;

// one warp per row: softmax over the candidates, then k argmax passes
__global__ void __launch_bounds__(32 * kTopkWarps) candidate_topk_kernel(const float* __restrict__ logits,
                                                                         const int32_t* __restrict__ ids, int rows,
                                                                         int c, int k, float* __restrict__ probs,
                                                                         int64_t* __restrict__ tokens) {
  __shared__ float vals[kTopkWarps][kMaxCand];
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int r = blockIdx.x * kTopkWarps + warp;
  if (r >= rows) return;
  float* v = vals[warp];
  const float* lr = logits + int64_t(r) * c;
  float mx = -FLT_MAX;
  for (int i = lane; i < c; i += 32) {
    v[i] = lr[i];
    mx = fmaxf(mx, v[i]);
  }
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  float sum = 0.0f;
  for (int i = lane; i < c; i += 32) sum += v[i] > -FLT_MAX ? __expf(v[i] - mx) : 0.0f;
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  __syncwarp();
  for (int p = 0; p < k; ++p) {
    float best = -INFINITY;
    int bi = 0x7fffffff;
    for (int i = lane; i < c; i += 32)
      if (v[i] > best || (v[i] == best && i < bi)) best = v[i], bi = i;
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ob > best || (ob == best && oi < bi)) best = ob, bi = oi;
    }
    if (lane == 0) {
      const bool ok = bi < c && best > -FLT_MAX;
      probs[int64_t(r) * k + p] = ok ? __expf(best - mx) / sum : 0.0f;
      tokens[int64_t(r) * k + p] = ok ? int64_t(ids[int64_t(r) * c + bi]) : 0;
      if (bi < c) v[bi] = -INFINITY;
    }
    __syncwarp();
  }
}

__device__ __forceinline__ int tree_slot(const blz_draft_tree& t, int item, int depth, int lane) {
  return t.slot_base + (item * t.max_depth + depth - 1) * t.top_k + lane;
}

// rows of `dim` bf16 values: dst row r = src row
__device__ __forceinline__ void copy_row(uint16_t* dst, const uint16_t* src, int dim) {
  if ((dim & 1) == 0) {
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    for (int i = threadIdx.x; i < dim / 2; i += blockDim.x) d[i] = s[i];
  } else {
    for (int i = threadIdx.x; i < dim; i += blockDim.x) dst[i] = src[i];
  }
}

// grid: items
__global__ void tree_begin_kernel(blz_draft_tree t, const float* __restrict__ probs, const int64_t* __restrict__ tokens,
                                  const uint16_t* __restrict__ root_hidden, int dim, uint16_t* __restrict__ front_hidden) {
  const int i = blockIdx.x, K = t.top_k, l = threadIdx.x;
  if (l < K) {
    const float p = probs[i * K + l];
    const int64_t tok = tokens[i * K + l];
    t.scores[int64_t(i) * t.cap + l] = p;
    t.tokens[int64_t(i) * t.cap + l] = tok;
    t.parents[int64_t(i) * t.cap + l] = -1;
    t.front_scores[i * K + l] = p;
    t.front_tokens[i * K + l] = tok;
    t.front_node[i * K + l] = l;
    t.ancestors[(i * K + l) * t.max_depth] = tree_slot(t, i, 1, l);
  }
  for (int r = 0; r < K; ++r) copy_row(front_hidden + int64_t(i * K + r) * dim, root_hidden + int64_t(i) * dim, dim);
}

constexpr int kMaxTopK = 16;
constexpr int kMaxDepth = 32;

// grid: items, block 256 (K * K <= 256)
__global__ void tree_step_kernel(blz_draft_tree t, int step, const float* __restrict__ probs,
                                 const int64_t* __restrict__ tokens, const uint16_t* __restrict__ hidden, int dim,
                                 uint16_t* __restrict__ front_hidden) {
  __shared__ float ss[kMaxTopK * kMaxTopK];
  __shared__ int src_lane[kMaxTopK];
  __shared__ int32_t anc[kMaxTopK][kMaxDepth];
  const int i = blockIdx.x, K = t.top_k, KK = K * K, tid = threadIdx.x;
  const int base = K + (step - 1) * KK;
  float s = 0.0f;
  int64_t tok = 0;
  int node = 0;
  if (tid < KK) {
    const int l = tid / K;
    s = t.front_scores[i * K + l] * probs[int64_t(i * K + l) * K + tid % K];
    tok = tokens[int64_t(i * K + l) * K + tid % K];
    node = base + tid;
    t.scores[int64_t(i) * t.cap + node] = s;
    t.tokens[int64_t(i) * t.cap + node] = tok;
    t.parents[int64_t(i) * t.cap + node] = t.front_node[i * K + l];
    ss[tid] = s;
  }
  for (int e = tid; e < K * t.max_depth; e += blockDim.x) anc[e / t.max_depth][e % t.max_depth] = t.ancestors[i * K * t.max_depth + e];
  __syncthreads();
  if (tid < KK) {
    int rank = 0;
    for (int u = 0; u < KK; ++u) rank += ss[u] > s || (ss[u] == s && u < tid);
    if (rank < K) {
      t.front_scores[i * K + rank] = s;
      t.front_tokens[i * K + rank] = tok;
      t.front_node[i * K + rank] = node;
      src_lane[rank] = tid / K;
    }
  }
  __syncthreads();
  for (int e = tid; e < K * t.max_depth; e += blockDim.x) {
    const int r = e / t.max_depth, d = e % t.max_depth;
    int32_t v = d < step ? anc[src_lane[r]][d] : 0;
    if (d == step) v = tree_slot(t, i, step + 1, r);
    t.ancestors[i * K * t.max_depth + e] = v;
  }
  for (int r = 0; r < K; ++r)
    copy_row(front_hidden + int64_t(i * K + r) * dim, hidden + int64_t(i * K + src_lane[r]) * dim, dim);
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_gdn_conv_tree(const uint16_t* x, int64_t ldx, const int32_t* parent, const int32_t* seq_start,
                                         const int32_t* slots, int seqs, int channels, int kernel,
                                         const uint16_t* weight, const uint16_t* state, uint16_t* out,
                                         cudaStream_t stream) {
  if (seqs == 0) return cudaSuccess;
  if (kernel < 2 || kernel > blaze::kMaxKernel) return cudaErrorInvalidValue;
  const dim3 grid(unsigned((channels + 127) / 128), unsigned(seqs));
  blaze::gdn_conv_tree_kernel<<<grid, 128, 0, stream>>>(x, ldx, parent, seq_start, slots, channels, kernel, weight,
                                                        state, out);
  return cudaGetLastError();
}

extern "C" size_t blz_gdn_tree_scratch_floats(int seqs, int v_heads, int dv, int levels) {
  return size_t(seqs) * v_heads * (dv / blaze::kSlice) * levels * blaze::kRows * 128;
}

extern "C" cudaError_t blz_gdn_recurrent_tree(const uint16_t* qkv, const float* g, const float* beta,
                                              const int32_t* prog, const int32_t* seq_start, const int32_t* slots,
                                              int seqs, int k_heads, int v_heads, int dk, int dv, const float* state,
                                              float* scratch, int levels, uint16_t* out, cudaStream_t stream) {
  if (seqs == 0) return cudaSuccess;
  if (dk != blaze::kDk || dv % blaze::kSlice || v_heads % k_heads || levels < 1) return cudaErrorInvalidValue;
  const dim3 grid(unsigned(seqs), unsigned(v_heads), unsigned(dv / blaze::kSlice));
  blaze::gdn_recurrent_tree_kernel<<<grid, 128, 0, stream>>>(qkv, g, beta, prog, seq_start, slots, k_heads, v_heads,
                                                             dv, state, scratch, levels, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_move_rows(const blz_row_buf* bufs, int nbuf, const int32_t* pairs, int n,
                                     cudaStream_t stream) {
  if (nbuf == 0 || n == 0) return cudaSuccess;
  blaze::move_rows_kernel<<<nbuf, 256, 0, stream>>>(bufs, pairs, n);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_candidate_topk(const float* logits, const int32_t* ids, int rows, int c, int k, float* probs,
                                          int64_t* tokens, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  if (c < 1 || c > blaze::kMaxCand || k < 1 || k > 16 || k > c) return cudaErrorInvalidValue;
  blaze::candidate_topk_kernel<<<(rows + blaze::kTopkWarps - 1) / blaze::kTopkWarps, 32 * blaze::kTopkWarps, 0,
                                 stream>>>(logits, ids, rows, c, k, probs, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_tree_begin(blz_draft_tree t, int items, const float* probs, const int64_t* tokens,
                                      const uint16_t* root_hidden, int dim, uint16_t* front_hidden,
                                      cudaStream_t stream) {
  if (items == 0) return cudaSuccess;
  if (t.top_k < 1 || t.top_k > blaze::kMaxTopK || t.max_depth < 1 || t.max_depth > blaze::kMaxDepth)
    return cudaErrorInvalidValue;
  blaze::tree_begin_kernel<<<items, 256, 0, stream>>>(t, probs, tokens, root_hidden, dim, front_hidden);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_tree_step(blz_draft_tree t, int items, int step, const float* probs, const int64_t* tokens,
                                     const uint16_t* hidden, int dim, uint16_t* front_hidden, cudaStream_t stream) {
  if (items == 0) return cudaSuccess;
  if (t.top_k < 1 || t.top_k > blaze::kMaxTopK || step < 1 || step >= t.max_depth || t.max_depth > blaze::kMaxDepth)
    return cudaErrorInvalidValue;
  blaze::tree_step_kernel<<<items, 256, 0, stream>>>(t, step, probs, tokens, hidden, dim, front_hidden);
  return cudaGetLastError();
}
