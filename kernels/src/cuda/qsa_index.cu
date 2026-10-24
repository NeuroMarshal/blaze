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

// Tensor-core scores of a prefill span (one sequence): S = Q K^T with Q [rows * heads, 128] (the heads of a row
// consecutive) and K the span's compressed keys [blocks, 128]; per CTA 32 rows x 4 heads (8 m16 tiles) against 64
// blocks (8 n8 tiles), Q and K in shared memory, ldmatrix + mma.m16n8k16 bf16 -> f32; relu and the sum over the four
// heads in the fragments (the heads of a row sit in four consecutive fragment rows: lanes 4 and 8 apart).
constexpr int kTcRows = 32, kTcBlocks = 64, kTcHeads = 4, kTcStride = 128 + 8;  // bf16 per smem row (padding)

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const uint16_t* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const uint16_t* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// grid: (block tiles, row tiles), block: 128 (4 warps; warp w: m tiles 2w, 2w + 1, all 8 n tiles)
__global__ void __launch_bounds__(128) scores_tc_kernel(const uint16_t* __restrict__ q,
                                                        const int64_t* __restrict__ positions,
                                                        const int32_t* __restrict__ tok_row, int row0, int rows,
                                                        int ratio, const int32_t* __restrict__ rt, int64_t rt_stride,
                                                        const uint16_t* __restrict__ comp, int max_blocks,
                                                        float* __restrict__ scores, int64_t ld) {
  extern __shared__ __align__(16) uint16_t tc_smem[];  // Q [32 * 4][136], K [64][136]
  uint16_t* sq = tc_smem;
  uint16_t* sk = tc_smem + kTcRows * kTcHeads * kTcStride;
  const int r0 = blockIdx.y * kTcRows, b0 = blockIdx.x * kTcBlocks;
  const int nr = min(kTcRows, rows - r0);
  const int seq = tok_row[row0];
  // the last row of the tile sees the most blocks (one sequence, ascending positions)
  const int64_t vis_last = (positions[row0 + r0 + nr - 1] + 1) / ratio;
  const float scale = rsqrtf(128.0f);
  if (b0 >= vis_last) {
    for (int i = threadIdx.x; i < nr * kTcBlocks; i += blockDim.x) {
      const int r = i / kTcBlocks, b = b0 + i % kTcBlocks;
      if (b < max_blocks) scores[int64_t(r0 + r) * ld + b] = -FLT_MAX;
    }
    return;
  }
  // Q rows (row, head) and K rows (block), 16 bytes per thread step
  for (int i = threadIdx.x; i < kTcRows * kTcHeads * 16; i += blockDim.x) {
    const int m = i / 16, c = i % 16;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (m / kTcHeads < nr) v = *reinterpret_cast<const uint4*>(q + (int64_t(row0 + r0) * kTcHeads + m) * 128 + c * 8);
    *reinterpret_cast<uint4*>(sq + m * kTcStride + c * 8) = v;
  }
  for (int i = threadIdx.x; i < kTcBlocks * 16; i += blockDim.x) {
    const int n = i / 16, c = i % 16, b = b0 + n;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (b < vis_last && b < max_blocks)
      v = *reinterpret_cast<const uint4*>(comp + int64_t(rt[int64_t(seq) * rt_stride + int64_t(b) * ratio] / ratio) * 128 + c * 8);
    *reinterpret_cast<uint4*>(sk + n * kTcStride + c * 8) = v;
  }
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  float acc[2][8][4] = {};
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    uint32_t a[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const int m = (2 * warp + mt) * 16 + (lane & 15);
      ldsm_x4(a[mt], sq + m * kTcStride + ks * 16 + (lane >> 4) * 8);
    }
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
      uint32_t bb[2];
      const int n = nt * 8 + (lane & 7);
      ldsm_x2(bb, sk + n * kTcStride + ks * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) mma_bf16(acc[mt][nt], a[mt], bb);
    }
  }
  // fragment: lane = 4 gr + c holds rows gr, gr + 8 (m = 4 row + head) and columns 2c, 2c + 1
  const int gr = lane / 4, c = lane % 4;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        float v = fmaxf(acc[mt][nt][e], 0.0f);
        v += __shfl_xor_sync(0xffffffffu, v, 4);
        v += __shfl_xor_sync(0xffffffffu, v, 8);
        if ((gr & 3) == 0) {
          // m = 16 (2 warp + mt) + gr + 8 (e >> 1): tile row m / 4
          const int r = (16 * (2 * warp + mt) + gr + 8 * (e >> 1)) / 4;
          const int b = b0 + nt * 8 + 2 * c + (e & 1);
          if (r < nr && b < max_blocks) {
            const int64_t vis = (positions[row0 + r0 + r] + 1) / ratio;
            scores[int64_t(r0 + r) * ld + b] = b < vis ? v * scale : -FLT_MAX;
          }
        }
      }
}

constexpr int kSelThreads = 1024;
constexpr int kBins = 2048;      // radix digits of 11 bits (the last one of 10)
constexpr int kCand = 2048;      // candidates the fast path keeps
constexpr int kBitWords = 8192;  // row bitmap of the fast path: rows of up to 262144 blocks
constexpr int kUnroll = 8;       // independent loads in flight per thread

struct Cand {
  uint32_t key;
  int idx;
};

// dynamic smem: bitmap (the first histogram lives there before it, the result after it) | candidates | level histograms
constexpr size_t kSelSmem = size_t(kBitWords) * 4 + kCand * sizeof(Cand) + (kBins + kBins / 2) * 4;

__device__ __forceinline__ uint32_t fkey(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive scan of one int per thread over the block; returns the thread's offset and sets *total. One barrier: every
// warp scans the warp sums itself; `Slot` gives each call site its own sums, so that no barrier is needed between two.
template <int Slot>
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
  int s = warp_sums[lane];  // kSelThreads / 32 == 32
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, s, o);
    if (lane >= o) s += y;
  }
  *total = __shfl_sync(0xffffffffu, s, 31);
  return __shfl_sync(0xffffffffu, s, (warp + 31) & 31) * (warp > 0) + x - v;
}

// The largest bin b of hist[0, NB) with at least `need` elements in the bins >= b; *above = elements in the bins > b,
// *in_bin = elements in b. Warp 0 scans it: lane l owns the bins NB - (l + 1) * NB / 32 .. NB - l * NB / 32 - 1. The
// lanes sum their bins starting at different vectors (rotated), which keeps the 16-byte loads free of bank conflicts;
// the lane holding the answer walks its bins from the top.
template <int NB>
__device__ int find_bin(const int* hist, int need, int* above, int* in_bin) {
  constexpr int vecs = NB / 32 / 4;
  __shared__ int found[3];
  if (threadIdx.x < 32) {
    const int lane = threadIdx.x;
    const int4* mine = reinterpret_cast<const int4*>(hist + NB - (lane + 1) * (NB / 32));
    int sum = 0;
#pragma unroll
    for (int j = 0; j < vecs; ++j) {
      const int4 q = mine[(j + lane) % vecs];
      sum += q.x + q.y + q.z + q.w;
    }
    int x = sum;  // inclusive scan from lane 0 (the top bins) down
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, x, o);
      if (lane >= o) x += y;
    }
    const int before = x - sum;
    if (before < need && x >= need) {
      int4 q[vecs];
#pragma unroll
      for (int j = 0; j < vecs; ++j) q[j] = mine[j];
      int c = before;
#pragma unroll
      for (int j = vecs - 1; j >= 0; --j) {
        const int hh[4] = {q[j].w, q[j].z, q[j].y, q[j].x};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (c < need && c + hh[e] >= need) {
            found[0] = NB - (lane + 1) * (NB / 32) + 4 * j + 3 - e, found[1] = c, found[2] = hh[e];
          }
          c += hh[e];
        }
      }
    }
  }
  __syncthreads();
  const int bin = found[0];
  *above = found[1];
  *in_bin = found[2];
  __syncthreads();
  return bin;
}

// Bitmap of the blocks with fkey(score) >= t1: word w holds blocks 32 w .. 32 w + 31, the ballot of one warp's loads.
// Non-negative t1 > 0x80000000 only needs the float bits compared as signed integers (a negative float is below any).
template <bool NonNeg>
__device__ void mark_ge(const float* __restrict__ sc, int nvis, uint32_t t1, uint32_t* bm) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = (nvis + 31) >> 5;
  const int ti = int(t1 ^ 0x80000000u);
  auto ge = [&](float v) { return NonNeg ? int(__float_as_uint(v)) >= ti : fkey(v) >= t1; };
  const int full = nvis / (kSelThreads * kUnroll) * (kSelThreads * kUnroll);
  for (int base = 0; base < full; base += kSelThreads * kUnroll) {
    float v[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) v[u] = sc[base + u * kSelThreads + tid];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const uint32_t m = __ballot_sync(0xffffffffu, ge(v[u]));
      if (lane == 0) bm[(base >> 5) + u * 32 + warp] = m;
    }
  }
  for (int i = full + tid; i < ((nvis + kSelThreads - 1) / kSelThreads) * kSelThreads; i += kSelThreads) {
    const uint32_t m = __ballot_sync(0xffffffffu, i < nvis && ge(sc[i]));
    if (lane == 0 && (i >> 5) < nw) bm[i >> 5] = m;
  }
}

// One block per row: the best block_topk visible blocks (largest score; the lowest blocks among equal scores), taken in
// block order (above the threshold, then the lowest ties), expanded to KV slots, plus the tail.
// Radix select: a histogram of the top 11 bits of the ordered float keys gives the bin of the block_topk-th key; the
// blocks from that bin up are marked in a bitmap (a ballot per 32 blocks), listed in block order and settled in shared
// memory with two more radix levels. When more than kCand blocks share the best bins the levels run over the row.
__global__ void __launch_bounds__(kSelThreads) select_kernel(const float* __restrict__ scores, int64_t ld,
                                                             const int64_t* __restrict__ positions,
                                                             const int32_t* __restrict__ tok_row, int row0, int ratio,
                                                             int block_topk, const int32_t* __restrict__ rt,
                                                             int64_t rt_stride, int32_t* __restrict__ table,
                                                             int64_t table_stride, int32_t* __restrict__ counts) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  uint32_t* bm = reinterpret_cast<uint32_t*>(smem_raw);
  int* hist = reinterpret_cast<int*>(smem_raw);
  int* sel = reinterpret_cast<int*>(smem_raw);
  Cand* cand = reinterpret_cast<Cand*>(smem_raw + kBitWords * 4);
  int* lh = reinterpret_cast<int*>(smem_raw + kBitWords * 4 + kCand * sizeof(Cand));  // kBins + kBins / 2 bins
  const int r = blockIdx.x, row = row0 + r, tid = threadIdx.x;
  const int64_t pos = positions[row];
  const int nvis = int((pos + 1) / ratio);
  const int seq = tok_row[row];
  const float* sc = scores ? scores + int64_t(r) * ld : nullptr;
  const int K = block_topk;
  int nsel;
  if (sc == nullptr || nvis <= K) {
    for (int i = tid; i < nvis; i += kSelThreads) sel[i] = i;
    nsel = nvis;
  } else {
    for (int i = tid; i < kBins; i += kSelThreads) hist[i] = 0;
    __syncthreads();
    {
      const int full = nvis / (kSelThreads * kUnroll) * (kSelThreads * kUnroll);
      for (int base = 0; base < full; base += kSelThreads * kUnroll) {
        float v[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) v[u] = sc[base + u * kSelThreads + tid];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) atomicAdd(&hist[fkey(v[u]) >> 21], 1);
      }
      for (int i = full + tid; i < nvis; i += kSelThreads) atomicAdd(&hist[fkey(sc[i]) >> 21], 1);
    }
    __syncthreads();
    int above1, in_bin1;
    const int b1 = find_bin<kBins>(hist, K, &above1, &in_bin1);
    if (above1 + in_bin1 <= kCand && nvis <= kBitWords * 32) {
      // blocks with a key in bin b1 or above: the bitmap word of 32 blocks is the warp's ballot
      const uint32_t t1 = uint32_t(b1) << 21;
      const int nw = (nvis + 31) >> 5;
      if (t1 >= 0x80000000u) mark_ge<true>(sc, nvis, t1, bm);
      else mark_ge<false>(sc, nvis, t1, bm);
      for (int i = tid; i < kBins + kBins / 2; i += kSelThreads) lh[i] = 0;
      __syncthreads();
      // candidates in block order: thread t lists the set bits of its words
      const int wp = (nw + kSelThreads - 1) / kSelThreads;
      const int w0 = min(nw, tid * wp), w1 = min(nw, w0 + wp);
      int cnt = 0;
      for (int w = w0; w < w1; ++w) cnt += __popc(bm[w]);
      int n;
      int o = block_scan<0>(cnt, &n);
      for (int w = w0; w < w1; ++w) {
        uint32_t m = bm[w];
        while (m) {
          cand[o++].idx = w * 32 + __ffs(m) - 1;
          m &= m - 1;
        }
      }
      __syncthreads();
      for (int e = tid; e < n; e += kSelThreads) cand[e].key = fkey(sc[cand[e].idx]);
      __syncthreads();
      // two more radix levels among the candidates
      const int need1 = K - above1;
      for (int e = tid; e < n; e += kSelThreads) {
        const uint32_t k = cand[e].key;
        if ((k >> 21) == uint32_t(b1)) atomicAdd(&lh[(k >> 10) & 2047], 1);
      }
      __syncthreads();
      int above2, in_bin2;
      const int b2 = find_bin<kBins>(lh, need1, &above2, &in_bin2);
      const uint32_t p2 = t1 | (uint32_t(b2) << 10);
      for (int e = tid; e < n; e += kSelThreads) {
        const uint32_t k = cand[e].key;
        if ((k >> 10) == (p2 >> 10)) atomicAdd(&lh[kBins + (k & 1023)], 1);
      }
      __syncthreads();
      int above3, in_bin3;
      const int b3 = find_bin<kBins / 2>(lh + kBins, need1 - above2, &above3, &in_bin3);
      const uint32_t thr = p2 | uint32_t(b3);
      // above thr, then the lowest ties, in block order: entries 2 tid, 2 tid + 1 of the list
      int idx2[2];
      uint32_t k2[2];
      int v = 0;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int e = 2 * tid + j;
        idx2[j] = e < n ? cand[e].idx : 0;
        k2[j] = e < n ? cand[e].key : 0;
        v += (e < n && k2[j] > thr) + ((e < n && k2[j] == thr) << 16);
      }
      int tot;
      const int off = block_scan<1>(v, &tot);
      const int n_gt = tot & 0xffff, ties = K - n_gt;
      int og = off & 0xffff, oe = off >> 16;
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        const int e = 2 * tid + j;
        if (e < n && k2[j] > thr) sel[og++] = idx2[j];
        else if (e < n && k2[j] == thr) {
          if (oe < ties) sel[n_gt + oe] = idx2[j];
          ++oe;
        }
      }
      nsel = K;
    } else {
      // many blocks in the best bins: the next levels over the row, then the collection
      uint32_t prefix = uint32_t(b1) << 21, mask = 0xFFE00000u;
      int need = K - above1;
      for (int pass = 0; pass < 2; ++pass) {
        const int sh = pass == 0 ? 10 : 0, bins = pass == 0 ? 2048 : 1024;
        for (int i = tid; i < kBins; i += kSelThreads) lh[i] = 0;
        __syncthreads();
        for (int i = tid; i < nvis; i += kSelThreads) {
          const uint32_t k = fkey(sc[i]);
          if ((k & mask) == prefix) atomicAdd(&lh[(k >> sh) & (bins - 1)], 1);
        }
        __syncthreads();
        int above, in_bin;
        const int bin = pass == 0 ? find_bin<kBins>(lh, need, &above, &in_bin) : find_bin<kBins / 2>(lh, need, &above, &in_bin);
        need -= above;
        prefix |= uint32_t(bin) << sh;
        mask |= uint32_t(bins - 1) << sh;
      }
      const uint32_t thr = prefix;
      const int ties = need;  // elements equal to thr to take
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
      const int off_gt = block_scan<2>(gt, &n_gt);
      const int off_eq = block_scan<3>(eq, &n_eq);
      int og = off_gt, oe = off_eq;
      for (int i = a; i < b; ++i) {
        const uint32_t k = fkey(sc[i]);
        if (k > thr) sel[og++] = i;
        else if (k == thr) {
          if (oe < ties) sel[n_gt + oe] = i;
          ++oe;
        }
      }
      nsel = n_gt + min(n_eq, ties);
    }
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
  if (same_seq && heads == kTcHeads && rows >= 16) {
    const dim3 grid(unsigned((max_blocks + kTcBlocks - 1) / kTcBlocks), unsigned((rows + kTcRows - 1) / kTcRows));
    const size_t smem = size_t(kTcRows * kTcHeads + kTcBlocks) * kTcStride * 2;
    // per device: the opt-in to the large shared memory is a property of the kernel on each GPU
    static bool attr[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 16 || !attr[dev]) {
      cudaFuncSetAttribute(scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
      if (dev < 16) attr[dev] = true;
    }
    scores_tc_kernel<<<grid, 128, smem, stream>>>(q, positions, tok_row, row0, rows, ratio, req_to_token, rt_stride,
                                                  comp, max_blocks, scores, ld_scores);
    return cudaGetLastError();
  }
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
  static bool attr[16] = {};
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev < 16 && !attr[dev]) {
    cudaFuncSetAttribute(select_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSelSmem));
    attr[dev] = true;
  }
  select_kernel<<<rows, kSelThreads, kSelSmem, stream>>>(scores, ld_scores, positions, tok_row, row0, ratio, block_topk,
                                                  req_to_token, rt_stride, table + int64_t(row0) * table_stride,
                                                  table_stride, counts + row0);
  return cudaGetLastError();
}
