// Dense attention straight from an NVFP4, FP8 or BF16 KV pool (see blaze/mha.h).
//
// Decode / verify, FlashAttention-2 style with the dequantization in registers. One warp = one 16-row MMA tile of
// query rows (the request's query tokens x the KV head's q heads, GQA 6: 2 tokens x 6 heads, each row with its own
// causal limit) and one part of the context; a CTA holds the MT tiles of a request's query tokens times KW parts, so
// the tiles' warps read the same keys (L1). A warp walks its part in chunks of 16 positions: the raw codes and block
// scales go from the pool (slots through req_to_token) straight into registers, E2M1 pairs convert to f16x2 (one
// instruction on SM100/SM120, cvt.rn.f16x2.e2m1x2) times the E4M3 block scale (exact in FP16), and feed m16n8k16 MMAs
// with an online softmax whose probabilities stay in registers as the A operand of P V. No shared memory but Q, no
// barrier in the loop. Head dims are permuted so that every thread's bytes are contiguous: in k-step s the B rows 2c,
// 2c+1, 2c+8, 2c+9 of thread c are dims 64c + 4s + 0..3 (bytes 32c + 2s, +1), and Q is stored in shared memory in
// that order; output tiles 2j, 2j+1 column n are dims 32n + 2j, +1 (V byte 16n + j). The next chunk's K loads are
// issued before the softmax and P V of the current one, its V loads before its Q K^T. The global scales go into q (K)
// and the output (V). Each warp leaves its unnormalized output, running max (log2 units) and sum; a second kernel
// merges the parts.
//
// FP8 and BF16 pools go through the same kernel (the format a template parameter): FP8 E4M3 pairs convert to f16x2
// exactly (cvt.rn.f16x2.e4m3x2), BF16 pairs through FP32 (exact in FP16 for |x| in [2^-14, 65504], the range of
// normed keys and values). FP8 chunks are prefetched like NVFP4 ones (twice the registers); BF16 rows are read where
// they are used (prefetching 64 registers per side would not fit next to O).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <climits>
#include <type_traits>

#include "fa_tile.cuh"
#include "nvfp4_attn.cuh"
#include "blaze/mha.h"

namespace blaze {
namespace {

using namespace attn;

__device__ __forceinline__ uint32_t pack_f16(float lo, float hi) {
  const __half2 v = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// FP16 accumulation (blz_mha_set_f16_accumulate): twice the MMA rate of FP32 accumulation on GeForce GPUs.
__device__ __forceinline__ void mma_f16h(uint32_t d[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(d[0]), "+r"(d[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void add_h2(float& x, float& y, uint32_t h) {
  const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
  x += f.x, y += f.y;
}

// Two E2M1 codes (one byte, first in the low nibble) as f16x2.
__device__ __forceinline__ __half2 e2m1x2_to_f16x2(uint32_t byte) {
#if __CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__
  const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t(byte & 0xff), __NV_E2M1);
  return *reinterpret_cast<const __half2*>(&r);
#else
  // Code s e1 e0 m into FP16 bits s << 15 | e1 e0 m << 9: times 2^14 that is the E2M1 value (0.5 a subnormal).
  const uint32_t bits = ((byte & 0x8u) << 12) | ((byte & 0x7u) << 9) | ((byte & 0x80u) << 24) | ((byte & 0x70u) << 21);
  return __hmul2(*reinterpret_cast<const __half2*>(&bits), __half2half2(__float2half_rn(16384.0f)));
#endif
}

// Two E4M3 values (low byte first) as f16x2, exactly.
__device__ __forceinline__ __half2 e4m3x2_to_f16x2(uint32_t two) {
  const __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(two & 0xffffu), __NV_E4M3);
  return *reinterpret_cast<const __half2*>(&r);
}

// Two bf16 values (low half first) as f16x2.
__device__ __forceinline__ __half2 bf16x2_to_f16x2(uint32_t two) {
  return __float22half2_rn(__bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&two)));
}

__device__ __forceinline__ int64_t load_index(blz_index_ptr p, int64_t i) {
  return p.type == BLZ_I64 ? __ldg(static_cast<const int64_t*>(p.ptr) + i)
                           : int64_t(__ldg(static_cast<const int32_t*>(p.ptr) + i));
}

// Four 8x8 b16 matrices from shared memory; lane l gives the row address of matrix l / 8.
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const __half* p) {
  const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}

// f16 per Q row in shared memory (padding: conflict-free ldmatrix), head dim D
template <int D>
constexpr int q_stride() { return D + 8; }
constexpr int kChunk = 16;            // positions per warp iteration (two n8 tiles of S, one k-step of P V)
// The running max of a row moves only when a score exceeds it by this much (log2 units), so O is rescaled rarely; the
// probabilities stay below 2^8, well inside FP16 (the lazy rescaling of FlashAttention 4).
constexpr float kRescaleSlack = 8.0f;

// Warps per CTA along the context for MT query tiles: 4 warps in all where the tiles allow.
__host__ __device__ constexpr int parts_per_cta(int mt) { return mt == 1 ? 4 : mt == 2 ? 2 : 1; }

// Head dim of logical column r (0..15) of k-step s: thread c's B rows 2c, 2c+1, 2c+8, 2c+9 are (D/4) c + 4s + 0..3.
template <int D>
__device__ __forceinline__ int permuted_dim(int s, int r) {
  return (D / 4) * ((r & 7) >> 1) + 4 * s + 2 * (r >> 3) + (r & 1);
}

__device__ __forceinline__ __half2 scale_h2(uint32_t e4m3_byte) {
  return __half2half2(__float2half_rn(e4m3_to_float(uint8_t(e4m3_byte))));
}

__device__ __forceinline__ uint32_t h2_bits(__half2 v) { return *reinterpret_cast<const uint32_t*>(&v); }

// Raw bytes of one thread for one chunk. K: positions gr and 8 + gr, bytes [32c, 32c + 32) and scale bytes
// [4c, 4c + 4). V: positions 2c, 2c + 1, 2c + 8, 2c + 9, bytes [16 gr, 16 gr + 16) and scale bytes [2 gr, 2 gr + 2).
struct ChunkK {
  uint4 c[2][2];
  uint32_t sc[2];
};
struct ChunkV {
  uint4 c[4];
  uint32_t sc[4];
};

// Pool row (slot * heads + head) of chunk position i, -1 past the part.
__device__ __forceinline__ int64_t chunk_row(int32_t slot_lane, int i) {
  const int32_t r = __shfl_sync(0xffffffffu, slot_lane, i);
  return r;
}

__device__ __forceinline__ void load_k(const blz_nvfp4_pool& pool, int64_t row, int c, int n, ChunkK& k) {
  if (row < 0) {
    k.c[n][0] = k.c[n][1] = make_uint4(0, 0, 0, 0), k.sc[n] = 0;
    return;
  }
  const uint4* src = reinterpret_cast<const uint4*>(pool.k_data + row * (kDim / 2) + 32 * c);
  k.c[n][0] = __ldg(src), k.c[n][1] = __ldg(src + 1);
  k.sc[n] = __ldg(reinterpret_cast<const uint32_t*>(pool.k_scale + row * (kDim / kNvfp4Block) + 4 * c));
}

__device__ __forceinline__ void load_v(const blz_nvfp4_pool& pool, int64_t row, int gr, int i, ChunkV& v) {
  if (row < 0) {
    v.c[i] = make_uint4(0, 0, 0, 0), v.sc[i] = 0;
    return;
  }
  v.c[i] = __ldg(reinterpret_cast<const uint4*>(pool.v_data + row * (kDim / 2) + 16 * gr));
  v.sc[i] = __ldg(reinterpret_cast<const uint16_t*>(pool.v_scale + row * (kDim / kNvfp4Block) + 2 * gr));
}

__device__ __forceinline__ uint32_t word_of(const uint4& u, int w) {
  return w == 0 ? u.x : w == 1 ? u.y : w == 2 ? u.z : u.w;
}

// FP8 chunks: K bytes [64c, 64c + 64) of positions gr and 8 + gr; V bytes [32 gr, 32 gr + 32) of positions 2c, 2c + 1,
// 2c + 8, 2c + 9 (the same dims as the NVFP4 shares, one byte per value).
struct ChunkK8 {
  uint4 c[2][4];
};
struct ChunkV8 {
  uint4 c[4][2];
};
struct NoChunk {};

template <int F>
struct Chunks {
  using K = NoChunk;
  using V = NoChunk;
};
template <>
struct Chunks<BLZ_KV_NVFP4> {
  using K = ChunkK;
  using V = ChunkV;
};
template <>
struct Chunks<BLZ_KV_FP8> {
  using K = ChunkK8;
  using V = ChunkV8;
};

template <int F, typename CK>
__device__ __forceinline__ void load_k_any(const blz_nvfp4_pool& pool, int64_t row, int c, int n, CK& k) {
  if constexpr (F == BLZ_KV_NVFP4) {
    load_k(pool, row, c, n, k);
  } else if constexpr (F == BLZ_KV_FP8) {
    const uint4* src = reinterpret_cast<const uint4*>(pool.k_data + row * kDim + 64 * c);
#pragma unroll
    for (int j = 0; j < 4; ++j) k.c[n][j] = row < 0 ? make_uint4(0, 0, 0, 0) : __ldg(src + j);
  }
}

template <int F, typename CV>
__device__ __forceinline__ void load_v_any(const blz_nvfp4_pool& pool, int64_t row, int gr, int i, CV& v) {
  if constexpr (F == BLZ_KV_NVFP4) {
    load_v(pool, row, gr, i, v);
  } else if constexpr (F == BLZ_KV_FP8) {
    const uint4* src = reinterpret_cast<const uint4*>(pool.v_data + row * kDim + 32 * gr);
#pragma unroll
    for (int j = 0; j < 2; ++j) v.c[i][j] = row < 0 ? make_uint4(0, 0, 0, 0) : __ldg(src + j);
  }
}

// BF16 pools: 8 values (16 bytes) of a row from `d`, zeros for a row past the part.
template <int D>
__device__ __forceinline__ uint4 load_bf16x8(const uint8_t* data, int64_t row, int d) {
  return row < 0 ? make_uint4(0, 0, 0, 0) : __ldg(reinterpret_cast<const uint4*>(data + (row * D + d) * 2));
}

// grid: (batch * tiles, kv_heads, splits), block: 32 * MT * KW, dynamic shared memory: MT * 16 * q_stride<D> f16.
// F: pool format; D: head dim (256; 128 for BF16 pools); window > 0: a row at position p sees p - window + 1 .. p.
// H: FP16 accumulation (Q K^T in runs of 4 k-steps, P V per step, summed in FP32) with an exact running max, so P <= 1
// and a step of P V stays inside FP16 (16 x 6 x 448 for NVFP4 codes times scales; normed FP8 / BF16 values far below).
template <int F, int D, int MT, int KW, bool H = false>
__global__ void __launch_bounds__(32 * MT * KW) mha_decode_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t sq_tok, int64_t sq_head, int group, int tokens_per_tile, int tiles,
    int q_heads, blz_nvfp4_pool pool, const int32_t* __restrict__ req_to_token, int64_t req_stride,
    blz_index_ptr req_indices, blz_index_ptr kv_lens, int q_len, int splits, float scale, int window, int causal,
    const uint64_t* __restrict__ tree_mask, const int32_t* __restrict__ tail, int tail_len, int tail_stride,
    float* __restrict__ part_o, float* __restrict__ part_ml) {
  static_assert(D == 256 || (D == 128 && F == BLZ_KV_BF16), "head dim 128 only from BF16 pools");
  constexpr int kRows = 16 * MT;
  constexpr int kQStride = q_stride<D>();
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __half* sq = reinterpret_cast<__half*>(smem_raw);
  const int b = blockIdx.x / tiles, qt = blockIdx.x % tiles, g = blockIdx.y, s = blockIdx.z;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, mt = warp % MT, kw = warp / MT;
  const int gr = lane / 4, c = lane % 4;  // MMA fragment row / column group

  const int qi0 = qt * tokens_per_tile, nq = min(tokens_per_tile, q_len - qi0), valid_rows = nq * group;
  const int64_t kv_len = load_index(kv_lens, b), req = load_index(req_indices, b);
  // Rows 16 mt + gr and + 8: query token qi0 + r / group, q head g * group + r % group, positions <= the limit.
  int64_t lim[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = 16 * mt + gr + 8 * h;
    lim[h] = r < valid_rows ? kv_len - q_len + qi0 + r / group : -1;
  }
  // one past the last position any row of the CTA sees (non-causal: the whole sequence, the query block included)
  const int64_t vis_end = causal ? kv_len - q_len + qi0 + nq : kv_len;
  // the first position any row of the CTA sees (its first query token's window)
  const int64_t vis_begin = window > 0 ? max(int64_t(0), kv_len - q_len + qi0 - window + 1) : 0;
  const int64_t span = ((vis_end - vis_begin + splits - 1) / splits + kChunk * KW - 1) / (kChunk * KW) * (kChunk * KW);
  const int64_t t0 = vis_begin + int64_t(s) * span, t1 = min(t0 + span, vis_end);
  // One past the last position each of this thread's rows sees in the part, and the first one (window).
  const int64_t row_end[2] = {causal ? min(t1, lim[0] + 1) : (lim[0] >= 0 ? t1 : 0),
                              causal ? min(t1, lim[1] + 1) : (lim[1] >= 0 ? t1 : 0)};
  const int64_t row_begin[2] = {window > 0 ? lim[0] - window + 1 : 0, window > 0 ? lim[1] - window + 1 : 0};
  const float qscale = scale * kLog2e * __ldg(pool.k_g), vg = __ldg(pool.v_g);
  // Tree verify: the query rows see the step's own positions (the last q_len) only where their ancestor mask has the
  // bit (row 0 of the step at bit 0).
  const int64_t tree_base = kv_len - q_len;
  uint64_t tmask[2] = {~0ull, ~0ull};
  if (tree_mask != nullptr) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = 16 * mt + gr + 8 * h;
      if (r < valid_rows) tmask[h] = __ldg(tree_mask + int64_t(b) * q_len + qi0 + r / group);
    }
  }
  // Positions past kv_len - tail_len come from the row's own slot list (draft tree steps: the path's KV slots).
  const int64_t tail_from = kv_len - tail_len;
  const int32_t* tail_row = tail != nullptr ? tail + int64_t(b) * tail_stride : nullptr;

  // Q of every tile into shared memory in the permuted head-dim order: f16(q * scale * log2 e * K global scale).
  if (t0 < t1) {
    for (int i = threadIdx.x; i < kRows * (D / 2); i += 32 * MT * KW) {
      const int r = i / (D / 2), l = 2 * (i % (D / 2));
      const int d = permuted_dim<D>(l / 16, l % 16);  // l even: d and d + 1
      float x0 = 0.0f, x1 = 0.0f;
      if (r < valid_rows) {
        const __nv_bfloat16* qrow =
            q + (int64_t(b) * q_len + qi0 + r / group) * sq_tok + int64_t(g * group + r % group) * sq_head;
        x0 = __bfloat162float(qrow[d]) * qscale, x1 = __bfloat162float(qrow[d + 1]) * qscale;
      }
      *reinterpret_cast<uint32_t*>(&sq[r * kQStride + l]) = pack_f16(x0, x1);
    }
  }
  __syncthreads();

  float o[D / 8][4];
#pragma unroll
  for (int nt = 0; nt < D / 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

  const int32_t* r2t = req_to_token + req * req_stride;
  const int64_t head_rows = pool.heads;
  // Slot rows of chunk positions: lane i (and i + 16) holds position i.
  auto slots_of = [&](int64_t tc) -> int32_t {
    const int64_t t = tc + (lane & 15);
    if (t >= t1) return -1;
    const int32_t sl = tail_row != nullptr && t >= tail_from ? __ldg(tail_row + (t - tail_from)) : __ldg(r2t + t);
    return int32_t(sl * head_rows + g);
  };
  typename Chunks<F>::K kc;
  typename Chunks<F>::V vc;
  int64_t tc = t0 + int64_t(kw) * kChunk;
  int32_t slot = tc < t1 ? slots_of(tc) : -1;
  if (tc < t1) {
#pragma unroll
    for (int n = 0; n < 2; ++n) load_k_any<F>(pool, chunk_row(slot, 8 * n + gr), c, n, kc);
#pragma unroll
    for (int i = 0; i < 4; ++i) load_v_any<F>(pool, chunk_row(slot, 2 * c + (i & 1) + 8 * (i >> 1)), gr, i, vc);
  }
  for (; tc < t1; tc += int64_t(kChunk) * KW) {
    // S = Q K^T over the chunk's 16 positions (two n8 tiles), two accumulation chains per tile.
    float sacc[2][2][4] = {};
    uint32_t hacc[2][2] = {};  // H: the current run of 4 k-steps, per n8 tile
    {
      __half2 ks[2][4];
      if constexpr (F == BLZ_KV_NVFP4) {
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int j = 0; j < 4; ++j) ks[n][j] = scale_h2(kc.sc[n] >> (8 * j));
      }
      int64_t krow[2] = {-1, -1};
      if constexpr (F == BLZ_KV_BF16) {
#pragma unroll
        for (int n = 0; n < 2; ++n) krow[n] = chunk_row(slot, 8 * n + gr);
      }
      const int arow = lane & 15, acol = (lane >> 4) * 8;
      uint4 kb[2];  // BF16: dims (D/4) c + 4 ks_ .. + 7 of both positions, for k-steps ks_ and ks_ + 1
#pragma unroll
      for (int ks_ = 0; ks_ < D / 16; ++ks_) {
        uint32_t qa[4];
        ldsm_x4(qa, &sq[(16 * mt + arow) * kQStride + 16 * ks_ + acol]);
        if constexpr (F == BLZ_KV_BF16) {
          if ((ks_ & 1) == 0) {
#pragma unroll
            for (int n = 0; n < 2; ++n) kb[n] = load_bf16x8<D>(pool.k_data, krow[n], (D / 4) * c + 4 * ks_);
          }
        }
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          uint32_t b0, b1;
          if constexpr (F == BLZ_KV_NVFP4) {
            const uint32_t w = word_of(kc.c[n][ks_ / 8], (ks_ / 2) % 4) >> (16 * (ks_ & 1));
            b0 = h2_bits(__hmul2(e2m1x2_to_f16x2(w), ks[n][ks_ / 4]));
            b1 = h2_bits(__hmul2(e2m1x2_to_f16x2(w >> 8), ks[n][ks_ / 4]));
          } else if constexpr (F == BLZ_KV_FP8) {
            const uint32_t w = word_of(kc.c[n][ks_ / 4], ks_ % 4);
            b0 = h2_bits(e4m3x2_to_f16x2(w));
            b1 = h2_bits(e4m3x2_to_f16x2(w >> 16));
          } else {
            b0 = h2_bits(bf16x2_to_f16x2(word_of(kb[n], 2 * (ks_ & 1))));
            b1 = h2_bits(bf16x2_to_f16x2(word_of(kb[n], 2 * (ks_ & 1) + 1)));
          }
          if constexpr (H) {
            mma_f16h(hacc[n], qa, b0, b1);
          } else {
            mma_f16(sacc[ks_ & 1][n], qa, b0, b1);
          }
        }
        if constexpr (H) {
          if ((ks_ & 3) == 3) {
#pragma unroll
            for (int n = 0; n < 2; ++n) {
              add_h2(sacc[0][n][0], sacc[0][n][1], hacc[n][0]);
              add_h2(sacc[0][n][2], sacc[0][n][3], hacc[n][1]);
              hacc[n][0] = hacc[n][1] = 0;
            }
          }
        }
      }
    }
    // The next chunk's slots and K, in flight during the softmax and P V.
    const int64_t tn = tc + int64_t(kChunk) * KW;
    const int32_t slot_next = tn < t1 ? slots_of(tn) : -1;
    if (tn < t1) {
#pragma unroll
      for (int n = 0; n < 2; ++n) load_k_any<F>(pool, chunk_row(slot_next, 8 * n + gr), c, n, kc);
    }

    // Online softmax; rows gr (h = 0) and gr + 8 (h = 1); this thread's positions 8n + 2c, + 1.
    uint32_t pa[4];
    {
      // Positions of the chunk visible to rows gr and gr + 8 (16: all of them).
      int vis[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int64_t v = row_end[h] - tc;
        vis[h] = v >= kChunk ? kChunk : v < 0 ? 0 : int(v);
      }
      float sv[2][4];
#pragma unroll
      for (int n = 0; n < 2; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const float x = sacc[0][n][e] + sacc[1][n][e];
          const int idx = 8 * n + 2 * c + (e & 1);
          const int64_t rel = tc + idx - tree_base;
          const bool in_tree = rel < 0 || ((tmask[e >> 1] >> (rel & 63)) & 1);
          sv[n][e] = idx < vis[e >> 1] && tc + idx >= row_begin[e >> 1] && in_tree ? x : -INFINITY;
        }
      float p[2][4], alpha[2];
      bool grow = false;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float cm = fmaxf(fmaxf(sv[0][2 * h], sv[0][2 * h + 1]), fmaxf(sv[1][2 * h], sv[1][2 * h + 1]));
        cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 1));
        cm = fmaxf(cm, __shfl_xor_sync(0xffffffffu, cm, 2));
        alpha[h] = 1.0f;
        if (cm > m[h] + (H ? 0.0f : kRescaleSlack) || (m[h] == -INFINITY && cm != -INFINITY)) {
          alpha[h] = m[h] == -INFINITY ? 0.0f : exp2f(m[h] - cm);
          m[h] = cm;
          grow = true;
        }
        // A row with nothing visible yet keeps its zero state (exp2(-inf - -inf) would be NaN).
        const float base = m[h] == -INFINITY ? 0.0f : m[h];
        float sum = 0.0f;
#pragma unroll
        for (int n = 0; n < 2; ++n)
#pragma unroll
          for (int e = 0; e < 2; ++e) sum += (p[n][2 * h + e] = exp2f(sv[n][2 * h + e] - base));
        l[h] = l[h] * alpha[h] + sum;
      }
      if (__any_sync(0xffffffffu, grow)) {
#pragma unroll
        for (int nt = 0; nt < D / 8; ++nt) {
          o[nt][0] *= alpha[0], o[nt][1] *= alpha[0];
          o[nt][2] *= alpha[1], o[nt][3] *= alpha[1];
        }
      }
      pa[0] = pack_f16(p[0][0], p[0][1]), pa[1] = pack_f16(p[0][2], p[0][3]);
      pa[2] = pack_f16(p[1][0], p[1][1]), pa[3] = pack_f16(p[1][2], p[1][3]);
    }

    // O += P V: B rows (positions) 2c, 2c + 1 / 2c + 8, 2c + 9; byte j of V row bytes [16 gr, +16) holds dims
    // 32 gr + 2j, + 1 of output tiles 2j, 2j + 1.
    int64_t vrow[4] = {-1, -1, -1, -1};
    if constexpr (F == BLZ_KV_BF16) {
#pragma unroll
      for (int i = 0; i < 4; ++i) vrow[i] = chunk_row(slot, 2 * c + (i & 1) + 8 * (i >> 1));
    }
#pragma unroll
    for (int half = 0; half < D / 128; ++half) {
      __half2 vs[4];
      if constexpr (F == BLZ_KV_NVFP4) {
#pragma unroll
        for (int i = 0; i < 4; ++i) vs[i] = scale_h2(vc.sc[i] >> (8 * half));
      }
      uint4 vb[4];  // BF16: dims (D/8) gr + 2j .. + 7 (pairs j .. j + 3) of the four positions
#pragma unroll
      for (int jj = 0; jj < 8; ++jj) {
        const int j = 8 * half + jj;
        if constexpr (F == BLZ_KV_BF16) {
          if (j % 4 == 0) {
#pragma unroll
            for (int i = 0; i < 4; ++i) vb[i] = load_bf16x8<D>(pool.v_data, vrow[i], (D / 8) * gr + 2 * j);
          }
        }
        __half2 f[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          if constexpr (F == BLZ_KV_NVFP4)
            f[i] = __hmul2(e2m1x2_to_f16x2(word_of(vc.c[i], j / 4) >> (8 * (j % 4))), vs[i]);
          else if constexpr (F == BLZ_KV_FP8)
            f[i] = e4m3x2_to_f16x2(word_of(vc.c[i][j / 8], (j / 2) % 4) >> (16 * (j & 1)));
          else
            f[i] = bf16x2_to_f16x2(word_of(vb[i], j % 4));
        }
        if constexpr (H) {
          uint32_t h0[2] = {0, 0}, h1[2] = {0, 0};
          mma_f16h(h0, pa, h2_bits(__lows2half2(f[0], f[1])), h2_bits(__lows2half2(f[2], f[3])));
          mma_f16h(h1, pa, h2_bits(__highs2half2(f[0], f[1])), h2_bits(__highs2half2(f[2], f[3])));
          add_h2(o[2 * j][0], o[2 * j][1], h0[0]), add_h2(o[2 * j][2], o[2 * j][3], h0[1]);
          add_h2(o[2 * j + 1][0], o[2 * j + 1][1], h1[0]), add_h2(o[2 * j + 1][2], o[2 * j + 1][3], h1[1]);
        } else {
          mma_f16(o[2 * j], pa, h2_bits(__lows2half2(f[0], f[1])), h2_bits(__lows2half2(f[2], f[3])));
          mma_f16(o[2 * j + 1], pa, h2_bits(__highs2half2(f[0], f[1])), h2_bits(__highs2half2(f[2], f[3])));
        }
      }
    }
    // The next chunk's V, in flight during its Q K^T.
    if (tn < t1) {
#pragma unroll
      for (int i = 0; i < 4; ++i) load_v_any<F>(pool, chunk_row(slot_next, 2 * c + (i & 1) + 8 * (i >> 1)), gr, i, vc);
    }
    slot = slot_next;
  }

  // This warp's partial results of the valid rows (part s * KW + kw).
  const int parts = splits * KW, part = s * KW + kw;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 1);
    l[h] += __shfl_xor_sync(0xffffffffu, l[h], 2);
    const int r = 16 * mt + gr + 8 * h;
    if (r >= valid_rows) continue;
    const int64_t tok = int64_t(b) * q_len + qi0 + r / group, head = int64_t(g) * group + r % group;
    const int64_t base = (tok * q_heads + head) * parts + part;
    float* orow = part_o + base * D;
#pragma unroll
    for (int j = 0; j < D / 16; ++j) {
      *reinterpret_cast<float2*>(orow + (D / 4) * c + 2 * j) =
          make_float2(o[2 * j][2 * h] * vg, o[2 * j + 1][2 * h] * vg);
      *reinterpret_cast<float2*>(orow + (D / 4) * c + D / 8 + 2 * j) =
          make_float2(o[2 * j][2 * h + 1] * vg, o[2 * j + 1][2 * h + 1] * vg);
    }
    if (c == 0) *reinterpret_cast<float2*>(part_ml + base * 2) = make_float2(m[h], l[h]);
  }
}

template <typename T>
__device__ __forceinline__ T to_half(float v);
template <>
__device__ __forceinline__ __nv_bfloat16 to_half(float v) { return __float2bfloat16_rn(v); }
template <>
__device__ __forceinline__ __half to_half(float v) { return __float2half_rn(v); }

// grid: (tokens, q_heads), block: D (one output value each)
template <typename T, int D>
__global__ void mha_combine_kernel(const float* __restrict__ part_o, const float* __restrict__ part_ml, int splits,
                                   int q_heads, T* __restrict__ out, int64_t so_tok, int64_t so_head) {
  const int64_t base = (int64_t(blockIdx.x) * q_heads + blockIdx.y) * splits;
  const int d = threadIdx.x;
  float mx = -INFINITY;
  for (int s = 0; s < splits; ++s) mx = fmaxf(mx, __ldg(part_ml + (base + s) * 2));
  float sum = 0.0f, acc = 0.0f;
  if (mx != -INFINITY)
    for (int s = 0; s < splits; ++s) {
      const float m = __ldg(part_ml + (base + s) * 2);
      if (m == -INFINITY) continue;
      const float w = exp2f(m - mx);
      sum += __ldg(part_ml + (base + s) * 2 + 1) * w;
      acc += __ldg(part_o + (base + s) * D + d) * w;
    }
  out[int64_t(blockIdx.x) * so_tok + int64_t(blockIdx.y) * so_head + d] = to_half<T>(sum > 0.0f ? acc / sum : 0.0f);
}

// One thread = one 16-value block of one (row, head), K and V.
template <typename T>
__global__ void dequant_rows_kernel(blz_nvfp4_pool pool, blz_index_ptr slots, int64_t n, T* __restrict__ out_k,
                                    T* __restrict__ out_v) {
  const int blocks = pool.dim / kNvfp4Block, per_row = pool.heads * blocks;
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n * per_row) return;
  const int64_t r = i / per_row;
  const int rem = int(i - r * per_row), h = rem / blocks, j = rem % blocks;
  const int64_t prow = load_index(slots, r) * pool.heads + h;
  const int64_t orow = (r * pool.heads + h) * pool.dim + int64_t(j) * kNvfp4Block;
#pragma unroll
  for (int side = 0; side < 2; ++side) {
    const uint8_t* data = side ? pool.v_data : pool.k_data;
    const uint8_t* scale = side ? pool.v_scale : pool.k_scale;
    const float sc = e4m3_to_float(__ldg(scale + prow * blocks + j)) * __ldg(side ? pool.v_g : pool.k_g);
    const uint2 c = __ldg(reinterpret_cast<const uint2*>(data + prow * (pool.dim / 2) + j * (kNvfp4Block / 2)));
    const uint32_t words[2] = {c.x, c.y};
    T vals[kNvfp4Block];
#pragma unroll
    for (int w = 0; w < 2; ++w)
#pragma unroll
      for (int k = 0; k < 8; ++k) vals[8 * w + k] = to_half<T>(e2m1_value((words[w] >> (4 * k)) & 15) * sc);
    uint4* dst = reinterpret_cast<uint4*>((side ? out_v : out_k) + orow);
    dst[0] = *reinterpret_cast<const uint4*>(&vals[0]);
    dst[1] = *reinterpret_cast<const uint4*>(&vals[8]);
  }
}

}  // namespace
}  // namespace blaze

namespace {
// TP ranks load and launch on independent host threads, possibly on different GPU architectures.
thread_local bool g_f16_accumulate = false;
// dense causal NVFP4 calls of more than one MMA tile of rows: 0 the register kernel, 1 / 2 the shared-dequantization
// kernel (mha_verify.cu) with FP32 / FP16 accumulation, 3 / 4 its pair kernel (two warps a tile) with FP32 / FP16,
// 5 / 6 paired warps only above 48 rows (the register kernel is faster for shorter calls on SM120).
thread_local int g_verify = 1;
bool verify_pair() { return g_verify >= 3; }
}

extern "C" void blz_mha_set_f16_accumulate(int on) { g_f16_accumulate = on != 0; }

extern "C" void blz_mha_set_verify(int mode) { g_verify = std::clamp(mode, 0, 6); }

extern "C" int blz_nvfp4_mha_verify_tiles(int q_heads, int kv_heads, int q_len) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads || q_len < 1) return 0;
  if (g_verify >= 5 && int64_t(q_len) * (q_heads / kv_heads) <= 48) return 0;
  return g_verify == 0 ? 0 : blaze::fa::verify_tiles(q_heads, kv_heads, q_len, verify_pair(), nullptr, nullptr);
}

extern "C" int blz_nvfp4_mha_parts(int q_heads, int kv_heads, int q_len, int splits) {
  if (q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 || q_heads / kv_heads > 16 ||
      q_len < 1 || splits < 1 || splits > 64) return 0;
  const int mt = int(std::min<int64_t>(3, (int64_t(q_len) * (q_heads / kv_heads) + 15) / 16));
  return splits * blaze::parts_per_cta(mt);
}

namespace {
cudaError_t mha_launch(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads, blz_nvfp4_pool pool,
                       const int32_t* req_to_token, int64_t req_stride, blz_index_ptr req_indices,
                       blz_index_ptr kv_lens, int batch, int q_len, int splits, float scale, int window, int causal,
                       const uint64_t* tree_mask, const int32_t* tail, int tail_len, int tail_stride,
                       float* partial_o, float* partial_ml, blz_half_t dtype, void* out, int64_t out_stride_tok,
                       int64_t out_stride_head, cudaStream_t stream) {
  if (tree_mask != nullptr && q_len > 64) return cudaErrorNotSupported;
  const bool dim_ok = pool.dim == 256 || (pool.dim == 128 && pool.format == BLZ_KV_BF16);
  if (dtype != BLZ_BF16 || !dim_ok || pool.heads <= 0 || q_heads <= 0 || q_heads % pool.heads != 0 || q_heads / pool.heads > 16 ||
      batch < 0 || q_len < 1 || splits < 1 || splits > 64 || pool.format < BLZ_KV_NVFP4 || pool.format > BLZ_KV_BF16 || window < 0)
    return cudaErrorNotSupported;
  if (batch == 0) return cudaSuccess;
  // Both the final merge grid and query offsets use batch * q_len; reject an overflowing launch before enqueueing.
  if (int64_t(batch) * q_len > INT_MAX) return cudaErrorNotSupported;
  const int group = q_heads / pool.heads;
  if (g_verify != 0 && pool.format == BLZ_KV_NVFP4 && pool.dim == 256 && window == 0 && causal && tree_mask == nullptr &&
      tail == nullptr && q_stride_tok % 8 == 0 && q_stride_head % 8 == 0 && reinterpret_cast<uintptr_t>(q) % 16 == 0 &&
      blz_nvfp4_mha_verify_tiles(q_heads, pool.heads, q_len) > 0) {
    cudaError_t err = blaze::fa::verify_launch(q, q_stride_tok, q_stride_head, q_heads, pool, req_to_token, req_stride,
                                               req_indices, kv_lens, batch, q_len, splits, scale, g_verify % 2 == 0,
                                               verify_pair(), partial_o, partial_ml, stream);
    if (err != cudaSuccess) return err;
    const dim3 cgrid(static_cast<unsigned>(batch * q_len), static_cast<unsigned>(q_heads));
    blaze::mha_combine_kernel<__nv_bfloat16, 256><<<cgrid, 256, 0, stream>>>(
        partial_o, partial_ml, splits, q_heads, static_cast<__nv_bfloat16*>(out), out_stride_tok, out_stride_head);
    return cudaGetLastError();
  }
  // All query tokens of a request in one CTA when they fit three tiles (a verify block of 8 with GQA 6: 48 rows).
  const int mt = int(std::min<int64_t>(3, (int64_t(q_len) * group + 15) / 16));
  const int per_tile = std::max(1, 16 * mt / group);
  const int tiles = 1 + (q_len - 1) / per_tile;
  const dim3 grid(static_cast<unsigned>(batch * tiles), static_cast<unsigned>(pool.heads),
                  static_cast<unsigned>(splits));
  const int D = pool.dim;
  const size_t smem = size_t(16) * mt * (D + 8) * sizeof(__half);
  auto launch = [&](auto kernel, int warps) {
    kernel<<<grid, 32 * warps, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q), q_stride_tok, q_stride_head, group, per_tile, tiles, q_heads, pool,
        req_to_token, req_stride, req_indices, kv_lens, q_len, splits, scale, window, causal, tree_mask, tail,
        tail_len, tail_stride, partial_o, partial_ml);
  };
  auto by_tiles = [&](auto fmt, auto dim) {
    constexpr int F = decltype(fmt)::value, Dc = decltype(dim)::value;
    auto go = [&](auto h) {
      constexpr bool H = decltype(h)::value;
      if (mt == 1)
        launch(blaze::mha_decode_kernel<F, Dc, 1, blaze::parts_per_cta(1), H>, blaze::parts_per_cta(1));
      else if (mt == 2)
        launch(blaze::mha_decode_kernel<F, Dc, 2, blaze::parts_per_cta(2), H>, 2 * blaze::parts_per_cta(2));
      else
        launch(blaze::mha_decode_kernel<F, Dc, 3, blaze::parts_per_cta(3), H>, 3 * blaze::parts_per_cta(3));
    };
    if (g_f16_accumulate) go(std::true_type{}); else go(std::false_type{});
  };
  using D256 = std::integral_constant<int, 256>;
  if (D == 128)
    by_tiles(std::integral_constant<int, BLZ_KV_BF16>{}, std::integral_constant<int, 128>{});
  else if (pool.format == BLZ_KV_FP8)
    by_tiles(std::integral_constant<int, BLZ_KV_FP8>{}, D256{});
  else if (pool.format == BLZ_KV_BF16)
    by_tiles(std::integral_constant<int, BLZ_KV_BF16>{}, D256{});
  else
    by_tiles(std::integral_constant<int, BLZ_KV_NVFP4>{}, D256{});
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return err;
  const dim3 cgrid(static_cast<unsigned>(batch * q_len), static_cast<unsigned>(q_heads));
  const int parts = splits * blaze::parts_per_cta(mt);
  auto* o = static_cast<__nv_bfloat16*>(out);
  if (D == 128)
    blaze::mha_combine_kernel<__nv_bfloat16, 128><<<cgrid, 128, 0, stream>>>(partial_o, partial_ml, parts, q_heads, o,
                                                                            out_stride_tok, out_stride_head);
  else
    blaze::mha_combine_kernel<__nv_bfloat16, 256><<<cgrid, 256, 0, stream>>>(partial_o, partial_ml, parts, q_heads, o,
                                                                            out_stride_tok, out_stride_head);
  return cudaGetLastError();
}
}  // namespace

extern "C" cudaError_t blz_mha_decode_window(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                             blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                             blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len,
                                             int splits, float scale, int window, int causal, float* partial_o,
                                             float* partial_ml, blz_half_t dtype, void* out, int64_t out_stride_tok,
                                             int64_t out_stride_head, cudaStream_t stream) {
  return mha_launch(q, q_stride_tok, q_stride_head, q_heads, pool, req_to_token, req_stride, req_indices, kv_lens,
                    batch, q_len, splits, scale, window, causal, nullptr, nullptr, 0, 0, partial_o, partial_ml, dtype,
                    out, out_stride_tok, out_stride_head, stream);
}

extern "C" cudaError_t blz_mha_decode_tail(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                           blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                           blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int splits,
                                           float scale, int window, const int32_t* tail, int tail_len,
                                           int tail_stride, float* partial_o, float* partial_ml, blz_half_t dtype,
                                           void* out, int64_t out_stride_tok, int64_t out_stride_head,
                                           cudaStream_t stream) {
  return mha_launch(q, q_stride_tok, q_stride_head, q_heads, pool, req_to_token, req_stride, req_indices, kv_lens,
                    batch, 1, splits, scale, window, 1, nullptr, tail, tail_len, tail_stride, partial_o, partial_ml,
                    dtype, out, out_stride_tok, out_stride_head, stream);
}

extern "C" cudaError_t blz_nvfp4_mha_tree(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                          blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                          blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len,
                                          int splits, float scale, const uint64_t* tree_mask, float* partial_o,
                                          float* partial_ml, blz_half_t dtype, void* out, int64_t out_stride_tok,
                                          int64_t out_stride_head, cudaStream_t stream) {
  return mha_launch(q, q_stride_tok, q_stride_head, q_heads, pool, req_to_token, req_stride, req_indices, kv_lens,
                    batch, q_len, splits, scale, 0, 1, tree_mask, nullptr, 0, 0, partial_o, partial_ml, dtype, out,
                    out_stride_tok, out_stride_head, stream);
}

extern "C" cudaError_t blz_nvfp4_mha_decode(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                            blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                            blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len,
                                            int splits, float scale, float* partial_o, float* partial_ml,
                                            blz_half_t dtype, void* out, int64_t out_stride_tok,
                                            int64_t out_stride_head, cudaStream_t stream) {
  return blz_mha_decode_window(q, q_stride_tok, q_stride_head, q_heads, pool, req_to_token, req_stride, req_indices,
                               kv_lens, batch, q_len, splits, scale, 0, 1, partial_o, partial_ml, dtype, out,
                               out_stride_tok, out_stride_head, stream);
}

extern "C" cudaError_t blz_nvfp4_dequant_rows(blz_nvfp4_pool pool, blz_index_ptr slots, int64_t n, blz_half_t dtype,
                                              void* out_k, void* out_v, cudaStream_t stream) {
  if (pool.dim % 16 != 0 || pool.heads <= 0 || pool.format != BLZ_KV_NVFP4) return cudaErrorNotSupported;
  if (n == 0) return cudaSuccess;
  const int64_t threads = n * pool.heads * (pool.dim / blaze::kNvfp4Block);
  const int block = 256;
  const unsigned grid = static_cast<unsigned>((threads + block - 1) / block);
  if (dtype == BLZ_BF16)
    blaze::dequant_rows_kernel<__nv_bfloat16><<<grid, block, 0, stream>>>(
        pool, slots, n, static_cast<__nv_bfloat16*>(out_k), static_cast<__nv_bfloat16*>(out_v));
  else
    blaze::dequant_rows_kernel<__half>
        <<<grid, block, 0, stream>>>(pool, slots, n, static_cast<__half*>(out_k), static_cast<__half*>(out_v));
  return cudaGetLastError();
}
