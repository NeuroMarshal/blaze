// Wide route of the hybrid MoE (blaze/moe_wide.h): exact W4A16 on int8 tensor cores.
//
// Fragment mapping (mma.m16n8k16, row.col, int8): lane l holds A rows l/4 and l/4 + 8 at k = 4 (l % 4) .. + 3 and
// B column l/4 at the same k. In an rg16 unit, code word L (byte 4L) holds rows L % 8 (low nibbles) and L % 8 + 8
// (high nibbles) at k = 4 (L / 8) .. + 3, so lane l reads word (l % 4) * 8 + l / 4. D: c0, c1 = row l/4, columns
// 2 (l % 4) and +1; c2, c3 = row l/4 + 8, same columns.
#include <cuda_runtime.h>

#include "blaze/canon.h"
#include "blaze/expert.h"
#include "blaze/moe_wide.h"

namespace blaze::wide {
namespace {

constexpr int kWarps = 4;   // row groups per block
constexpr int kNTiles = BLZ_WIDE_TILE_COLS / 8;
constexpr int kUnroll = 4;  // units in flight per lane

// Four 4-bit codes (one per byte, 0..15) -> four signed bytes c2(code) (same table as the narrow route).
__device__ __forceinline__ uint32_t c2x4(uint32_t codes) {
  const uint32_t idx = codes & 0x07070707u;
  const uint32_t t = idx | (idx >> 4);
  const uint32_t sel = (t & 0xFFu) | ((t >> 8) & 0xFF00u);
  const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel);  // {0,1,2,3,4,6,8,12}
  const uint32_t neg = ((codes >> 3) & 0x01010101u) * 0xFFu;
  return (mag & ~neg) | (__vneg4(mag) & neg);
}

__device__ __forceinline__ uint32_t ld_u32(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned*>(p)); }

__device__ __forceinline__ void mma_su(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

__device__ __forceinline__ void mma_ss(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

__device__ __forceinline__ int64_t shfl_xor_i64(int64_t v, int m) {
  const int lo = __shfl_xor_sync(0xffffffffu, int(v & 0xffffffff), m);
  const int hi = __shfl_xor_sync(0xffffffffu, int(v >> 32), m);
  return (int64_t(hi) << 32) | uint32_t(lo);
}

// int64 sums of one 16-row group of `matrix` (k_dim / 16 units) against up to 32 columns, whose limb planes start at
// col_planes[nt] (plane stride k_dim). acc[nt][j]: j = 0, 1 row l/4, columns 2 (l % 4) + j; j = 2, 3 row l/4 + 8.
template <bool kA8>
__device__ __forceinline__ void group_sums(const uint8_t* __restrict__ group, int blocks, int k_dim,
                                           const uint8_t* const (&col_planes)[kNTiles], int ntiles,
                                           int64_t (&acc)[kNTiles][4]) {
  const int lane = threadIdx.x & 31;
  const int word = (lane & 3) * 8 + (lane >> 2);
  const int r = lane >> 2;
  const int koff = (lane & 3) * 4;
  float facc[kNTiles][4];
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[nt][j] = 0, facc[nt][j] = 0.0f;
  for (int b0 = 0; b0 < blocks; b0 += kUnroll) {
    uint32_t w[kUnroll];
    int32_t s0[kUnroll], s1[kUnroll];
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (b0 + u < blocks) {
        const uint8_t* un = group + size_t(b0 + u) * expert::kUnitBytes;
        w[u] = ld_u32(un + 4 * word);
        s0[u] = canon::s_hat(un[expert::kUnitCodeBytes + r]);
        s1[u] = canon::s_hat(un[expert::kUnitCodeBytes + r + 8]);
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      if (b0 + u >= blocks) break;
      const uint32_t a0 = c2x4(w[u] & 0x0F0F0F0Fu), a1 = c2x4((w[u] >> 4) & 0x0F0F0F0Fu);
      const int k = 16 * (b0 + u) + koff;
#pragma unroll
      for (int nt = 0; nt < kNTiles; ++nt) {
        if (nt >= ntiles) break;
        const uint8_t* p = col_planes[nt] + k;
        if constexpr (kA8) {
          // D columns 2 (l % 4) + (j & 1); lane 4c holds column c's block shift. FP32 FMAs instead of int64 MACs
          // (the bottleneck of this loop): ~1e-7 relative rounding per term, far below the A8 input error.
          int32_t d[4];
          mma_ss(d, a0, a1, ld_u32(p));
          const int sh_own = col_planes[nt][k_dim + b0 + u];
          const float p0 = __int_as_float((127 + __shfl_sync(0xffffffffu, sh_own, 8 * (lane & 3))) << 23);
          const float p1 = __int_as_float((127 + __shfl_sync(0xffffffffu, sh_own, 8 * (lane & 3) + 4)) << 23);
          const float fs0 = float(s0[u]), fs1 = float(s1[u]);
          facc[nt][0] = fmaf(float(d[0]), fs0 * p0, facc[nt][0]);
          facc[nt][1] = fmaf(float(d[1]), fs0 * p1, facc[nt][1]);
          facc[nt][2] = fmaf(float(d[2]), fs1 * p0, facc[nt][2]);
          facc[nt][3] = fmaf(float(d[3]), fs1 * p1, facc[nt][3]);
          continue;
        }
        int32_t dl[4], dm[4], dh[4];
        mma_su(dl, a0, a1, ld_u32(p));
        mma_su(dm, a0, a1, ld_u32(p + k_dim));
        mma_ss(dh, a0, a1, ld_u32(p + 2 * k_dim));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          // Exact P in wrapping int32 (|P| < 2^31), as limb_dot in the narrow route.
          const int32_t pv = int32_t(uint32_t(dl[j]) + (uint32_t(dm[j]) << 8) + (uint32_t(dh[j]) << 16));
          acc[nt][j] += int64_t(pv) * (j < 2 ? s0[u] : s1[u]);
        }
      }
    }
  }
  if constexpr (kA8) {
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[nt][j] = __float2ll_rn(facc[nt][j]);
  }
}

// grid (gu_groups / kWarps, n_tiles); warp = one 16-row group of the gate/up matrix (rows 2p gate, 2p + 1 up).
__global__ void __launch_bounds__(32 * kWarps) gate_up_kernel(blz_wide_batch bt) {
  const int2 tile = bt.tiles[blockIdx.y];
  const blz_wide_item it = bt.items[tile.x];
  const expert::Geometry geo{bt.hidden, bt.inter};
  const int g = blockIdx.x * kWarps + threadIdx.x / 32;
  if (g >= geo.gu_groups()) return;
  const int lane = threadIdx.x & 31;
  const int H = bt.hidden, I = bt.inter;
  const int cols = min(BLZ_WIDE_TILE_COLS, it.ncols - tile.y);
  const int ntiles = (cols + 7) / 8;
  const uint8_t* col_planes[kNTiles];
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
    const int c = min(nt * 8 + (lane >> 2), cols - 1);  // padding columns repeat the last one (not stored)
    const int token = bt.route_token[it.route_begin + tile.y + c];
    col_planes[nt] = bt.x_planes + size_t(token) * 3 * H;
  }
  int64_t acc[kNTiles][4];
  if (bt.a8)
    group_sums<true>(it.record + expert::unit_offset(g, 0, geo.gu_blocks()), geo.gu_blocks(), H, col_planes, ntiles, acc);
  else
    group_sums<false>(it.record + expert::unit_offset(g, 0, geo.gu_blocks()), geo.gu_blocks(), H, col_planes, ntiles, acc);
  const float mg = bt.mult[size_t(it.expert) * 3], mu = bt.mult[size_t(it.expert) * 3 + 1];
  const int r = lane >> 2;
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
    if (nt >= ntiles) break;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int64_t partner = shfl_xor_i64(acc[nt][j], 4);  // row r ^ 1: gate <-> up
      if (r & 1) continue;
      const int c = nt * 8 + 2 * (lane & 3) + (j & 1);
      if (c >= cols) continue;
      const int route = it.route_begin + tile.y + c;
      const int x_e = bt.x_emax[bt.route_token[route]];
      const int row = 16 * g + r + (j < 2 ? 0 : 8);  // even: gate row of p = row / 2
      uint16_t v = canon::kCanonicalNaN;
      if (x_e >= 0) {
        const uint16_t gv = canon::a16_row_output(acc[nt][j], x_e, mg);
        const uint16_t uv = canon::a16_row_output(partner, x_e, mu);
        v = canon::swiglu_bf16(gv, uv);
      }
      bt.h[size_t(route - bt.route0) * I + row / 2] = v;
    }
  }
}

// One block per route: exponent and A16 limb planes of h (as the narrow finalize).
__global__ void h_encode_kernel(blz_wide_batch bt) {
  const int lr = blockIdx.x;
  const int I = bt.inter;
  const uint16_t* h = bt.h + size_t(lr) * I;
  __shared__ int s_emax;
  int e = 1;
  bool nonfinite = false;
  for (int p = threadIdx.x; p < I; p += blockDim.x) {
    const uint16_t v = h[p];
    nonfinite |= canon::bf16_nonfinite(v);
    e = max(e, canon::a16_exp(v));
  }
  e = __reduce_max_sync(0xffffffffu, e);
  nonfinite = __any_sync(0xffffffffu, nonfinite);
  if (threadIdx.x == 0) s_emax = 0;
  __syncthreads();
  if ((threadIdx.x & 31) == 0) atomicMax(&s_emax, nonfinite ? 0x7fffffff : e);
  __syncthreads();
  const int emax = s_emax == 0x7fffffff ? -1 : s_emax;
  if (threadIdx.x == 0) bt.h_emax[lr] = emax;
  uint8_t* planes = bt.h_planes + size_t(lr) * 3 * I;
  for (int p = threadIdx.x; p < I; p += blockDim.x) {
    const int32_t v = emax < 0 ? 0 : canon::a16_encode(h[p], emax);
    planes[p] = uint8_t(v & 0xFF);
    planes[I + p] = uint8_t((v >> 8) & 0xFF);
    planes[2 * I + p] = uint8_t((v >> 16) & 0xFF);
  }
}

// grid (d_groups / kWarps, n_tiles); warp = one 16-row group of the down matrix.
__global__ void __launch_bounds__(32 * kWarps) down_kernel(blz_wide_batch bt) {
  const int2 tile = bt.tiles[blockIdx.y];
  const blz_wide_item it = bt.items[tile.x];
  const expert::Geometry geo{bt.hidden, bt.inter};
  const int g = blockIdx.x * kWarps + threadIdx.x / 32;
  if (g >= geo.d_groups()) return;
  const int lane = threadIdx.x & 31;
  const int H = bt.hidden, I = bt.inter;
  const int cols = min(BLZ_WIDE_TILE_COLS, it.ncols - tile.y);
  const int ntiles = (cols + 7) / 8;
  const uint8_t* col_planes[kNTiles];
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
    const int c = min(nt * 8 + (lane >> 2), cols - 1);
    col_planes[nt] = bt.h_planes + size_t(it.route_begin + tile.y + c - bt.route0) * 3 * I;
  }
  int64_t acc[kNTiles][4];
  const uint8_t* down = it.record + geo.gu_bytes() + expert::unit_offset(g, 0, geo.d_blocks());
  if (bt.a8)
    group_sums<true>(down, geo.d_blocks(), I, col_planes, ntiles, acc);
  else
    group_sums<false>(down, geo.d_blocks(), I, col_planes, ntiles, acc);
  const float md = bt.mult[size_t(it.expert) * 3 + 2];
  const int r = lane >> 2;
#pragma unroll
  for (int nt = 0; nt < kNTiles; ++nt) {
    if (nt >= ntiles) break;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = nt * 8 + 2 * (lane & 3) + (j & 1);
      if (c >= cols) continue;
      const int lr = it.route_begin + tile.y + c - bt.route0;
      const int he = bt.h_emax[lr];
      const int row = 16 * g + r + (j < 2 ? 0 : 8);
      bt.y[size_t(lr) * H + row] = he < 0 ? canon::kCanonicalNaN : canon::a16_row_output(acc[nt][j], he, md);
    }
  }
}

// One block per token: emax and limb planes (same encoding as the narrow plan kernel).
__global__ void encode_kernel(const uint16_t* __restrict__ x, int H, uint8_t* planes_out, int32_t* emax_out) {
  const int t = blockIdx.x, tid = threadIdx.x, chunks = H / 8;
  const uint4* xv = reinterpret_cast<const uint4*>(x + size_t(t) * H);
  __shared__ int s_emax;
  int e = 1;
  bool nonfinite = false;
  for (int c = tid; c < chunks; c += blockDim.x) {
    const uint4 v = xv[c];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t b = uint16_t(w[i / 2] >> (16 * (i % 2)));
      nonfinite |= canon::bf16_nonfinite(b);
      e = max(e, canon::a16_exp(b));
    }
  }
  e = __reduce_max_sync(0xffffffffu, e);
  nonfinite = __any_sync(0xffffffffu, nonfinite);
  if (tid == 0) s_emax = 0;
  __syncthreads();
  if ((tid & 31) == 0) atomicMax(&s_emax, nonfinite ? 0x7fffffff : e);
  __syncthreads();
  const int emax = s_emax;
  uint8_t* planes = planes_out + size_t(t) * 3 * H;
  for (int c = tid; c < chunks; c += blockDim.x) {
    const uint4 v = xv[c];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t lo[2] = {0, 0}, mid[2] = {0, 0}, hi[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t b = uint16_t(w[i / 2] >> (16 * (i % 2)));
      const int32_t q = emax == 0x7fffffff ? 0 : canon::a16_encode(b, emax);
      const int sh = 8 * (i % 4);
      lo[i / 4] |= uint32_t(q & 0xFF) << sh;
      mid[i / 4] |= uint32_t((q >> 8) & 0xFF) << sh;
      hi[i / 4] |= uint32_t((q >> 16) & 0xFF) << sh;
    }
    const size_t o = size_t(c) * 8;
    *reinterpret_cast<uint2*>(planes + o) = make_uint2(lo[0], lo[1]);
    *reinterpret_cast<uint2*>(planes + H + o) = make_uint2(mid[0], mid[1]);
    *reinterpret_cast<uint2*>(planes + 2 * H + o) = make_uint2(hi[0], hi[1]);
  }
  if (tid == 0) emax_out[t] = emax == 0x7fffffff ? -1 : emax;
}

// One block per row, one thread per 16-value block: all three limb planes are read before any byte is rewritten.
__global__ void to_a8_kernel(uint8_t* planes, int k_dim) {
  uint8_t* row = planes + size_t(blockIdx.x) * 3 * k_dim;
  const int b = threadIdx.x, blocks = k_dim / 16;
  int32_t x[16];
  int32_t amax = 0;
  if (b < blocks) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      const int k = 16 * b + j;
      x[j] = int32_t(row[k]) | (int32_t(row[k_dim + k]) << 8) | (int32_t(int8_t(row[2 * k_dim + k])) * 65536);
      amax = max(amax, abs(x[j]));
    }
  }
  __syncthreads();
  if (b >= blocks) return;
  int sh = 0;
  while ((amax >> sh) > 127) ++sh;
  const int32_t half = sh ? 1 << (sh - 1) : 0;
#pragma unroll
  for (int j = 0; j < 16; ++j) row[16 * b + j] = uint8_t(int8_t(min(127, max(-127, (x[j] + half) >> sh))));
  row[k_dim + b] = uint8_t(sh);
}

// One thread per (token, 8 outputs): all K route outputs loaded, then summed in k order (as the narrow combine).
__global__ void combine_kernel(const int32_t* __restrict__ route_of, const float* __restrict__ w,
                               const uint16_t* __restrict__ y, int topk, int H, uint16_t* __restrict__ out) {
  const int t = blockIdx.y;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= H / 8) return;
  constexpr int kMaxK = 16;
  uint4 yv[kMaxK];
#pragma unroll
  for (int k = 0; k < kMaxK; ++k) {
    if (k >= topk) break;
    yv[k] = *reinterpret_cast<const uint4*>(y + size_t(route_of[size_t(t) * topk + k]) * H + size_t(c) * 8);
  }
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int k = 0; k < kMaxK; ++k) {
    if (k >= topk) break;
    const float wk = w[size_t(t) * topk + k];
    const uint32_t yw[4] = {yv[k].x, yv[k].y, yv[k].z, yv[k].w};
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[i] = __fadd_rn(acc[i], __fmul_rn(wk, canon::bf16_to_f32(uint16_t(yw[i / 2] >> (16 * (i % 2))))));
  }
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    o[i] = uint32_t(canon::bf16_rn(acc[2 * i])) | (uint32_t(canon::bf16_rn(acc[2 * i + 1])) << 16);
  *reinterpret_cast<uint4*>(out + size_t(t) * H + size_t(c) * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

}  // namespace
}  // namespace blaze::wide

using namespace blaze::wide;

extern "C" cudaError_t blz_wide_encode(const uint16_t* x, int tokens, int hidden, uint8_t* planes, int32_t* emax,
                                       cudaStream_t stream) {
  if (tokens <= 0) return cudaSuccess;
  if (hidden % 16 != 0) return cudaErrorInvalidValue;
  encode_kernel<<<tokens, 256, 0, stream>>>(x, hidden, planes, emax);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_wide_to_a8(uint8_t* planes, int rows, int k_dim, cudaStream_t stream) {
  if (rows <= 0) return cudaSuccess;
  if (k_dim % 16 != 0 || k_dim / 16 > 1024) return cudaErrorInvalidValue;
  to_a8_kernel<<<rows, (k_dim / 16 + 31) / 32 * 32, 0, stream>>>(planes, k_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_wide_gate_up(const blz_wide_batch* bt, cudaStream_t stream) {
  if (bt->n_tiles <= 0) return cudaSuccess;
  const blaze::expert::Geometry geo{bt->hidden, bt->inter};
  const dim3 grid((geo.gu_groups() + kWarps - 1) / kWarps, bt->n_tiles);
  gate_up_kernel<<<grid, 32 * kWarps, 0, stream>>>(*bt);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_wide_h_encode(const blz_wide_batch* bt, cudaStream_t stream) {
  if (bt->n_routes <= 0) return cudaSuccess;
  h_encode_kernel<<<bt->n_routes, 128, 0, stream>>>(*bt);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_wide_down(const blz_wide_batch* bt, cudaStream_t stream) {
  if (bt->n_tiles <= 0) return cudaSuccess;
  const blaze::expert::Geometry geo{bt->hidden, bt->inter};
  const dim3 grid((geo.d_groups() + kWarps - 1) / kWarps, bt->n_tiles);
  down_kernel<<<grid, 32 * kWarps, 0, stream>>>(*bt);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_wide_combine(const int32_t* route_of, const float* w, const uint16_t* y, int tokens,
                                        int topk, int hidden, uint16_t* out, cudaStream_t stream) {
  if (tokens <= 0) return cudaSuccess;
  if (hidden % 8 != 0 || topk > 16) return cudaErrorInvalidValue;
  constexpr int kBlock = 64;
  const dim3 grid((hidden / 8 + kBlock - 1) / kBlock, tokens);
  combine_kernel<<<grid, kBlock, 0, stream>>>(route_of, w, y, topk, hidden, out);
  return cudaGetLastError();
}
