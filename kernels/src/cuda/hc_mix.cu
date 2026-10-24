// Hyper-connection low-rank mix (blaze/hc_mix.h).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "blaze/hc_mix.h"

namespace blaze {
namespace {

constexpr int kWarps = 8;
constexpr int kDownSplit = 4;               // warps per down row (K quarters), reduced in fixed order
constexpr int kDownRows = kWarps / kDownSplit;
constexpr int kBatch = 8;                   // weight vectors in flight per lane

__device__ __forceinline__ float bf16_bits_to_float(uint32_t bits16) { return __uint_as_float(bits16 << 16); }

__device__ __forceinline__ void unpack8(const uint4& v, float f[8]) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    f[2 * i] = __uint_as_float(w[i] << 16);
    f[2 * i + 1] = __uint_as_float(w[i] & 0xFFFF0000u);
  }
}

__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
  return v;
}

__device__ __forceinline__ uint16_t to_bf16_bits(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }

// Down pass: block = kDownRows rows of w_down; each row's K is split over kDownSplit warps.
template <int kRows>
__global__ void __launch_bounds__(32 * kWarps) hc_down_kernel(const uint16_t* __restrict__ x,
                                                              const uint16_t* __restrict__ w_down,
                                                              uint16_t* __restrict__ t, int rows, int K, int R,
                                                              float inv_hc) {
  __shared__ float part[kWarps][kRows];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * kDownRows + warp / kDownSplit;
  const int q = warp % kDownSplit;
  const int kq = K / kDownSplit, vecs = kq / 8;
  float acc[kRows];
#pragma unroll
  for (int m = 0; m < kRows; ++m) acc[m] = 0.0f;
  if (n < R) {
    const uint4* w = reinterpret_cast<const uint4*>(w_down + size_t(n) * K + size_t(q) * kq);
    const uint16_t* xq = x + size_t(q) * kq;
    for (int base = 0; base < vecs; base += 32 * kBatch) {
      uint4 wv[kBatch];
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const int v = base + b * 32 + lane;
        wv[b] = v < vecs ? ld_stream(w + v) : make_uint4(0, 0, 0, 0);
      }
#pragma unroll
      for (int b = 0; b < kBatch; ++b) {
        const int v = base + b * 32 + lane;
        if (v >= vecs) break;
        float wf[8];
        unpack8(wv[b], wf);
#pragma unroll
        for (int m = 0; m < kRows; ++m) {
          if (m >= rows) break;
          float xf[8];
          unpack8(*reinterpret_cast<const uint4*>(xq + size_t(m) * K + size_t(v) * 8), xf);
          float s = acc[m];
#pragma unroll
          for (int i = 0; i < 8; ++i) s = fmaf(xf[i], wf[i], s);
          acc[m] = s;
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < kRows; ++m) acc[m] = warp_sum(acc[m]);
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < kRows; ++m) part[warp][m] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < kDownRows * kRows) {
    const int r2 = threadIdx.x / kRows, m = threadIdx.x % kRows;
    const int n2 = blockIdx.x * kDownRows + r2;
    if (n2 < R && m < rows) {
      float s = part[r2 * kDownSplit][m];
#pragma unroll
      for (int p = 1; p < kDownSplit; ++p) s += part[r2 * kDownSplit + p][m];
      const float a = s * inv_hc;
      t[size_t(m) * R + n2] = to_bf16_bits(a / (1.0f + __expf(-a)));
    }
  }
}

// Up pass: one warp per output column j; it reads the hc rows (g * hs + j) of w_up.
template <int kRows, int kHc>
__global__ void __launch_bounds__(32 * kWarps) hc_up_kernel(const uint16_t* __restrict__ x,
                                                            const uint16_t* __restrict__ t,
                                                            const uint16_t* __restrict__ w_up,
                                                            uint16_t* __restrict__ out, int rows, int R, int hs,
                                                            float inv_hc) {
  extern __shared__ float ts[];  // [kRows][R]
  for (int i = threadIdx.x; i < rows * R; i += blockDim.x) ts[i] = bf16_bits_to_float(t[i]);
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int j = blockIdx.x * kWarps + warp;
  if (j >= hs) return;
  const int vecs = R / 8;
  float acc[kHc][kRows];
#pragma unroll
  for (int g = 0; g < kHc; ++g)
#pragma unroll
    for (int m = 0; m < kRows; ++m) acc[g][m] = 0.0f;
  for (int base = 0; base < vecs; base += 32) {
    const int v = base + lane;
    uint4 wv[kHc];
#pragma unroll
    for (int g = 0; g < kHc; ++g)
      wv[g] = v < vecs ? ld_stream(reinterpret_cast<const uint4*>(w_up + (size_t(g) * hs + j) * R) + v)
                       : make_uint4(0, 0, 0, 0);
    if (v < vecs) {
#pragma unroll
      for (int g = 0; g < kHc; ++g) {
        float wf[8];
        unpack8(wv[g], wf);
#pragma unroll
        for (int m = 0; m < kRows; ++m) {
          if (m >= rows) break;
          const float* tm = ts + m * R + v * 8;
          float s = acc[g][m];
#pragma unroll
          for (int i = 0; i < 8; ++i) s = fmaf(tm[i], wf[i], s);
          acc[g][m] = s;
        }
      }
    }
  }
#pragma unroll
  for (int g = 0; g < kHc; ++g)
#pragma unroll
    for (int m = 0; m < kRows; ++m) acc[g][m] = warp_sum(acc[g][m]);
  if (lane < kRows && lane < rows) {
    // Every lane holds the reduced sums; lane m writes row m.
    float o = 0.0f;
#pragma unroll
    for (int g = 0; g < kHc; ++g) {
      float u = 0.0f;
#pragma unroll
      for (int m = 0; m < kRows; ++m)
        if (m == lane) u = acc[g][m];
      const float gate = 1.0f / (1.0f + __expf(-u));
      o = fmaf(gate, bf16_bits_to_float(x[size_t(lane) * kHc * hs + size_t(g) * hs + j]), o);
    }
    out[size_t(lane) * hs + j] = to_bf16_bits(o * inv_hc);
  }
}

template <int kRows, int kHc>
cudaError_t launch(const uint16_t* x, const uint16_t* w_down, const uint16_t* w_up, uint16_t* t, uint16_t* out,
                   int rows, int hs, int R, cudaStream_t stream) {
  const int K = kHc * hs;
  const float inv_hc = 1.0f / float(kHc);
  hc_down_kernel<kRows><<<(R + kDownRows - 1) / kDownRows, 32 * kWarps, 0, stream>>>(x, w_down, t, rows, K, R, inv_hc);
  const size_t smem = size_t(kRows) * R * sizeof(float);
  // 16 rows of a 1024 low rank: 64 KiB, past the default 48 KiB of dynamic shared memory
  if (smem > 48 * 1024) cudaFuncSetAttribute(hc_up_kernel<kRows, kHc>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
  hc_up_kernel<kRows, kHc><<<(hs + kWarps - 1) / kWarps, 32 * kWarps, smem, stream>>>(x, t, w_up, out, rows, R, hs,
                                                                                     inv_hc);
  return cudaGetLastError();
}

template <int kHc>
cudaError_t dispatch_rows(const uint16_t* x, const uint16_t* w_down, const uint16_t* w_up, uint16_t* t, uint16_t* out,
                          int rows, int hs, int R, cudaStream_t s) {
  if (rows <= 1) return launch<1, kHc>(x, w_down, w_up, t, out, rows, hs, R, s);
  if (rows <= 2) return launch<2, kHc>(x, w_down, w_up, t, out, rows, hs, R, s);
  if (rows <= 4) return launch<4, kHc>(x, w_down, w_up, t, out, rows, hs, R, s);
  if (rows <= 8) return launch<8, kHc>(x, w_down, w_up, t, out, rows, hs, R, s);
  return launch<16, kHc>(x, w_down, w_up, t, out, rows, hs, R, s);
}

// tensor cores
// hc = 4, any row count: rows go in 16-row M tiles of bf16 mma.m16n8k16 (FP32 accumulation, fixed order).
//   down : t_raw[m][n] = x[m] . w_down[n]; per (8-row N tile of w_down, K split) one warp, partials [split][M][R]
//   t    : t = bf16(silu(sum over splits (fixed order) / hc))
//   up   : u[m][g * hs + j] for one warp's 8 w_up rows {g * hs + j0 + jj : g < 4, jj < 2}; the gate products
//          sigmoid(u) * x are summed over g across each lane quad (lane l holds g = l % 4) and stored once.
// Fragments (lane l, gq = l / 4, tq = l % 4): A a0 = (gq, 2tq..+1), a1 = (gq+8, ..), a2 = (gq, 2tq+8..), a3 = (gq+8, ..+8);
// B b0 = (k 2tq..+1, n gq), b1 = (k 2tq+8..+9, n gq); D c0,c1 = (gq, 2tq..+1), c2,c3 = (gq+8, ..).
constexpr int kMmaHc = 4;
constexpr int kMaxSplit = 16;

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ld_u32_nc(const uint16_t* p) {
  return __ldg(reinterpret_cast<const unsigned*>(p));
}

int mma_splits(int rows) {
  const int mtiles = (rows + 15) / 16;
  // Split K at least 4 ways (shorter FP32 chains), more while there are few M tiles to fill the GPU.
  int s = kMaxSplit;
  while (s > 4 && s * mtiles > kMaxSplit) s /= 2;
  return s;
}

// grid (R / 8, splits, M tiles), 1 warp per block.
__global__ void __launch_bounds__(32) hc_down_mma_kernel(const uint16_t* __restrict__ x,
                                                         const uint16_t* __restrict__ w_down, float* __restrict__ part,
                                                         int rows, int K, int R) {
  const int lane = threadIdx.x, gq = lane >> 2, tq = lane & 3;
  const int n0 = blockIdx.x * 8, split = blockIdx.y, m0 = blockIdx.z * 16;
  const int splits = gridDim.y, steps = K / 16 / splits;
  const int k_begin = split * steps * 16;
  const int ma = m0 + gq, mb = m0 + gq + 8;
  const uint16_t* xa = x + size_t(min(ma, rows - 1)) * K;
  const uint16_t* xb = x + size_t(min(mb, rows - 1)) * K;
  const bool va = ma < rows, vb = mb < rows;
  const uint16_t* w = w_down + size_t(n0 + gq) * K;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  constexpr int kU = 4;
  for (int s0 = 0; s0 < steps; s0 += kU) {
    uint32_t a[kU][4], b[kU][2];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const int k = k_begin + (s0 + u) * 16 + 2 * tq;
      if (s0 + u < steps) {
        a[u][0] = va ? ld_u32_nc(xa + k) : 0u;
        a[u][1] = vb ? ld_u32_nc(xb + k) : 0u;
        a[u][2] = va ? ld_u32_nc(xa + k + 8) : 0u;
        a[u][3] = vb ? ld_u32_nc(xb + k + 8) : 0u;
        b[u][0] = ld_u32_nc(w + k);
        b[u][1] = ld_u32_nc(w + k + 8);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u)
      if (s0 + u < steps) mma_bf16(d, a[u], b[u][0], b[u][1]);
  }
  const int mpad = gridDim.z * 16;
  float* p = part + (size_t(split) * mpad) * R;
  p[size_t(ma) * R + n0 + 2 * tq] = d[0];
  p[size_t(ma) * R + n0 + 2 * tq + 1] = d[1];
  p[size_t(mb) * R + n0 + 2 * tq] = d[2];
  p[size_t(mb) * R + n0 + 2 * tq + 1] = d[3];
}

__global__ void hc_t_kernel(const float* __restrict__ part, uint16_t* __restrict__ t, int rows, int mpad, int R,
                            int splits, float inv_hc) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * R) return;
  float s = part[i];
  for (int k = 1; k < splits; ++k) s += part[size_t(k) * mpad * R + i];
  const float a = s * inv_hc;
  t[i] = to_bf16_bits(a / (1.0f + __expf(-a)));
}

// grid (hs / 2 / 8 warps, M tiles); block 8 warps; t of the M tile in shared memory ([16][R] bf16, zero padded).
__global__ void __launch_bounds__(256) hc_up_mma_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ t,
                                                        const uint16_t* __restrict__ w_up, uint16_t* __restrict__ out,
                                                        int rows, int R, int hs, float inv_hc) {
  extern __shared__ uint16_t ts16[];  // [16][R]
  const int m0 = blockIdx.y * 16;
  for (int i = threadIdx.x; i < 16 * R; i += blockDim.x) {
    const int m = m0 + i / R;
    ts16[i] = m < rows ? t[size_t(m) * R + i % R] : uint16_t(0);
  }
  __syncthreads();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gq = lane >> 2, tq = lane & 3;
  const int j0 = (blockIdx.x * 8 + warp) * 2;
  if (j0 >= hs) return;
  // B rows: n = gq -> (g = gq / 2, jj = gq % 2).
  const uint16_t* w = w_up + (size_t(gq / 2) * hs + j0 + (gq & 1)) * R;
  float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int k0 = 0; k0 < R; k0 += 16) {
    const int k = k0 + 2 * tq;
    const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(ts16 + gq * R + k),
                           *reinterpret_cast<const uint32_t*>(ts16 + (gq + 8) * R + k),
                           *reinterpret_cast<const uint32_t*>(ts16 + gq * R + k + 8),
                           *reinterpret_cast<const uint32_t*>(ts16 + (gq + 8) * R + k + 8)};
    mma_bf16(d, a, ld_u32_nc(w + k), ld_u32_nc(w + k + 8));
  }
  // Lane holds n = 2tq (g = tq, jj = 0) and 2tq + 1 (g = tq, jj = 1) for rows gq and gq + 8.
  const int K = kMmaHc * hs;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int m = m0 + gq + 8 * half;
    const int mr = min(m, rows - 1);
    float v[2];
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const float u = d[2 * half + jj];
      const float xv = bf16_bits_to_float(x[size_t(mr) * K + size_t(tq) * hs + j0 + jj]);
      v[jj] = (1.0f / (1.0f + __expf(-u))) * xv;
    }
    // Sum over g = tq in a fixed order: g0 + g1, then + (g2 + g3).
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      v[jj] += __shfl_xor_sync(0xFFFFFFFFu, v[jj], 1);
      v[jj] += __shfl_xor_sync(0xFFFFFFFFu, v[jj], 2);
    }
    if (tq == 0 && m < rows) {
      out[size_t(m) * hs + j0] = to_bf16_bits(v[0] * inv_hc);
      out[size_t(m) * hs + j0 + 1] = to_bf16_bits(v[1] * inv_hc);
    }
  }
}

bool mma_ok(int hc, int hs, int R) {
  return hc == kMmaHc && R % 16 == 0 && R <= 1024 && hs % 16 == 0 && (hc * hs / 16) % kMaxSplit == 0;
}

cudaError_t launch_mma(const uint16_t* x, const uint16_t* w_down, const uint16_t* w_up, uint16_t* t, uint16_t* out,
                       int rows, int hs, int R, float* part, cudaStream_t stream) {
  const int K = kMmaHc * hs, mtiles = (rows + 15) / 16, splits = mma_splits(rows);
  hc_down_mma_kernel<<<dim3(R / 8, splits, mtiles), 32, 0, stream>>>(x, w_down, part, rows, K, R);
  hc_t_kernel<<<(rows * R + 255) / 256, 256, 0, stream>>>(part, t, rows, mtiles * 16, R, splits, 1.0f / kMmaHc);
  hc_up_mma_kernel<<<dim3((hs / 2 + 7) / 8, mtiles), 256, size_t(16) * R * 2, stream>>>(x, t, w_up, out, rows, R, hs,
                                                                                       1.0f / kMmaHc);
  return cudaGetLastError();
}

}  // namespace
}  // namespace blaze

extern "C" int blz_hc_mix_supported(int rows, int hc, int hs, int lowrank) {
  if (rows < 0 || lowrank <= 0 || lowrank % 8 != 0 || lowrank > 1024 || hs % 8 != 0) return 0;
  if (blaze::mma_ok(hc, hs, lowrank)) return 1;  // tensor cores: any row count
  return rows <= BLZ_HC_MIX_MAX_ROWS && (hc == 2 || hc == 4 || hc == 8) && (hc * hs) % (blaze::kDownSplit * 8) == 0;
}

extern "C" size_t blz_hc_mix_workspace_bytes(int rows, int hc, int hs, int lowrank) {
  if (rows <= 2 || !blaze::mma_ok(hc, hs, lowrank)) return 0;
  return size_t(blaze::mma_splits(rows)) * size_t((rows + 15) / 16 * 16) * lowrank * sizeof(float);
}

extern "C" cudaError_t blz_hc_mix(const uint16_t* x, const uint16_t* w_down, const uint16_t* w_up, uint16_t* t,
                                  uint16_t* out, int rows, int hc, int hs, int lowrank, void* workspace,
                                  cudaStream_t stream) {
  if (!blz_hc_mix_supported(rows, hc, hs, lowrank)) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  // Tensor cores from 3 rows (1-2 rows: the FFMA kernels read the weights faster).
  if ((rows > 2 || rows > BLZ_HC_MIX_MAX_ROWS) && blaze::mma_ok(hc, hs, lowrank)) {
    if (!workspace) return cudaErrorInvalidValue;
    return blaze::launch_mma(x, w_down, w_up, t, out, rows, hs, lowrank, static_cast<float*>(workspace), stream);
  }
  switch (hc) {
    case 2: return blaze::dispatch_rows<2>(x, w_down, w_up, t, out, rows, hs, lowrank, stream);
    case 4: return blaze::dispatch_rows<4>(x, w_down, w_up, t, out, rows, hs, lowrank, stream);
    default: return blaze::dispatch_rows<8>(x, w_down, w_up, t, out, rows, hs, lowrank, stream);
  }
}
