// Vision tower kernels (blaze/vision.h). The GEMMs are cuBLAS (blz_gemm_bf16); everything around them is here. Rounding
// points follow the bf16 reference (each linear layer's output, the residual sums, the rotary output are bf16; layer
// norm, softmax and the rotary angles are fp32).
#include <math.h>
#include <stdint.h>

#include "blaze/vision.h"

namespace {

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }

__device__ __forceinline__ uint16_t to_bf(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}

__device__ __forceinline__ float rbf(float f) { return bf(to_bf(f)); }

// layer norm

// One block per row; the row sits in shared memory between the statistics and the output.
__global__ void layernorm_kernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ gamma,
                                 const uint16_t* __restrict__ beta, float eps, int dim, uint16_t* __restrict__ out) {
  extern __shared__ float row[];
  __shared__ float red[32];
  const int64_t r = blockIdx.x;
  const uint16_t* xr = x + r * dim;
  float s = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = bf(xr[i]);
    row[i] = v;
    s += v;
  }
  auto block_sum = [&](float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < int(blockDim.x) / 32; ++w) t += red[w];
    return t;
  };
  const float mean = block_sum(s) / float(dim);
  float q = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float d = row[i] - mean;
    q += d * d;
  }
  const float inv = rsqrtf(block_sum(q) / float(dim) + eps);
  uint16_t* orow = out + r * dim;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) orow[i] = to_bf((row[i] - mean) * inv * bf(gamma[i]) + bf(beta[i]));
}

// patch + position embedding

// Patch i (merge order) of a grid_h x grid_w grid: its row and column in the grid.
__device__ __forceinline__ void patch_rc(int64_t i, int grid_w, int merge, int* row, int* col) {
  const int unit = merge * merge;
  const int64_t b = i / unit;
  const int j = int(i % unit);
  const int bw = grid_w / merge;
  *row = int(b / bw) * merge + j / merge;
  *col = int(b % bw) * merge + j % merge;
}

// One block per patch. Taps and weights as get_vision_interpolation_indices_and_weights(bilinear, align_corners=True):
// src = index * (side - 1) / max(size - 1, 1) in fp32, taps floor and floor + 1 (clamped), hat-kernel weights.
__global__ void embed_pos_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ bias,
                                 const uint16_t* __restrict__ table, int side, int grid_h, int grid_w, int merge, int dim) {
  const int64_t i = blockIdx.x;
  int row, col;
  patch_rc(i, grid_w, merge, &row, &col);
  const float sh = float(row) * float(side - 1) / float(max(grid_h - 1, 1));
  const float sw = float(col) * float(side - 1) / float(max(grid_w - 1, 1));
  const float fh = floorf(sh), fw = floorf(sw);
  int th[2], tw[2];
  float wh[2], ww[2];
  for (int o = 0; o < 2; ++o) {
    th[o] = min(max(int(fh) + o, 0), side - 1);
    tw[o] = min(max(int(fw) + o, 0), side - 1);
    wh[o] = fmaxf(1.0f - fabsf(sh - fh - float(o)), 0.0f);
    ww[o] = fmaxf(1.0f - fabsf(sw - fw - float(o)), 0.0f);
  }
  uint16_t* xr = x + i * dim;
  for (int c = threadIdx.x; c < dim; c += blockDim.x) {
    // tap order of the reference: (h0, w0), (h0, w1), (h1, w0), (h1, w1)
    float pe = 0.0f;
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) pe += bf(table[int64_t(th[a]) * side * dim + int64_t(tw[b]) * dim + c]) * (wh[a] * ww[b]);
    xr[c] = to_bf(rbf(bf(xr[c]) + bf(bias[c])) + rbf(pe));
  }
}

// qkv split + rotary

// One warp per (patch, head). q and k: bias, bf16 rounding, rotation in fp32 (pairs (d, d + head_dim / 2); the first
// half of the frequencies belongs to the patch's row, the second to its column); v: bias only.
__global__ void qkv_rope_kernel(const uint16_t* __restrict__ qkv, const uint16_t* __restrict__ bias, int grid_w, int merge,
                                int heads, int head_dim, float theta, uint16_t* __restrict__ q, uint16_t* __restrict__ k,
                                uint16_t* __restrict__ v, int64_t items) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int64_t item = int64_t(blockIdx.x) * (blockDim.x / 32) + warp;
  if (item >= items) return;
  const int64_t i = item / heads;
  const int h = int(item % heads);
  int row, col;
  patch_rc(i, grid_w, merge, &row, &col);
  const int hd2 = head_dim / 2;       // rotated pair distance (36)
  const int nf = head_dim / 4;        // frequencies per axis (18)
  const int64_t width = int64_t(heads) * head_dim;
  const uint16_t* base = qkv + i * 3 * width + int64_t(h) * head_dim;
  for (int d = lane; d < hd2; d += 32) {
    const int f = d % nf;
    const float pos = float(d < nf ? row : col);
    // torch: 1 / (theta ** (arange(0, spatial_dim, 2) / spatial_dim)) in fp32, spatial_dim = head_dim / 2
    const float inv_freq = 1.0f / powf(theta, float(2 * f) / float(hd2));
    float s, c;
    sincosf(pos * inv_freq, &s, &c);
    for (int t = 0; t < 2; ++t) {
      const uint16_t* src = base + int64_t(t) * width;
      const uint16_t* b = bias + int64_t(t) * width + int64_t(h) * head_dim;
      const float x1 = rbf(bf(src[d]) + bf(b[d])), x2 = rbf(bf(src[d + hd2]) + bf(b[d + hd2]));
      uint16_t* dst = (t == 0 ? q : k) + i * width + int64_t(h) * head_dim;
      dst[d] = to_bf(x1 * c - x2 * s);
      dst[d + hd2] = to_bf(x2 * c + x1 * s);
    }
  }
  for (int d = lane; d < head_dim; d += 32) {
    v[i * width + int64_t(h) * head_dim + d] =
        to_bf(bf(base[2 * width + d]) + bf(bias[2 * width + int64_t(h) * head_dim + d]));
  }
}

// attention

constexpr int kBQ = 64;   // queries per block (4 warps x 16 rows)
constexpr int kBK = 64;   // keys per tile
constexpr int kD = 72;    // head dim
constexpr int kDP = 80;   // padded to the mma's k = 16 steps
constexpr int kLD = 88;   // shared row stride (bf16): 176 B, ldmatrix and 32-bit fragment loads are conflict free

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int bytes) {
  const uint32_t s = uint32_t(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const void* smem) {
  const uint32_t s = uint32_t(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ uint32_t pack_bf(float lo, float hi) { return uint32_t(to_bf(lo)) | (uint32_t(to_bf(hi)) << 16); }

// Flash-style softmax attention, grid (query tiles, heads), 4 warps of 16 query rows. K and V tiles double-buffered
// through cp.async; the score tile lives in registers and is reused as the A operand of P V.
__global__ void __launch_bounds__(128) attention_kernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ k,
                                                        const uint16_t* __restrict__ v, uint16_t* __restrict__ o, int n,
                                                        int heads, float scale_log2) {
  __shared__ __align__(16) uint16_t ks[2][kBK * kLD];
  __shared__ __align__(16) uint16_t vs[2][kBK * kLD];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * kBQ;
  const int64_t ld = int64_t(heads) * kD;
  const uint16_t* qh = q + int64_t(h) * kD;
  const uint16_t* kh = k + int64_t(h) * kD;
  const uint16_t* vh = v + int64_t(h) * kD;

  const int r0 = q0 + warp * 16 + lane / 4, r1 = r0 + 8;
  uint32_t qa[kDP / 16][4];
#pragma unroll
  for (int ks_ = 0; ks_ < kDP / 16; ++ks_) {
    const int c = ks_ * 16 + (lane % 4) * 2;
    auto ld32 = [&](int row, int col) -> uint32_t {
      return (row < n && col < kD) ? *reinterpret_cast<const uint32_t*>(qh + int64_t(row) * ld + col) : 0u;
    };
    qa[ks_][0] = ld32(r0, c);
    qa[ks_][1] = ld32(r1, c);
    qa[ks_][2] = ld32(r0, c + 8);
    qa[ks_][3] = ld32(r1, c + 8);
  }

  auto load_tile = [&](int kb, int stage) {
    for (int t = tid; t < kBK * (kDP / 8); t += 128) {
      const int r = t / (kDP / 8), ch = t % (kDP / 8);
      const int key = kb * kBK + r;
      const bool ok = key < n && ch < kD / 8;
      const size_t src = ok ? int64_t(key) * ld + ch * 8 : 0;
      cp_async16(&ks[stage][r * kLD + ch * 8], kh + src, ok ? 16 : 0);
      cp_async16(&vs[stage][r * kLD + ch * 8], vh + src, ok ? 16 : 0);
    }
  };

  float acc[kDP / 8][4];
#pragma unroll
  for (int i = 0; i < kDP / 8; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;

  const int nkb = (n + kBK - 1) / kBK;
  load_tile(0, 0);
  cp_async_commit();
  for (int kb = 0; kb < nkb; ++kb) {
    const int stage = kb & 1;
    if (kb + 1 < nkb) {
      load_tile(kb + 1, stage ^ 1);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    float s[kBK / 8][4];
#pragma unroll
    for (int nt = 0; nt < kBK / 8; ++nt) {
      s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
      const uint16_t* krow = &ks[stage][(nt * 8 + lane / 4) * kLD + (lane % 4) * 2];
#pragma unroll
      for (int ks_ = 0; ks_ < kDP / 16; ++ks_) {
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(krow + ks_ * 16);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(krow + ks_ * 16 + 8);
        mma16816(s[nt], qa[ks_], b0, b1);
      }
    }
    // scale (into the exp2 domain), mask the keys past the end, row maxima
    float mx0 = -1e30f, mx1 = -1e30f;
#pragma unroll
    for (int nt = 0; nt < kBK / 8; ++nt) {
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int key = kb * kBK + nt * 8 + (lane % 4) * 2 + (e & 1);
        const float val = key < n ? s[nt][e] * scale_log2 : -1e30f;
        s[nt][e] = val;
        if (e < 2) mx0 = fmaxf(mx0, val); else mx1 = fmaxf(mx1, val);
      }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = exp2f(m0 - mn0), al1 = exp2f(m1 - mn1);
    m0 = mn0;
    m1 = mn1;
    float ps0 = 0.0f, ps1 = 0.0f;
#pragma unroll
    for (int nt = 0; nt < kBK / 8; ++nt) {
      s[nt][0] = exp2f(s[nt][0] - mn0);
      s[nt][1] = exp2f(s[nt][1] - mn0);
      s[nt][2] = exp2f(s[nt][2] - mn1);
      s[nt][3] = exp2f(s[nt][3] - mn1);
      ps0 += s[nt][0] + s[nt][1];
      ps1 += s[nt][2] + s[nt][3];
    }
    l0 = l0 * al0 + ps0;
    l1 = l1 * al1 + ps1;
#pragma unroll
    for (int i = 0; i < kDP / 8; ++i) {
      acc[i][0] *= al0;
      acc[i][1] *= al0;
      acc[i][2] *= al1;
      acc[i][3] *= al1;
    }
    // P V: A fragments straight from the score registers, V through ldmatrix.trans
#pragma unroll
    for (int kk = 0; kk < kBK / 16; ++kk) {
      uint32_t a[4];
      a[0] = pack_bf(s[2 * kk][0], s[2 * kk][1]);
      a[1] = pack_bf(s[2 * kk][2], s[2 * kk][3]);
      a[2] = pack_bf(s[2 * kk + 1][0], s[2 * kk + 1][1]);
      a[3] = pack_bf(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
      for (int np = 0; np < kDP / 16; ++np) {
        uint32_t b[4];
        const int vr = kk * 16 + (lane % 8) + ((lane / 8) & 1) * 8;
        const int vc = np * 16 + (lane / 16) * 8;
        ldmatrix_x4_trans(b, &vs[stage][vr * kLD + vc]);
        mma16816(acc[2 * np], a, b[0], b[1]);
        mma16816(acc[2 * np + 1], a, b[2], b[3]);
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 1; off <= 2; off <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, off);
    l1 += __shfl_xor_sync(0xffffffffu, l1, off);
  }
  const float i0 = 1.0f / l0, i1 = 1.0f / l1;
#pragma unroll
  for (int nt = 0; nt < kD / 8; ++nt) {
    const int c = nt * 8 + (lane % 4) * 2;
    if (r0 < n) *reinterpret_cast<uint32_t*>(o + int64_t(r0) * ld + int64_t(h) * kD + c) = pack_bf(acc[nt][0] * i0, acc[nt][1] * i0);
    if (r1 < n) *reinterpret_cast<uint32_t*>(o + int64_t(r1) * ld + int64_t(h) * kD + c) = pack_bf(acc[nt][2] * i1, acc[nt][3] * i1);
  }
}

// bias, residual, activation

__global__ void bias_residual_kernel(const uint16_t* __restrict__ g, const uint16_t* __restrict__ bias,
                                     const uint16_t* res, int64_t total, int dim, uint16_t* out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x)
    out[i] = to_bf(bf(res[i]) + rbf(bf(g[i]) + bf(bias[i % dim])));
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2 / pi)
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }

__global__ void bias_act_kernel(uint16_t* __restrict__ x, const uint16_t* __restrict__ bias, int64_t total, int dim, int act) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    float v = rbf(bf(x[i]) + bf(bias[i % dim]));
    if (act == 1) v = gelu_tanh(v);
    else if (act == 2) v = gelu_erf(v);
    x[i] = to_bf(v);
  }
}

unsigned grid_for(int64_t n, int block) {
  const int64_t g = (n + block - 1) / block;
  return unsigned(g < 65535 * 4 ? g : 65535 * 4);
}

}  // namespace

extern "C" cudaError_t blz_vis_layernorm(const uint16_t* x, const uint16_t* gamma, const uint16_t* beta, float eps, int rows,
                                         int dim, uint16_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  layernorm_kernel<<<unsigned(rows), 128, size_t(dim) * sizeof(float), stream>>>(x, gamma, beta, eps, dim, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_vis_embed_pos(uint16_t* x, const uint16_t* bias, const uint16_t* table, int side, int grid_h,
                                         int grid_w, int merge, int dim, cudaStream_t stream) {
  if (grid_h % merge || grid_w % merge) return cudaErrorInvalidValue;
  embed_pos_kernel<<<unsigned(grid_h) * unsigned(grid_w), 128, 0, stream>>>(x, bias, table, side, grid_h, grid_w, merge, dim);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_vis_qkv_rope(const uint16_t* qkv, const uint16_t* bias, int grid_h, int grid_w, int merge, int heads,
                                        int head_dim, float theta, uint16_t* q, uint16_t* k, uint16_t* v, cudaStream_t stream) {
  if (grid_h % merge || grid_w % merge || head_dim % 4) return cudaErrorInvalidValue;
  const int64_t items = int64_t(grid_h) * grid_w * heads;
  qkv_rope_kernel<<<unsigned((items + 3) / 4), 128, 0, stream>>>(qkv, bias, grid_w, merge, heads, head_dim, theta, q, k, v, items);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_vis_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int n, int heads,
                                         int head_dim, uint16_t* o, cudaStream_t stream) {
  if (head_dim != kD) return cudaErrorInvalidValue;
  if (n == 0) return cudaSuccess;
  const float scale_log2 = 1.4426950408889634f / sqrtf(float(kD));
  attention_kernel<<<dim3(unsigned((n + kBQ - 1) / kBQ), unsigned(heads)), 128, 0, stream>>>(q, k, v, o, n, heads, scale_log2);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_vis_bias_residual(const uint16_t* g, const uint16_t* bias, const uint16_t* res, int rows, int dim,
                                             uint16_t* out, cudaStream_t stream) {
  const int64_t total = int64_t(rows) * dim;
  if (total == 0) return cudaSuccess;
  bias_residual_kernel<<<grid_for(total, 256), 256, 0, stream>>>(g, bias, res, total, dim, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_vis_bias_act(uint16_t* x, const uint16_t* bias, int rows, int dim, int act, cudaStream_t stream) {
  const int64_t total = int64_t(rows) * dim;
  if (total == 0) return cudaSuccess;
  bias_act_kernel<<<grid_for(total, 256), 256, 0, stream>>>(x, bias, total, dim, act);
  return cudaGetLastError();
}
