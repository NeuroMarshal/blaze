// NVFP4 KV cache store (plain or searched block scales) and NVFP4 row dequantization.
//
// One thread owns 8 consecutive elements (one 16-byte load), so a 16-element scale block is a lane pair;
// amax and the candidates' squared errors are combined with one shuffle each. With four over six every block
// is quantized with its maximum mapped to 6 and to 4 (Cook et al., arXiv 2512.02010) and with the E4M3 scales a few
// steps around max/6 (kScaleSearchFirst..kScaleSearchLast), and the candidate with the lowest squared error is
// stored; the storage format does not change, so every NVFP4 reader works unchanged. Division is IEEE (the target
// is built without fast math), so the output matches nvfp4_quantize_row_ref in blaze/nvfp4.h bit for bit, except
// where two candidates' errors tie to the last ulp.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "blaze/kv.h"
#include "blaze/nvfp4.h"
#include "blaze/tq3.h"

namespace blaze {
namespace {

constexpr int kPerThread = 8;

__device__ __forceinline__ uint8_t e4m3_bits(float v) {
  const __nv_fp8_e4m3 q(fminf(v, kE4m3Max));  // cvt.rn.satfinite.e4m3x2.f32
  return reinterpret_cast<const uint8_t&>(q);
}

struct Candidate {
  float err;       // squared error over this thread's 8 elements
  uint32_t codes;  // element i in bits [4i, 4i + 4): byte j = elements 2j (low) and 2j+1 (high)
  uint8_t scale;   // E4M3 bits
};

__device__ __forceinline__ Candidate quantize8(const float (&x)[kPerThread], uint8_t scale, float g) {
  Candidate c;
  c.scale = scale;
  const float t = e4m3_to_float(scale) * g;
  const float inv = t > 0.0f ? 1.0f / t : 0.0f;
  c.err = 0.0f;
  c.codes = 0;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) {
    const float v = x[i] * inv;
    const int idx = e2m1_index(fabsf(v));
    const float lvl = e2m1_level(idx);
    const float d = (v < 0.0f ? -lvl : lvl) * t - x[i];
    c.err += d * d;
    c.codes |= uint32_t(idx | (v < 0.0f ? 8 : 0)) << (4 * i);
  }
  return c;
}

template <typename T>
__device__ __forceinline__ void load8(const T* p, float (&x)[kPerThread]) {
  const uint4 raw = __ldg(reinterpret_cast<const uint4*>(p));
  const T* e = reinterpret_cast<const T*>(&raw);
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) x[i] = static_cast<float>(e[i]);
}

template <typename T, typename IdxT, bool kScaleSearch>
__global__ void __launch_bounds__(512) kv_store_kernel(
    const T* __restrict__ k, const T* __restrict__ v, int64_t k_stride, int64_t v_stride,
    const IdxT* __restrict__ loc, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data,
    uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_scale, const float* __restrict__ k_g,
    const float* __restrict__ v_g, int heads, int dim) {
  const int64_t token = blockIdx.x;
  const int64_t slot = static_cast<int64_t>(loc[token]);
  const int per_side = heads * dim / kPerThread;  // threads for K, then the same number for V
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  // Idle lanes still reach every shuffle (with zeros) and only skip the stores.
  const bool active = tid < 2 * per_side && slot >= 0;
  const bool is_v = tid >= per_side;
  const int e = (is_v ? tid - per_side : tid) * kPerThread;  // element offset in [heads, dim]

  float x[kPerThread] = {};
  if (active) load8((is_v ? v + token * v_stride : k + token * k_stride) + e, x);
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kPerThread; ++i) amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const float g = active ? __ldg(is_v ? v_g : k_g) : 1.0f;

  const int lo = e4m3_bits(amax / (6.0f * g));
  Candidate c = quantize8(x, uint8_t(lo), g);
  if constexpr (kScaleSearch) {
    // Both lanes of a block add the same two partial sums, so they pick the same candidate. Every lane reaches every
    // shuffle; an out-of-range code is evaluated as max/6 and discarded.
    float best = c.err + __shfl_xor_sync(0xffffffffu, c.err, 1);
    auto consider = [&](int code) {
      const bool valid = code >= 0 && code <= 0x7E;
      const Candidate t = quantize8(x, uint8_t(valid ? code : lo), g);
      const float err = t.err + __shfl_xor_sync(0xffffffffu, t.err, 1);
      if (valid && err < best) {
        best = err;
        c = t;
      }
    };
    consider(e4m3_bits(amax / (4.0f * g)));
#pragma unroll
    for (int s = kScaleSearchFirst; s <= kScaleSearchLast; ++s) {
      if (s != 0) consider(lo + s);
    }
  }
  if (!active) return;

  const int head = e / dim;
  const int d = e - head * dim;
  const int64_t row = slot * heads + head;
  *reinterpret_cast<uint32_t*>((is_v ? v_data : k_data) + row * (dim / 2) + d / 2) = c.codes;
  if ((d & (kNvfp4Block - 1)) == 0) (is_v ? v_scale : k_scale)[row * (dim / kNvfp4Block) + d / kNvfp4Block] = c.scale;
}

template <typename T>
__global__ void dequant_kernel(const uint8_t* __restrict__ data, const uint8_t* __restrict__ scale,
                               const float* __restrict__ g_ptr, T* __restrict__ out, int64_t groups) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;  // one thread per 8 values
  if (i >= groups) return;
  const int64_t e = i * kPerThread;
  const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(data + e / 2));
  const float s = e4m3_to_float(__ldg(scale + e / kNvfp4Block)) * __ldg(g_ptr);
  alignas(16) T vals[kPerThread];
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) vals[j] = static_cast<T>(e2m1_value((codes >> (4 * j)) & 15) * s);
  *reinterpret_cast<uint4*>(out + e) = *reinterpret_cast<const uint4*>(vals);
}

template <typename T, typename IdxT>
cudaError_t launch_store(const void* k, const void* v, int64_t k_stride, int64_t v_stride, const void* loc,
                         int64_t tokens, int heads, int dim, uint8_t* k_data, uint8_t* v_data, uint8_t* k_scale,
                         uint8_t* v_scale, const float* k_g, const float* v_g, bool four_over_six,
                         cudaStream_t stream) {
  const int needed = 2 * heads * dim / kPerThread;
  const int block = needed < 512 ? (needed + 31) / 32 * 32 : 512;
  const dim3 grid(static_cast<unsigned>(tokens), (needed + block - 1) / block);
  auto args = [&](auto kernel) {
    kernel<<<grid, block, 0, stream>>>(static_cast<const T*>(k), static_cast<const T*>(v), k_stride, v_stride,
                                       static_cast<const IdxT*>(loc), k_data, v_data, k_scale, v_scale, k_g, v_g,
                                       heads, dim);
  };
  if (four_over_six) {
    args(kv_store_kernel<T, IdxT, true>);
  } else {
    args(kv_store_kernel<T, IdxT, false>);
  }
  return cudaGetLastError();
}

template <typename T>
cudaError_t store_for_index(blz_index_t loc_type, const void* k, const void* v, int64_t k_stride,
                            int64_t v_stride, const void* loc, int64_t tokens, int heads, int dim, uint8_t* k_data,
                            uint8_t* v_data, uint8_t* k_scale, uint8_t* v_scale, const float* k_g, const float* v_g,
                            bool four_over_six, cudaStream_t stream) {
  if (loc_type == BLZ_I64) {
    return launch_store<T, int64_t>(k, v, k_stride, v_stride, loc, tokens, heads, dim, k_data, v_data, k_scale,
                                    v_scale, k_g, v_g, four_over_six, stream);
  }
  return launch_store<T, int32_t>(k, v, k_stride, v_stride, loc, tokens, heads, dim, k_data, v_data, k_scale,
                                  v_scale, k_g, v_g, four_over_six, stream);
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_nvfp4_kv_store(const void* k, const void* v, int64_t k_stride, int64_t v_stride,
                                          blz_half_t dtype, const void* loc, blz_index_t loc_type, int64_t tokens,
                                          int heads, int dim, uint8_t* k_data, uint8_t* v_data, uint8_t* k_scale,
                                          uint8_t* v_scale, const float* k_g, const float* v_g, int four_over_six,
                                          cudaStream_t stream) {
  if (dim % blaze::kNvfp4Block != 0 || k_stride % 8 != 0 || v_stride % 8 != 0 || heads <= 0) {
    return cudaErrorInvalidValue;
  }
  if (tokens == 0) return cudaSuccess;
  if (dtype == BLZ_BF16) {
    return blaze::store_for_index<__nv_bfloat16>(loc_type, k, v, k_stride, v_stride, loc, tokens, heads, dim, k_data,
                                               v_data, k_scale, v_scale, k_g, v_g, four_over_six != 0, stream);
  }
  return blaze::store_for_index<__half>(loc_type, k, v, k_stride, v_stride, loc, tokens, heads, dim, k_data, v_data,
                                      k_scale, v_scale, k_g, v_g, four_over_six != 0, stream);
}

namespace blaze {
namespace {

// FP8 / BF16 rows: one thread = 8 consecutive values of K (first half of the threads) or V
template <typename IdxT, bool kFp8>
__global__ void __launch_bounds__(512) kv_store_dense_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v, int64_t k_stride, int64_t v_stride,
    const IdxT* __restrict__ loc, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data,
    const float* __restrict__ k_g, const float* __restrict__ v_g, int heads, int dim) {
  const int64_t token = blockIdx.x;
  const int64_t slot = static_cast<int64_t>(loc[token]);
  const int per_side = heads * dim / kPerThread;
  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  if (slot < 0 || tid >= 2 * per_side) return;
  const bool is_v = tid >= per_side;
  const int e = (is_v ? tid - per_side : tid) * kPerThread;
  const uint4 raw = __ldg(reinterpret_cast<const uint4*>((is_v ? v + token * v_stride : k + token * k_stride) + e));
  const int64_t off = slot * heads * dim + e;
  uint8_t* dst = is_v ? v_data : k_data;
  if constexpr (kFp8) {
    const float inv = 1.0f / __ldg(is_v ? v_g : k_g);
    const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(&raw);
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
      const __nv_fp8_e4m3 q(__bfloat162float(x[i]) * inv);  // round to nearest, saturated
      w[i / 4] |= uint32_t(reinterpret_cast<const uint8_t&>(q)) << (8 * (i % 4));
    }
    *reinterpret_cast<uint2*>(dst + off) = make_uint2(w[0], w[1]);
  } else {
    *reinterpret_cast<uint4*>(dst + 2 * off) = raw;
  }
}

template <typename IdxT>
cudaError_t store_dense(const void* k, const void* v, int64_t k_stride, int64_t v_stride, const void* loc,
                        int64_t tokens, int heads, int dim, bool fp8, uint8_t* k_data, uint8_t* v_data,
                        const float* k_g, const float* v_g, cudaStream_t stream) {
  const int threads = 2 * heads * dim / kPerThread, block = threads < 512 ? threads : 512;
  const dim3 grid{unsigned(tokens), unsigned((threads + block - 1) / block)};
  auto kk = static_cast<const __nv_bfloat16*>(k);
  auto vv = static_cast<const __nv_bfloat16*>(v);
  auto ll = static_cast<const IdxT*>(loc);
  if (fp8)
    kv_store_dense_kernel<IdxT, true><<<grid, block, 0, stream>>>(kk, vv, k_stride, v_stride, ll, k_data, v_data, k_g,
                                                                  v_g, heads, dim);
  else
    kv_store_dense_kernel<IdxT, false><<<grid, block, 0, stream>>>(kk, vv, k_stride, v_stride, ll, k_data, v_data,
                                                                   k_g, v_g, heads, dim);
  return cudaGetLastError();
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_kv_store_dense(const void* k, const void* v, int64_t k_stride, int64_t v_stride,
                                          const void* loc, blz_index_t loc_type, int64_t tokens, int heads, int dim,
                                          blz_kv_format_t format, uint8_t* k_data, uint8_t* v_data, const float* k_g,
                                          const float* v_g, cudaStream_t stream) {
  if (dim % blaze::kPerThread != 0 || k_stride % 8 != 0 || v_stride % 8 != 0 || heads <= 0 ||
      (format != BLZ_KV_FP8 && format != BLZ_KV_BF16))
    return cudaErrorInvalidValue;
  if (tokens == 0) return cudaSuccess;
  const bool fp8 = format == BLZ_KV_FP8;
  return loc_type == BLZ_I64
             ? blaze::store_dense<int64_t>(k, v, k_stride, v_stride, loc, tokens, heads, dim, fp8, k_data, v_data, k_g,
                                           v_g, stream)
             : blaze::store_dense<int32_t>(k, v, k_stride, v_stride, loc, tokens, heads, dim, fp8, k_data, v_data, k_g,
                                           v_g, stream);
}

extern "C" cudaError_t blz_nvfp4_dequant(const uint8_t* data, const uint8_t* scale, const float* g, int64_t rows,
                                         int dim, blz_half_t out_dtype, void* out, cudaStream_t stream) {
  if (dim % blaze::kNvfp4Block != 0) return cudaErrorInvalidValue;
  const int64_t groups = rows * dim / blaze::kPerThread;
  if (groups == 0) return cudaSuccess;
  const int block = 256;
  const auto grid = static_cast<unsigned>((groups + block - 1) / block);
  if (out_dtype == BLZ_BF16) {
    blaze::dequant_kernel<<<grid, block, 0, stream>>>(data, scale, g, static_cast<__nv_bfloat16*>(out), groups);
  } else {
    blaze::dequant_kernel<<<grid, block, 0, stream>>>(data, scale, g, static_cast<__half*>(out), groups);
  }
  return cudaGetLastError();
}

namespace blaze {
namespace {

// D: the sign of index i, a hash of it (fixed: the cache and the queries must agree across processes)
__device__ __forceinline__ float rht_sign(int i) {
  uint32_t x = uint32_t(i) * 0x9E3779B1u + 0x7F4A7C15u;
  x ^= x >> 15, x *= 0x2C1B3C6Du, x ^= x >> 12, x *= 0x297A2D39u, x ^= x >> 15;
  return (x & 1u) ? -1.0f : 1.0f;
}

// Uniform in [-1, 1) from a 32-bit hash of (index, salt): the fixed parameters of the block rotations.
__device__ __forceinline__ float rht_uniform(int i, uint32_t salt) {
  uint32_t x = uint32_t(i) * 0x85EBCA6Bu + salt;
  x ^= x >> 16, x *= 0x7FEB352Du, x ^= x >> 15, x *= 0x846CA68Bu, x ^= x >> 16;
  return float(x) * (2.0f / 4294967296.0f) - 1.0f;
}

// Block-diagonal rotations of RotorQuant's family on a lane's E consecutive values (element index base + e), fixed
// pseudo-random per block; `inverse`: the transpose. kind 1: Givens rotations of pairs (PlanarQuant); 2: left
// multiplication by a unit quaternion on groups of 4 (IsoQuant); 3: 3D rotations of groups of 3 from a unit
// quaternion (the rotor sandwich R x R~ of Cl(3,0), RotorQuant) and a Givens pair for the lane's last two values.
template <int E>
__device__ __forceinline__ void block_rotate(float (&v)[E], int base, int kind, bool inverse) {
  auto givens = [&](int e) {
    float sn, cs;
    __sincosf(3.14159265f * rht_uniform((base + e) / 2, 0x2545F491u), &sn, &cs);
    if (inverse) sn = -sn;
    const float a = v[e], b = v[e + 1];
    v[e] = cs * a - sn * b, v[e + 1] = sn * a + cs * b;
  };
  auto quat = [&](int g, uint32_t salt, float (&qq)[4]) {
    float n = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) qq[j] = rht_uniform(4 * g + j, salt), n += qq[j] * qq[j];
    n = rsqrtf(fmaxf(n, 1e-12f));
#pragma unroll
    for (int j = 0; j < 4; ++j) qq[j] *= n;
  };
  if (kind == 1 || (kind == 2 && E % 4 != 0)) {
#pragma unroll
    for (int e = 0; e + 1 < E; e += 2) givens(e);
  } else if (kind == 2) {
#pragma unroll
    for (int e = 0; e < E; e += 4) {
      float qq[4];
      quat((base + e) / 4, 0x9E3779B9u, qq);
      const float w = qq[0], x = inverse ? -qq[1] : qq[1], y = inverse ? -qq[2] : qq[2], z = inverse ? -qq[3] : qq[3];
      const float a = v[e], b = v[e + 1], c = v[e + 2], d = v[e + 3];  // q (x) (a + b i + c j + d k)
      v[e] = w * a - x * b - y * c - z * d;
      v[e + 1] = w * b + x * a + y * d - z * c;
      v[e + 2] = w * c - x * d + y * a + z * b;
      v[e + 3] = w * d + x * c - y * b + z * a;
    }
  } else if (kind == 3) {
    // groups of 3 over the lane's values, then a pair (E % 3 == 2) or one value left as is (E % 3 == 1)
    constexpr int kGroups = (E - (E % 3 == 2 ? 2 : E % 3)) / 3;
    int e = 0;
#pragma unroll
    for (int gi = 0; gi < kGroups; ++gi, e += 3) {
      float qq[4];
      quat(base + e, 0x6A09E667u, qq);
      const float w = qq[0], x = qq[1], y = qq[2], z = qq[3];
      // rotation matrix of the rotor (unit quaternion); its transpose undoes it
      const float r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                             {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                             {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
      const float a[3] = {v[e], v[e + 1], v[e + 2]};
#pragma unroll
      for (int i = 0; i < 3; ++i)
        v[e + i] = inverse ? r[0][i] * a[0] + r[1][i] * a[1] + r[2][i] * a[2] : r[i][0] * a[0] + r[i][1] * a[1] + r[i][2] * a[2];
    }
    if (E % 3 == 2) givens(e);
  }
}

// Rows of one tensor for the rotation: row r is (token r / heads, head r % heads).
struct RhtRows {
  __nv_bfloat16* x;
  int64_t rows;
  int heads;
  int64_t tok_stride, head_stride;
};

// A warp a row of 32 E values: lane l holds elements l E .. l E + E - 1; the butterflies of the low log2 E index bits
// run in registers, those of the lane bits through shuffles. Up to three tensors' rows in one launch (one after the
// other); `gate` (inverse, one dense tensor): the result times sigmoid(gate) of the same element.
template <int E>
__global__ void rht_kernel(RhtRows a, RhtRows b, RhtRows c, int inverse, int kind,
                           const __nv_bfloat16* __restrict__ gate) {
  const int lane = threadIdx.x % 32;
  int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  const RhtRows* t = &a;
  if (row >= a.rows) {
    row -= a.rows, t = &b;
    if (row >= b.rows) {
      row -= b.rows, t = &c;
      if (row >= c.rows) return;
    }
  }
  const int64_t off = (row / t->heads) * t->tok_stride + (row % t->heads) * t->head_stride + lane * E;
  __nv_bfloat16* p = t->x + off;
  float v[E];
#pragma unroll
  for (int e = 0; e < E; e += 2) {
    const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p + e));
    v[e] = f.x, v[e + 1] = f.y;
  }
  if (kind != 0) {
    block_rotate<E>(v, lane * E, kind, inverse != 0);
  } else {
  if (!inverse) {
#pragma unroll
    for (int e = 0; e < E; ++e) v[e] *= rht_sign(lane * E + e);
  }
#pragma unroll
  for (int m = 1; m < E; m <<= 1)
#pragma unroll
    for (int e = 0; e < E; ++e)
      if (!(e & m)) {
        const float x0 = v[e], x1 = v[e + m];
        v[e] = x0 + x1, v[e + m] = x0 - x1;
      }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    const bool high = lane & m;
#pragma unroll
    for (int e = 0; e < E; ++e) {
      const float o = __shfl_xor_sync(0xffffffffu, v[e], m);
      v[e] = high ? o - v[e] : v[e] + o;
    }
  }
  const float norm = rsqrtf(float(32 * E));
#pragma unroll
  for (int e = 0; e < E; ++e) v[e] *= inverse ? norm * rht_sign(lane * E + e) : norm;
  }
  if (gate != nullptr) {
#pragma unroll
    for (int e = 0; e < E; ++e) v[e] *= 1.0f / (1.0f + __expf(-__bfloat162float(gate[off + e])));
  }
#pragma unroll
  for (int e = 0; e < E; e += 2)
    *reinterpret_cast<__nv_bfloat162*>(p + e) = __floats2bfloat162_rn(v[e], v[e + 1]);
}

cudaError_t rht_launch(RhtRows a, RhtRows b, RhtRows c, int dim, int inverse, int kind, const __nv_bfloat16* gate,
                       cudaStream_t stream) {
  if (kind < 0 || kind > 3) return cudaErrorInvalidValue;
  for (const RhtRows* t : {&a, &b, &c})
    if (t->rows < 0 || (t->rows > 0 && (t->heads <= 0 || t->tok_stride % 2 || t->head_stride % 2 ||
                                        reinterpret_cast<uintptr_t>(t->x) % 4)))
      return cudaErrorInvalidValue;
  const int64_t rows = a.rows + b.rows + c.rows;
  if (rows == 0) return cudaSuccess;
  const int warps = 8;
  const dim3 grid(static_cast<unsigned>((rows + warps - 1) / warps));
  switch (dim) {
    case 64: rht_kernel<2><<<grid, 32 * warps, 0, stream>>>(a, b, c, inverse, kind, gate); break;
    case 128: rht_kernel<4><<<grid, 32 * warps, 0, stream>>>(a, b, c, inverse, kind, gate); break;
    case 256: rht_kernel<8><<<grid, 32 * warps, 0, stream>>>(a, b, c, inverse, kind, gate); break;
    case 512: rht_kernel<16><<<grid, 32 * warps, 0, stream>>>(a, b, c, inverse, kind, gate); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_rht_rows(uint16_t* x, int64_t tokens, int heads, int64_t tok_stride, int64_t head_stride,
                                    int dim, int inverse, int kind, cudaStream_t stream) {
  if (tokens < 0 || heads <= 0) return cudaErrorInvalidValue;
  const blaze::RhtRows a{reinterpret_cast<__nv_bfloat16*>(x), tokens * heads, heads, tok_stride, head_stride}, none{};
  return blaze::rht_launch(a, none, none, dim, inverse, kind, nullptr, stream);
}

extern "C" cudaError_t blz_rht_qkv(uint16_t* q, int q_heads, int64_t q_tok_stride, uint16_t* k, uint16_t* v, int kv_heads,
                                   int64_t kv_tok_stride, int64_t tokens, int dim, int kind, cudaStream_t stream) {
  if (tokens < 0 || q_heads <= 0 || kv_heads <= 0) return cudaErrorInvalidValue;
  using blaze::RhtRows;
  const RhtRows a{reinterpret_cast<__nv_bfloat16*>(q), tokens * q_heads, q_heads, q_tok_stride, dim},
      b{reinterpret_cast<__nv_bfloat16*>(k), tokens * kv_heads, kv_heads, kv_tok_stride, dim},
      c{reinterpret_cast<__nv_bfloat16*>(v), tokens * kv_heads, kv_heads, kv_tok_stride, dim};
  return blaze::rht_launch(a, b, c, dim, 0, kind, nullptr, stream);
}

extern "C" cudaError_t blz_rht_inverse_mul_sigmoid(uint16_t* x, const uint16_t* gate, int64_t tokens, int heads, int dim,
                                                   int kind, cudaStream_t stream) {
  if (tokens < 0 || heads <= 0) return cudaErrorInvalidValue;
  const blaze::RhtRows a{reinterpret_cast<__nv_bfloat16*>(x), tokens * heads, heads, int64_t(heads) * dim, dim}, none{};
  return blaze::rht_launch(a, none, none, dim, 1, kind, reinterpret_cast<const __nv_bfloat16*>(gate), stream);
}

namespace blaze {
namespace {

// A warp a row (token, head, side): lane l holds dims 8l .. 8l + 7, its group l / 8's magnitude bytes 2 (l % 8), + 1
// and sign byte 16 + l % 8.
__global__ void tq3_store_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                                 int64_t k_stride, int64_t v_stride, const void* __restrict__ loc, int loc_i64,
                                 int64_t tokens, int heads, uint8_t* __restrict__ k_data, uint8_t* __restrict__ v_data,
                                 uint8_t* __restrict__ k_scale, uint8_t* __restrict__ v_scale) {
  const int lane = threadIdx.x % 32;
  const int64_t w = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
  if (w >= tokens * heads * 2) return;
  const int side = int(w & 1);
  const int64_t th = w >> 1, t = th / heads;
  const int h = int(th % heads);
  const int64_t slot = loc_i64 ? static_cast<const int64_t*>(loc)[t] : int64_t(static_cast<const int32_t*>(loc)[t]);
  if (slot < 0) return;
  const __nv_bfloat16* src = (side ? v + t * v_stride : k + t * k_stride) + int64_t(h) * kTq3Dim + lane * 8;
  const uint4 raw = *reinterpret_cast<const uint4*>(src);
  const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
  float x[8], ss = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 f = __bfloat1622float2(b2[i]);
    x[2 * i] = f.x, x[2 * i + 1] = f.y;
    ss = fmaf(f.x, f.x, fmaf(f.y, f.y, ss));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  const int64_t row = slot * heads + h;
  uint8_t* codes = (side ? v_data : k_data) + row * kTq3RowBytes + (lane / 8) * kTq3GroupBytes;
  uint16_t* scale = reinterpret_cast<uint16_t*>(side ? v_scale : k_scale) + row;
  const float rms = sqrtf(ss / kTq3Dim);
  uint32_t mags = 0, signs = 0;
  float num = 0.0f, den = 0.0f;
  if (rms > 0.0f) {
    const float inv = 1.0f / rms;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int m = tq3_magnitude(fabsf(x[i]) * inv);
      const float level = m == 0 ? 0.245361328125f : m == 1 ? 0.7568359375f : m == 2 ? 1.3447265625f : 2.154296875f;
      num = fmaf(fabsf(x[i]), level, num), den = fmaf(level, level, den);
      mags |= uint32_t(m) << (2 * i);
      signs |= uint32_t(x[i] < 0.0f) << i;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    num += __shfl_xor_sync(0xffffffffu, num, o);
    den += __shfl_xor_sync(0xffffffffu, den, o);
  }
  *reinterpret_cast<uint16_t*>(codes + 2 * (lane % 8)) = uint16_t(mags);
  codes[16 + lane % 8] = uint8_t(signs);
  if (lane == 0) *scale = __half_as_ushort(__float2half_rn(den > 0.0f ? num / den : 0.0f));
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_tq3_kv_store(const void* k, const void* v, int64_t k_stride, int64_t v_stride, const void* loc,
                                        blz_index_t loc_type, int64_t tokens, int heads, int dim, uint8_t* k_data,
                                        uint8_t* v_data, uint8_t* k_scale, uint8_t* v_scale, cudaStream_t stream) {
  if (dim != blaze::kTq3Dim || tokens < 0 || heads <= 0 || k_stride % 8 || v_stride % 8 ||
      reinterpret_cast<uintptr_t>(k) % 16 || reinterpret_cast<uintptr_t>(v) % 16)
    return cudaErrorInvalidValue;
  const int64_t warps = tokens * heads * 2;
  if (warps == 0) return cudaSuccess;
  const int per_block = 8;
  blaze::tq3_store_kernel<<<static_cast<unsigned>((warps + per_block - 1) / per_block), 32 * per_block, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(k), static_cast<const __nv_bfloat16*>(v), k_stride, v_stride, loc,
      loc_type == BLZ_I64, tokens, heads, k_data, v_data, k_scale, v_scale);
  return cudaGetLastError();
}
