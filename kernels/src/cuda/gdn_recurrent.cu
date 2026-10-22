// Gated delta rule, token by token (blaze/ops.h: blz_gdn_recurrent).
//
// One CTA per (sequence, v head, 32-column slice of the value dim); the slice of the state, [dk = 128 rows][32
// columns], lives in registers: warp w holds columns 8w..8w+7, lane (c, g) = (lane % 8, lane / 8) rows 32g..32g+31 of
// column 8w + c. Column sums over the rows are two shuffles across the four row groups of a warp, so a token costs
// two barriers (staging q, k, v) and no cross-warp reduction. The L2 norms of q and k come out of the same
// shuffles.

#include <cuda_bf16.h>

#include "blaze/ops.h"

namespace blaze {
namespace {

constexpr int kDk = 128;
constexpr int kSlice = 32;
constexpr int kRows = kDk / 4;   // rows per thread
constexpr int kPad = kRows + 1;  // shared-memory stride of a row group (bank-conflict free)

__device__ __forceinline__ float bf(uint16_t b) { return __uint_as_float(uint32_t(b) << 16); }
__device__ __forceinline__ uint16_t to_bf(float f) {
  const __nv_bfloat16 h = __float2bfloat16_rn(f);
  return *reinterpret_cast<const uint16_t*>(&h);
}

__global__ void __launch_bounds__(128) gdn_recurrent_kernel(const uint16_t* __restrict__ qkv,
                                                            const float* __restrict__ g,
                                                            const float* __restrict__ beta,
                                                            const int32_t* __restrict__ seq_start,
                                                            const int32_t* __restrict__ slots, int k_heads,
                                                            int v_heads, int dv, float* __restrict__ state,
                                                            uint16_t* __restrict__ out) {
  __shared__ float sq[4 * kPad], sk[4 * kPad], sv[kSlice], sgb[2];
  const int s = blockIdx.x, vh = blockIdx.y, slice = blockIdx.z;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int col = warp * 8 + (lane & 7), rg = lane >> 3;
  const int j = slice * kSlice + col;
  const int kh = vh / (v_heads / k_heads);
  const int64_t ld = int64_t(2) * k_heads * kDk + int64_t(v_heads) * dv;
  float* st = state + (int64_t(slots[s]) * v_heads + vh) * kDk * dv;

  float S[kRows];
#pragma unroll
  for (int r = 0; r < kRows; ++r) S[r] = st[int64_t(rg * kRows + r) * dv + j];

  const float q_scale = rsqrtf(float(kDk));
  const int t0 = seq_start[s], t1 = seq_start[s + 1];
  for (int t = t0; t < t1; ++t) {
    const uint16_t* row = qkv + int64_t(t) * ld;
    __syncthreads();  // the previous token's reads are done
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
  }
#pragma unroll
  for (int r = 0; r < kRows; ++r) st[int64_t(rg * kRows + r) * dv + j] = S[r];
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_gdn_recurrent(const uint16_t* qkv, const float* g, const float* beta,
                                         const int32_t* seq_start, const int32_t* slots, int seqs, int tokens,
                                         int k_heads, int v_heads, int dk, int dv, float* state, uint16_t* out,
                                         cudaStream_t stream) {
  (void)tokens;
  if (seqs == 0) return cudaSuccess;
  if (dk != blaze::kDk || dv % blaze::kSlice || v_heads % k_heads) return cudaErrorInvalidValue;
  const dim3 grid(unsigned(seqs), unsigned(v_heads), unsigned(dv / blaze::kSlice));
  blaze::gdn_recurrent_kernel<<<grid, 128, 0, stream>>>(qkv, g, beta, seq_start, slots, k_heads, v_heads, dv, state,
                                                        out);
  return cudaGetLastError();
}
