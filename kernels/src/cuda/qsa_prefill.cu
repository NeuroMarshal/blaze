// QSA sparse attention for prefill straight from an NVFP4 KV pool (see blaze/qsa.h).
//
// One CTA = one query row x one KV group (the group's q heads padded to the 16 rows of an MMA tile), 4 warps.
// Prologue: the pool rows of all the row's selected positions (index -> req_to_token) are resolved into shared memory
// at once, so the per-block gathers are single loads. Blocks of kBlockN positions: the raw NVFP4 bytes of the next
// block are loaded into registers while the current block computes (a thread: 32 values of 2 positions); a block is
// dequantized into shared memory (K as [token][dim], V transposed as [dim][token], stored as pairs of positions) and
// consumed by bf16 m16n8k16 MMAs: S = Q K^T (warp w: tokens 8w .. 8w+7, two accumulators), online softmax,
// O += P V (warp w: dims 64w .. 64w+63).
// The arithmetic follows SGLang's _sparse_gqa_chunk_prefill: q * scale * log2(e) rounded to bf16, fp32 scores, exp2
// softmax with fp32 sums, probabilities rounded to bf16 for P V, output = acc / sum.

#include <cuda_bf16.h>

#include "nvfp4_attn.cuh"
#include "blaze/nvfp4.h"
#include "blaze/qsa.h"

namespace blaze {

__device__ __forceinline__ int64_t qsa_load_index(blz_index_ptr p, int64_t i) {
  return p.type == BLZ_I64 ? __ldg(static_cast<const int64_t*>(p.ptr) + i)
                           : int64_t(__ldg(static_cast<const int32_t*>(p.ptr) + i));
}

namespace {

using namespace attn;

struct Smem {
  __nv_bfloat16 k[kBlockN * kKStride];
  __nv_bfloat16 vt[kDim * kVStride];
  __nv_bfloat16 p[16 * kPStride];
  float red_max[kWarps][16];
  float red_sum[kWarps][16];
  // followed by int32_t rows[ceil(topk / kBlockN) * kBlockN]: pool row (slot * heads + group) per position, -1: masked
};

size_t smem_bytes(int topk) {
  return sizeof(Smem) + size_t((topk + kBlockN - 1) / kBlockN * kBlockN) * sizeof(int32_t);
}

__device__ __forceinline__ void load_share(const blz_nvfp4_pool& pool, const int32_t* rows, int pair, int d0, Raw& r) {
  load_raw(pool, rows[2 * pair], d0, r.k[0], r.v[0], r.ks[0], r.vs[0]);
  load_raw(pool, rows[2 * pair + 1], d0, r.k[1], r.v[1], r.ks[1], r.vs[1]);
}

// grid: (total_q, kv_heads), block: kThreads, dynamic shared memory: smem_bytes(topk)
__global__ void __launch_bounds__(kThreads) qsa_sparse_prefill_kernel(
    const __nv_bfloat16* __restrict__ q, int64_t sq_row, int64_t sq_head, int group, blz_nvfp4_pool pool,
    const int32_t* __restrict__ req_to_token, int64_t req_stride, blz_index_ptr req_indices,
    const int32_t* __restrict__ indices, int64_t si_row, int64_t si_group, int topk, const int32_t* __restrict__ cu_q,
    const int32_t* __restrict__ kv_lens, const int32_t* __restrict__ stage_cu, int batch, float scale,
    __nv_bfloat16* __restrict__ out, int64_t so_row, int64_t so_head) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
  int32_t* rows = reinterpret_cast<int32_t*>(smem_raw + sizeof(Smem));
  const int i = blockIdx.x, g = blockIdx.y;
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  const int gr = lane / 4, gc = (lane % 4) * 2;  // MMA fragment row / column pair

  int b = 0;
  while (b + 1 < batch && __ldg(cu_q + b + 1) <= i) ++b;
  const int q_rel = i - __ldg(cu_q + b), q_len = __ldg(cu_q + b + 1) - __ldg(cu_q + b);
  const int kv_len = __ldg(kv_lens + b);
  const int row_topk = max(0, min(topk, q_rel + kv_len - q_len + 1));
  const int nblocks = (row_topk + kBlockN - 1) / kBlockN;
  const int64_t req = stage_cu ? 0 : qsa_load_index(req_indices, b);
  const int32_t stage_base = stage_cu ? __ldg(stage_cu + b) : 0;
  const int32_t* idx = indices + int64_t(i) * si_row + int64_t(g) * si_group;
  const float kg = __ldg(pool.k_g), vg = __ldg(pool.v_g);

  // Pool rows of every selected position (padding to whole blocks masked).
  for (int j = tid; j < nblocks * kBlockN; j += kThreads) {
    const int t = j < row_topk ? __ldg(idx + j) : -1;
    const int32_t slot = (t < 0 || t >= kv_len) ? -1 : stage_cu ? stage_base + t : __ldg(req_to_token + req * req_stride + t);
    rows[j] = slot < 0 ? -1 : slot * pool.heads + g;
  }

  // Q fragments of all 16 k-steps (rows >= group are zero): bf16(q * scale * log2 e).
  uint32_t qa[kDim / 16][4];
  {
    const __nv_bfloat16* qrow = q + int64_t(i) * sq_row + int64_t(g) * group * sq_head;
    auto qv = [&](int r, int d) {
      return r < group ? __bfloat162float(qrow[int64_t(r) * sq_head + d]) * scale * kLog2e : 0.0f;
    };
#pragma unroll
    for (int ks = 0; ks < kDim / 16; ++ks) {
      const int d = 16 * ks + gc;
      qa[ks][0] = pack_bf16(qv(gr, d), qv(gr, d + 1));
      qa[ks][1] = pack_bf16(qv(gr + 8, d), qv(gr + 8, d + 1));
      qa[ks][2] = pack_bf16(qv(gr, d + 8), qv(gr, d + 9));
      qa[ks][3] = pack_bf16(qv(gr + 8, d + 8), qv(gr + 8, d + 9));
    }
  }
  __syncthreads();

  float o[8][4];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) o[nt][0] = o[nt][1] = o[nt][2] = o[nt][3] = 0.0f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;  // rows gr and gr + 8

  // This thread's share of each block: positions 2p, 2p + 1 (p = tid / 8), values [32 * (tid % 8), +32).
  const int pair = tid / 8, d0 = 32 * (tid % 8);
  Raw raw{};
  if (nblocks > 0) load_share(pool, rows, pair, d0, raw);
  for (int blk = 0; blk < nblocks; ++blk) {
    const int j0 = blk * kBlockN;
    {
      float a[32], c[32];
      const bool ok0 = rows[j0 + 2 * pair] >= 0, ok1 = rows[j0 + 2 * pair + 1] >= 0;
#pragma unroll
      for (int side = 0; side < 2; ++side) {  // K rows of both positions
        const bool ok = side ? ok1 : ok0;
        if (ok) dequant32(raw.k[side], raw.ks[side], kg, a);
        uint4* dst = reinterpret_cast<uint4*>(&sm.k[(2 * pair + side) * kKStride + d0]);
#pragma unroll
        for (int w = 0; w < 4; ++w)
          dst[w] = ok ? make_uint4(pack_bf16(a[8 * w], a[8 * w + 1]), pack_bf16(a[8 * w + 2], a[8 * w + 3]),
                                   pack_bf16(a[8 * w + 4], a[8 * w + 5]), pack_bf16(a[8 * w + 6], a[8 * w + 7]))
                      : make_uint4(0, 0, 0, 0);
      }
      // V transposed: one 32-bit store holds dim d of both positions.
      if (ok0) dequant32(raw.v[0], raw.vs[0], vg, a);
      if (ok1) dequant32(raw.v[1], raw.vs[1], vg, c);
#pragma unroll
      for (int d = 0; d < 32; ++d)
        *reinterpret_cast<uint32_t*>(&sm.vt[(d0 + d) * kVStride + 2 * pair]) =
            pack_bf16(ok0 ? a[d] : 0.0f, ok1 ? c[d] : 0.0f);
    }
    __syncthreads();
    if (blk + 1 < nblocks) load_share(pool, rows + j0 + kBlockN, pair, d0, raw);  // in flight while this block computes

    // S for positions 8w .. 8w+7 of the block.
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    {
      // Two accumulators (even / odd k-steps) halve the dependent MMA chain.
      float s2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
      const __nv_bfloat16* kr = &sm.k[(8 * warp + gr) * kKStride + gc];
#pragma unroll
      for (int ks = 0; ks < kDim / 16; ks += 2) {
        mma_bf16(s, qa[ks], *reinterpret_cast<const uint32_t*>(kr + 16 * ks),
                 *reinterpret_cast<const uint32_t*>(kr + 16 * ks + 8));
        mma_bf16(s2, qa[ks + 1], *reinterpret_cast<const uint32_t*>(kr + 16 * ks + 16),
                 *reinterpret_cast<const uint32_t*>(kr + 16 * ks + 24));
      }
#pragma unroll
      for (int e = 0; e < 4; ++e) s[e] += s2[e];
    }
    if (rows[j0 + 8 * warp + gc] < 0) s[0] = s[2] = -INFINITY;
    if (rows[j0 + 8 * warp + gc + 1] < 0) s[1] = s[3] = -INFINITY;
    float bm0 = fmaxf(s[0], s[1]), bm1 = fmaxf(s[2], s[3]);
    bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 1)), bm0 = fmaxf(bm0, __shfl_xor_sync(0xffffffffu, bm0, 2));
    bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 1)), bm1 = fmaxf(bm1, __shfl_xor_sync(0xffffffffu, bm1, 2));
    if (lane % 4 == 0) sm.red_max[warp][gr] = bm0, sm.red_max[warp][gr + 8] = bm1;
    __syncthreads();
    float n0 = m0, n1 = m1;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) n0 = fmaxf(n0, sm.red_max[w][gr]), n1 = fmaxf(n1, sm.red_max[w][gr + 8]);
    // A row with nothing visible yet keeps its zero state (exp2(-inf - -inf) would be NaN).
    const float base0 = n0 == -INFINITY ? 0.0f : n0, base1 = n1 == -INFINITY ? 0.0f : n1;
    const float alpha0 = m0 == -INFINITY ? 0.0f : exp2f(m0 - base0);
    const float alpha1 = m1 == -INFINITY ? 0.0f : exp2f(m1 - base1);
    const float p0 = exp2f(s[0] - base0), p1 = exp2f(s[1] - base0);
    const float p2 = exp2f(s[2] - base1), p3 = exp2f(s[3] - base1);
    *reinterpret_cast<uint32_t*>(&sm.p[gr * kPStride + 8 * warp + gc]) = pack_bf16(p0, p1);
    *reinterpret_cast<uint32_t*>(&sm.p[(gr + 8) * kPStride + 8 * warp + gc]) = pack_bf16(p2, p3);
    float bs0 = p0 + p1, bs1 = p2 + p3;
    bs0 += __shfl_xor_sync(0xffffffffu, bs0, 1), bs0 += __shfl_xor_sync(0xffffffffu, bs0, 2);
    bs1 += __shfl_xor_sync(0xffffffffu, bs1, 1), bs1 += __shfl_xor_sync(0xffffffffu, bs1, 2);
    if (lane % 4 == 0) sm.red_sum[warp][gr] = bs0, sm.red_sum[warp][gr + 8] = bs1;
    __syncthreads();
    float t0 = 0.0f, t1 = 0.0f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) t0 += sm.red_sum[w][gr], t1 += sm.red_sum[w][gr + 8];
    l0 = l0 * alpha0 + t0, l1 = l1 * alpha1 + t1;
    m0 = n0, m1 = n1;

    // O (dims 64w .. 64w+63) = O * alpha + P V.
#pragma unroll
    for (int nt = 0; nt < 8; ++nt) o[nt][0] *= alpha0, o[nt][1] *= alpha0, o[nt][2] *= alpha1, o[nt][3] *= alpha1;
#pragma unroll
    for (int ks = 0; ks < kBlockN / 16; ++ks) {
      uint32_t pa[4];
      const __nv_bfloat16* pr = &sm.p[gr * kPStride + 16 * ks + gc];
      pa[0] = *reinterpret_cast<const uint32_t*>(pr);
      pa[1] = *reinterpret_cast<const uint32_t*>(pr + 8 * kPStride);
      pa[2] = *reinterpret_cast<const uint32_t*>(pr + 8);
      pa[3] = *reinterpret_cast<const uint32_t*>(pr + 8 * kPStride + 8);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) {
        const __nv_bfloat16* vr = &sm.vt[(64 * warp + 8 * nt + gr) * kVStride + 16 * ks + gc];
        mma_bf16(o[nt], pa, *reinterpret_cast<const uint32_t*>(vr), *reinterpret_cast<const uint32_t*>(vr + 8));
      }
    }
    __syncthreads();
  }

  const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
  __nv_bfloat16* orow = out + int64_t(i) * so_row + int64_t(g) * group * so_head;
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int d = 64 * warp + 8 * nt + gc;
    if (gr < group)
      *reinterpret_cast<__nv_bfloat162*>(orow + int64_t(gr) * so_head + d) =
          __floats2bfloat162_rn(o[nt][0] * inv0, o[nt][1] * inv0);
    if (gr + 8 < group)
      *reinterpret_cast<__nv_bfloat162*>(orow + int64_t(gr + 8) * so_head + d) =
          __floats2bfloat162_rn(o[nt][2] * inv1, o[nt][3] * inv1);
  }
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_nvfp4_qsa_sparse_prefill(const void* q, int64_t q_stride_row, int64_t q_stride_head,
                                                    int q_heads, blz_nvfp4_pool pool, const int32_t* req_to_token,
                                                    int64_t req_stride, blz_index_ptr req_indices,
                                                    const int32_t* indices, int64_t idx_stride_row,
                                                    int64_t idx_stride_group, int topk, const int32_t* cu_q,
                                                    const int32_t* kv_lens, const int32_t* stage_cu, int batch,
                                                    int total_q, float scale, blz_half_t dtype, void* out,
                                                    int64_t out_stride_row, int64_t out_stride_head,
                                                    cudaStream_t stream) {
  if (dtype != BLZ_BF16 || pool.dim != blaze::kDim || pool.heads <= 0 || q_heads % pool.heads != 0 ||
      q_heads / pool.heads > 16 || topk < 0)
    return cudaErrorNotSupported;
  if (total_q == 0 || batch == 0) return cudaSuccess;
  const size_t smem = blaze::smem_bytes(topk);
  // The opt-in above 48 KB is a per-device function attribute: raised when a launch needs more than set so far.
  static size_t configured[64] = {};
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return err;
  if (device >= 64 || configured[device] < smem) {
    err = cudaFuncSetAttribute(blaze::qsa_sparse_prefill_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               int(smem));
    if (err != cudaSuccess) return err;
    if (device < 64) configured[device] = smem;
  }
  const dim3 grid(static_cast<unsigned>(total_q), static_cast<unsigned>(pool.heads));
  blaze::qsa_sparse_prefill_kernel<<<grid, blaze::kThreads, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(q), q_stride_row, q_stride_head, q_heads / pool.heads, pool, req_to_token,
      req_stride, req_indices, indices, idx_stride_row, idx_stride_group, topk, cu_q, kv_lens, stage_cu, batch, scale,
      static_cast<__nv_bfloat16*>(out), out_stride_row, out_stride_head);
  return cudaGetLastError();
}

namespace blaze {
namespace {

// grid: (ceil(max_k * per_slot / 16 / blockDim), batch): copies the raw NVFP4 rows of positions [0, len) of request b
// (data then scales, K and V) to slot cu_k[b] + t of the staging pool. per_slot = 2 * heads * (dim / 2 + dim / 16).
__global__ void qsa_stage_kernel(blz_nvfp4_pool pool, const int32_t* __restrict__ req_to_token, int64_t req_stride,
                                 blz_index_ptr req_indices, const int32_t* __restrict__ cu_k, blz_nvfp4_pool out) {
  const int b = blockIdx.y;
  const int64_t start = __ldg(cu_k + b), len = __ldg(cu_k + b + 1) - start;
  const int data16 = pool.heads * (pool.dim / 2) / 16, scale16 = pool.heads * (pool.dim / kNvfp4Block) / 16;
  const int per_token = 2 * (data16 + scale16);  // uint4 words per position: K data, V data, K scales, V scales
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t t = i / per_token;
  if (t >= len) return;
  int w = int(i - t * per_token);
  const int64_t slot = __ldg(req_to_token + qsa_load_index(req_indices, b) * req_stride + t), dst = start + t;
  const uint8_t* src;
  uint8_t* to;
  int64_t row_bytes;
  if (w < 2 * data16) {
    const bool v = w >= data16;
    w -= v ? data16 : 0;
    src = v ? pool.v_data : pool.k_data, to = const_cast<uint8_t*>(v ? out.v_data : out.k_data);
    row_bytes = int64_t(data16) * 16;
  } else {
    w -= 2 * data16;
    const bool v = w >= scale16;
    w -= v ? scale16 : 0;
    src = v ? pool.v_scale : pool.k_scale, to = const_cast<uint8_t*>(v ? out.v_scale : out.k_scale);
    row_bytes = int64_t(scale16) * 16;
  }
  reinterpret_cast<uint4*>(to + dst * row_bytes)[w] = reinterpret_cast<const uint4*>(src + slot * row_bytes)[w];
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_nvfp4_qsa_stage(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                           blz_index_ptr req_indices, const int32_t* cu_k, int batch, int64_t max_k,
                                           blz_nvfp4_pool out, cudaStream_t stream) {
  if (pool.dim % 32 != 0 || pool.heads <= 0 || (pool.heads * (pool.dim / blaze::kNvfp4Block)) % 16 != 0)
    return cudaErrorNotSupported;
  if (batch == 0 || max_k == 0) return cudaSuccess;
  const int64_t per_token = 2 * (pool.heads * (pool.dim / 2) / 16 + pool.heads * (pool.dim / blaze::kNvfp4Block) / 16);
  const int block = 256;
  const dim3 grid(static_cast<unsigned>((max_k * per_token + block - 1) / block), static_cast<unsigned>(batch));
  blaze::qsa_stage_kernel<<<grid, block, 0, stream>>>(pool, req_to_token, req_stride, req_indices, cu_k, out);
  return cudaGetLastError();
}
