// NVFP4 KV readers for Qwen Sparse Attention: gather + dequantize selected rows into 16-bit scratch.
//
// Thread mapping: one thread owns 8 consecutive values of one (row, head): it loads 4 bytes of codes and
// one block scale, and stores one 16-byte vector. A warp covers 256 values, i.e. one head of dim 256.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "blaze/nvfp4.h"
#include "blaze/qsa.h"

namespace blaze {
namespace {

constexpr int kPerThread = 8;

__device__ __forceinline__ int64_t load_index(blz_index_ptr p, int64_t i) {
  return p.type == BLZ_I64 ? __ldg(static_cast<const int64_t*>(p.ptr) + i)
                           : int64_t(__ldg(static_cast<const int32_t*>(p.ptr) + i));
}

// Dequantize the 8 values starting at element `e` of pool row (slot, head) and store them at `dst`.
template <typename T>
__device__ __forceinline__ void dequant8_to(const uint8_t* data, const uint8_t* scale, float g, int64_t row,
                                            int dim, int d, T* dst) {
  const uint32_t codes = __ldg(reinterpret_cast<const uint32_t*>(data + row * (dim / 2) + d / 2));
  const float s = e4m3_to_float(__ldg(scale + row * (dim / kNvfp4Block) + d / kNvfp4Block)) * g;
  alignas(16) T vals[kPerThread];
#pragma unroll
  for (int j = 0; j < kPerThread; ++j) vals[j] = static_cast<T>(e2m1_value((codes >> (4 * j)) & 15) * s);
  *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(vals);
}

template <typename T>
__device__ __forceinline__ void store_zero8(T* dst) {
  *reinterpret_cast<uint4*>(dst) = make_uint4(0, 0, 0, 0);
}

// Row cache fill. grid: (batch, ceil(topk / blockDim)); one thread per selected position. A missing row's line is
// claimed by swapping its tag to -2 (one winner per line per launch), filled from the pool, then tagged.
__global__ void qsa_cache_fill_kernel(blz_nvfp4_pool pool, blz_nvfp4_row_cache cache,
                                      const int32_t* __restrict__ req_to_token, int64_t req_stride,
                                      blz_index_ptr req_indices, const int32_t* __restrict__ indices, int64_t idx_stride,
                                      blz_index_ptr seq_lens, const int32_t* __restrict__ cu_k, int topk) {
  const int b = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= topk) return;
  const int64_t valid_count = __ldg(cu_k + b + 1) - __ldg(cu_k + b);
  const int64_t position = __ldg(indices + b * idx_stride + col);
  if (col >= valid_count || position < 0 || position >= load_index(seq_lens, b)) return;
  const int32_t slot = __ldg(req_to_token + load_index(req_indices, b) * req_stride + position);
  const int line = int(slot % cache.lines);
  const int32_t seen = atomicAdd(&cache.tag[line], 0);
  if (seen == slot || seen == -2 || atomicCAS(&cache.tag[line], seen, -2) != seen) return;
  const int data16 = pool.heads * (pool.dim / 2) / 16, scale_bytes = pool.heads * (pool.dim / kNvfp4Block);
  const uint4* kd = reinterpret_cast<const uint4*>(pool.k_data + int64_t(slot) * data16 * 16);
  const uint4* vd = reinterpret_cast<const uint4*>(pool.v_data + int64_t(slot) * data16 * 16);
  uint4* ckd = reinterpret_cast<uint4*>(cache.k_data + int64_t(line) * data16 * 16);
  uint4* cvd = reinterpret_cast<uint4*>(cache.v_data + int64_t(line) * data16 * 16);
  for (int w = 0; w < data16; ++w) ckd[w] = kd[w], cvd[w] = vd[w];
  for (int i = 0; i < scale_bytes; ++i) {
    cache.k_scale[int64_t(line) * scale_bytes + i] = pool.k_scale[int64_t(slot) * scale_bytes + i];
    cache.v_scale[int64_t(line) * scale_bytes + i] = pool.v_scale[int64_t(slot) * scale_bytes + i];
  }
  __threadfence();
  atomicExch(&cache.tag[line], slot);
}

__global__ void row_cache_invalidate_kernel(blz_nvfp4_row_cache cache, blz_index_ptr loc, int64_t n) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int64_t slot = load_index(loc, i);
  if (slot < 0) return;
  atomicCAS(&cache.tag[slot % cache.lines], int32_t(slot), -1);
}

// grid: (batch, heads, ceil(num_cols * threads_per_head / blockDim))
template <typename T>
__global__ void qsa_compact_kernel(blz_nvfp4_pool pool, blz_nvfp4_row_cache cache,
                                   const int32_t* __restrict__ req_to_token, int64_t req_stride,
                                   blz_index_ptr req_indices, const int32_t* __restrict__ indices, int64_t idx_stride,
                                   blz_index_ptr seq_lens, const int32_t* __restrict__ cu_k, int topk, int num_cols,
                                   bool zero_fill, T* __restrict__ out_k, T* __restrict__ out_v) {
  const int b = blockIdx.x, head = blockIdx.y;
  const int per_head = pool.dim / kPerThread;
  const int t = blockIdx.z * blockDim.x + threadIdx.x;
  const int col = t / per_head;
  if (col >= num_cols) return;
  const int d = (t - col * per_head) * kPerThread;

  const int64_t length = load_index(seq_lens, b);
  const int64_t pack_start = __ldg(cu_k + b);
  const int64_t valid_count = __ldg(cu_k + b + 1) - pack_start;
  const int64_t position = col < topk ? __ldg(indices + b * idx_stride + col) : -1;
  const bool valid = col < valid_count && position >= 0 && position < length;
  if (!valid && !zero_fill) return;

  const int64_t dst = ((pack_start + col) * pool.heads + head) * pool.dim + d;
  if (!valid) {
    store_zero8(out_k + dst);
    store_zero8(out_v + dst);
    return;
  }
  const int64_t req = load_index(req_indices, b);
  const int64_t slot = __ldg(req_to_token + req * req_stride + position);
  blz_nvfp4_pool src = pool;
  int64_t row = slot * pool.heads + head;
  if (cache.tag != nullptr) {
    const int line = int(slot % cache.lines);
    if (cache.tag[line] == int32_t(slot)) {
      src.k_data = cache.k_data, src.v_data = cache.v_data, src.k_scale = cache.k_scale, src.v_scale = cache.v_scale;
      row = int64_t(line) * pool.heads + head;
    }
  }
  dequant8_to(src.k_data, src.k_scale, __ldg(pool.k_g), row, pool.dim, d, out_k + dst);
  dequant8_to(src.v_data, src.v_scale, __ldg(pool.v_g), row, pool.dim, d, out_v + dst);
}

// grid: (ceil(max_k * heads * dim / 8 / blockDim), batch)
template <typename T>
__global__ void qsa_pack_kernel(blz_nvfp4_pool pool, const int32_t* __restrict__ req_to_token, int64_t sr0,
                                int64_t sr1, blz_index_ptr req_indices, const int32_t* __restrict__ cu_k,
                                T* __restrict__ out_k, T* __restrict__ out_v) {
  const int b = blockIdx.y;
  const int64_t start = __ldg(cu_k + b);
  const int64_t len = __ldg(cu_k + b + 1) - start;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_token = int64_t(pool.heads) * pool.dim / kPerThread;
  const int64_t token = i / per_token;
  if (token >= len) return;
  const int64_t e = (i - token * per_token) * kPerThread;  // offset in [heads, dim]
  const int head = int(e / pool.dim);
  const int d = int(e - int64_t(head) * pool.dim);
  const int64_t req = load_index(req_indices, b);
  const int64_t slot = __ldg(req_to_token + req * sr0 + token * sr1);
  const int64_t row = slot * pool.heads + head;
  const int64_t dst = (start + token) * pool.heads * pool.dim + e;
  dequant8_to(pool.k_data, pool.k_scale, __ldg(pool.k_g), row, pool.dim, d, out_k + dst);
  dequant8_to(pool.v_data, pool.v_scale, __ldg(pool.v_g), row, pool.dim, d, out_v + dst);
}

bool valid_pool(const blz_nvfp4_pool& p) { return p.heads > 0 && p.dim > 0 && p.dim % kNvfp4Block == 0; }

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_nvfp4_qsa_compact(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                             blz_index_ptr req_indices, const int32_t* indices, int64_t idx_stride,
                                             blz_index_ptr seq_lens, const int32_t* cu_k, int batch, int topk,
                                             int zero_fill_cols, const blz_nvfp4_row_cache* cache,
                                             blz_half_t out_dtype, void* out_k, void* out_v, cudaStream_t stream) {
  if (!blaze::valid_pool(pool)) return cudaErrorInvalidValue;
  const bool zero_fill = zero_fill_cols > 0;
  const int num_cols = zero_fill ? zero_fill_cols : topk;
  if (batch == 0 || num_cols == 0) return cudaSuccess;
  const blz_nvfp4_row_cache rc = cache ? *cache : blz_nvfp4_row_cache{nullptr, nullptr, nullptr, nullptr, nullptr, 1};
  if (cache && topk > 0) {
    const dim3 fill_grid(batch, static_cast<unsigned>((topk + 127) / 128));
    blaze::qsa_cache_fill_kernel<<<fill_grid, 128, 0, stream>>>(pool, rc, req_to_token, req_stride, req_indices,
                                                              indices, idx_stride, seq_lens, cu_k, topk);
  }
  const int block = 256;
  const int64_t threads = int64_t(num_cols) * (pool.dim / blaze::kPerThread);
  const dim3 grid(batch, pool.heads, static_cast<unsigned>((threads + block - 1) / block));
  if (out_dtype == BLZ_BF16) {
    blaze::qsa_compact_kernel<<<grid, block, 0, stream>>>(
        pool, rc, req_to_token, req_stride, req_indices, indices, idx_stride, seq_lens, cu_k, topk, num_cols,
        zero_fill, static_cast<__nv_bfloat16*>(out_k), static_cast<__nv_bfloat16*>(out_v));
  } else {
    blaze::qsa_compact_kernel<<<grid, block, 0, stream>>>(
        pool, rc, req_to_token, req_stride, req_indices, indices, idx_stride, seq_lens, cu_k, topk, num_cols,
        zero_fill, static_cast<__half*>(out_k), static_cast<__half*>(out_v));
  }
  return cudaGetLastError();
}

extern "C" cudaError_t blz_nvfp4_row_cache_invalidate(blz_nvfp4_row_cache cache, blz_index_ptr loc, int64_t n,
                                                      cudaStream_t stream) {
  if (n <= 0 || cache.tag == nullptr || cache.lines <= 0) return cudaSuccess;
  const int block = 256;
  blaze::row_cache_invalidate_kernel<<<static_cast<unsigned>((n + block - 1) / block), block, 0, stream>>>(cache, loc, n);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_nvfp4_qsa_pack(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride0,
                                          int64_t req_stride1, blz_index_ptr req_indices, const int32_t* cu_k,
                                          int batch, int64_t max_k, blz_half_t out_dtype, void* out_k, void* out_v,
                                          cudaStream_t stream) {
  if (!blaze::valid_pool(pool)) return cudaErrorInvalidValue;
  if (batch == 0 || max_k == 0) return cudaSuccess;
  const int block = 256;
  const int64_t threads = max_k * pool.heads * (pool.dim / blaze::kPerThread);
  const dim3 grid(static_cast<unsigned>((threads + block - 1) / block), batch);
  if (out_dtype == BLZ_BF16) {
    blaze::qsa_pack_kernel<<<grid, block, 0, stream>>>(pool, req_to_token, req_stride0, req_stride1, req_indices,
                                                     cu_k, static_cast<__nv_bfloat16*>(out_k),
                                                     static_cast<__nv_bfloat16*>(out_v));
  } else {
    blaze::qsa_pack_kernel<<<grid, block, 0, stream>>>(pool, req_to_token, req_stride0, req_stride1, req_indices,
                                                     cu_k, static_cast<__half*>(out_k), static_cast<__half*>(out_v));
  }
  return cudaGetLastError();
}
