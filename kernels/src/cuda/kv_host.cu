// KV pools in pinned host memory: gathers through a device row cache (see blaze/kv_host.h).

#include "blaze/kv_host.h"

namespace blaze {
namespace {

constexpr int kWarps = 8;

// Words of one row: data heads * dim / 2 bytes, scales heads * dim / 16 bytes (NVFP4), copied as 32-bit words.
struct RowWords {
  int data, scale;
};

__device__ __forceinline__ RowWords row_words(const blz_nvfp4_pool& p) {
  return {p.heads * p.dim / 8, p.heads * p.dim / 64};
}

template <typename W>
__device__ __forceinline__ void copy_words(const uint8_t* a, int64_t src, uint8_t* b, int64_t dst, int bytes, int lane) {
  const int n = bytes / int(sizeof(W));
  const W* s = reinterpret_cast<const W*>(a + src * bytes);
  W* d = reinterpret_cast<W*>(b + dst * bytes);
  for (int i = lane; i < n; i += 32) d[i] = s[i];
}

// Copy row `src` of pool `a` to row `dst` of pool `b` (K and V, data and scales) with one warp: 16-byte words when the
// row sizes allow (a host pool is read over PCIe: 512-byte requests per warp instruction instead of 128), else 4-byte.
__device__ __forceinline__ void copy_row(const uint8_t* ak, const uint8_t* av, const uint8_t* aks, const uint8_t* avs,
                                         int64_t src, uint8_t* bk, uint8_t* bv, uint8_t* bks, uint8_t* bvs, int64_t dst,
                                         RowWords rw, int lane) {
  const int db = rw.data * 4, sb = rw.scale * 4;
  if (db % 16 == 0 && sb % 16 == 0) {
    copy_words<uint4>(ak, src, bk, dst, db, lane);
    copy_words<uint4>(av, src, bv, dst, db, lane);
    if (sb > 0) {
      copy_words<uint4>(aks, src, bks, dst, sb, lane);
      copy_words<uint4>(avs, src, bvs, dst, sb, lane);
    }
    return;
  }
  const uint32_t* sk = reinterpret_cast<const uint32_t*>(ak) + src * rw.data;
  const uint32_t* sv = reinterpret_cast<const uint32_t*>(av) + src * rw.data;
  uint32_t* dk = reinterpret_cast<uint32_t*>(bk) + dst * rw.data;
  uint32_t* dv = reinterpret_cast<uint32_t*>(bv) + dst * rw.data;
  for (int i = lane; i < rw.data; i += 32) dk[i] = sk[i], dv[i] = sv[i];
  if (rw.scale > 0) {
    const uint32_t* ss = reinterpret_cast<const uint32_t*>(aks) + src * rw.scale;
    const uint32_t* sw = reinterpret_cast<const uint32_t*>(avs) + src * rw.scale;
    uint32_t* ds = reinterpret_cast<uint32_t*>(bks) + dst * rw.scale;
    uint32_t* dw = reinterpret_cast<uint32_t*>(bvs) + dst * rw.scale;
    for (int i = lane; i < rw.scale; i += 32) ds[i] = ss[i], dw[i] = sw[i];
  }
}

// grid: (ceil(max_count / kWarps), rows), block: 32 * kWarps; a warp per (row, selected slot)
__global__ void kv_host_gather_kernel(blz_nvfp4_pool pool, blz_nvfp4_row_cache cache, const int32_t* __restrict__ table,
                                      int64_t ldt, const int32_t* __restrict__ counts, blz_nvfp4_pool out,
                                      int64_t out_stride, int32_t* __restrict__ fills, int32_t* __restrict__ fill_n,
                                      int fill_cap) {
  const int r = blockIdx.y, lane = threadIdx.x & 31;
  const int j = blockIdx.x * kWarps + (threadIdx.x >> 5);
  if (j >= counts[r]) return;
  const int32_t slot = table[int64_t(r) * ldt + j];
  const int64_t dst = int64_t(r) * out_stride + j;
  const RowWords rw = row_words(pool);
  uint8_t* ok = const_cast<uint8_t*>(out.k_data);
  uint8_t* ov = const_cast<uint8_t*>(out.v_data);
  uint8_t* oks = const_cast<uint8_t*>(out.k_scale);
  uint8_t* ovs = const_cast<uint8_t*>(out.v_scale);
  const int line = cache.lines > 0 ? slot % cache.lines : 0;
  if (cache.lines > 0 && cache.tag[line] == slot) {
    copy_row(cache.k_data, cache.v_data, cache.k_scale, cache.v_scale, line, ok, ov, oks, ovs, dst, rw, lane);
    return;
  }
  copy_row(pool.k_data, pool.v_data, pool.k_scale, pool.v_scale, slot, ok, ov, oks, ovs, dst, rw, lane);
  if (cache.lines > 0 && lane == 0) {
    const int i = atomicAdd(fill_n, 1);
    if (i < fill_cap) fills[3 * i] = line, fills[3 * i + 1] = slot, fills[3 * i + 2] = int32_t(dst);
  }
}

__global__ void kv_host_clear_kernel(blz_nvfp4_row_cache cache, const int32_t* __restrict__ fills,
                                     const int32_t* __restrict__ fill_n, int fill_cap) {
  const int n = min(*fill_n, fill_cap);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) cache.tag[fills[3 * i]] = -1;
}

// a warp per listed row: the first claimant of a cleared line writes it
__global__ void kv_host_claim_kernel(blz_nvfp4_row_cache cache, blz_nvfp4_pool out, const int32_t* __restrict__ fills,
                                     const int32_t* __restrict__ fill_n, int fill_cap) {
  const int n = min(*fill_n, fill_cap), lane = threadIdx.x & 31;
  const RowWords rw = row_words(out);
  for (int i = blockIdx.x * kWarps + (threadIdx.x >> 5); i < n; i += gridDim.x * kWarps) {
    const int line = fills[3 * i], slot = fills[3 * i + 1], src = fills[3 * i + 2];
    int won = 0;
    if (lane == 0) won = atomicCAS(&cache.tag[line], -1, slot) == -1;
    won = __shfl_sync(0xffffffffu, won, 0);
    if (!won) continue;
    copy_row(out.k_data, out.v_data, out.k_scale, out.v_scale, src, cache.k_data, cache.v_data, cache.k_scale,
             cache.v_scale, line, rw, lane);
  }
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_kv_host_gather(blz_nvfp4_pool pool, blz_nvfp4_row_cache cache, const int32_t* table,
                                          int64_t ldt, const int32_t* counts, int rows, int max_count,
                                          blz_nvfp4_pool out, int64_t out_stride, int32_t* fills, int32_t* fill_n,
                                          int fill_cap, cudaStream_t stream) {
  if (rows <= 0 || max_count <= 0) return cudaSuccess;
  if ((pool.heads * pool.dim) % 64 != 0) return cudaErrorInvalidValue;
  const dim3 grid((max_count + blaze::kWarps - 1) / blaze::kWarps, rows);
  blaze::kv_host_gather_kernel<<<grid, 32 * blaze::kWarps, 0, stream>>>(pool, cache, table, ldt, counts, out, out_stride,
                                                                        fills, fill_n, fill_cap);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_kv_host_fill(blz_nvfp4_row_cache cache, blz_nvfp4_pool out, const int32_t* fills,
                                        const int32_t* fill_n, int fill_cap, cudaStream_t stream) {
  if (cache.lines <= 0 || fill_cap <= 0) return cudaSuccess;
  const int blocks = (fill_cap + 255) / 256 < 1024 ? (fill_cap + 255) / 256 : 1024;
  blaze::kv_host_clear_kernel<<<blocks, 256, 0, stream>>>(cache, fills, fill_n, fill_cap);
  blaze::kv_host_claim_kernel<<<blocks * 4, 32 * blaze::kWarps, 0, stream>>>(cache, out, fills, fill_n, fill_cap);
  return cudaGetLastError();
}
