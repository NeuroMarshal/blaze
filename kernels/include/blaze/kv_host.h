// KV pools in pinned host memory (--kv-host): the decode reads the slots QSA selected through a device row cache into
// a compact device pool, so the attention kernels read device memory and only missing rows cross PCIe. The cache is
// blz_nvfp4_row_cache (blaze/qsa.h): direct-mapped, row `slot` at line slot % lines when tag[line] == slot.
//
// A step's rows go through three kernels so that the cache is never read and written at once:
//   blz_kv_host_gather  copies every selected row into the compact pool (from its line when cached, else from host
//                       memory) and lists the missing ones (line, slot, compact row);
//   blz_kv_host_fill    clears the listed lines' tags, then one claimant per line (atomicCAS) writes its row and tag.
// Rows rewritten by the KV store are dropped with blz_nvfp4_row_cache_invalidate first.
// Raw device pointers (the host pool: its mapped device addresses), caller's stream.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "blaze/qsa.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Row r < rows reads the slots table[r * ldt + j], j < counts[r], of `pool` (host memory) into rows
/// r * out_stride + j of `out` (device; same heads / dim / format, contiguous rows). Missing rows are appended to
/// fills (int32 triples: line, slot, out row; at most fill_cap, the count in *fill_n, which the caller zeroes first).
cudaError_t blz_kv_host_gather(blz_nvfp4_pool pool, blz_nvfp4_row_cache cache, const int32_t* table, int64_t ldt,
                               const int32_t* counts, int rows, int max_count, blz_nvfp4_pool out, int64_t out_stride,
                               int32_t* fills, int32_t* fill_n, int fill_cap, cudaStream_t stream);

/// \brief Put the rows blz_kv_host_gather listed (now in `out`) into the cache. `fill_cap`: the grid bound (the count
/// is read on the device).
cudaError_t blz_kv_host_fill(blz_nvfp4_row_cache cache, blz_nvfp4_pool out, const int32_t* fills, const int32_t* fill_n,
                             int fill_cap, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
