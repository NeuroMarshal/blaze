// C API of the NVFP4 KV readers for Qwen Sparse Attention (QSA). They mirror SGLang's Triton gathers
// (_compact_kv and _pack_qsa_prefill_kv in layers/attention/qsa/sparse_attn.py) for an NVFP4 pool and
// dequantize on the way into the 16-bit scratch the attention kernels read.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "blaze/kv.h"

#ifdef __cplusplus
extern "C" {
#endif

/// \brief A device index array whose element type is decided at run time (int32 or int64).
typedef struct {
  const void* ptr;
  blz_index_t type;
} blz_index_ptr;

/// \brief KV pool of one layer: data [slots, heads, dim/2], scale [slots, heads, dim/16], global scales; with format
/// BLZ_KV_FP8 / BLZ_KV_BF16 (blaze/kv.h; only blz_nvfp4_mha_decode reads those) data rows [slots, heads, dim] of 1 / 2
/// bytes per value and no scales. Zero-initialized: NVFP4.
typedef struct {
  const uint8_t* k_data;
  const uint8_t* v_data;
  const uint8_t* k_scale;
  const uint8_t* v_scale;
  const float* k_g;
  const float* v_g;
  int heads;
  int dim;
  int format;
} blz_nvfp4_pool;

/// \brief Device cache of pool rows for a pool kept in host memory (--blaze-kv-host): decode gathers read cached rows
/// instead of crossing PCIe. Direct-mapped: row `slot` lives at line slot % lines when tag[line] == slot (-1: empty).
/// Lines are filled by blz_nvfp4_qsa_compact before it reads; writes to a slot must invalidate it
/// (blz_nvfp4_row_cache_invalidate) since a freed slot is reused for another token.
typedef struct {
  int32_t* tag;        // [lines]
  uint8_t* k_data;     // [lines, heads, dim/2]
  uint8_t* v_data;
  uint8_t* k_scale;    // [lines, heads, dim/16]
  uint8_t* v_scale;
  int lines;
} blz_nvfp4_row_cache;

/// \brief Gather the selected K/V rows of each decode row into a packed 16-bit scratch.
/// Row b reads positions indices[b, 0:topk] (-1 = unused) of request req_indices[b] through req_to_token,
/// keeps those with position < seq_lens[b] and column < cu_k[b+1] - cu_k[b], and writes them at
/// out[cu_k[b] + col]. With zero_fill_cols > 0 (page-aligned layout) every column below zero_fill_cols is
/// written, zeros for the invalid ones; otherwise only valid columns are written.
/// \param cache        null, or a row cache of `pool` (rows missing from it are copied in first, then read from it)
/// \param out_k,out_v  [rows, heads, dim] 16-bit scratch
cudaError_t blz_nvfp4_qsa_compact(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                  blz_index_ptr req_indices, const int32_t* indices, int64_t idx_stride,
                                  blz_index_ptr seq_lens, const int32_t* cu_k, int batch, int topk,
                                  int zero_fill_cols, const blz_nvfp4_row_cache* cache, blz_half_t out_dtype,
                                  void* out_k, void* out_v, cudaStream_t stream);

/// \brief Drop the cached rows of the n slots in `loc` (they are being rewritten).
cudaError_t blz_nvfp4_row_cache_invalidate(blz_nvfp4_row_cache cache, blz_index_ptr loc, int64_t n,
                                           cudaStream_t stream);

/// \brief Pack the whole K/V of each request (prefix + current chunk) for QSA prefill:
/// out[cu_k[b] + t] = pool[req_to_token[req_indices[b], t]] for t < cu_k[b+1] - cu_k[b].
/// \param max_k  largest cu_k[b+1] - cu_k[b] (bounds the grid)
cudaError_t blz_nvfp4_qsa_pack(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride0,
                               int64_t req_stride1, blz_index_ptr req_indices, const int32_t* cu_k, int batch,
                               int64_t max_k, blz_half_t out_dtype, void* out_k, void* out_v, cudaStream_t stream);

/// \brief QSA sparse attention of prefill rows straight from the NVFP4 pool (no 16-bit copy of the prefix).
/// Query row i of request b (cu_q[b] <= i < cu_q[b+1]) and q head h of KV group g = h / (q_heads / pool.heads) attends
/// to the positions t = indices[i * idx_stride_row + g * idx_stride_group + j] for j < min(topk, its causally visible
/// count kv_lens[b] - (cu_q[b+1] - i) + 1), skipping t < 0 and t >= kv_lens[b]; their K/V are
/// pool[req_to_token[req_indices[b] * req_stride + t]]. Same arithmetic as SGLang's _sparse_gqa_chunk_prefill
/// (q * scale * log2 e rounded to 16 bits, fp32 scores and sums, probabilities rounded to 16 bits for P V); a row
/// with no valid position gets zeros. bf16, head dim 256, up to 16 q heads per KV head; else cudaErrorNotSupported.
/// \param q         [total_q, q_heads, 256] (head dim contiguous)
/// \param stage_cu  null: `pool` is the KV pool, rows found through req_to_token; else `pool` is a staging pool
///                  (blz_nvfp4_qsa_stage) and position t of request b is its row stage_cu[b] + t
/// \param out       same layout as q
cudaError_t blz_nvfp4_qsa_sparse_prefill(const void* q, int64_t q_stride_row, int64_t q_stride_head, int q_heads,
                                         blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                         blz_index_ptr req_indices, const int32_t* indices, int64_t idx_stride_row,
                                         int64_t idx_stride_group, int topk, const int32_t* cu_q,
                                         const int32_t* kv_lens, const int32_t* stage_cu, int batch, int total_q,
                                         float scale, blz_half_t dtype, void* out, int64_t out_stride_row,
                                         int64_t out_stride_head, cudaStream_t stream);

/// \brief Copy the raw NVFP4 rows (data and scales, K and V) of positions [0, cu_k[b+1] - cu_k[b]) of each request
/// to rows cu_k[b] + t of `out` (same heads / dim, contiguous). Reads a pool kept in mapped host memory with
/// sequential PCIe transfers, so prefill attention runs on device memory.
/// \param max_k  largest cu_k[b+1] - cu_k[b] (bounds the grid)
cudaError_t blz_nvfp4_qsa_stage(blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                blz_index_ptr req_indices, const int32_t* cu_k, int batch, int64_t max_k,
                                blz_nvfp4_pool out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
