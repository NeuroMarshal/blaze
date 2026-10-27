// C API of dense attention straight from an NVFP4 KV pool (full-attention layers of Qwen3.5/3.6/3.8 dense models,
// --kv-cache-dtype nvfp4_4over6): decode and speculative verify read the 4-bit codes directly, flash-decode style
// (the context split over blocks, partial results merged by log-sum-exp); prefill dequantizes the prefix rows into a
// 16-bit workspace for the extend kernel. Raw device pointers, caller's stream; no torch dependency.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "blaze/qsa.h"  // blz_nvfp4_pool, blz_index_ptr

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Attention of q_len query tokens per request (decode: 1; verify: the draft tokens, causal among themselves)
/// over positions [0, kv_len) of the request, all of them already in the pool.
/// \param q            [batch * q_len, q_heads, 256] 16-bit, query token i of request b at row b * q_len + i
/// \param pool         K/V codes and scales; head dim 256, q_heads / pool.heads <= 16
/// \param req_to_token [requests, req_stride] pool slot of every position
/// \param kv_lens      [batch] positions per request, the q_len query tokens included (the last q_len)
/// \param splits       CTAs per (request, kv head, query group) along the context; 1 .. 64
/// \param partial_o    float32 [batch * q_len, q_heads, parts, 256] workspace, parts = blz_nvfp4_mha_parts(...)
/// \param partial_ml   float32 [batch * q_len, q_heads, parts, 2] workspace (running max in log2 units, sum)
/// \param out          [batch * q_len, q_heads, 256] with the given strides (elements)
cudaError_t blz_nvfp4_mha_decode(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                 blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                 blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len, int splits,
                                 float scale, float* partial_o, float* partial_ml, blz_half_t dtype, void* out,
                                 int64_t out_stride_tok, int64_t out_stride_head, cudaStream_t stream);

/// \brief blz_nvfp4_mha_decode with a head dim of 128 (BF16 pools only) and a sliding window: with window > 0 the
/// query at position p attends to positions p - window + 1 .. p (an EAGLE-4 draft layer); with causal == 0 every query
/// row also sees the positions after its own up to kv_len (a DFlash block: bidirectional within it, the window still
/// counted back from the query). The workspaces hold pool.dim values per (row, head, part).
cudaError_t blz_mha_decode_window(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                  blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                  blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len, int splits,
                                  float scale, int window, int causal, float* partial_o, float* partial_ml, blz_half_t dtype,
                                  void* out, int64_t out_stride_tok, int64_t out_stride_head, cudaStream_t stream);

/// \brief blz_mha_decode_window for one query row per batch entry (q_len 1, causal) whose last tail_len positions come
/// from the entry's own slot list tail[b * tail_stride + j] (position kv_len - tail_len + j) instead of req_to_token:
/// a draft tree step, its rows sharing the sequence's committed window and each with its own path.
cudaError_t blz_mha_decode_tail(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                                blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int splits, float scale,
                                int window, const int32_t* tail, int tail_len, int tail_stride, float* partial_o,
                                float* partial_ml, blz_half_t dtype, void* out, int64_t out_stride_tok,
                                int64_t out_stride_head, cudaStream_t stream);

/// \brief blz_nvfp4_mha_decode of a tree verify: query row i of request b sees the earlier positions and, of the
/// request's last q_len positions (the verify rows), those whose bit is set in tree_mask[b * q_len + i] (bit j: row j;
/// its ancestors and itself). q_len <= 64.
cudaError_t blz_nvfp4_mha_tree(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                               blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride,
                               blz_index_ptr req_indices, blz_index_ptr kv_lens, int batch, int q_len, int splits,
                               float scale, const uint64_t* tree_mask, float* partial_o, float* partial_ml,
                               blz_half_t dtype, void* out, int64_t out_stride_tok, int64_t out_stride_head,
                               cudaStream_t stream);

/// \brief Causal prefill attention of one sequence's q_len query tokens (its last q_len positions of kv_len, all in the
/// pool; slots in row `req` of req_to_token) from an NVFP4 pool, FlashAttention-2 style: a CTA of 16 tokens x the
/// query heads of a KV head reads each key tile once, dequantized once in shared memory; no split, no workspace.
/// q / out [q_len, q_heads, 256] bf16 (strides in elements: q multiple of 8, out of 2). A CTA takes 6, 4, 2 or 1 of a
/// KV head's query heads (the largest dividing the group).
/// f16_accumulate: the MMAs accumulate in FP16 in short runs summed in FP32 (twice the MMA rate on GeForce GPUs).
cudaError_t blz_nvfp4_mha_prefill(const void* q, int64_t q_stride_tok, int64_t q_stride_head, int q_heads,
                                  blz_nvfp4_pool pool, const int32_t* req_to_token, int64_t req_stride, int64_t req,
                                  int kv_len, int q_len, float scale, int f16_accumulate, blz_half_t dtype, void* out,
                                  int64_t out_stride_tok, int64_t out_stride_head, cudaStream_t stream);

/// \brief 1 when blz_nvfp4_mha_prefill runs on this build.
int blz_nvfp4_mha_prefill_supported(void);

/// \brief 1 when blz_nvfp4_mha_prefill takes this head geometry.
int blz_nvfp4_mha_prefill_group_ok(int q_heads, int kv_heads);

/// \brief The decode / verify kernels (blz_nvfp4_mha_decode, _tree, blz_mha_decode_window, _tail) accumulate in FP16 in
/// short runs summed in FP32 from now on (twice the MMA rate on GeForce GPUs: a verify of 8 tokens over a long context
/// is compute-bound), process-wide; 0: FP32 accumulation (the default).
void blz_mha_set_f16_accumulate(int on);

/// \brief The kernel of the dense causal NVFP4 calls (blz_nvfp4_mha_decode) of more than one MMA tile of rows per
/// request (a verify, a short prefill chunk; q_len x GQA group > 16), process-wide: 0 the register-dequantization kernel,
/// 1 the shared-dequantization kernel with FP32 accumulation (the default), 2 with FP16 accumulation.
void blz_mha_set_verify(int mode);

/// \brief CTAs a request takes per context split in a blz_nvfp4_mha_decode call of this shape when the
/// shared-dequantization kernel takes it (its occupancy is one CTA an SM); 0: the register kernel does.
int blz_nvfp4_mha_verify_tiles(int q_heads, int kv_heads, int q_len);

/// \brief Context parts of blz_nvfp4_mha_decode (the partial workspaces' third dimension): splits times the warps a CTA
/// runs along the context for this query geometry; 0 for an unsupported one.
int blz_nvfp4_mha_parts(int q_heads, int kv_heads, int q_len, int splits);

/// \brief Dequantize the pool rows of n slots (all heads) into [n, heads, dim] 16-bit K and V.
cudaError_t blz_nvfp4_dequant_rows(blz_nvfp4_pool pool, blz_index_ptr slots, int64_t n, blz_half_t dtype,
                                   void* out_k, void* out_v, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
