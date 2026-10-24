// Qwen sparse attention (QSA) indexer of Qwen4-Exp: per token an index query (heads x dim) and one index key; every
// `ratio` consecutive keys of a sequence are averaged into a compressed block key; a query scores the sequence's
// visible blocks (sum over heads of relu(q . k) / sqrt(dim)), keeps the best `block_topk` and attends to their tokens
// plus the tokens of its own incomplete block. Arithmetic follows SGLang's QSAIndexer (Gemma norms, rope on the
// first rotary_dim values, FP32 averages / scores, BF16 storage).
//
// KV slots come in pages of `ratio` slots (position p of a sequence lives in slot page[p / ratio] * ratio +
// p % ratio), so the compressed key of block b is stored at index page[b] = slot(b * ratio) / ratio of the compressed
// cache, next to its tokens, and is shared like them.
// C API (raw device pointers, caller's stream; CUDA-graph capturable).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Index queries: q = rope(gemma_norm(proj[t, col + h * dim ...], w)) for every token t and head h.
/// \param proj   [tokens, ld] bf16, the index projection at column `col` (heads * dim query values, then the key)
/// \param out    [tokens, heads, dim] bf16
cudaError_t blz_qsa_index_q(const uint16_t* proj, int64_t ld, int col, const int64_t* positions, const uint16_t* norm_w,
                            float eps, int tokens, int heads, int dim, int rotary_dim, const float* inv_freq,
                            float rope_scale, uint16_t* out, cudaStream_t stream);

/// \brief Index keys of a step: the blocks completed by its tokens are compressed (mean of the block's raw keys, Gemma
/// norm, rope at the block's first position) into comp[page]; then the raw keys of every sequence's trailing
/// incomplete block are kept in its ring (ring[row, p % ratio]) for the next step.
/// \param proj       [tokens, ld] bf16, the raw key at column `kcol`
/// \param tok_row    [tokens] the KV-table row (sequence) of each token
/// \param kv_lens    per token: its sequence's length after the step (positions < kv_len are written)
/// \param req_to_token [rows, rt_stride] KV slot of each position
/// \param ring       [rows, ratio, dim] bf16
/// \param comp       [pages, dim] bf16
cudaError_t blz_qsa_compress(const uint16_t* proj, int64_t ld, int kcol, const int64_t* positions,
                             const int32_t* tok_row, const int32_t* tok_kv_len, int tokens, int dim, int ratio,
                             const uint16_t* norm_w, float eps, int rotary_dim, const float* inv_freq, float rope_scale,
                             const int32_t* req_to_token, int64_t rt_stride, uint16_t* ring, uint16_t* comp,
                             cudaStream_t stream);

/// \brief Block scores of query rows [row0, row0 + rows): scores[r - row0, b] = sum_h relu(q[r, h] . K[b]) / sqrt(dim)
/// for blocks b < visible(r) = (positions[r] + 1) / ratio of the row's sequence, -inf beyond (up to max_blocks).
/// K[b] = comp[req_to_token[tok_row[r], b * ratio] / ratio]. dim = 128, heads <= 8. With same_seq (every row of the
/// call belongs to one sequence, a prefill span) groups of 16 rows share their key loads.
cudaError_t blz_qsa_scores(const uint16_t* q, int heads, int dim, const int64_t* positions, const int32_t* tok_row,
                           int row0, int rows, int same_seq, int ratio, const int32_t* req_to_token, int64_t rt_stride,
                           const uint16_t* comp, int max_blocks, float* scores, int64_t ld_scores, cudaStream_t stream);

/// \brief The positions each query row attends to, as KV slots: the best `block_topk` visible blocks (all of them when
/// fewer; the scores of rows [row0, row0 + rows) at `scores`) expanded to their `ratio` tokens, plus the tokens of the
/// row's own incomplete block; slots written to table[r, 0 .. count) and count to counts[r] (r - row0 rows of
/// table / counts are written, at row r). With scores == null the visible blocks are taken, the newest block_topk of
/// them when there are more. A row takes up to block_topk * ratio + ratio - 1 slots: table_stride must hold them.
cudaError_t blz_qsa_select(const float* scores, int64_t ld_scores, const int64_t* positions, const int32_t* tok_row,
                           int row0, int rows, int ratio, int block_topk, const int32_t* req_to_token,
                           int64_t rt_stride, int32_t* table, int64_t table_stride, int32_t* counts,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
