// Device-side sampling and logit utilities. Raw device pointers, caller's stream, CUDA-graph capturable.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_TOPK_MAX 64
/// Largest k of the standalone BF16 blz_topk utility.
#define BLZ_TOPK_LIMIT 128

/// \brief Per-token logit edits (vLLM's logits processors in one pass): for every entry e of row r's list
/// (ids[off[r] .. off[r+1]], ids outside the shard skipped) logits[r, id - col0] = (x > 0 ? x * a : x * b) + c with
/// (a, b, c) = coef[3 * e .. 3 * e + 3]: repetition penalty (a = 1 / p, b = p), presence / frequency / logit bias
/// (c), bans (c = -inf).
cudaError_t blz_logit_edits(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, const int32_t* ids,
                            const float* coef, const int32_t* off, cudaStream_t stream);

/// \brief Structured outputs: tokens a grammar forbids get -inf. Row r uses mask row mask_row[r] (< 0: no mask) of
/// `words` 32-bit words, bit t of the mask allowing global token t (tokens past words * 32 are forbidden); the shard
/// holds tokens col0 .. col0 + ncols.
cudaError_t blz_token_mask(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, int words, const uint32_t* mask,
                           const int32_t* mask_row, cudaStream_t stream);

/// \brief log(sum(exp(logits))) of each row of a shard (the softmax normalizer, for logprobs): lse[r].
cudaError_t blz_row_lse(const uint16_t* logits, int64_t ld, int rows, int ncols, float* lse, cudaStream_t stream);

/// \brief OpenAI presence / frequency penalties on bf16 logits [rows, ld] of a vocabulary shard starting at col0 with
/// ncols columns: for every (id, count) of row r's list (ids[off[r] .. off[r+1]], counts likewise; ids outside the shard
/// are skipped) logits[r, id - col0] -= presence[r] + frequency[r] * count.
cudaError_t blz_apply_penalties(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, const int32_t* ids,
                                const int32_t* counts, const int32_t* off, const float* presence,
                                const float* frequency, cudaStream_t stream);

/// \brief Row r of `pairs` ([rows][k + 1][2] floats): (vals[r, j], float(ids[r, j])) for j < k (ids < 2^24 are exact),
/// then (lse[r], -2): candidates of a vocabulary shard in a form a float all-reduce can merge (every other rank's slot
/// is zero). lse may be null (then 0).
cudaError_t blz_topk_pack(const float* vals, const int32_t* ids, const float* lse, int rows, int k, float* pairs,
                          cudaStream_t stream);

/// \brief Bytes of workspace blz_topk needs for `rows` rows of `vocab` columns.
size_t blz_topk_workspace_bytes(int rows, int vocab);

/// \brief The K largest logits of each row, sorted descending (ties: lower index first): vals [rows, K] f32,
/// ids [rows, K] int32 (column + col0; -1 past the row's size). Exact. K <= BLZ_TOPK_LIMIT.
cudaError_t blz_topk(const uint16_t* logits, int64_t ld, int rows, int vocab, int col0, int k, float* vals,
                     int32_t* ids, void* workspace, cudaStream_t stream);

typedef struct {
  int32_t stages[8], len, top_k;
  float temperature, top_p, min_p, top_nsigma, top_h, draw;
  int32_t need_lse;
} blz_sampling_row;

size_t blz_sampling_workspace_bytes(int rows, int vocab);
cudaError_t blz_sampling_pack(const uint16_t* local, int local_vocab, int vocab, int rank,
                             int rows, float* full, cudaStream_t stream);
/// Full logits stay on the device. Results are 64 alternatives, their LSE, and (canonical chosen logit, id).
/// Alternatives and LSE are canonical when logprobs is true; otherwise alternatives are sampling diagnostics.
/// execution: slot bits 0..7 active, 8..15 need ranking, 16..23 edit logits. Bit 24 permits top-64 selection
/// only when every row's first active prefix filter is top_k <= 64. valid_ids may be null (all IDs exist).
cudaError_t blz_sampling_run(const float* raw, int rows, int vocab, int true_vocab, const uint32_t* valid_ids,
                            const blz_sampling_row* programs, const int32_t* edit_ids,
                            const float* edits, const int32_t* offsets,
                            const uint32_t* masks, const int32_t* mask_rows, int mask_words,
                            uint32_t execution, bool logprobs, void* workspace, size_t bytes,
                            float* results, cudaStream_t stream);
#ifdef __cplusplus
}
#endif
