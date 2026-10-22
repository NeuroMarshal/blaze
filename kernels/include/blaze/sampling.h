// Sampling support: penalties applied to the logits in place, and the exact top-K candidates of each row (K <= 64),
// from which the host draws with temperature / top-k / top-p / min-p. C API (raw device pointers, caller's stream;
// CUDA-graph capturable).
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_TOPK_MAX 64

/// \brief OpenAI presence / frequency penalties on bf16 logits [rows, ld] of a vocabulary shard starting at col0 with
/// ncols columns: for every (id, count) of row r's list (ids[off[r] .. off[r+1]], counts likewise; ids outside the shard
/// are skipped) logits[r, id - col0] -= presence[r] + frequency[r] * count.
cudaError_t blz_apply_penalties(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, const int32_t* ids,
                                const int32_t* counts, const int32_t* off, const float* presence,
                                const float* frequency, cudaStream_t stream);

/// \brief pairs[i] = (vals[i], float(ids[i])) for n entries (ids < 2^24 are exact): candidates of a vocabulary shard
/// in a form a float all-reduce can merge (every other rank's slot is zero).
cudaError_t blz_topk_pack(const float* vals, const int32_t* ids, int n, float* pairs, cudaStream_t stream);

/// \brief Bytes of workspace blz_topk needs for `rows` rows of `vocab` columns.
size_t blz_topk_workspace_bytes(int rows, int vocab);

/// \brief The K largest logits of each row, sorted descending (ties: lower index first): vals [rows, K] f32,
/// ids [rows, K] int32 (column + col0; -1 past the row's size). Exact. K <= BLZ_TOPK_MAX.
cudaError_t blz_topk(const uint16_t* logits, int64_t ld, int rows, int vocab, int col0, int k, float* vals,
                     int32_t* ids, void* workspace, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
