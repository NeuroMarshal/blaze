// Kernels of the EAGLE-4 draft (one Llama-style layer over the target's hidden states and the next token, a low-rank
// candidate head, a full LM head read only at the candidates' rows). Raw device pointers (the LM head may be mapped
// host memory), caller's stream.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief RMSNorm of `rows` rows of `dim` bf16 values with row strides: out[r * ldo + i] = x[r * ldx + i] / rms * w[i]
/// (FP32, one rounding).
cudaError_t blz_rmsnorm_ld(const uint16_t* x, int64_t ldx, const uint16_t* w, float eps, int rows, int dim,
                           uint16_t* out, int64_t ldo, cudaStream_t stream);

/// \brief Rotary embedding in place, halves rotated (neox / Llama): for every token t and head h of
/// x[t * ld + h * head_dim ...], pairs (i, i + head_dim / 2) rotate by positions[t] * inv_freq[i].
cudaError_t blz_rope_neox(uint16_t* x, int64_t ld, int tokens, int heads, int head_dim, const int64_t* positions,
                          const float* inv_freq, cudaStream_t stream);

/// \brief Exact logits of the candidates: out[r, j] = hidden[r] . table[ids[r, j]] in FP32, for `c` candidates per row
/// (ids < 0: -inf). `table` [vocab, dim] bf16 may be mapped host memory (only the candidates' rows are read).
cudaError_t blz_candidate_logits(const uint16_t* hidden, int64_t ld_hidden, int rows, int dim, const uint16_t* table,
                                 const int32_t* ids, int c, float* out, cudaStream_t stream);

/// \brief Greedy pick per row: the candidate with the largest logit (ties: lower id) as token[r] (int64), and, with
/// `record` non-null, also record[r] = token (a chain step's draft output).
cudaError_t blz_candidate_pick(const float* logits, const int32_t* ids, int rows, int c, int64_t* token,
                               int64_t* record, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
