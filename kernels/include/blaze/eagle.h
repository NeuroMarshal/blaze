// Kernels of the drafters: EAGLE-4 (one Llama-style layer over the target's hidden states and the next token, a
// low-rank candidate head, a full LM head read only at the candidates' rows) and DFlash2 (a block of mask tokens
// drafted in one pass: grouped convolutions over the block's positions, a candidate selector walking pairwise scores).
// Raw device pointers (the LM head may be mapped host memory), caller's stream.
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

/// \brief The best candidate of every row: pairs[r] = (max logit, id + col0) over logits[r][0..c) with ids[r][j] >= 0 (ties:
/// the lower id; no candidate: (-3e38, 0)), the (value, index) pairs of blz_argmax_merge.
cudaError_t blz_cand_argmax(const float* logits, const int32_t* ids, int rows, int c, int col0, float* pairs, cudaStream_t stream);

/// \brief a[i] = b[i] = (int64) src[i] for i < n (a greedy pick as the next draft input and as the call's record).
cudaError_t blz_widen_i32(const int32_t* src, int64_t* a, int64_t* b, int n, cudaStream_t stream);

/// \brief blz_rmsnorm_ld with the zero-centered weight: out = x / rms * (1 + w) (the Qwen3.5 / MTP norms).
cudaError_t blz_rmsnorm_ld_zc(const uint16_t* x, int64_t ldx, const uint16_t* w, float eps, int rows, int dim,
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

/// \brief RMSNorm of every head in place: x[t * ld + h * head_dim + i] for heads h < heads, the same weight w[head_dim]
/// for every head (q_norm / k_norm).
cudaError_t blz_head_rmsnorm(uint16_t* x, int64_t ld, int tokens, int heads, int head_dim, const uint16_t* w, float eps,
                             cudaStream_t stream);

/// \brief DFlash2 grouped convolution over the positions of each query block (rows of a request's block are
/// consecutive; position = row % block_size): out[r, c] = sum over taps t <= position of
/// (base[t, c] + delta[r, t, c / group]) * x[r - t, c], FP32 sums. delta [rows, taps, groups] bf16 with the given row
/// and tap strides (elements); base [taps, channels] bf16.
cudaError_t blz_dflash_grouped_conv(const uint16_t* x, int64_t ldx, const uint16_t* delta, int64_t delta_row,
                                    int64_t delta_tap, const uint16_t* base, uint16_t* out, int64_t ldo, int rows,
                                    int channels, int block_size, int group_size, int taps, cudaStream_t stream);

/// \brief DFlash2 candidate walk (greedy): for every item, step s picks the candidate c maximizing
/// unary[s, c] + sum_r pred[prev, r] * hproj[s, r] * succ[cand[s, c], r], where prev is the anchor token at step 0 and
/// the token picked at step s - 1 after it; ties: the lower candidate index.
/// \param cand    [items, steps, k] int32 token ids (k <= 32)
/// \param unary   [items, steps, k] f32 (the candidates' logits)
/// \param hproj   [items * steps, rank] bf16 (the hidden states through the selector's projection)
/// \param anchor  [items] int64: the token before each block's first drafted position
/// \param pred, succ  [vocab, rank] bf16 codebooks
/// \param out     [items, steps] int64 drafted tokens
cudaError_t blz_dflash_walk(const int32_t* cand, const float* unary, const uint16_t* hproj, const int64_t* anchor,
                            const uint16_t* pred, const uint16_t* succ, int items, int steps, int k, int rank,
                            int64_t* out, cudaStream_t stream);

/// \brief Merge the ranks' top-k lists of every row (pairs [world][rows][k + 1][2] floats as blz_topk_pack writes them,
/// all-reduced) into the row's best k: ids [rows][k] int32 and values [rows][k] f32 (ties: lower id). world * k <= 64.
cudaError_t blz_cand_merge(const float* pairs, int world, int rows, int k, int32_t* ids, float* vals,
                           cudaStream_t stream);

#ifdef __cplusplus
}
#endif
