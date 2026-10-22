// Elementwise, normalization, routing and recurrent-attention kernels of the model graph, and the 16-bit GEMM.
// bf16 tensors are passed as uint16_t bit patterns; all pointers are device memory; rows are contiguous unless a
// leading dimension (in elements) is given.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// normalization

/// \brief RMSNorm over `rows` rows of `dim` values. With `residual` non-NULL the input is h = x + residual (bf16 sum,
/// written back to `residual`), otherwise h = x. out = h / rms(h) * (zero_centered ? 1 + w : w), in fp32, rounded
/// once to bf16. `out` may alias `x`.
cudaError_t blz_rmsnorm(const uint16_t* x, uint16_t* residual, const uint16_t* w, float eps, int rows, int dim,
                        int zero_centered, uint16_t* out, cudaStream_t stream);

/// \brief Gated RMSNorm of linear attention, per (token, head) row of `dim` values: out = (w * bf16(x / rms(x))) *
/// silu(z). Head h of token t is at x + t * ld_x + h * dim (likewise z); out is dense [tokens, heads * dim].
cudaError_t blz_gated_rmsnorm(const uint16_t* x, int64_t ld_x, const uint16_t* z, int64_t ld_z, const uint16_t* w,
                              float eps, int tokens, int heads, int dim, uint16_t* out, cudaStream_t stream);

// attention

/// \brief Inputs of gated softmax attention: per-head RMSNorm of q and k, rotary embedding of the first `rotary_dim`
/// dims (halves rotated, cos / sin of position * inv_freq[i] times `rope_scale`), split of the output gate.
/// qg [tokens, heads, 2 * head_dim] (per head: query then gate), token rows `ld_qg` apart; k [tokens, kv_heads,
/// head_dim], token rows `ld_k` apart, rotated in place; positions [tokens] int64; inv_freq [rotary_dim / 2].
/// q_out [tokens, heads, head_dim], gate [tokens, heads * head_dim].
cudaError_t blz_attn_prep(const uint16_t* qg, int64_t ld_qg, uint16_t* k, int64_t ld_k, const int64_t* positions,
                          const uint16_t* q_norm,
                          const uint16_t* k_norm, float eps, int zero_centered, int tokens, int heads, int kv_heads,
                          int head_dim, int rotary_dim, const float* inv_freq, float rope_scale, uint16_t* q_out,
                          uint16_t* gate, cudaStream_t stream);

/// \brief x <- x * sigmoid(gate), `n` values.
cudaError_t blz_mul_sigmoid(uint16_t* x, const uint16_t* gate, int64_t n, cudaStream_t stream);

// MLP / MoE

/// \brief out[r, i] = silu(gate[r, i]) * up[r, i] for i < n; gate / up rows `ld_gate` / `ld_up` apart, out dense.
cudaError_t blz_silu_mul(const uint16_t* gate, int64_t ld_gate, const uint16_t* up, int64_t ld_up, int rows, int n,
                         uint16_t* out, cudaStream_t stream);

/// \brief out[r, :] = a[r, :] + sigmoid(gate_logit[r * ld_gate]) * b[r, :] (shared expert added to the routed
/// experts).
cudaError_t blz_add_sigmoid_scaled(const uint16_t* a, const uint16_t* b, const uint16_t* gate_logit, int64_t ld_gate,
                                   int rows, int dim, uint16_t* out, cudaStream_t stream);

/// \brief Softmax router: probabilities over `experts` logits, the `k` largest (ties: lower index), renormalized to
/// sum 1 when `renorm`. Logit rows `ld` apart; ids [tokens, k] int32, w [tokens, k] float.
cudaError_t blz_router_topk(const uint16_t* logits, int64_t ld, int tokens, int experts, int k, int renorm,
                            int32_t* ids, float* w, cudaStream_t stream);

/// \brief out[i] = a[i] + b[i] (bf16).
cudaError_t blz_add(const uint16_t* a, const uint16_t* b, int64_t n, uint16_t* out, cudaStream_t stream);

// gated deltanet

/// \brief beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias); a, b [tokens, v_heads] bf16.
cudaError_t blz_gdn_gate(const uint16_t* a, int64_t ld_a, const uint16_t* b, int64_t ld_b, const uint16_t* a_log,
                         const uint16_t* dt_bias, int tokens, int v_heads, float* g, float* beta, cudaStream_t stream);

/// \brief Causal depthwise convolution of `seqs` sequences with a rolling state, then SiLU. Sequence s covers tokens
/// [seq_start[s], seq_start[s + 1]) and state slot slots[s]: state [slots, channels, kernel - 1] bf16 holds the
/// previous inputs (zero for a new sequence) and is updated. x rows `ldx` apart, weight [channels, kernel],
/// out [tokens, channels].
cudaError_t blz_gdn_conv(const uint16_t* x, int64_t ldx, const int32_t* seq_start, const int32_t* slots, int seqs,
                         int channels, int kernel, const uint16_t* weight, uint16_t* state, uint16_t* out,
                         cudaStream_t stream);

/// \brief Gated delta rule over `seqs` sequences, token by token: S <- S * exp(g); delta = (v - S^T k) * beta;
/// S <- S + k delta^T; o = S^T q, with q and k L2-normalized and q scaled by dk^-1/2. qkv [tokens, 2 * k_heads * dk
/// + v_heads * dv] (q, k, v blocks; v head h reads k head h / (v_heads / k_heads)); g, beta [tokens, v_heads];
/// state [slots, v_heads, dk, dv] float, read and written; out [tokens, v_heads, dv]. dk = 128, dv % 32 == 0.
cudaError_t blz_gdn_recurrent(const uint16_t* qkv, const float* g, const float* beta, const int32_t* seq_start,
                              const int32_t* slots, int seqs, int tokens, int k_heads, int v_heads, int dk, int dv,
                              float* state, uint16_t* out, cudaStream_t stream);

// misc

/// \brief req_to_token[index[i]] = int32(slot[i]) for i < n: records the KV slots of a forward's new tokens.
cudaError_t blz_set_req_to_token(int32_t* req_to_token, const int64_t* index, const int64_t* slot, int n,
                                 cudaStream_t stream);

/// \brief Pull [p, p + bytes) into the L2 cache (SM90+: bulk prefetches of 64 KiB, else one per 128-byte line);
/// meant for a side stream, to stream the next layer's weights while the main stream waits on communication.
cudaError_t blz_l2_prefetch(const void* p, size_t bytes, cudaStream_t stream);

/// \brief dst[i, :] = src[rows[i], :] for i < n (rows of `dim` bf16 values, source rows `ld_src` apart).
cudaError_t blz_gather_rows(const uint16_t* src, int64_t ld_src, const int32_t* rows, int n, int dim, uint16_t* dst,
                            cudaStream_t stream);

/// \brief Row-wise argmax of bf16 logits (ties: lower index); rows `ld` apart.
cudaError_t blz_argmax_bf16(const uint16_t* logits, int rows, int vocab, int64_t ld, int32_t* out, cudaStream_t stream);

/// \brief FP8 E4M3 weight [n, k] with one scale per [block_n, block_k] block (bf16 or float, multiplied in) -> bf16.
cudaError_t blz_fp8_block_dequant(const uint8_t* w, const void* scale, int scale_is_f32, int n, int k, int block_n,
                                  int block_k, uint16_t* out, cudaStream_t stream);

// GEMM

typedef struct blz_gemm blz_gemm;

int blz_gemm_create(blz_gemm** out);
void blz_gemm_destroy(blz_gemm* g);

/// \brief out[m, n] = sum_k x[m, k] * w[n, k] (w row-major [n, k], i.e. a linear layer), bf16 in and out, fp32
/// accumulation. x rows `ldx` apart, out rows `ldo` apart.
cudaError_t blz_gemm_bf16(blz_gemm* g, const uint16_t* x, int64_t ldx, const uint16_t* w, uint16_t* out, int64_t ldo,
                          int m, int n, int k, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
