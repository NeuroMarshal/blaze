// Kernels of the Qwen4-Exp (Qwen3.8 Flash Next) layer structure: hyper-connection streams, and the per-layer n-gram
// embedding (PLE). Arithmetic follows SGLang's Qwen4ExpForConditionalGeneration (FP32 inside, BF16 storage).
// C API (raw device pointers, caller's stream; CUDA-graph capturable).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief Gemma-style RMSNorm over groups: x [rows, groups * dim] bf16, each group of `dim` normalized on its own,
/// out = bf16(x * rsqrt(mean(x^2) + eps) * (1 + w)), w [groups * dim].
cudaError_t blz_grouped_rmsnorm(const uint16_t* x, const uint16_t* w, float eps, int rows, int groups, int dim,
                                uint16_t* out, cudaStream_t stream);

/// \brief Hyper-connection gated combine, in place on the streams:
/// a[m, c] = 2 * sigmoid(dot(normed[m], inject[c]) / hc), streams[m, c * dim + i] = bf16(streams + a[m, c] * y[m, i]).
/// streams, normed [rows, hc * dim]; inject [hc, hc * dim]; y [rows, dim], rows `ld_y` apart.
cudaError_t blz_hc_combine(uint16_t* streams, const uint16_t* normed, const uint16_t* inject, const uint16_t* y,
                           int64_t ld_y, int rows, int hc, int dim, cudaStream_t stream);

/// \brief The same combine for few rows (decode): the gate dot products split over column chunks of 1024 (partials in
/// `partial`, [rows, ceil(hc * dim / 1024), hc] floats), summed in a fixed order by every block of the update.
cudaError_t blz_hc_combine_split(uint16_t* streams, const uint16_t* normed, const uint16_t* inject, const uint16_t* y,
                                 int64_t ld_y, int rows, int hc, int dim, float* partial, cudaStream_t stream);

/// \brief streams[m, c * dim + i] = x[m, i] for every stream c (the layer stack's input).
cudaError_t blz_hc_widen(const uint16_t* x, int rows, int hc, int dim, uint16_t* streams, cudaStream_t stream);

/// \brief N-gram table rows of every token (PLE): for token j of sequence s (rows seq_start[s] .. seq_start[s+1]) the
/// window is the sequence's two previous tokens hist[slots[s]] followed by the step's tokens; an n-gram does not reach
/// across an EOS (a missing neighbour reads as eos). 2-grams mix = t0 * m0 ^ t1 * m1, 3-grams ^= t2 * m2 (wrapping
/// int64); head h's row = mix mod sizes[h] (non-negative) + offsets[h], heads_per_ngram heads per n-gram size.
/// \param max_tokens  upper bound of seq_start[seqs] (the grid)
/// \param ids  [tokens, 2 * heads_per_ngram] int64
/// \param hist [slots, 2] int64, updated to the last two tokens of each sequence when `update_hist`
cudaError_t blz_ngram_hash(const int64_t* tokens, const int32_t* seq_start, const int32_t* slots, int seqs,
                           int max_tokens, int64_t* hist, int update_hist, const int64_t* multipliers,
                           const int64_t* sizes, const int64_t* offsets, int heads_per_ngram, int64_t eos,
                           int64_t* ids, cudaStream_t stream);

/// \brief PLE gate: g[m, c] = sigmoid(signed_sqrt(dot(key_n[m, c], query_n[m, c]) / sqrt(dim))),
/// gated[m, c * dim + i] = bf16(g[m, c] * value[m, i]); key_n, query_n, gated [rows, hc * dim], value [rows, dim].
cudaError_t blz_ple_gate(const uint16_t* key_n, const uint16_t* query_n, const uint16_t* value, int rows, int hc,
                         int dim, uint16_t* gated, cudaStream_t stream);

/// \brief PLE short convolution and residual add: per channel c of gated_n ([rows, channels]), a depthwise conv of
/// `kernel` taps `dilation` rows apart over the sequence (state [slots, channels, (kernel - 1) * dilation] bf16 holds the
/// previous inputs), streams[m, c] = bf16(streams + bf16(gated + bf16(silu(conv)))). With `write_state` the state
/// advances to the sequence's last inputs.
cudaError_t blz_ple_conv(const uint16_t* gated_n, const uint16_t* gated, const int32_t* seq_start,
                         const int32_t* slots, int seqs, int channels, int kernel, int dilation,
                         const uint16_t* weight, uint16_t* state, int write_state, uint16_t* streams,
                         cudaStream_t stream);

/// \brief PLE n-gram table service (blaze::PleService): rows gathered from the safetensors shards on demand.
typedef struct blz_ple blz_ple;
int blz_ple_create(const char* path, int max_ids, int threads, double cache_gib, blz_ple** out, char* err, size_t err_len);
void blz_ple_destroy(blz_ple* p);
int blz_ple_row_dim(const blz_ple* p);
/// \brief Publish n row ids (device int64); the CPU service starts reading. Capturable.
cudaError_t blz_ple_request(blz_ple* p, const int64_t* ids, int n, cudaStream_t stream);
/// \brief Wait for the rows of the last request and copy them into out [n, row_dim] bf16. Capturable.
cudaError_t blz_ple_fetch(blz_ple* p, uint16_t* out, int n, cudaStream_t stream);
/// \brief Host: start reading these rows (host ids) into the page cache in the background.
void blz_ple_warm(blz_ple* p, const int64_t* host_ids, int n);

#ifdef __cplusplus
}
#endif
