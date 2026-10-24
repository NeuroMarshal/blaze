// Qwen3-VL / Qwen3.5 vision tower on the GPU: the elementwise kernels, the 2D rotary attention and the position-embedding
// interpolation around the cuBLAS GEMMs (blz_gemm_bf16). bf16 tensors are uint16_t bit patterns; all pointers are device
// memory. Patches are in spatial-merge order (each merge x merge window of the patch grid consecutive), as the image
// processor lays them out.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief LayerNorm over `dim` values of `rows` rows, fp32 statistics, out = (x - mean) / sqrt(var + eps) * gamma + beta
/// rounded once to bf16. `out` may alias `x`.
cudaError_t blz_vis_layernorm(const uint16_t* x, const uint16_t* gamma, const uint16_t* beta, float eps, int rows, int dim,
                              uint16_t* out, cudaStream_t stream);

/// \brief The input of the first block: x[i] = bf16(bf16(x[i] + bias) + bf16(pos)) in place, with x the patch-embedding
/// GEMM's output [grid_h * grid_w, dim] and pos the learned table [side * side, dim] resampled bilinearly (corners
/// aligned, like F.interpolate(align_corners=True)) to the grid, every patch in merge order.
cudaError_t blz_vis_embed_pos(uint16_t* x, const uint16_t* bias, const uint16_t* table, int side, int grid_h, int grid_w,
                              int merge, int dim, cudaStream_t stream);

/// \brief Splits the QKV GEMM's output [n, 3, heads, head_dim] (bias added here), rotates q and k by the 2D rotary
/// embedding (head_dim / 4 frequencies of theta for the patch's row, as many for its column, over the two halves of the
/// head) in fp32 and writes q, k, v as [n, heads * head_dim] bf16. n = grid_h * grid_w patches in merge order.
cudaError_t blz_vis_qkv_rope(const uint16_t* qkv, const uint16_t* bias, int grid_h, int grid_w, int merge, int heads,
                             int head_dim, float theta, uint16_t* q, uint16_t* k, uint16_t* v, cudaStream_t stream);

/// \brief Bidirectional softmax attention of one image (every patch sees every patch), head_dim 72: o = softmax(q k^T /
/// sqrt(d)) v per head, flash-style tiles on the tensor cores. q, k, v, o [n, heads * 72] bf16.
cudaError_t blz_vis_attention(const uint16_t* q, const uint16_t* k, const uint16_t* v, int n, int heads, int head_dim,
                              uint16_t* o, cudaStream_t stream);

/// \brief out[i] = bf16(res[i] + bf16(g[i] + bias[col])) for rows x dim (a projection's bias and the residual add).
/// `out` may alias `res`.
cudaError_t blz_vis_bias_residual(const uint16_t* g, const uint16_t* bias, const uint16_t* res, int rows, int dim,
                                  uint16_t* out, cudaStream_t stream);

/// \brief x[i] = bf16(act(bf16(x[i] + bias[col]))) in place; act: 0 none, 1 GELU tanh approximation, 2 exact GELU.
cudaError_t blz_vis_bias_act(uint16_t* x, const uint16_t* bias, int rows, int dim, int act, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
