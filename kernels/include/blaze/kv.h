// C API of the NVFP4 KV cache kernels (raw device pointers, caller's stream; no torch dependency).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { BLZ_BF16 = 0, BLZ_FP16 = 1 } blz_half_t;
typedef enum { BLZ_I32 = 0, BLZ_I64 = 1 } blz_index_t;
/// KV pool formats: NVFP4 (codes [slots, heads, dim/2] + E4M3 block scales [.., dim/16]); FP8 E4M3 and BF16 rows
/// [slots, heads, dim] (no block scales). The per-layer global scales k_g / v_g apply to every format.
typedef enum { BLZ_KV_NVFP4 = 0, BLZ_KV_FP8 = 1, BLZ_KV_BF16 = 2 } blz_kv_format_t;

/// \brief Quantize K and V rows to NVFP4 and scatter them into a paged pool.
/// \param k,v          [tokens, heads, dim] 16-bit rows; the inner [heads, dim] block is contiguous
/// \param k_stride     elements between consecutive tokens of k (multiple of 8)
/// \param v_stride     elements between consecutive tokens of v (multiple of 8)
/// \param loc          [tokens] pool slot per token (int32 or int64); a negative slot is skipped
/// \param k_data       [slots, heads, dim/2] packed codes (the same for v_data)
/// \param k_scale      [slots, heads, dim/16] E4M3 block scales (the same for v_scale)
/// \param k_g,v_g      device pointers to the per-layer FP32 global scales
/// \param four_over_six nonzero: per block, the scale with the lowest squared error among max/6, max/4 and the E4M3
///                     codes kScaleSearchFirst..kScaleSearchLast steps from max/6's (blaze/nvfp4.h)
/// \return cudaSuccess or the launch error
cudaError_t blz_nvfp4_kv_store(const void* k, const void* v, int64_t k_stride, int64_t v_stride,
                               blz_half_t dtype, const void* loc, blz_index_t loc_type, int64_t tokens,
                               int heads, int dim, uint8_t* k_data, uint8_t* v_data, uint8_t* k_scale,
                               uint8_t* v_scale, const float* k_g, const float* v_g, int four_over_six,
                               cudaStream_t stream);

/// \brief Store bf16 K and V rows into an FP8 (value / g, E4M3 round to nearest, saturated) or BF16 (copied) pool
/// [slots, heads, dim]; arguments as blz_nvfp4_kv_store.
cudaError_t blz_kv_store_dense(const void* k, const void* v, int64_t k_stride, int64_t v_stride, const void* loc,
                               blz_index_t loc_type, int64_t tokens, int heads, int dim, blz_kv_format_t format,
                               uint8_t* k_data, uint8_t* v_data, const float* k_g, const float* v_g,
                               cudaStream_t stream);

/// \brief Randomized Hadamard rotation of bf16 head vectors in place: x <- H D x / sqrt(dim) (forward) or
/// D H x / sqrt(dim) (inverse), H the Walsh-Hadamard matrix, D a fixed +-1 diagonal (a hash of the index), so the
/// inverse undoes the forward exactly in real arithmetic. A cache of rotated K and V with Q rotated the same way gives
/// the same scores and a rotated output (orthogonal map): a few large channels spread over all of them before the
/// quantizer sees them (QuaRot, TurboQuant). Row (token t, head h) starts at x + t * tok_stride + h * head_stride.
/// \param dim 64, 128, 256 or 512
/// \param kind 0: the Hadamard rotation above; RotorQuant's block-diagonal family instead (fixed pseudo-random blocks,
///             the inverse their transpose): 1 Givens rotations of pairs (PlanarQuant), 2 unit quaternions on groups of
///             4 (IsoQuant), 3 3D rotors on groups of 3 (RotorQuant, Cl(3,0)); a block mixes only its own values
cudaError_t blz_rht_rows(uint16_t* x, int64_t tokens, int heads, int64_t tok_stride, int64_t head_stride, int dim,
                         int inverse, int kind, cudaStream_t stream);

/// \brief The forward rotation of Q [tokens, q_heads, dim] (rows q_tok_stride apart) and of K and V [tokens, kv_heads,
/// dim] (rows kv_tok_stride apart) in one launch.
cudaError_t blz_rht_qkv(uint16_t* q, int q_heads, int64_t q_tok_stride, uint16_t* k, uint16_t* v, int kv_heads,
                        int64_t kv_tok_stride, int64_t tokens, int dim, int kind, cudaStream_t stream);

/// \brief x <- inverse rotation of x, times sigmoid(gate) (x and gate dense [tokens, heads, dim]): the attention
/// output back from the rotated cache's basis and its output gate in one pass.
cudaError_t blz_rht_inverse_mul_sigmoid(uint16_t* x, const uint16_t* gate, int64_t tokens, int heads, int dim,
                                        int kind, cudaStream_t stream);

/// \brief Dequantize `rows` contiguous NVFP4 rows of `dim` values into 16-bit output [rows, dim].
cudaError_t blz_nvfp4_dequant(const uint8_t* data, const uint8_t* scale, const float* g, int64_t rows, int dim,
                              blz_half_t out_dtype, void* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
