// C API of the NVFP4 KV cache kernels (raw device pointers, caller's stream; no torch dependency).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { BLZ_BF16 = 0, BLZ_FP16 = 1 } blz_half_t;
typedef enum { BLZ_I32 = 0, BLZ_I64 = 1 } blz_index_t;

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

/// \brief Dequantize `rows` contiguous NVFP4 rows of `dim` values into 16-bit output [rows, dim].
cudaError_t blz_nvfp4_dequant(const uint8_t* data, const uint8_t* scale, const float* g, int64_t rows, int dim,
                              blz_half_t out_dtype, void* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
