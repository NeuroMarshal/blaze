// Linear layers with FP8 E4M3 weights and 16-bit activations (W8A16), weights as stored: codes [n, k], an optional
// FP32 scale per [block_n, 128] block (block-quantized checkpoints) and an FP32 multiplier per output row (per-channel
// scales, or 1):
//     y[m, j] = alpha[j] * sum_c block_scale[j / block_n, c] * sum_{i in chunk c} x[m, i] * e4m3(w[j, i])
// C API (raw device pointers, caller's stream).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_FP8_MMA_MAX_ROWS 16

/// \brief y = x . W^T for 1..16 rows on tensor cores (BF16 MMAs over codes widened in registers, FP32 accumulation;
/// block scales applied to each 128-wide chunk's FP32 partial sums).
/// \param x            [rows, k] BF16, row stride ldx (elements), 16-byte aligned rows; k a multiple of 128
/// \param block_scale  null, or [n / block_n, k / 128] FP32 (block_n a multiple of 8)
/// \param alpha        [n] FP32
/// \param y            [rows, n] BF16, row stride ldy
cudaError_t blz_fp8_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const float* block_scale, int block_n,
                             const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                             cudaStream_t stream);

/// \brief out[j, i] = bf16(e4m3(w[j, i]) * block_scale[j / block_n, i / 128] * alpha[j]) (block_scale may be null).
cudaError_t blz_fp8_weight_dequant(const uint8_t* w, const float* block_scale, int block_n, const float* alpha, int n,
                                   int k, uint16_t* out, cudaStream_t stream);

#define BLZ_MXFP8_MMA_MAX_ROWS 64

/// \brief MXFP8 (OCP microscaling, ModelOpt quant_algo MXFP8) W8A16: y = x . W^T with W[j, i] = e4m3(w[j, i]) *
/// 2^(scale[j, i / 32] - 127), for 1..64 rows on tensor cores (BF16 MMAs over the codes widened and scaled exactly in
/// registers, FP32 accumulation). x [rows, k] BF16 (16-byte aligned rows, k a multiple of 128), w [n, k] E4M3,
/// scale [n, k / 32] E8M0, y [rows, n] BF16.
cudaError_t blz_mxfp8_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, uint16_t* y,
                               int64_t ldy, int rows, int n, int k, cudaStream_t stream);

/// \brief The MXFP8 weight as a BF16 matrix [n, k] (k a multiple of 32).
cudaError_t blz_mxfp8_weight_dequant(const uint8_t* w, const uint8_t* scale, int n, int k, uint16_t* out,
                                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
