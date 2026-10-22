// Linear layers with NVFP4 weights and 16-bit activations (W4A16), weights as stored in ModelOpt / compressed-tensors
// checkpoints: codes [n, k / 2] (two E2M1 per byte, even element in the low nibble), E4M3 block scales [n, k / 16]
// and one FP32 multiplier per output row (the module's global scale; stacked modules keep their own):
//     y[m, j] = sum_i x[m, i] * e2m1(w[j, i]) * e4m3(scale[j, i / 16]) * alpha[j]
// C API (raw device pointers, caller's stream).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Most rows of x one GEMV launch takes (decode and verify steps); longer inputs go through blz_nvfp4_weight_dequant + GEMM.
#define BLZ_NVFP4_GEMV_MAX_ROWS 8

/// \brief y = x . W^T for rows <= BLZ_NVFP4_GEMV_MAX_ROWS, FP32 accumulation, BF16 output rounded once.
/// \param x     [rows, k] BF16, row stride ldx (elements), 16-byte aligned rows
/// \param w     [n, k / 2] codes, scale [n, k / 16] E4M3, alpha [n] FP32; k a multiple of 32
/// \param y     [rows, n] BF16, row stride ldy
cudaError_t blz_nvfp4_gemv(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, const float* alpha,
                           uint16_t* y, int64_t ldy, int rows, int n, int k, cudaStream_t stream);

/// \brief The same product on tensor cores for 1..16 rows (k a multiple of 128): BF16 MMAs over weights widened
/// in registers, FP32 accumulation.
cudaError_t blz_nvfp4_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                               const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                               cudaStream_t stream);

/// \brief out[j, i] = bf16(e2m1(w[j, i]) * e4m3(scale[j, i / 16]) * alpha[j]): the weight as a BF16 matrix [n, k].
cudaError_t blz_nvfp4_weight_dequant(const uint8_t* w, const uint8_t* scale, const float* alpha, int n, int k, uint16_t* out,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif
