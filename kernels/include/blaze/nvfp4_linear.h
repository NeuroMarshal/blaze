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
/// Most rows of blz_nvfp4_gemv_mma (four 16-row tiles).
#define BLZ_NVFP4_MMA_MAX_ROWS 64

/// \brief y = x . W^T for rows <= BLZ_NVFP4_GEMV_MAX_ROWS, FP32 accumulation, BF16 output rounded once.
/// \param x     [rows, k] BF16, row stride ldx (elements), 16-byte aligned rows
/// \param w     [n, k / 2] codes, scale [n, k / 16] E4M3, alpha [n] FP32; k a multiple of 32
/// \param y     [rows, n] BF16, row stride ldy
cudaError_t blz_nvfp4_gemv(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale, const float* alpha,
                           uint16_t* y, int64_t ldy, int rows, int n, int k, cudaStream_t stream);

/// \brief The same product on tensor cores for 1..BLZ_NVFP4_MMA_MAX_ROWS rows (k a multiple of 128): BF16 MMAs over
/// weights widened in registers once for every 16-row tile of the input, FP32 accumulation.
cudaError_t blz_nvfp4_gemv_mma(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                               const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k,
                               cudaStream_t stream);

/// \brief blz_nvfp4_gemv_mma with an explicit launch: `split` warps along K (2, 4, 8) and `tiles` 8-row weight tiles per
/// block (1, 2, 4); 0 for either picks blz_nvfp4_mma_default_cfg. cudaErrorInvalidValue for a combination that does not
/// fit (blz_nvfp4_mma_cfg_valid).
cudaError_t blz_nvfp4_gemv_mma_cfg(const uint16_t* x, int64_t ldx, const uint8_t* w, const uint8_t* scale,
                                   const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k, int split,
                                   int tiles, cudaStream_t stream);
int blz_nvfp4_mma_default_cfg(int rows, int k, int* split, int* tiles);
int blz_nvfp4_mma_cfg_valid(int rows, int split, int tiles);

/// \brief Inputs of blz_nvfp4_gemv_mma_f16_cfg: BF16 rows of x (row stride ldx) as FP16 [rows, k] (contiguous), each
/// row times a power of two 2^-shift[r] that brings its max |x| into [32, 64) (exact; the GEMV multiplies its outputs back).
cudaError_t blz_nvfp4_f16_rows(const uint16_t* x, int64_t ldx, int rows, int k, uint16_t* x16, int32_t* shift,
                               cudaStream_t stream);

/// \brief blz_nvfp4_gemv_mma_cfg on blz_nvfp4_f16_rows inputs with FP16 accumulation within each 128-wide chunk of K
/// and FP32 across chunks (twice the tensor rate of FP32 accumulation on GeForce cards). Partial sums stay finite for
/// block scales <= 1 (|code x block scale| <= 6); the caller keeps weights with larger scales on the BF16 kernel.
cudaError_t blz_nvfp4_gemv_mma_f16_cfg(const uint16_t* x16, const int32_t* shift, const uint8_t* w, const uint8_t* scale,
                                       const float* alpha, uint16_t* y, int64_t ldy, int rows, int n, int k, int split,
                                       int tiles, cudaStream_t stream);

/// \brief out[j, i] = bf16(e2m1(w[j, i]) * e4m3(scale[j, i / 16]) * alpha[j]): the weight as a BF16 matrix [n, k].
cudaError_t blz_nvfp4_weight_dequant(const uint8_t* w, const uint8_t* scale, const float* alpha, int n, int k, uint16_t* out,
                              cudaStream_t stream);

#ifdef __cplusplus
}
#endif
