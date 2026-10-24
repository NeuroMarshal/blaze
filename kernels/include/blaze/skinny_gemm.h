// Skinny 16-bit linear layers: y[M, N] = x[M, K] . w[N, K]^T for a few rows (decode / verify tokens) and a small or
// moderate N, where cuBLAS's GEMV launches too few blocks (a 48 x 5120 BF16 projection: ~25 us instead of ~3).
// C API (raw device pointers, caller's stream; no torch dependency).
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "blaze/kv.h"  // blz_half_t

#ifdef __cplusplus
extern "C" {
#endif

/// Most rows of x blz_skinny_gemm takes (and the rows of one pass of blz_skinny_gemm_ex).
#define BLZ_SKINNY_MAX_ROWS 16
/// Most rows of x blz_skinny_gemm_ex takes (passes of BLZ_SKINNY_MAX_ROWS; beyond a few passes a tiled GEMM wins).
#define BLZ_SKINNY_EX_MAX_ROWS 64

/// \brief y = x . w^T with FP32 accumulation, rounded to the 16-bit type once.
/// \param x      [rows, k], row stride x_stride (elements); rows <= BLZ_SKINNY_MAX_ROWS
/// \param w      [n, k] contiguous; k a multiple of 8, x and w 16-byte aligned
/// \param y      [rows, n] contiguous
cudaError_t blz_skinny_gemm(const void* x, int64_t x_stride, const void* w, void* y, int rows, int n, int k,
                            blz_half_t dtype, cudaStream_t stream);

/// \brief blz_skinny_gemm with a row stride of y (a part of a wider output: y_stride >= n) and up to
/// BLZ_SKINNY_EX_MAX_ROWS rows (verify and draft-tree steps). Wide n shares each load of x among 2 or 4 columns.
cudaError_t blz_skinny_gemm_ex(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride, int rows,
                               int n, int k, blz_half_t dtype, cudaStream_t stream);

/// \brief blz_skinny_gemm_ex with the columns per block given (1, 2, 4; 0: chosen from n and the SM count), the
/// configuration the autotuner measures (crates/blaze/src/tune.rs).
cudaError_t blz_skinny_gemm_cfg(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride, int rows,
                                int n, int k, blz_half_t dtype, int cols_per_block, cudaStream_t stream);

/// Most rows and largest k of blz_gemv_wide.
#define BLZ_GEMV_WIDE_MAX_ROWS 4
#define BLZ_GEMV_WIDE_MAX_K 16384
/// Shared memory blz_gemv_wide may take for x (rows * k * 2 bytes).
#define BLZ_GEMV_WIDE_MAX_SMEM (96 * 1024)

/// \brief BF16 y = x . w^T for a few rows and a large n (an LM head: 248k x 2048): one warp per output column, x in
/// shared memory, every lane with all its 16-byte weight loads in flight at once; FP32 sums, one rounding.
/// \param x  [rows, k] bf16, row stride x_stride; rows <= BLZ_GEMV_WIDE_MAX_ROWS, k <= BLZ_GEMV_WIDE_MAX_K, k % 256 == 0,
///           rows * k * 2 <= BLZ_GEMV_WIDE_MAX_SMEM
/// \param w  [n, k] bf16 contiguous, 16-byte aligned
/// \param y  [rows, n] bf16, row stride y_stride
cudaError_t blz_gemv_wide(const void* x, int64_t x_stride, const void* w, void* y, int64_t y_stride, int rows, int n,
                          int k, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
