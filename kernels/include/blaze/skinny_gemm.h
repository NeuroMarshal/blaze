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

/// Most rows of x one launch takes.
#define BLZ_SKINNY_MAX_ROWS 16

/// \brief y = x . w^T with FP32 accumulation, rounded to the 16-bit type once.
/// \param x      [rows, k], row stride x_stride (elements); rows <= BLZ_SKINNY_MAX_ROWS
/// \param w      [n, k] contiguous; k a multiple of 8, x and w 16-byte aligned
/// \param y      [rows, n] contiguous
cudaError_t blz_skinny_gemm(const void* x, int64_t x_stride, const void* w, void* y, int rows, int n, int k,
                            blz_half_t dtype, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
