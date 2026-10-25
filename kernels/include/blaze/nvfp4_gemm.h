// NVFP4 GEMMs of many rows (prefill): y[m, n] = x[m, k] . W[n, k]^T with W an NVFP4 weight (E2M1 codes two per
// byte [n, k / 2], an E4M3 scale per 16 values [n, k / 16], an FP32 factor per row alpha[n]).
//
// W4A4 (blz_nvfp4_gemm_w4a4): the rows of x quantized to NVFP4 first (blz_nvfp4_quant_rows: an FP32 scale per row
// from its largest value, E4M3 scales per 16 values below it), then the FP4 tensor cores with their block scales
// (mma.sync kind::mxf4nvf4.block_scale, SM120a), FP32 accumulation, y = acc * row scale * alpha in BF16.
// blz_nvfp4_gemm_w4a4_supported says whether this build and the current GPU have it.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief x [m, k] bf16 (row stride ldx) to NVFP4: codes [m, k / 2], E4M3 block scales [m, k / 16], FP32 row scales
/// [m]. k a multiple of 16.
cudaError_t blz_nvfp4_quant_rows(const uint16_t* x, int64_t ldx, int m, int k, uint8_t* codes, uint8_t* scales,
                                 float* row_scale, cudaStream_t stream);

/// \brief 1 when the W4A4 GEMM runs on the current GPU (built for its architecture-specific variant, SM120a).
int blz_nvfp4_gemm_w4a4_supported(void);

/// \brief y [m, n] bf16 (row stride ldy) from quantized rows (blz_nvfp4_quant_rows) and an NVFP4 weight. k a multiple
/// of 128; any m and n.
cudaError_t blz_nvfp4_gemm_w4a4(const uint8_t* x_codes, const uint8_t* x_scales, const float* x_row_scale,
                                const uint8_t* w_codes, const uint8_t* w_scales, const float* w_alpha, uint16_t* y,
                                int64_t ldy, int m, int n, int k, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
