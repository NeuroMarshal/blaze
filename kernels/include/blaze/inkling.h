#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

// Byte offsets in the resident expert pool for one packed matrix.
struct blz_inkling_matrix {
  uint64_t codes, scales, alpha;
};

struct blz_moe_matrix_tile {
  const uint8_t* codes;
  const uint8_t* scales;
  const float* alpha;
  int32_t begin, rows;
};

extern "C" cudaError_t blz_moe_grouped_nvfp4(const uint16_t* x, const blz_moe_matrix_tile* tiles,
    uint16_t* y, int count, int row_class, int n, int k, cudaStream_t stream);
extern "C" cudaError_t blz_moe_grouped_mxfp8(const uint16_t* x, const blz_moe_matrix_tile* tiles,
    uint16_t* y, int count, int row_class, int n, int k, cudaStream_t stream);

extern "C" cudaError_t blz_inkling_grouped_nvfp4(const uint16_t* x, int64_t x_step, const uint8_t* pool,
                                               const blz_inkling_matrix* matrices, uint16_t* y,
                                               int experts, int n, int k, cudaStream_t stream);
extern "C" cudaError_t blz_inkling_grouped_mxfp8(const uint16_t* x, int64_t x_step, const uint8_t* pool,
                                               const blz_inkling_matrix* matrices, uint16_t* y,
                                               int experts, int n, int k, cudaStream_t stream);
