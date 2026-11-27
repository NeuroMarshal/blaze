#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

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
extern "C" cudaError_t blz_mxfp8_quant_rows(const uint16_t* input, uint8_t* codes, uint8_t* scales,
    int rows, int inner, cudaStream_t stream);
extern "C" cudaError_t blz_mxfp8_grouped_native(const uint8_t* input, const uint8_t* scales,
    const blz_moe_matrix_tile* tiles, uint16_t* out, int count, int n, int k, cudaStream_t stream);
