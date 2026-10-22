// Token embedding gathered from a table in pinned host memory (mapped, UVA): the GPU reads the rows it needs over
// PCIe, so the table needs no VRAM. CUDA-graph capturable.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief out[i] = table[ids[i]] for i < n; ids outside [0, rows) give zero rows.
/// \param table      device-visible address of the host table [rows][row_bytes] (cudaHostGetDevicePointer)
/// \param row_bytes  multiple of 16
cudaError_t blz_embed_gather(const int64_t* ids, int64_t n, const uint8_t* table, int64_t rows, int64_t row_bytes,
                             uint8_t* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
