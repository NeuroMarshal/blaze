// Hyper-connection low-rank input mix for decode / verify batches (Qwen4-Exp attn/mlp hyper connections).
//
//   t[m]      = bf16(silu(fp32(x[m] . w_down^T) / hc))                      [rows][lowrank]
//   u[m]      = fp32(t[m] . w_up^T)                                           [rows][hc * hs]
//   out[m][j] = bf16(sum_g sigmoid(u[m][g * hs + j]) * x[m][g * hs + j] / hc)
//
// Two bandwidth-bound passes over the BF16 weights, no atomics: every sum has a fixed order, so results are
// bitwise reproducible (replays, eager vs graph). CUDA-graph capturable.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_HC_MIX_MAX_ROWS 16

/// \brief Whether blz_hc_mix supports these sizes: hc = 4 with lowrank % 16 == 0 and (4 hs / 16) % 16 == 0 runs on
/// tensor cores for any row count; otherwise rows <= 16, hc in {2, 4, 8}, (hc * hs) % 32 == 0, lowrank % 8 == 0.
int blz_hc_mix_supported(int rows, int hc, int hs, int lowrank);

/// \brief Bytes of float workspace blz_hc_mix needs (0: none).
size_t blz_hc_mix_workspace_bytes(int rows, int hc, int hs, int lowrank);

/// \param x       [rows][hc * hs] BF16 bits (the normalized hyper input)
/// \param w_down  [lowrank][hc * hs] BF16 bits
/// \param w_up    [hc * hs][lowrank] BF16 bits
/// \param t       workspace [rows][lowrank] BF16 bits
/// \param out     [rows][hs] BF16 bits
/// \param workspace blz_hc_mix_workspace_bytes(...) bytes (may be null when that is 0)
cudaError_t blz_hc_mix(const uint16_t* x, const uint16_t* w_down, const uint16_t* w_up, uint16_t* t, uint16_t* out,
                       int rows, int hc, int hs, int lowrank, void* workspace, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
