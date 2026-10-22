// Wide route of the hybrid MoE (prefill): all routed (token, k) pairs of a layer, grouped by expert, computed with
// int8 tensor-core MMAs in the narrow route's exact arithmetic, so both routes produce the same bits.
//
// A8 (approximate, option): one s8 x s8 MMA per unit on 8-bit block-floating-point inputs (blz_wide_to_a8).
//
// Per 16-row x 16-k unit, three mma.m16n8k16 (A = the unit's c2 codes, B = the lo / mid / hi byte limbs of the
// A16-encoded inputs) give the exact int32 dot product P of every (row, column); P * s^ (the unit's scale) is added
// in int64, as the narrow rows kernel does lane by lane.
//
//   encode   : token columns -> A16 limb planes [T][3][H] and exponents [T]
//   gate_up  : per (item, 32-column tile) and 16-row group: gate / up sums -> SwiGLU -> h (BF16) per route
//   h_encode : per route: exponent and limb planes of h
//   down     : per (item, tile) and 16-row group: y (BF16) per route
//   combine  : out[t] = sum_k w[t][k] * y[route_of[t][k]] in the fixed k order (FP32, then BF16)
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_WIDE_TILE_COLS 32

/// \brief One expert of a batch: its record and its routes (route_begin .. route_begin + ncols - 1, global indices).
typedef struct {
  const uint8_t* record;  // device address of the rg16 record (pool slot or streaming buffer) on this GPU
  int32_t expert;         // expert id within the layer (multipliers)
  int32_t route_begin;
  int32_t ncols;
  int32_t pad;
} blz_wide_item;

/// \brief A batch of items processed by one gate_up / h_encode / down sequence on one GPU.
typedef struct {
  const blz_wide_item* items;  // device
  const int2* tiles;           // device: (item index, first column) per 32-column tile
  int n_tiles;
  int n_routes;                // routes of the batch: global indices route0 .. route0 + n_routes - 1
  int route0;
  const int32_t* route_token;  // device [all routes]: token of each route (global index)
  const uint8_t* x_planes;     // [T][3][H]
  const int32_t* x_emax;       // [T]
  const float* mult;           // [E][3] gate, up, down multipliers of this layer
  int hidden, inter;
  uint16_t* h;                 // [n_routes][I] BF16 bits (batch-local route index)
  uint8_t* h_planes;           // [n_routes][3][I]
  int32_t* h_emax;             // [n_routes]
  uint16_t* y;                 // [n_routes][H] BF16 bits (batch-local route index)
  int a8;                      // nonzero: planes hold the A8 form (blz_wide_to_a8), one MMA per unit (approximate)
} blz_wide_batch;

cudaError_t blz_wide_encode(const uint16_t* x, int tokens, int hidden, uint8_t* planes, int32_t* emax,
                            cudaStream_t stream);
/// \brief In place, rows of limb planes ([rows][3][k_dim]) -> A8 form: per 16-value block the A16 integer keeps its
/// top 8 bits, q = round(X >> shift) in [-127, 127] (as blaze::cpu::quantize_column_q8); q at [row][0][k], shift at
/// [row][1][k / 16]. Sums from it are P' << shift, in the exact sums' fixed point, so outputs convert unchanged and
/// match the CPU's approximate kernels bit for bit.
cudaError_t blz_wide_to_a8(uint8_t* planes, int rows, int k_dim, cudaStream_t stream);
cudaError_t blz_wide_gate_up(const blz_wide_batch* batch, cudaStream_t stream);
cudaError_t blz_wide_h_encode(const blz_wide_batch* batch, cudaStream_t stream);
cudaError_t blz_wide_down(const blz_wide_batch* batch, cudaStream_t stream);
/// \brief out[t] (BF16 bits, [T][H]) from y ([routes][H]) and route_of / w ([T][K]).
cudaError_t blz_wide_combine(const int32_t* route_of, const float* w, const uint16_t* y, int tokens, int topk,
                             int hidden, uint16_t* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
