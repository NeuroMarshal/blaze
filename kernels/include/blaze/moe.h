// Hybrid routed-MoE: device-side planning and the GPU narrow route (decode / verify, <= 8 token columns).
//
// Per layer and forward of T tokens (T <= BLZ_MOE_MAX_TOKENS):
//   plan      : group the T x topk routes by expert; resident experts (slot_of[e] >= 0) become GPU items,
//               the others CPU items; encode every token's x to A16 limb planes.
//   gate_up   : one warp per (GPU item, 16-row group): exact int64 row sums for all the item's columns.
//   finalize  : SwiGLU -> h (BF16) and its A16 exponent, per (item, column).
//   down      : one warp per (GPU item, 16-row group) of the down matrix -> y (BF16) per (item, column).
//   combine   : out[t] = sum_k w[t][k] * y(e_k, t) in the fixed order k = 0..topk-1 (FP32, then BF16), taking
//               y from the GPU items or from the CPU items' result buffer.
// All buffers are preallocated and sized for the maximum, so the sequence is CUDA-graph capturable.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BLZ_MOE_MAX_TOKENS 8
#define BLZ_MOE_MAX_TOPK 16

/// \brief One expert's routed columns in a forward.
typedef struct {
  int32_t expert;
  int32_t slot;                         // VRAM slot (GPU items) or -1 (CPU items)
  int32_t ncols;
  int32_t cols[BLZ_MOE_MAX_TOKENS];     // token indices
} blz_moe_item;

/// \brief Device-written plan of one layer forward (lives in device memory; CPU items are mirrored to the host).
typedef struct {
  int32_t n_gpu;
  int32_t n_cpu;
  int32_t tokens;
  uint32_t seq;                         // mailbox sequence number of this layer call
  int32_t n_gpu1;
  int32_t n_imm;                        // CPU items 0 .. n_imm - 1 are immediate; the others are deferred
  blz_moe_item gpu[BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK];
  blz_moe_item cpu[BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK];
  blz_moe_item gpu1[BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK];
  // route (t, k) -> (source, item, column): source 0 = this GPU's item, 1 = CPU item, 2 = GPU1 item
  int16_t src[BLZ_MOE_MAX_TOKENS][BLZ_MOE_MAX_TOPK];
  int16_t item[BLZ_MOE_MAX_TOKENS][BLZ_MOE_MAX_TOPK];
  int16_t col[BLZ_MOE_MAX_TOKENS][BLZ_MOE_MAX_TOPK];
} blz_moe_plan;

/// \brief Static description of a layer's experts.
typedef struct {
  int hidden;
  int inter;
  int num_experts;
  int topk;
  const int32_t* slot_of;   // [num_experts] device: slot in this GPU's pool, -1 (RAM) or -2 (SSD tier)
  const int32_t* slot1_of;  // [num_experts] device: slot in GPU1's pool, or -1 (null: no GPU1 tier)
  const uint8_t* pool;      // device base of the VRAM record slots
  int64_t slot_stride;      // bytes between slots (>= record bytes, 256-aligned)
  const float* mult;        // [num_experts][3] device: gate, up, down multipliers
  uint32_t* heat;           // [num_experts] device routing counters (one atomic add per route), or null
  // Expert deferral of this call (blz_moe_defer): CPU items none of whose routes rank < defer_min_rank by route_w
  // in their token are deferred to the next call. 0: off.
  int32_t defer;
  int32_t defer_min_rank;
  const float* route_w;     // [T][topk] device routing weights (deferral only)
  // Extra heat for routes ranking < defer_min_rank (they are waited for even with deferral), so the swap engine keeps
  // the experts that are often among a token's top routes in VRAM. 0: every route counts 1.
  int32_t imm_heat;
  // With deferral: CPU items of the SSD tier (slot_of == -2) are waited for only when some route ranks below this
  // instead of defer_min_rank (their reads are the slow part, so they can be deferred more readily).
  int32_t defer_ssd_rank;
} blz_moe_layer;

/// \brief Expert-deferral state (KTransformers' Expert Deferral, applied to the CPU tier): a layer call whose CPU
/// routes are deferred does not wait for the CPU engine; the next call collects their weighted sum (before its plan
/// overwrites the mailbox) and its combine adds it, so the CPU computes layer k while the GPU runs layer k + 1.
typedef struct {
  uint32_t pending_seq;  // seq of a call whose CPU routes were deferred (consumed only by the next call)
  uint32_t acc_seq;      // seq of the call whose combine adds acc
  float w[BLZ_MOE_MAX_TOKENS][BLZ_MOE_MAX_TOPK];  // routing weights of the pending call
} blz_moe_defer;

/// \brief Device workspace (sizes from blz_moe_workspace_bytes).
typedef struct {
  blz_moe_plan* plan;       // device
  uint8_t* x_planes;        // [T][3][H] bytes: lo, mid, hi limbs of the A16 code
  int32_t* x_emax;          // [T] (-1: non-finite column)
  int64_t* gu_sums;         // [items][cols][2I]
  uint16_t* h;              // [items][cols][I] BF16 bits
  int32_t* h_emax;          // [items][cols]
  uint16_t* y;              // [items][cols][H] BF16 bits (GPU items)
  uint8_t* h_planes;        // [items][cols][3][I] A16 limbs of h
  uint8_t* mail;            // device view of the mapped mailbox (blaze/mailbox.h), or null: no CPU engine
  uint32_t* seq;            // device counter of layer calls (mailbox sequence)
  uint32_t* blocks_done;    // device counter for the plan kernel's last-block signal (kept at 0 between calls)
  blz_moe_defer* defer;     // device, zeroed by the owner
  float* defer_acc;         // [T][H] FP32 weighted sum of the pending call's CPU routes
} blz_moe_workspace;

size_t blz_moe_workspace_bytes(int hidden, int inter);
void blz_moe_workspace_carve(void* base, int hidden, int inter, blz_moe_workspace* ws);

/// \brief plan + gate_up + finalize + down for the GPU items. With a mailbox, the plan also publishes the CPU items
/// of layer `layer_id` to the CPU engine.
cudaError_t blz_moe_gpu_route(const blz_moe_layer* layer, const uint16_t* x, int tokens, const int32_t* topk_ids,
                              const blz_moe_workspace* ws, cudaStream_t stream);
cudaError_t blz_moe_gpu_route_layer(const blz_moe_layer* layer, int layer_id, const uint16_t* x, int tokens,
                                    const int32_t* topk_ids, const blz_moe_workspace* ws, cudaStream_t stream);

/// \brief Wait (on the GPU) for the CPU engine's result of the current layer call, if it has CPU items.
cudaError_t blz_moe_wait_cpu(const blz_moe_workspace* ws, cudaStream_t stream);

/// \brief Deferral mode, before blz_moe_gpu_route_layer: if the previous layer call deferred its CPU routes, wait for
/// the CPU engine and sum them (w * y, k order) into defer_acc for this call's combine.
cudaError_t blz_moe_defer_collect(int hidden, int tokens, const blz_moe_workspace* ws, cudaStream_t stream);

/// \brief Deferral mode, instead of blz_moe_wait_cpu: wait for GPU1 and for the CPU engine's immediate items only.
cudaError_t blz_moe_wait_defer(const blz_moe_workspace* ws, cudaStream_t stream);

/// \brief blz_moe_gpu_route with per-kernel timings (plan, gate_up, finalize, down) in kernel_ms[4]; synchronizes.
cudaError_t blz_moe_gpu_route_timed(const blz_moe_layer* layer, const uint16_t* x, int tokens,
                                    const int32_t* topk_ids, const blz_moe_workspace* ws, cudaStream_t stream,
                                    float* kernel_ms);

/// \brief GPU1 tier: run the GPU1 items of the mailbox's current request on this (helper) GPU and publish done1.
/// `layer` describes the helper's pool; its mult covers every layer ([num_layers * num_experts][3]); `mail` is the
/// mailbox's device address on this GPU. Graph-capturable per token count.
cudaError_t blz_moe_gpu1_route(const blz_moe_layer* layer, const blz_moe_workspace* ws, uint8_t* mail, int tokens,
                               cudaStream_t stream);

/// \brief Weighted sum of the expert outputs in fixed k order. y_cpu: [cpu items][cols][H] BF16 bits, used only
/// without a mailbox (with one, the CPU and GPU1 outputs are read from it). out: [T][H] BF16 bits.
cudaError_t blz_moe_combine(const blz_moe_layer* layer, int tokens, const float* topk_w,
                            const blz_moe_workspace* ws, const uint16_t* y_cpu, uint16_t* out,
                            cudaStream_t stream);

#ifdef __cplusplus
}
#endif
