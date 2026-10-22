// C ABI of the routed-expert engine (blaze::HybridMoe): experts placed over GPU0 / GPU1 VRAM, pinned RAM (served by
// the CPU) and the SSD, moved between tiers by routing heat. One engine per process and model.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blz_moe_engine blz_moe_engine;

typedef struct {
  const char* model_dir;      // checkpoint directory (NVFP4 experts read as stored)
  int topk;                   // routed experts per token
  int gpu0;                   // device of the model
  int gpu1;                   // helper device for expert slots, -1: none
  double gpu0_gib;            // VRAM for expert slots on gpu0
  double gpu1_gib;            // VRAM for expert slots on gpu1
  double ram_gib;             // pinned RAM for experts; < 0: every expert not in VRAM (no SSD tier)
  int cpu_threads;            // CPU expert workers
  const char* cpu_cores;      // worker affinity ("3-4,6,9-17"; NULL or "": none)
  const char* heat_path;      // routing profile file (initial placement, saved periodically); NULL: none
  const char* swap_options;   // "key=value,..." of the swap engine (blaze/hybrid.h); NULL: defaults
  const char* options;        // "key=value,..." runtime options (blaze/hybrid.h); NULL: defaults
} blz_moe_engine_config;

/// \brief Build the engine (loads every expert into its tier). On failure returns nonzero and writes a message.
int blz_moe_engine_create(const blz_moe_engine_config* cfg, blz_moe_engine** out, char* err, size_t err_len);
void blz_moe_engine_destroy(blz_moe_engine* e);
int blz_moe_engine_layers(const blz_moe_engine* e);
int blz_moe_engine_hidden(const blz_moe_engine* e);

/// \brief out[t] = sum_k w[t][k] * expert_{ids[t][k]}(x[t]) for one MoE layer of `tokens` tokens.
/// x, out: [tokens, hidden] bf16; ids: [tokens, topk] int32; w: [tokens, topk] float; device pointers on gpu0.
/// Long inputs outside CUDA-graph capture take the wide (tensor-core, every tier) route and return when done;
/// otherwise chunks of BLZ_MOE_MAX_TOKENS run asynchronously on `stream` (graph capturable).
cudaError_t blz_moe_engine_forward(blz_moe_engine* e, int layer, const uint16_t* x, int tokens, const int32_t* ids,
                                   const float* w, uint16_t* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
