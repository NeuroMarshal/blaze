// Qwen3-VL / Qwen3.5 vision tower on the CPU (AVX2 + FMA, threads): the same math as the GPU path (blaze/vision.h) with fp32
// activations, so it uses no VRAM. The weights are bf16 tensors read in place (the checkpoint's own memory map); a
// GEMM converts them to fp32 in cache-sized packed panels.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blz_vis_cfg {
  int depth, hidden, heads, intermediate;
  int patch, temporal, merge, channels;
  int side;        ///< the learned position table is side x side
  int out_hidden;  ///< width of the merged embeddings (the language model's hidden size)
  float eps;       ///< layer norm epsilon
  float theta;     ///< rotary base of the 2D rope
  int act_tanh;    ///< block MLP activation: 1 GELU tanh approximation, 0 exact GELU
} blz_vis_cfg;

typedef struct blz_vis_layer {
  const uint16_t *n1_w, *n1_b, *qkv_w, *qkv_b, *proj_w, *proj_b, *n2_w, *n2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
} blz_vis_layer;

/// Pointers to bf16 tensors in the layouts of the checkpoint (Linear weights [out, in]; the patch weight is the Conv3d
/// kernel [hidden, channels, temporal, patch, patch] = [hidden, channels * temporal * patch * patch]).
typedef struct blz_vis_weights {
  blz_vis_cfg cfg;
  const uint16_t *patch_w, *patch_b, *pos_table;
  const blz_vis_layer* layers;  ///< cfg.depth entries
  const uint16_t *mrg_norm_w, *mrg_norm_b, *mrg_fc1_w, *mrg_fc1_b, *mrg_fc2_w, *mrg_fc2_b;
} blz_vis_weights;

/// \brief Encode one image: `pixels` [grid_h * grid_w, channels * temporal * patch * patch] bf16 (normalized, patches in
/// spatial-merge order, the still image repeated over the temporal patch) -> `out` [grid_h * grid_w / merge^2,
/// out_hidden] bf16. `stages` (NULL, or 3 * patches * hidden floats): the activations after the position embedding, after
/// the first block and after the last block (debugging). Returns 0, or a nonzero code for unsupported shapes.
int blz_vis_cpu_encode(const blz_vis_weights* w, const uint16_t* pixels, int grid_h, int grid_w, int threads, uint16_t* out,
                       float* stages);

#ifdef __cplusplus
}
#endif
