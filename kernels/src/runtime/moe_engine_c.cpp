#include "blaze/moe_engine.h"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <memory>

#include "blaze/hybrid.h"
#include "blaze/moe.h"

struct blz_moe_engine {
  std::unique_ptr<blaze::HybridMoe> moe;
  int topk = 0;
};

extern "C" int blz_moe_engine_create(const blz_moe_engine_config* c, blz_moe_engine** out, char* err, size_t err_len) {
  try {
    blaze::HybridConfig cfg;
    cfg.model_dir = c->model_dir ? c->model_dir : "";
    cfg.topk = c->topk;
    cfg.gpu0 = c->gpu0;
    cfg.gpu1 = c->gpu1;
    cfg.gpu0_bytes = size_t(c->gpu0_gib * double(1ull << 30));
    cfg.gpu1_bytes = size_t(c->gpu1_gib * double(1ull << 30));
    cfg.ram_bytes = c->ram_gib < 0 ? blaze::HybridConfig::kRamAuto : size_t(c->ram_gib * double(1ull << 30));
    cfg.cpu_threads = c->cpu_threads;
    if (c->cpu_cores && *c->cpu_cores) cfg.cpu_cores = blaze::cpu::parse_cpu_list(c->cpu_cores);
    cfg.load_threads = 16;
    cfg.swap = blaze::parse_swap_options(c->swap_options ? c->swap_options : "");
    if (c->heat_path) cfg.swap.heat_path = c->heat_path;
    blaze::apply_hybrid_options(c->options ? c->options : "", cfg);
    auto* e = new blz_moe_engine;
    e->moe = std::make_unique<blaze::HybridMoe>(cfg);
    e->topk = cfg.topk;
    *out = e;
    return 0;
  } catch (const std::exception& ex) {
    if (err && err_len) std::snprintf(err, err_len, "%s", ex.what());
    return 1;
  }
}

extern "C" void blz_moe_engine_destroy(blz_moe_engine* e) { delete e; }

extern "C" int blz_moe_engine_layers(const blz_moe_engine* e) { return e->moe->num_layers(); }

extern "C" int blz_moe_engine_hidden(const blz_moe_engine* e) { return e->moe->geometry().hidden; }

extern "C" cudaError_t blz_moe_engine_forward(blz_moe_engine* e, int layer, const uint16_t* x, int tokens,
                                              const int32_t* ids, const float* w, uint16_t* out, cudaStream_t stream) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaError_t err = cudaStreamIsCapturing(stream, &capture);
  if (err != cudaSuccess) return err;
  if (tokens >= e->moe->wide_min_tokens() && capture == cudaStreamCaptureStatusNone) {
    // Prefill: every tier on tensor cores (same bits as the narrow route), host-synchronous.
    return e->moe->forward_wide(layer, x, tokens, ids, w, out, stream);
  }
  const int hidden = e->moe->geometry().hidden, k = e->topk;
  for (int t0 = 0; t0 < tokens; t0 += BLZ_MOE_MAX_TOKENS) {
    const int n = std::min(BLZ_MOE_MAX_TOKENS, tokens - t0);
    err = e->moe->forward(layer, x + size_t(t0) * hidden, n, ids + size_t(t0) * k, w + size_t(t0) * k,
                          out + size_t(t0) * hidden, stream, /*allow_defer=*/tokens <= BLZ_MOE_MAX_TOKENS);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}
