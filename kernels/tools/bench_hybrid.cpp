// bench-hybrid: MoE part of a decode step on the real experts (read from the checkpoint), all layers in one CUDA graph.
//
//   bench-hybrid --model DIR [--gpu0-gib 5] [--gpu1-gib 14] [--ram-gib 40|auto] [--threads 32] [--cores 0-23,48-55]
//                [--gpu1 1|-1] [--tokens 1] [--steps 50] [--zipf 0.0] [--swap OPTIONS] [--no-profile] [--wide T]
//
// --wide T: prefill route instead: checks forward_wide against forward (narrow, 8-token chunks) bit for bit on two
// layers, then times forward_wide over all layers for T tokens.
// --swap runs the swap engine (options as parse_swap_options, "" for defaults); --no-profile starts from the uniform
// placement instead of the routing law, so the swap engine has to find the hot experts itself.
//
// Routes: every step draws new top-k ids per (layer, token) from a Zipf(s) law over expert ranks (s = 0: uniform);
// the placement is profiled with the same law, so hot experts sit in VRAM. Prints ms per step (all layers) and the
// share of expert uses served by each tier.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "blaze/canon.h"
#include "blaze/hybrid.h"

using namespace blaze;

static void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Prefill route: bit-exactness against the narrow route on real records, then layers x T tokens timing.
static int bench_wide(HybridMoe& moe, const std::vector<std::vector<double>>& weight, int T, double zipf,
                      std::mt19937& rng) {
  const int L = moe.num_layers(), H = moe.geometry().hidden, K = 10;
  std::vector<std::discrete_distribution<int>> dists;
  for (int l = 0; l < L; ++l) dists.emplace_back(weight[l].begin(), weight[l].end());
  std::vector<uint16_t> x(size_t(T) * H);
  std::normal_distribution<float> nd(0, 1);
  for (auto& b : x) b = canon::bf16_rn(nd(rng));
  std::vector<float> w(size_t(T) * K);
  std::uniform_real_distribution<float> wd(0.01f, 0.3f);
  for (auto& v : w) v = wd(rng);
  auto draw = [&](int l, std::vector<int32_t>& ids) {
    ids.resize(size_t(T) * K);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < K; ++k) {
        int e;
        do e = dists[l](rng); while (std::find(&ids[size_t(t) * K], &ids[size_t(t) * K] + k, e) != &ids[size_t(t) * K] + k);
        ids[size_t(t) * K + k] = e;
      }
  };
  uint16_t *dx, *dn, *dw_out;
  int32_t* dids;
  float* dw;
  check(cudaMalloc(&dx, x.size() * 2), "x");
  check(cudaMalloc(&dn, x.size() * 2), "out");
  check(cudaMalloc(&dw_out, x.size() * 2), "out");
  check(cudaMalloc(&dids, size_t(T) * K * 4), "ids");
  check(cudaMalloc(&dw, w.size() * 4), "w");
  check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
  check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
  cudaStream_t s;
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
  std::vector<int32_t> ids;
  int bad = 0;
  for (int l : {0, L - 1}) {
    draw(l, ids);
    check(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice), "ids");
    for (int t0 = 0; t0 < T; t0 += BLZ_MOE_MAX_TOKENS) {
      const int n = std::min(BLZ_MOE_MAX_TOKENS, T - t0);
      check(moe.forward(l, dx + size_t(t0) * H, n, dids + size_t(t0) * K, dw + size_t(t0) * K, dn + size_t(t0) * H, s),
            "narrow");
    }
    check(moe.forward_wide(l, dx, T, dids, dw, dw_out, s), "wide");
    check(cudaStreamSynchronize(s), "sync");
    std::vector<uint16_t> a(x.size()), b(x.size());
    check(cudaMemcpy(a.data(), dn, a.size() * 2, cudaMemcpyDeviceToHost), "out");
    check(cudaMemcpy(b.data(), dw_out, b.size() * 2, cudaMemcpyDeviceToHost), "out");
    size_t diff = 0;
    for (size_t i = 0; i < a.size(); ++i) diff += a[i] != b[i];
    std::printf("layer %d, %d tokens: wide vs narrow: %zu / %zu outputs differ\n", l, T, diff, a.size());
    bad += diff != 0;
  }
  // Timing: every layer once (new routes per layer), as one prefill chunk would.
  std::vector<std::vector<int32_t>> all(L);
  for (int l = 0; l < L; ++l) draw(l, all[l]);
  double total = 0;
  for (int rep = 0; rep < 2; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0; l < L; ++l) {
      check(cudaMemcpyAsync(dids, all[l].data(), all[l].size() * 4, cudaMemcpyHostToDevice, s), "ids");
      check(moe.forward_wide(l, dx, T, dids, dw, dw_out, s), "wide");
    }
    check(cudaStreamSynchronize(s), "sync");
    total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  std::printf("wide: %d layers x %d tokens (zipf %.2f) in %.3f s -> %.0f tokens/s of MoE prefill\n", L, T, zipf, total,
              T / total);
  cudaFree(dx), cudaFree(dn), cudaFree(dw_out), cudaFree(dids), cudaFree(dw), cudaStreamDestroy(s);
  std::printf(bad ? "WIDE FAILED\n" : "WIDE PASSED\n");
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  HybridConfig cfg;
  bool swap = false, profile = true;
  std::string swap_options;
  double gpu0_gib = 5, gpu1_gib = 14, ram_gib = 40, zipf = 0.0;
  int tokens = 1, steps = 50, wide = 0;
  cfg.gpu1 = 1;
  cfg.cpu_threads = 32;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto v = [&] { return std::string(argv[++i]); };
    if (a == "--model") cfg.model_dir = v();
    else if (a == "--gpu0-gib") gpu0_gib = std::stod(v());
    else if (a == "--gpu1-gib") gpu1_gib = std::stod(v());
    else if (a == "--ram-gib") {
      const std::string r = v();
      ram_gib = r == "auto" ? -1.0 : std::stod(r);
    }
    else if (a == "--threads") cfg.cpu_threads = std::stoi(v());
    else if (a == "--gpu1") cfg.gpu1 = std::stoi(v());
    else if (a == "--tokens") tokens = std::stoi(v());
    else if (a == "--steps") steps = std::stoi(v());
    else if (a == "--zipf") zipf = std::stod(v());
    else if (a == "--swap") swap = true, swap_options = v();
    else if (a == "--no-profile") profile = false;
    else if (a == "--wide") wide = std::stoi(v());
    else if (a == "--cores") cfg.cpu_cores = cpu::parse_cpu_list(v());
    else {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      return 2;
    }
  }
  if (cfg.model_dir.empty()) {
    std::fprintf(stderr, "--model is required\n");
    return 2;
  }
  cfg.gpu0_bytes = size_t(gpu0_gib * (1ull << 30));
  cfg.gpu1_bytes = size_t(gpu1_gib * (1ull << 30));
  cfg.ram_bytes = ram_gib < 0 ? HybridConfig::kRamAuto : size_t(ram_gib * (1ull << 30));
  cfg.load_threads = 16;

  // Expert popularity: a random rank per (layer, expert); weight ~ 1 / rank^s.
  ExpertStore probe(cfg.model_dir);
  const int L = probe.num_layers(), E = probe.num_experts(), H = probe.geometry().hidden, K = cfg.topk;
  std::mt19937 rng(1234);
  std::vector<std::vector<double>> weight(L, std::vector<double>(E));
  for (int l = 0; l < L; ++l) {
    std::vector<int> rank(E);
    std::iota(rank.begin(), rank.end(), 1);
    std::shuffle(rank.begin(), rank.end(), rng);
    for (int e = 0; e < E; ++e) weight[l][e] = 1.0 / std::pow(double(rank[e]), zipf);
  }
  cfg.heat.resize(size_t(L) * E);
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) cfg.heat[size_t(l) * E + e] = weight[l][e];

  if (!profile) cfg.heat.clear();
  cfg.swap = parse_swap_options(swap_options);
  cfg.swap.enabled = swap;
  cfg.wide_tokens = std::max(4096, wide);
  HybridMoe moe(cfg);
  const Placement& pl = moe.placement();
  if (wide > 0) return bench_wide(moe, weight, wide, zipf, rng);

  // Device buffers: per-layer inputs (random, fixed), outputs, routes, weights.
  uint16_t *dx, *dout;
  int32_t* dids;
  float* dw;
  check(cudaMalloc(&dx, size_t(L) * tokens * H * 2), "x");
  check(cudaMalloc(&dout, size_t(L) * tokens * H * 2), "out");
  check(cudaMalloc(&dids, size_t(L) * tokens * K * 4), "ids");
  check(cudaMalloc(&dw, size_t(L) * tokens * K * 4), "w");
  {
    std::vector<uint16_t> x(size_t(L) * tokens * H);
    std::normal_distribution<float> nd(0, 1);
    for (auto& b : x) b = canon::bf16_rn(nd(rng));
    check(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice), "x");
    std::vector<float> w(size_t(L) * tokens * K, 1.0f / K);
    check(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice), "w");
  }
  // Route sets for every step, drawn without replacement per (layer, token).
  std::vector<std::discrete_distribution<int>> dists;
  for (int l = 0; l < L; ++l) dists.emplace_back(weight[l].begin(), weight[l].end());
  auto draw = [&](std::vector<int32_t>& ids, long tier_count[4]) {
    ids.resize(size_t(L) * tokens * K);
    for (int l = 0; l < L; ++l)
      for (int t = 0; t < tokens; ++t) {
        int32_t* row = &ids[(size_t(l) * tokens + t) * K];
        for (int k = 0; k < K; ++k) {
          int e;
          do e = dists[l](rng); while (std::find(row, row + k, e) != row + k);
          row[k] = e;
          if (!swap) ++tier_count[int(pl.tier_of(l, e))];
        }
      }
  };

  cudaStream_t s;
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  check(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), "capture");
  for (int l = 0; l < L; ++l)
    check(moe.forward(l, dx + size_t(l) * tokens * H, tokens, dids + size_t(l) * tokens * K,
                      dw + size_t(l) * tokens * K, dout + size_t(l) * tokens * H, s),
          "forward");
  check(cudaStreamEndCapture(s, &graph), "end capture");
  check(cudaGraphInstantiate(&exec, graph, 0), "instantiate");

  long tiers[4] = {0, 0, 0, 0};
  HybridStats st0{};
  std::vector<int32_t> ids;
  std::vector<double> ms;
  for (int step = 0; step < steps + 3; ++step) {
    long tc[4] = {0, 0, 0, 0};
    draw(ids, tc);
    check(cudaMemcpyAsync(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, s), "ids");
    check(cudaStreamSynchronize(s), "sync");
    const auto t0 = std::chrono::steady_clock::now();
    check(cudaGraphLaunch(exec, s), "launch");
    check(cudaStreamSynchronize(s), "sync");
    const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (step == 2) st0 = moe.stats();
    if (step >= 3) {  // warm-up
      ms.push_back(dt);
      for (int i = 0; i < 4; ++i) tiers[i] += tc[i];
    }
  }
  std::sort(ms.begin(), ms.end());
  const double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / ms.size();
  if (swap) {
    const HybridStats st = moe.stats();
    for (int i = 0; i < 4; ++i) tiers[i] = long(st.routes[i] - st0.routes[i]);
    std::printf("swap engine: %llu RAM<->GPU and %llu SSD<->RAM moves\n", (unsigned long long)st.moves_gpu,
                (unsigned long long)st.moves_ram);
  }
  const long uses = std::max(1L, tiers[0] + tiers[1] + tiers[2] + tiers[3]);
  std::printf("tokens=%d zipf=%.2f layers=%d: MoE step mean %.2f ms, median %.2f, p90 %.2f -> %.1f steps/s\n", tokens,
              zipf, L, mean, ms[ms.size() / 2], ms[ms.size() * 9 / 10], 1000.0 / mean);
  std::printf("expert uses by tier: GPU0 %.1f%%, GPU1 %.1f%%, RAM %.1f%%, SSD %.1f%%; cpu requests %llu, gpu1 requests %llu\n",
              100.0 * tiers[0] / uses, 100.0 * tiers[1] / uses, 100.0 * tiers[2] / uses, 100.0 * tiers[3] / uses,
              (unsigned long long)moe.cpu_requests(), (unsigned long long)moe.gpu1_requests());
  cudaGraphExecDestroy(exec), cudaGraphDestroy(graph), cudaStreamDestroy(s);
  cudaFree(dx), cudaFree(dout), cudaFree(dids), cudaFree(dw);
  return 0;
}
