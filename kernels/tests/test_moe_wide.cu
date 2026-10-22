// Wide (prefill) route of the hybrid MoE: bit-exact against the scalar reference (the narrow route's arithmetic),
// with the items split over several batches (route0 offsets), then throughput at prefill sizes.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "blaze/canon.h"
#include "blaze/expert.h"
#include "blaze/moe_wide.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

using namespace blaze;

namespace {

struct Bank {
  expert::Geometry geo;
  int n;
  size_t stride;
  std::vector<uint8_t> host;
  std::vector<float> mult;
  uint8_t* dev = nullptr;
  float* dmult = nullptr;
};

Bank make_bank(expert::Geometry geo, int n, std::mt19937& rng) {
  Bank b{geo, n, (geo.record_bytes() + 255) / 256 * 256};
  b.host.assign(b.stride * n, 0);
  for (int e = 0; e < n; ++e) {
    auto mat = [&](int rows, int k, std::vector<uint8_t>& c, std::vector<uint8_t>& s) {
      c.resize(size_t(rows) * k / 2), s.resize(size_t(rows) * k / 16);
      for (auto& v : c) v = uint8_t(rng());
      for (auto& v : s) v = uint8_t(0x28 + rng() % 0x30);
    };
    std::vector<uint8_t> gc, gs, uc, us, dc, ds;
    mat(geo.inter, geo.hidden, gc, gs), mat(geo.inter, geo.hidden, uc, us), mat(geo.hidden, geo.inter, dc, ds);
    expert::pack_record(geo, gc.data(), gs.data(), uc.data(), us.data(), dc.data(), ds.data(),
                        b.host.data() + b.stride * e);
    std::uniform_real_distribution<float> md(0.002f, 0.02f);
    b.mult.insert(b.mult.end(), {md(rng), md(rng), md(rng)});
  }
  CHECK_CUDA(cudaMalloc(&b.dev, b.host.size()));
  CHECK_CUDA(cudaMemcpy(b.dev, b.host.data(), b.host.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&b.dmult, b.mult.size() * 4));
  CHECK_CUDA(cudaMemcpy(b.dmult, b.mult.data(), b.mult.size() * 4, cudaMemcpyHostToDevice));
  return b;
}

std::vector<uint16_t> reference(const Bank& bank, int T, int K, const std::vector<uint16_t>& x,
                                const std::vector<int32_t>& ids, const std::vector<float>& w) {
  const int H = bank.geo.hidden;
  std::vector<uint16_t> out(size_t(T) * H);
  std::vector<std::vector<uint16_t>> y(K, std::vector<uint16_t>(H));
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      const int e = ids[t * K + k];
      const float* m = &bank.mult[size_t(e) * 3];
      expert::forward_ref(bank.geo, bank.host.data() + bank.stride * e, {m[0], m[1], m[2]}, &x[size_t(t) * H],
                          y[k].data());
    }
    for (int r = 0; r < H; ++r) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc = acc + w[t * K + k] * canon::bf16_to_f32(y[k][r]);
      out[size_t(t) * H + r] = canon::bf16_rn(acc);
    }
  }
  return out;
}

struct Layer {
  int T, K;
  std::vector<uint16_t> x;
  std::vector<int32_t> ids;
  std::vector<float> w;
};

Layer make_layer(int T, int K, int E, double zipf, std::mt19937& rng) {
  Layer l{T, K};
  std::normal_distribution<float> nd(0, 1);
  l.x.resize(size_t(T) * 2560);
  for (auto& v : l.x) v = canon::bf16_rn(nd(rng));
  std::vector<double> p(E);
  for (int e = 0; e < E; ++e) p[e] = 1.0 / std::pow(e + 1.0, zipf);
  std::discrete_distribution<int> dist(p.begin(), p.end());
  std::uniform_real_distribution<float> wd(0.01f, 0.3f);
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      int e;
      do e = dist(rng); while (std::find(l.ids.end() - k, l.ids.end(), e) != l.ids.end());
      l.ids.push_back(e);
      l.w.push_back(wd(rng));
    }
  }
  return l;
}

// Host side of a wide layer: routes grouped by expert (token order), items split into `batches` batches.
struct Plan {
  std::vector<blz_wide_item> items;
  std::vector<int2> tiles;
  std::vector<int> batch_items, batch_tiles, batch_route0, batch_routes;  // per batch: first item / tile, routes
  std::vector<int32_t> route_token, route_of;
};

Plan plan_layer(const Bank& bank, const Layer& l, int batches) {
  Plan p;
  std::vector<std::vector<int>> by_expert(bank.n);
  for (int t = 0; t < l.T; ++t)
    for (int k = 0; k < l.K; ++k) by_expert[l.ids[t * l.K + k]].push_back(t * l.K + k);
  p.route_of.assign(size_t(l.T) * l.K, -1);
  std::vector<int> used;
  for (int e = 0; e < bank.n; ++e)
    if (!by_expert[e].empty()) used.push_back(e);
  const int per = (int(used.size()) + batches - 1) / batches;
  for (int b = 0; b < batches; ++b) {
    p.batch_items.push_back(int(p.items.size()));
    p.batch_tiles.push_back(int(p.tiles.size()));
    p.batch_route0.push_back(int(p.route_token.size()));
    for (int u = b * per; u < std::min(int(used.size()), (b + 1) * per); ++u) {
      const int e = used[u];
      const int item = int(p.items.size()) - p.batch_items[b];
      blz_wide_item it{bank.dev + bank.stride * e, e, int(p.route_token.size()), int(by_expert[e].size()), 0};
      for (int tk : by_expert[e]) {
        p.route_of[tk] = int(p.route_token.size());
        p.route_token.push_back(tk / l.K);
      }
      for (int c = 0; c < it.ncols; c += BLZ_WIDE_TILE_COLS) p.tiles.push_back({item, c});
      p.items.push_back(it);
    }
    p.batch_routes.push_back(int(p.route_token.size()) - p.batch_route0[b]);
  }
  p.batch_items.push_back(int(p.items.size()));
  p.batch_tiles.push_back(int(p.tiles.size()));
  return p;
}

template <class T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  CHECK_CUDA(cudaMalloc(&d, std::max<size_t>(1, v.size() * sizeof(T))));
  if (!v.empty()) CHECK_CUDA(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

// Runs the wide route; returns out [T][H] and the kernel time (ms) of `reps` repetitions.
std::vector<uint16_t> run_wide(const Bank& bank, const Layer& l, const Plan& p, int reps, double* ms,
                               bool a8 = false) {
  const int H = bank.geo.hidden, I = bank.geo.inter, R = int(p.route_token.size());
  uint16_t* dx = upload(l.x);
  float* dw = upload(l.w);
  blz_wide_item* ditems = upload(p.items);
  int2* dtiles = upload(p.tiles);
  int32_t* droute_token = upload(p.route_token);
  int32_t* droute_of = upload(p.route_of);
  uint8_t *xplanes, *hplanes;
  int32_t *xemax, *hemax;
  uint16_t *h, *y, *dout;
  CHECK_CUDA(cudaMalloc(&xplanes, size_t(l.T) * 3 * H));
  CHECK_CUDA(cudaMalloc(&xemax, size_t(l.T) * 4));
  CHECK_CUDA(cudaMalloc(&h, size_t(R) * I * 2));
  CHECK_CUDA(cudaMalloc(&hplanes, size_t(R) * 3 * I));
  CHECK_CUDA(cudaMalloc(&hemax, size_t(R) * 4));
  CHECK_CUDA(cudaMalloc(&y, size_t(R) * H * 2));
  CHECK_CUDA(cudaMalloc(&dout, size_t(l.T) * H * 2));
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0), cudaEventCreate(&e1);
  for (int rep = 0; rep <= reps; ++rep) {
    if (rep == 1) cudaEventRecord(e0);
    CHECK_CUDA(blz_wide_encode(dx, l.T, H, xplanes, xemax, nullptr));
    if (a8) CHECK_CUDA(blz_wide_to_a8(xplanes, l.T, H, nullptr));
    for (size_t b = 0; b + 1 < p.batch_items.size(); ++b) {
      blz_wide_batch bt{};
      bt.items = ditems + p.batch_items[b];
      bt.tiles = dtiles + p.batch_tiles[b];
      bt.n_tiles = p.batch_tiles[b + 1] - p.batch_tiles[b];
      bt.n_routes = p.batch_routes[b];
      bt.route0 = p.batch_route0[b];
      bt.route_token = droute_token;
      bt.x_planes = xplanes, bt.x_emax = xemax, bt.mult = bank.dmult, bt.hidden = H, bt.inter = I;
      bt.h = h + size_t(bt.route0) * I, bt.h_planes = hplanes + size_t(bt.route0) * 3 * I;
      bt.h_emax = hemax + bt.route0, bt.y = y + size_t(bt.route0) * H;
      bt.a8 = a8;
      CHECK_CUDA(blz_wide_gate_up(&bt, nullptr));
      CHECK_CUDA(blz_wide_h_encode(&bt, nullptr));
      if (a8) CHECK_CUDA(blz_wide_to_a8(bt.h_planes, bt.n_routes, I, nullptr));
      CHECK_CUDA(blz_wide_down(&bt, nullptr));
    }
    CHECK_CUDA(blz_wide_combine(droute_of, dw, y, l.T, l.K, H, dout, nullptr));
  }
  cudaEventRecord(e1);
  CHECK_CUDA(cudaEventSynchronize(e1));
  float t = 0;
  if (reps > 0) cudaEventElapsedTime(&t, e0, e1);
  *ms = reps > 0 ? t / reps : 0;
  std::vector<uint16_t> out(size_t(l.T) * H);
  CHECK_CUDA(cudaMemcpy(out.data(), dout, out.size() * 2, cudaMemcpyDeviceToHost));
  for (void* ptr : {(void*)dx, (void*)dw, (void*)ditems, (void*)dtiles, (void*)droute_token, (void*)droute_of,
                    (void*)xplanes, (void*)xemax, (void*)h, (void*)hplanes, (void*)hemax, (void*)y, (void*)dout})
    cudaFree(ptr);
  return out;
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  int fail = 0;
  {
    Bank bank = make_bank(expert::kFlashNext, 24, rng);
    for (int T : {1, 5, 37, 70}) {
      Layer l = make_layer(T, 10, bank.n, 0.8, rng);
      const std::vector<uint16_t> ref = reference(bank, T, 10, l.x, l.ids, l.w);
      for (int batches : {1, 3}) {
        double ms;
        const std::vector<uint16_t> out = run_wide(bank, l, plan_layer(bank, l, batches), 0, &ms);
        size_t diff = 0;
        for (size_t i = 0; i < out.size(); ++i) diff += out[i] != ref[i];
        std::printf("T=%3d batches=%d: %zu / %zu outputs differ from the reference\n", T, batches, diff, out.size());
        fail += diff != 0;
        // A8: error of the layer output relative to its RMS (approximate by design).
        const std::vector<uint16_t> a8 = run_wide(bank, l, plan_layer(bank, l, batches), 0, &ms, true);
        double se = 0, sr = 0;
        for (size_t i = 0; i < a8.size(); ++i) {
          const double r = canon::bf16_to_f32(ref[i]), d = canon::bf16_to_f32(a8[i]) - r;
          se += d * d, sr += r * r;
        }
        const double rel = std::sqrt(se / std::max(sr, 1e-30));
        std::printf("            a8: relative RMS error %.2e\n", rel);
        fail += !(rel < 3e-2);
      }
    }
    cudaFree(bank.dev), cudaFree(bank.dmult);
  }
  {
    // Throughput: 128 resident experts, prefill-sized token counts.
    Bank bank = make_bank(expert::kFlashNext, 128, rng);
    for (int T : {512, 2048, 4096}) {
      Layer l = make_layer(T, 10, bank.n, 0.5, rng);
      double ms;
      const double flops = 2.0 * T * 10 * (2.0 * 640 * 2560 + 2560.0 * 640);
      for (bool a8 : {false, true}) {
        run_wide(bank, l, plan_layer(bank, l, 4), 3, &ms, a8);
        std::printf("bench T=%4d, 128 experts, %s: %.2f ms per layer -> %.0f tokens/s per layer, %.1f TOPS\n", T,
                    a8 ? "a8   " : "exact", ms, T / (ms / 1000.0), flops / (ms / 1000.0) / 1e12);
      }
    }
    cudaFree(bank.dev), cudaFree(bank.dmult);
  }
  std::printf(fail ? "FAILED\n" : "PASSED\n");
  return fail ? 1 : 0;
}
