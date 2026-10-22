// GPU narrow route of the hybrid MoE against the scalar reference.
//   - random experts, random residency (CPU items computed by forward_ref on the host), T = 1, 3, 8
//   - placement invariance: all-GPU, all-CPU and mixed residency give the same bits
//   - bandwidth of gate_up + down for T = 1 and T = 4 with every expert resident
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "blaze/canon.h"
#include "blaze/expert.h"
#include "blaze/moe.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

using namespace blaze;
static int g_fail = 0;

struct Bank {
  expert::Geometry geo;
  int n;
  size_t stride;
  std::vector<uint8_t> host;  // n records, `stride` apart
  std::vector<float> mult;    // [n][3]
  uint8_t* dev = nullptr;
  float* dmult = nullptr;
};

static Bank make_bank(expert::Geometry geo, int n, std::mt19937& rng) {
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

struct Run {
  std::vector<uint16_t> out;
  int n_gpu = 0, n_cpu = 0;
};

// slot_of: VRAM slot = expert id when resident (the bank doubles as the slot pool), else -1.
static Run run_layer(const Bank& bank, const std::vector<int32_t>& slot_of, int T, int K,
                     const std::vector<uint16_t>& x, const std::vector<int32_t>& ids, const std::vector<float>& w) {
  int32_t *dslot, *dids;
  uint16_t *dx, *dout, *dycpu;
  float* dw;
  void* wsmem;
  CHECK_CUDA(cudaMalloc(&dslot, slot_of.size() * 4));
  CHECK_CUDA(cudaMemcpy(dslot, slot_of.data(), slot_of.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dids, ids.size() * 4));
  CHECK_CUDA(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dx, x.size() * 2));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dw, w.size() * 4));
  CHECK_CUDA(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  const int H = bank.geo.hidden;
  CHECK_CUDA(cudaMalloc(&dout, size_t(T) * H * 2));
  const size_t ycpu_elems = size_t(BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK) * BLZ_MOE_MAX_TOKENS * H;
  CHECK_CUDA(cudaMalloc(&dycpu, ycpu_elems * 2));
  CHECK_CUDA(cudaMalloc(&wsmem, blz_moe_workspace_bytes(H, bank.geo.inter)));
  blz_moe_workspace ws;
  blz_moe_workspace_carve(wsmem, H, bank.geo.inter, &ws);
  const blz_moe_layer layer{H, bank.geo.inter, bank.n, K, dslot, nullptr, bank.dev, int64_t(bank.stride), bank.dmult};

  CHECK_CUDA(blz_moe_gpu_route(&layer, dx, T, dids, &ws, nullptr));
  // CPU side: read the plan, compute the CPU items with the reference, upload their outputs.
  blz_moe_plan plan;
  CHECK_CUDA(cudaMemcpy(&plan, ws.plan, sizeof(plan), cudaMemcpyDeviceToHost));
  std::vector<uint16_t> ycpu(ycpu_elems, 0);
  for (int it = 0; it < plan.n_cpu; ++it) {
    const blz_moe_item& item = plan.cpu[it];
    const float* m = &bank.mult[size_t(item.expert) * 3];
    for (int c = 0; c < item.ncols; ++c)
      expert::forward_ref(bank.geo, bank.host.data() + bank.stride * item.expert, {m[0], m[1], m[2]},
                          &x[size_t(item.cols[c]) * H], &ycpu[(size_t(it) * BLZ_MOE_MAX_TOKENS + c) * H]);
  }
  CHECK_CUDA(cudaMemcpy(dycpu, ycpu.data(), ycpu.size() * 2, cudaMemcpyHostToDevice));
  CHECK_CUDA(blz_moe_combine(&layer, T, dw, &ws, dycpu, dout, nullptr));
  Run r;
  r.out.resize(size_t(T) * H);
  CHECK_CUDA(cudaMemcpy(r.out.data(), dout, r.out.size() * 2, cudaMemcpyDeviceToHost));
  r.n_gpu = plan.n_gpu, r.n_cpu = plan.n_cpu;
  cudaFree(dslot), cudaFree(dids), cudaFree(dx), cudaFree(dw), cudaFree(dout), cudaFree(dycpu), cudaFree(wsmem);
  return r;
}

static std::vector<uint16_t> reference(const Bank& bank, int T, int K, const std::vector<uint16_t>& x,
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

static void make_routes(int T, int K, int E, std::mt19937& rng, std::vector<int32_t>& ids, std::vector<float>& w) {
  ids.resize(size_t(T) * K), w.resize(size_t(T) * K);
  std::vector<int> perm(E);
  std::iota(perm.begin(), perm.end(), 0);
  for (int t = 0; t < T; ++t) {
    std::shuffle(perm.begin(), perm.end(), rng);
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      ids[t * K + k] = perm[k];
      w[t * K + k] = std::uniform_real_distribution<float>(0.01f, 1.0f)(rng);
      sum += w[t * K + k];
    }
    for (int k = 0; k < K; ++k) w[t * K + k] /= sum;
  }
}

static void test_correctness(const Bank& bank, std::mt19937& rng) {
  const int K = 10, E = bank.n, H = bank.geo.hidden;
  for (int T : {1, 3, 8}) {
    std::vector<uint16_t> x(size_t(T) * H);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : x) v = canon::bf16_rn(nd(rng));
    std::vector<int32_t> ids;
    std::vector<float> w;
    make_routes(T, K, E, rng, ids, w);
    const auto ref = reference(bank, T, K, x, ids, w);
    std::vector<int32_t> all_gpu(E), all_cpu(E, -1), mixed(E);
    std::iota(all_gpu.begin(), all_gpu.end(), 0);
    for (int e = 0; e < E; ++e) mixed[e] = (rng() & 1) ? e : -1;
    for (const auto& [name, slots] : {std::pair{"all-gpu", all_gpu}, {"all-cpu", all_cpu}, {"mixed", mixed}}) {
      const Run r = run_layer(bank, slots, T, K, x, ids, w);
      int bad = 0;
      for (size_t i = 0; i < ref.size(); ++i) bad += r.out[i] != ref[i];
      std::printf("moe T=%d %-8s gpu items %2d, cpu items %2d: %d of %zu outputs differ %s\n", T, name, r.n_gpu,
                  r.n_cpu, bad, ref.size(), bad ? "FAIL" : "ok");
      if (bad) ++g_fail;
    }
  }
}

static void bench(const Bank& bank, int T) {
  const int K = 10, H = bank.geo.hidden;
  std::mt19937 rng(77);
  std::vector<int32_t> ids;
  std::vector<float> w;
  make_routes(T, K, bank.n, rng, ids, w);
  std::vector<uint16_t> x(size_t(T) * H);
  for (auto& v : x) v = canon::bf16_rn(std::normal_distribution<float>(0, 1)(rng));
  std::vector<int32_t> slots(bank.n);
  std::iota(slots.begin(), slots.end(), 0);
  int32_t *dslot, *dids;
  uint16_t* dx;
  void* wsmem;
  CHECK_CUDA(cudaMalloc(&dslot, slots.size() * 4));
  CHECK_CUDA(cudaMemcpy(dslot, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dids, ids.size() * 4));
  CHECK_CUDA(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dx, x.size() * 2));
  CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * 2, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&wsmem, blz_moe_workspace_bytes(H, bank.geo.inter)));
  blz_moe_workspace ws;
  blz_moe_workspace_carve(wsmem, H, bank.geo.inter, &ws);
  const blz_moe_layer layer{H, bank.geo.inter, bank.n, K, dslot, nullptr, bank.dev, int64_t(bank.stride), bank.dmult};
  cudaStream_t s;
  CHECK_CUDA(cudaStreamCreate(&s));
  // Graph of the route, replayed: what the server will do.
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  CHECK_CUDA(blz_moe_gpu_route(&layer, dx, T, dids, &ws, s));
  CHECK_CUDA(cudaStreamEndCapture(s, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  for (int i = 0; i < 10; ++i) CHECK_CUDA(cudaGraphLaunch(exec, s));
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  const int iters = 200;
  cudaEventRecord(a, s);
  for (int i = 0; i < iters; ++i) CHECK_CUDA(cudaGraphLaunch(exec, s));
  cudaEventRecord(b, s);
  CHECK_CUDA(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  blz_moe_plan plan;
  CHECK_CUDA(cudaMemcpy(&plan, ws.plan, sizeof(plan), cudaMemcpyDeviceToHost));
  const double us = 1000.0 * ms / iters;
  const double bytes = double(plan.n_gpu) * bank.geo.record_bytes();
  std::printf("bench moe gpu route T=%d: %d experts, %.1f us per layer, %.0f GB/s of expert records\n", T,
              plan.n_gpu, us, bytes / (us * 1e3));
  float acc[4] = {}, km[4];
  for (int i = 0; i < 20; ++i) {
    CHECK_CUDA(blz_moe_gpu_route_timed(&layer, dx, T, dids, &ws, s, km));
    for (int j = 0; j < 4; ++j) acc[j] += km[j] / 20;
  }
  std::printf("  per kernel (us): plan %.1f, gate_up %.1f (%.0f GB/s), finalize %.1f, down %.1f (%.0f GB/s)\n",
              1000 * acc[0], 1000 * acc[1], plan.n_gpu * bank.geo.gu_bytes() / (acc[1] * 1e6), 1000 * acc[2],
              1000 * acc[3], plan.n_gpu * bank.geo.d_bytes() / (acc[3] * 1e6));
  cudaGraphExecDestroy(exec), cudaGraphDestroy(graph), cudaStreamDestroy(s);
  cudaFree(dslot), cudaFree(dids), cudaFree(dx), cudaFree(wsmem);
}

int main() {
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d), %d SMs\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
  std::mt19937 rng(21);
  {
    Bank small = make_bank({256, 64}, 24, rng);
    test_correctness(small, rng);
    cudaFree(small.dev), cudaFree(small.dmult);
  }
  Bank bank = make_bank(expert::kFlashNext, 48, rng);
  test_correctness(bank, rng);
  bench(bank, 1);
  bench(bank, 4);
  cudaFree(bank.dev), cudaFree(bank.dmult);
  std::printf(g_fail ? "FAILED\n" : "PASSED\n");
  return g_fail ? 1 : 0;
}
