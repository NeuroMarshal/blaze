// End to end: L MoE layers in one CUDA graph, misses served by the CPU engine through the mailbox.
//   - outputs equal the scalar reference for every layer chain (all-GPU, mixed, all-CPU residency)
//   - timing per layer: GPU-only vs mixed vs all-CPU, i.e. what the in-graph CPU round trip costs
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "blaze/canon.h"
#include "blaze/cpu_engine.h"
#include "blaze/expert.h"
#include "blaze/gpu1_service.h"
#include "blaze/mailbox.h"
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

struct Model {
  expert::Geometry geo;
  int L, E, K;
  size_t stride;
  std::vector<uint8_t> host;  // [L][E] records (one shared random record per (l, e) % variety)
  std::vector<float> mult;    // [L][E][3]
};

static Model make_model(expert::Geometry geo, int L, int E, int K, std::mt19937& rng) {
  Model m{geo, L, E, K, (geo.record_bytes() + 255) / 256 * 256};
  // A few distinct random records, reused across (layer, expert) to keep host memory small.
  const int distinct = 8;
  std::vector<std::vector<uint8_t>> recs;
  for (int d = 0; d < distinct; ++d) {
    auto mat = [&](int rows, int k, std::vector<uint8_t>& c, std::vector<uint8_t>& s) {
      c.resize(size_t(rows) * k / 2), s.resize(size_t(rows) * k / 16);
      for (auto& v : c) v = uint8_t(rng());
      for (auto& v : s) v = uint8_t(0x28 + rng() % 0x30);
    };
    std::vector<uint8_t> gc, gs, uc, us, dc, ds, rec(geo.record_bytes());
    mat(geo.inter, geo.hidden, gc, gs), mat(geo.inter, geo.hidden, uc, us), mat(geo.hidden, geo.inter, dc, ds);
    expert::pack_record(geo, gc.data(), gs.data(), uc.data(), us.data(), dc.data(), ds.data(), rec.data());
    recs.push_back(std::move(rec));
  }
  m.host.resize(m.stride * L * E);
  std::uniform_real_distribution<float> md(0.004f, 0.012f);
  for (int i = 0; i < L * E; ++i) {
    std::memcpy(m.host.data() + m.stride * i, recs[rng() % distinct].data(), geo.record_bytes());
    m.mult.insert(m.mult.end(), {md(rng), md(rng), md(rng)});
  }
  return m;
}

// Reference chain: x_{l+1} = moe_l(x_l), per token, fixed-order combine.
static std::vector<uint16_t> reference(const Model& m, int T, std::vector<uint16_t> x, const std::vector<int32_t>& ids,
                                       const std::vector<float>& w) {
  const int H = m.geo.hidden;
  std::vector<uint16_t> y(H);
  std::vector<std::vector<uint16_t>> ys(m.K, std::vector<uint16_t>(H));
  for (int l = 0; l < m.L; ++l) {
    std::vector<uint16_t> next(size_t(T) * H);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < m.K; ++k) {
        const int e = ids[(size_t(l) * T + t) * m.K + k];
        const float* mu = &m.mult[(size_t(l) * m.E + e) * 3];
        expert::forward_ref(m.geo, m.host.data() + m.stride * (size_t(l) * m.E + e), {mu[0], mu[1], mu[2]},
                            &x[size_t(t) * H], ys[k].data());
      }
      for (int r = 0; r < H; ++r) {
        float acc = 0.0f;
        for (int k = 0; k < m.K; ++k) acc = acc + w[(size_t(l) * T + t) * m.K + k] * canon::bf16_to_f32(ys[k][r]);
        next[size_t(t) * H + r] = canon::bf16_rn(acc);
      }
    }
    x.swap(next);
  }
  return x;
}

// Tier of expert i (layer * E + e) for a residency (see run): 0 GPU0, 1 CPU, 2 GPU1.
static int tier_of(int residency, int i) {
  return residency == 0 ? 0 : residency == 1 ? (i % 2 == 0 ? 0 : 1) : residency == 2 ? 1 : (i % 3 == 0 ? 0 : i % 3 == 1 ? 2 : 1);
}

// Reference chain with expert deferral: in layers < L - 1, CPU experts none of whose routes rank < min_rank (by weight
// in its token) are left out, and the next layer adds their FP32 weighted sum after its own routes.
static std::vector<uint16_t> reference_deferred(const Model& m, int T, std::vector<uint16_t> x,
                                                const std::vector<int32_t>& ids, const std::vector<float>& w,
                                                int residency, int min_rank) {
  const int H = m.geo.hidden;
  std::vector<std::vector<uint16_t>> ys(m.K, std::vector<uint16_t>(H));
  std::vector<float> pending(size_t(T) * H, 0.0f);
  bool have_pending = false;
  for (int l = 0; l < m.L; ++l) {
    std::vector<char> imm(m.E, 0);
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < m.K; ++k) {
        const float* wt = &w[(size_t(l) * T + t) * m.K];
        int rank = 0;
        for (int j = 0; j < m.K; ++j) rank += wt[j] > wt[k] || (wt[j] == wt[k] && j < k);
        if (rank < min_rank) imm[ids[(size_t(l) * T + t) * m.K + k]] = 1;
      }
    auto deferred_route = [&](int e) { return l < m.L - 1 && tier_of(residency, l * m.E + e) == 1 && !imm[e]; };
    bool defer = false;
    for (int t = 0; t < T; ++t)
      for (int k = 0; k < m.K; ++k) defer |= deferred_route(ids[(size_t(l) * T + t) * m.K + k]);
    std::vector<uint16_t> next(size_t(T) * H);
    std::vector<float> acc_def(size_t(T) * H, 0.0f);
    for (int t = 0; t < T; ++t) {
      for (int k = 0; k < m.K; ++k) {
        const int e = ids[(size_t(l) * T + t) * m.K + k];
        const float* mu = &m.mult[(size_t(l) * m.E + e) * 3];
        expert::forward_ref(m.geo, m.host.data() + m.stride * (size_t(l) * m.E + e), {mu[0], mu[1], mu[2]},
                            &x[size_t(t) * H], ys[k].data());
      }
      for (int r = 0; r < H; ++r) {
        float acc = 0.0f, d = 0.0f;
        for (int k = 0; k < m.K; ++k) {
          const int e = ids[(size_t(l) * T + t) * m.K + k];
          const float v = w[(size_t(l) * T + t) * m.K + k] * canon::bf16_to_f32(ys[k][r]);
          if (deferred_route(e)) d = d + v;
          else acc = acc + v;
        }
        if (have_pending) acc = acc + pending[size_t(t) * H + r];
        acc_def[size_t(t) * H + r] = d;
        next[size_t(t) * H + r] = canon::bf16_rn(acc);
      }
    }
    have_pending = defer;
    pending.swap(acc_def);
    x.swap(next);
  }
  return x;
}

struct Result {
  std::vector<uint16_t> out;
  double us_per_layer;
  uint64_t served, served1 = 0;
};

// residency: 0 all-GPU, 1 mixed (every other expert), 2 all-CPU, 3 thirds: GPU0 / GPU1 (helper; the same device
// here, on its own stream and pool) / CPU
static Result run(const Model& m, int T, const std::vector<uint16_t>& x0, const std::vector<int32_t>& ids,
                  const std::vector<float>& w, int residency, int threads, int replays, int defer_rank = -1) {
  const bool defer = defer_rank >= 0;
  const int H = m.geo.hidden, L = m.L, E = m.E, K = m.K;
  // VRAM pool: resident experts only, slot table per layer.
  std::vector<int32_t> slot_of(size_t(L) * E, -1), slot1_of(size_t(L) * E, -1);
  std::vector<const uint8_t*> host_records(size_t(L) * E, nullptr);
  int slots = 0, slots1 = 0;
  for (int i = 0; i < L * E; ++i) {
    const int tier = tier_of(residency, i);
    if (tier == 0) {
      slot_of[i] = slots++;
    } else if (tier == 2) {
      slot1_of[i] = slots1++;
    } else {
      host_records[i] = m.host.data() + m.stride * i;
    }
  }
  uint8_t* pool1 = nullptr;
  CHECK_CUDA(cudaMalloc(&pool1, std::max<size_t>(1, m.stride * slots1)));
  for (int i = 0; i < L * E; ++i)
    if (slot1_of[i] >= 0)
      CHECK_CUDA(cudaMemcpy(pool1 + m.stride * slot1_of[i], m.host.data() + m.stride * i, m.geo.record_bytes(),
                            cudaMemcpyHostToDevice));
  int32_t* dslot1;
  CHECK_CUDA(cudaMalloc(&dslot1, slot1_of.size() * 4));
  CHECK_CUDA(cudaMemcpy(dslot1, slot1_of.data(), slot1_of.size() * 4, cudaMemcpyHostToDevice));
  uint8_t* pool = nullptr;
  CHECK_CUDA(cudaMalloc(&pool, std::max<size_t>(1, m.stride * slots)));
  for (int i = 0; i < L * E; ++i)
    if (slot_of[i] >= 0)
      CHECK_CUDA(cudaMemcpy(pool + m.stride * slot_of[i], m.host.data() + m.stride * i, m.geo.record_bytes(),
                            cudaMemcpyHostToDevice));
  int32_t *dslot, *dids;
  float *dmult, *dw;
  CHECK_CUDA(cudaMalloc(&dslot, slot_of.size() * 4));
  CHECK_CUDA(cudaMemcpy(dslot, slot_of.data(), slot_of.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dmult, m.mult.size() * 4));
  CHECK_CUDA(cudaMemcpy(dmult, m.mult.data(), m.mult.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dids, ids.size() * 4));
  CHECK_CUDA(cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMalloc(&dw, w.size() * 4));
  CHECK_CUDA(cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  // Activations: x_0 .. x_L.
  uint16_t* dx;
  CHECK_CUDA(cudaMalloc(&dx, size_t(L + 1) * T * H * 2));
  CHECK_CUDA(cudaMemcpy(dx, x0.data(), x0.size() * 2, cudaMemcpyHostToDevice));
  // Mailbox (mapped pinned) and counters.
  const blz_mail_layout ml = blz_mail_layout_for(H);
  uint8_t* mail_host;
  CHECK_CUDA(cudaHostAlloc(&mail_host, ml.total, cudaHostAllocMapped | cudaHostAllocPortable));
  std::memset(mail_host, 0, ml.total);
  uint8_t* mail_dev;
  CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mail_dev), mail_host, 0));
  void* wsmem;
  CHECK_CUDA(cudaMalloc(&wsmem, blz_moe_workspace_bytes(H, m.geo.inter)));
  blz_moe_workspace ws;
  blz_moe_workspace_carve(wsmem, H, m.geo.inter, &ws);
  uint32_t* counters;
  CHECK_CUDA(cudaMalloc(&counters, 8));
  CHECK_CUDA(cudaMemset(counters, 0, 8));
  ws.mail = mail_dev, ws.seq = counters, ws.blocks_done = counters + 1;
  CHECK_CUDA(cudaMemset(ws.defer, 0, sizeof(blz_moe_defer)));

  cpu::EngineConfig ec;
  ec.geo = m.geo, ec.num_layers = L, ec.num_experts = E, ec.threads = threads;
  if (const char* v = std::getenv("BLZ_GU_CHUNK")) ec.gu_chunk_groups = std::atoi(v);
  if (const char* v = std::getenv("BLZ_DOWN_CHUNK")) ec.down_chunk_groups = std::atoi(v);
  if (const char* v = std::getenv("BLZ_CORES")) {  // e.g. "0-11"
    int a = 0, b = -1;
    if (std::sscanf(v, "%d-%d", &a, &b) == 2)
      for (int c = a; c <= b; ++c) ec.cores.push_back(c);
  }
  auto engine = std::make_unique<cpu::Engine>(ec, mail_host, host_records.data(), m.mult.data());
  std::unique_ptr<Gpu1Service> gpu1;
  if (residency == 3) {
    const blz_moe_layer helper{H, m.geo.inter, E, K, nullptr, nullptr, pool1, int64_t(m.stride), dmult};
    gpu1 = std::make_unique<Gpu1Service>(0, helper, mail_host);
  }

  cudaStream_t s;
  CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CHECK_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int l = 0; l < L; ++l) {
    const blz_moe_layer layer{H, m.geo.inter, E, K, dslot + size_t(l) * E,
                              residency == 3 ? dslot1 + size_t(l) * E : nullptr, pool, int64_t(m.stride),
                              dmult + size_t(l) * E * 3, nullptr, defer && l < L - 1, defer_rank,
                              dw + size_t(l) * T * K};
    const uint16_t* xl = dx + size_t(l) * T * H;
    if (defer) CHECK_CUDA(blz_moe_defer_collect(H, T, &ws, s));
    CHECK_CUDA(blz_moe_gpu_route_layer(&layer, l, xl, T, dids + size_t(l) * T * K, &ws, s));
    if (defer)
      CHECK_CUDA(blz_moe_wait_defer(&ws, s));
    else
      CHECK_CUDA(blz_moe_wait_cpu(&ws, s));
    CHECK_CUDA(blz_moe_combine(&layer, T, dw + size_t(l) * T * K, &ws,
                               reinterpret_cast<const uint16_t*>(mail_dev + ml.y), dx + size_t(l + 1) * T * H, s));
  }
  CHECK_CUDA(cudaStreamEndCapture(s, &graph));
  CHECK_CUDA(cudaGraphInstantiate(&exec, graph, 0));
  CHECK_CUDA(cudaGraphLaunch(exec, s));  // warm-up
  CHECK_CUDA(cudaStreamSynchronize(s));
  const auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < replays; ++r) CHECK_CUDA(cudaGraphLaunch(exec, s));
  CHECK_CUDA(cudaStreamSynchronize(s));
  const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  Result res;
  res.out.resize(size_t(T) * H);
  CHECK_CUDA(cudaMemcpy(res.out.data(), dx + size_t(L) * T * H, res.out.size() * 2, cudaMemcpyDeviceToHost));
  res.us_per_layer = 1e6 * sec / replays / L;
  res.served = engine->served();
  res.served1 = gpu1 ? gpu1->served() : 0;
  gpu1.reset();
  engine.reset();  // stop the workers before the mailbox goes away
  cudaGraphExecDestroy(exec), cudaGraphDestroy(graph), cudaStreamDestroy(s);
  cudaFree(pool1), cudaFree(dslot1);
  cudaFree(pool), cudaFree(dslot), cudaFree(dmult), cudaFree(dids), cudaFree(dw), cudaFree(dx), cudaFree(wsmem);
  cudaFree(counters), cudaFreeHost(mail_host);
  return res;
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);  // Keep progress when a graph stalls or the harness times out.
  cudaDeviceProp prop;
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s; host threads: %u\n", prop.name, std::thread::hardware_concurrency());
  std::mt19937 rng(5);
  const int threads = std::getenv("BLZ_THREADS") ? std::atoi(std::getenv("BLZ_THREADS"))
                                                 : int(std::max(1u, std::thread::hardware_concurrency() - 2));
  const bool only_big = std::getenv("BLZ_ONLY_BIG") != nullptr;
  struct Case {
    expert::Geometry geo;
    int L, E, K, T, replays;
    bool check;
  };
  const Case cases[] = {
      {{256, 64}, 8, 16, 4, 2, 50, true},
      {expert::kFlashNext, 4, 12, 10, 1, 1, true},
      {expert::kFlashNext, 48, 16, 10, 1, 5, false},
  };
  // the GPU1 helper runs next to GPU0's graph on device 0 here (in serving it has a GPU of its own): on a small GPU
  // (the laptop's 24 SMs) the spinning wait kernels and the helper's kernels may not fit at once and block each other,
  // so gpu/gpu1/cpu runs only where a second GPU exists
  int n_dev = 0;
  CHECK_CUDA(cudaGetDeviceCount(&n_dev));
  const bool with_gpu1 = n_dev >= 2;
  if (!with_gpu1) std::printf("one GPU: the gpu/gpu1/cpu residency is skipped\n");
  for (const Case& c : cases) {
    if (only_big && c.L != 48) continue;
    Model m = make_model(c.geo, c.L, c.E, c.K, rng);
    std::vector<uint16_t> x0(size_t(c.T) * c.geo.hidden);
    for (auto& v : x0) v = canon::bf16_rn(std::normal_distribution<float>(0, 1)(rng));
    std::vector<int32_t> ids;
    std::vector<float> w;
    std::vector<int> perm(c.E);
    std::iota(perm.begin(), perm.end(), 0);
    for (int l = 0; l < c.L; ++l)
      for (int t = 0; t < c.T; ++t) {
        std::shuffle(perm.begin(), perm.end(), rng);
        float sum = 0;
        for (int k = 0; k < c.K; ++k) {
          ids.push_back(perm[k]);
          w.push_back(std::uniform_real_distribution<float>(0.05f, 1.0f)(rng));
          sum += w.back();
        }
        for (int k = 0; k < c.K; ++k) w[w.size() - 1 - k] /= sum;
      }
    std::vector<uint16_t> ref;
    if (c.check) ref = reference(m, c.T, x0, ids, w);
    for (int residency : {0, 1, 2, 3}) {
      if (residency == 3 && !with_gpu1) continue;
      const Result r = run(m, c.T, x0, ids, w, residency, threads, c.replays);
      int bad = 0;
      if (c.check)
        for (size_t i = 0; i < ref.size(); ++i) bad += r.out[i] != ref[i];
      static const char* names[] = {"all-gpu", "mixed", "all-cpu", "gpu/gpu1/cpu"};
      std::printf("hybrid H=%d L=%d E=%d K=%d T=%d %-12s: %.1f us/layer, cpu requests %llu, gpu1 requests %llu%s\n",
                  c.geo.hidden, c.L, c.E, c.K, c.T, names[residency], r.us_per_layer, (unsigned long long)r.served,
                  (unsigned long long)r.served1,
                  c.check ? (bad ? " MISMATCH" : ", output == reference") : "");
      if (bad) ++g_fail;
    }
    for (int rank : {0, 2})
      for (int residency : {1, 2, 3}) {
        if (residency == 3 && !with_gpu1) continue;
        const Result r = run(m, c.T, x0, ids, w, residency, threads, c.replays, rank);
        int bad = 0;
        if (c.check) {
          const std::vector<uint16_t> dref = reference_deferred(m, c.T, x0, ids, w, residency, rank);
          for (size_t i = 0; i < dref.size(); ++i) bad += r.out[i] != dref[i];
        }
        static const char* names[] = {"all-gpu", "mixed", "all-cpu", "gpu/gpu1/cpu"};
        std::printf("hybrid H=%d L=%d E=%d K=%d T=%d %-12s deferred (rank %d): %.1f us/layer%s\n", c.geo.hidden, c.L,
                    c.E, c.K, c.T, names[residency], rank, r.us_per_layer,
                    c.check ? (bad ? " MISMATCH" : ", output == deferral reference") : "");
        if (bad) ++g_fail;
      }
  }
  std::printf(g_fail ? "FAILED\n" : "PASSED\n");
  return g_fail ? 1 : 0;
}
