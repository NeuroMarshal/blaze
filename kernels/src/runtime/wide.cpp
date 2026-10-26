// Wide route of HybridMoe (see blaze/hybrid.h, blaze/moe_wide.h): prefill-sized layer calls over every tier.
//
// One pass (<= wide_tokens tokens):
//   1. routes grouped by expert on the host (token order inside an expert); placement read under swap_mu_
//   2. GPU0-resident experts run on GPU0, GPU1-resident ones on GPU1, RAM experts are streamed to whichever GPU is
//      expected to finish first (stream and compute estimates; GPU1 also pays for receiving x and returning y)
//   3. per GPU: batches of items -> gate_up -> h_encode -> down; streamed batches alternate two staging halves, so
//      the copy engine fills one while the kernels read the other
//   4. GPU1's y goes back through pinned host memory (two halves) into GPU0's route-indexed y; combine on GPU0
// SSD-tier experts are streamed like RAM ones from a ring of pinned slots that ssd_io reader threads fill ahead, in the
// order the batches consume them (a 4096-token chunk touches nearly every expert of a layer).
// The arithmetic is the narrow route's, so a token's output does not depend on the route or the placement.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <future>
#include <stdexcept>
#include <utility>

#include "blaze/hybrid.h"
#include "blaze/moe_wide.h"
#include "blaze/log.h"

namespace blaze {

namespace {

constexpr int kBatchRoutes = 8192;  // routes per batch: h / h_planes and GPU1's y halves are sized for this
// Cost model of the RAM-expert split (microseconds): one PCIe 4.0 x16 link per GPU, and the measured wide kernels.
constexpr double kLinkBytesPerUs = 22000.0;
constexpr double kUsPerRoute = 0.5;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("blaze wide: ") + what + ": " + cudaGetErrorString(e));
}

template <class T>
void dev_alloc(T** p, size_t n, const char* what) {
  check(cudaMalloc(reinterpret_cast<void**>(p), std::max<size_t>(1, n) * sizeof(T)), what);
}

template <class T>
void host_alloc(T** p, size_t n, const char* what) {
  check(cudaHostAlloc(reinterpret_cast<void**>(p), std::max<size_t>(1, n) * sizeof(T), cudaHostAllocPortable), what);
}

}  // namespace

struct HybridMoe::Wide {
  struct Gpu {
    int device = -1;
    cudaStream_t compute = nullptr;  // GPU1 only (GPU0 computes on the caller's stream)
    cudaStream_t copy = nullptr;     // record uploads
    cudaStream_t aux = nullptr;      // GPU0: imports of GPU1's y; GPU1: exports of its y
    uint8_t* staging = nullptr;      // [2][slots] records
    cudaEvent_t staged[2] = {}, freed[2] = {};
    uint16_t* x = nullptr;           // GPU1: token columns
    uint8_t* x_planes = nullptr;
    int32_t* x_emax = nullptr;
    blz_wide_item* items = nullptr;
    int2* tiles = nullptr;
    int32_t* route_token = nullptr;
    int32_t* route_of = nullptr;     // GPU0
    uint16_t* h = nullptr;
    uint8_t* h_planes = nullptr;
    int32_t* h_emax = nullptr;
    uint16_t* y = nullptr;           // GPU0: [routes_cap][H]; GPU1: [2][kBatchRoutes][H]
    cudaEvent_t y_ready[2] = {};     // GPU1: a y half was written (compute) / exported (aux)
    cudaEvent_t y_exported[2] = {};
    cudaEvent_t y_imported[2] = {};  // GPU0 aux: a y half was copied from the host
    cudaEvent_t kernels_done = nullptr;
  } gpu[2];
  int tokens_cap = 0, routes_cap = 0, items_cap = 0, tiles_cap = 0;
  // pinned host
  int32_t* ids = nullptr;
  uint16_t* x_host = nullptr;
  uint16_t* y_host = nullptr;  // [2][kBatchRoutes][H]
  blz_wide_item* items_host[2] = {};
  int2* tiles_host[2] = {};
  int32_t* route_token_host = nullptr;
  int32_t* route_of_host = nullptr;
  // SSD-tier read-ahead: pinned slots and, per GPU, an event per slot recorded after its record left for the device.
  std::vector<uint8_t*> ahead;
  std::vector<cudaEvent_t> ahead_copied[2];
  cudaEvent_t ids_ready = nullptr, x_ready = nullptr, imports_done = nullptr;
};

void HybridMoe::init_wide() {
  if (cfg_.wide_tokens <= 0) return;
  // one expert takes at most one route per token: wide_tokens <= kBatchRoutes keeps every batch within the buffers
  if (cfg_.wide_tokens > kBatchRoutes) {
    std::fprintf(stderr, "blaze moe: wide_tokens %d above %d: %d\n", cfg_.wide_tokens, kBatchRoutes, kBatchRoutes);
    cfg_.wide_tokens = kBatchRoutes;
  }
  const expert::Geometry geo = store_->geometry();
  const int H = geo.hidden, I = geo.inter, E = num_experts();
  auto* w = new Wide;
  wide_ = w;
  w->tokens_cap = cfg_.wide_tokens;
  w->routes_cap = cfg_.wide_tokens * cfg_.topk;
  w->items_cap = E;
  w->tiles_cap = w->routes_cap / BLZ_WIDE_TILE_COLS + E;
  const int slots = std::max(1, cfg_.wide_slots);
  for (int g = 0; g < 2; ++g) {
    Wide::Gpu& d = w->gpu[g];
    d.device = g == 0 ? cfg_.gpu0 : cfg_.gpu1;
    if (d.device < 0) continue;
    check(cudaSetDevice(d.device), "set device");
    for (cudaStream_t* s : {&d.copy, &d.aux}) check(cudaStreamCreateWithFlags(s, cudaStreamNonBlocking), "stream");
    if (g == 1) check(cudaStreamCreateWithFlags(&d.compute, cudaStreamNonBlocking), "stream");
    for (int h = 0; h < 2; ++h)
      for (cudaEvent_t* e : {&d.staged[h], &d.freed[h], &d.y_ready[h], &d.y_exported[h], &d.y_imported[h]})
        check(cudaEventCreateWithFlags(e, cudaEventDisableTiming), "event");
    check(cudaEventCreateWithFlags(&d.kernels_done, cudaEventDisableTiming), "event");
    dev_alloc(&d.staging, 2 * size_t(slots) * stride_, "staging");
    if (g == 1) dev_alloc(&d.x, size_t(w->tokens_cap) * H, "x");
    dev_alloc(&d.x_planes, size_t(w->tokens_cap) * 3 * H, "x planes");
    dev_alloc(&d.x_emax, size_t(w->tokens_cap), "x emax");
    dev_alloc(&d.items, size_t(w->items_cap), "items");
    dev_alloc(&d.tiles, size_t(w->tiles_cap), "tiles");
    dev_alloc(&d.route_token, size_t(w->routes_cap), "route tokens");
    if (g == 0) dev_alloc(&d.route_of, size_t(w->routes_cap), "route_of");
    dev_alloc(&d.h, size_t(kBatchRoutes) * I, "h");
    dev_alloc(&d.h_planes, size_t(kBatchRoutes) * 3 * I, "h planes");
    dev_alloc(&d.h_emax, size_t(kBatchRoutes), "h emax");
    dev_alloc(&d.y, g == 0 ? size_t(w->routes_cap) * H : 2 * size_t(kBatchRoutes) * H, "y");
    host_alloc(&w->items_host[g], size_t(w->items_cap), "items");
    host_alloc(&w->tiles_host[g], size_t(w->tiles_cap), "tiles");
    w->ahead_copied[g].resize(size_t(std::max(2, cfg_.ssd_io_threads)));
    for (cudaEvent_t& e : w->ahead_copied[g]) check(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "event");
  }
  check(cudaSetDevice(cfg_.gpu0), "set gpu0");
  check(cudaEventCreateWithFlags(&w->ids_ready, cudaEventDisableTiming), "event");
  check(cudaEventCreateWithFlags(&w->x_ready, cudaEventDisableTiming), "event");
  check(cudaEventCreateWithFlags(&w->imports_done, cudaEventDisableTiming), "event");
  host_alloc(&w->ids, size_t(w->routes_cap), "ids");
  host_alloc(&w->route_token_host, size_t(w->routes_cap), "route tokens");
  host_alloc(&w->route_of_host, size_t(w->routes_cap), "route_of");
  if (cfg_.gpu1 >= 0) {
    host_alloc(&w->x_host, size_t(w->tokens_cap) * H, "x");
    host_alloc(&w->y_host, 2 * size_t(kBatchRoutes) * H, "y");
  }
  w->ahead.resize(size_t(std::max(2, cfg_.ssd_io_threads)));
  for (uint8_t*& b : w->ahead) host_alloc(&b, stride_, "ssd read-ahead");
  prefill_routes_.assign(size_t(num_layers()) * E, 0);
}

void HybridMoe::free_wide() {
  Wide* w = wide_;
  if (!w) return;
  for (int g = 0; g < 2; ++g) {
    Wide::Gpu& d = w->gpu[g];
    if (d.device < 0) continue;
    cudaSetDevice(d.device);
    cudaDeviceSynchronize();
    for (cudaStream_t s : {d.compute, d.copy, d.aux})
      if (s) cudaStreamDestroy(s);
    for (int h = 0; h < 2; ++h)
      for (cudaEvent_t e : {d.staged[h], d.freed[h], d.y_ready[h], d.y_exported[h], d.y_imported[h]}) cudaEventDestroy(e);
    cudaEventDestroy(d.kernels_done);
    for (void* p : {(void*)d.staging, (void*)d.x, (void*)d.x_planes, (void*)d.x_emax, (void*)d.items,
                    (void*)d.tiles, (void*)d.route_token, (void*)d.route_of, (void*)d.h, (void*)d.h_planes,
                    (void*)d.h_emax, (void*)d.y})
      if (p) cudaFree(p);
    cudaFreeHost(w->items_host[g]);
    cudaFreeHost(w->tiles_host[g]);
    for (cudaEvent_t e : w->ahead_copied[g]) cudaEventDestroy(e);
  }
  cudaSetDevice(cfg_.gpu0);
  cudaEventDestroy(w->ids_ready);
  cudaEventDestroy(w->x_ready);
  cudaEventDestroy(w->imports_done);
  for (void* p : {(void*)w->ids, (void*)w->x_host, (void*)w->y_host, (void*)w->route_token_host,
                  (void*)w->route_of_host})
    if (p) cudaFreeHost(p);
  for (uint8_t* b : w->ahead) cudaFreeHost(b);
  delete w;
  wide_ = nullptr;
}

cudaError_t HybridMoe::forward_wide(int layer, const uint16_t* x, int tokens, const int32_t* ids, const float* w,
                                    uint16_t* out, cudaStream_t stream) {
  if (!wide_) return cudaErrorNotSupported;
  const int H = store_->geometry().hidden, K = cfg_.topk;
  for (int t0 = 0; t0 < tokens; t0 += wide_->tokens_cap) {
    const int n = std::min(wide_->tokens_cap, tokens - t0);
    const cudaError_t err = wide_pass(layer, x + size_t(t0) * H, n, ids + size_t(t0) * K, w + size_t(t0) * K,
                                      out + size_t(t0) * H, stream);
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

cudaError_t HybridMoe::wide_pass(int layer, const uint16_t* x, int T, const int32_t* ids, const float* wts,
                                 uint16_t* out, cudaStream_t stream) {
  Wide& W = *wide_;
  const expert::Geometry geo = store_->geometry();
  const int H = geo.hidden, I = geo.inter, K = cfg_.topk, E = num_experts();
  const int slots = std::max(1, cfg_.wide_slots);
  const int R = T * K;
  const bool has_gpu1 = cfg_.gpu1 >= 0;
  try {
    check(cudaSetDevice(cfg_.gpu0), "set gpu0");
    check(cudaMemcpyAsync(W.ids, ids, size_t(R) * 4, cudaMemcpyDeviceToHost, stream), "ids");
    // The CPU only needs route IDs before grouping. Mark that transfer separately so grouping overlaps GPU0's
    // encoding and GPU1's x staging; all GPU work still remains ordered on the caller stream.
    check(cudaEventRecord(W.ids_ready, stream), "ids ready");
    if (has_gpu1) {
      check(cudaMemcpyAsync(W.x_host, x, size_t(T) * H * 2, cudaMemcpyDeviceToHost, stream), "x to host");
      check(cudaEventRecord(W.x_ready, stream), "x ready");
    }
    check(blz_wide_encode(x, T, H, W.gpu[0].x_planes, W.gpu[0].x_emax, stream), "encode");
    if (cfg_.wide_a8) check(blz_wide_to_a8(W.gpu[0].x_planes, T, H, stream), "x to a8");
    check(cudaEventSynchronize(W.ids_ready), "ids ready");

    // Routes by expert, in token order.
    std::vector<int> count(E, 0);
    for (int r = 0; r < R; ++r) {
      const int e = W.ids[r];
      if (e < 0 || e >= E) throw std::runtime_error("expert id out of range");
      ++count[e];
    }
    std::vector<int> start(E + 1, 0);
    for (int e = 0; e < E; ++e) start[e + 1] = start[e] + count[e];
    std::vector<int> by_expert(R);
    {
      std::vector<int> fill(start.begin(), start.end() - 1);
      for (int r = 0; r < R; ++r) by_expert[fill[W.ids[r]]++] = r;
    }

    std::unique_lock<std::mutex> lock(swap_mu_);
    // Experts per GPU: resident ones first, then the streamed RAM experts.
    struct Item {
      int e;
      bool streamed;
      const uint8_t* src;  // device record (resident) or host record (streamed; null: read from SSD)
    };
    std::vector<Item> res[2], ram;
    for (int e = 0; e < E; ++e) {
      if (count[e] == 0) continue;
      const size_t i = size_t(layer) * E + e;
      const Tier t = placement_.tier[i];
      const int slot = placement_.slot[i];
      if (t == Tier::kGpu0) res[0].push_back({e, false, gpu_slot(Tier::kGpu0, slot)});
      else if (t == Tier::kGpu1) res[1].push_back({e, false, gpu_slot(Tier::kGpu1, slot)});
      else ram.push_back({e, true, host_records_[i]});
      prefill_routes_[i] += uint32_t(count[e]);
    }
    // Split the RAM experts. Each GPU's time ~ max(its link's host -> device bytes, its compute); GPU1's y comes back
    // over GPU0's host -> device direction (the same one GPU0 streams on), x goes to GPU1 over GPU1's.
    const double rec_us = double(stride_) / kLinkBytesPerUs, y_us = double(H) * 2 / kLinkBytesPerUs;
    double link[2] = {0, 0}, comp[2] = {0, 0};
    for (int g = 0; g < 2; ++g)
      for (const Item& it : res[g]) comp[g] += kUsPerRoute * count[it.e];
    if (has_gpu1) {
      link[1] += double(T) * H * 2 / kLinkBytesPerUs;
      for (const Item& it : res[1]) link[0] += y_us * count[it.e];
    }
    std::sort(ram.begin(), ram.end(), [&](const Item& a, const Item& b) { return count[a.e] > count[b.e]; });
    std::vector<Item> streamed[2];
    for (const Item& it : ram) {
      const double c = kUsPerRoute * count[it.e];
      int best = 0;
      if (has_gpu1) {
        const double t0 = std::max(link[0] + rec_us, comp[0] + c);
        const double t1 = std::max(link[1] + rec_us, comp[1] + c);
        const double t0_if1 = std::max(link[0] + y_us * count[it.e], comp[0]);  // GPU0 also imports GPU1's y
        best = std::max(t1, t0_if1) < std::max(t0, std::max(link[1], comp[1])) ? 1 : 0;
      }
      if (best == 0) {
        link[0] += rec_us, comp[0] += c;
      } else {
        link[1] += rec_us, comp[1] += c, link[0] += y_us * count[it.e];
      }
      streamed[best].push_back(it);
    }

    // Route numbering (GPU0's routes first) and batches.
    struct Batch {
      int item0, items, tile0, tiles, route0, routes;
      bool streamed;
      int half;
    };
    std::vector<Batch> batches[2];
    int next_route = 0;
    for (int g = 0; g < 2; ++g) {
      int n_items = 0, n_tiles = 0, half = 0;
      for (int pass = 0; pass < 2; ++pass) {
        const std::vector<Item>& list = pass == 0 ? res[g] : streamed[g];
        const bool is_streamed = pass == 1;
        Batch* b = nullptr;
        for (const Item& it : list) {
          const int c = count[it.e];
          if (!b || b->routes + c > kBatchRoutes || (is_streamed && b->items == slots)) {
            batches[g].push_back({n_items, 0, n_tiles, 0, next_route, 0, is_streamed, is_streamed ? half : 0});
            if (is_streamed) half ^= 1;
            b = &batches[g].back();
          }
          const uint8_t* rec = is_streamed
                                   ? W.gpu[g].staging + (size_t(b->half) * slots + size_t(b->items)) * stride_
                                   : it.src;
          W.items_host[g][n_items] = {rec, it.e, next_route, c, 0};
          for (int col = 0; col < c; col += BLZ_WIDE_TILE_COLS) {
            W.tiles_host[g][n_tiles++] = make_int2(b->items, col);
            ++b->tiles;
          }
          for (int j = start[it.e]; j < start[it.e + 1]; ++j) {
            const int r = by_expert[j];
            W.route_of_host[r] = next_route;
            W.route_token_host[next_route] = r / K;
            ++next_route;
          }
          ++b->items, ++n_items;
          b->routes += c;
        }
      }
    }

    // Tables.
    for (int g = 0; g < 2; ++g) {
      Wide::Gpu& d = W.gpu[g];
      if (d.device < 0) continue;
      check(cudaSetDevice(d.device), "set device");
      cudaStream_t cs = g == 0 ? stream : d.compute;
      const int n_items = batches[g].empty() ? 0 : batches[g].back().item0 + batches[g].back().items;
      const int n_tiles = batches[g].empty() ? 0 : batches[g].back().tile0 + batches[g].back().tiles;
      check(cudaMemcpyAsync(d.items, W.items_host[g], size_t(n_items) * sizeof(blz_wide_item), cudaMemcpyHostToDevice,
                            cs),
            "items");
      check(cudaMemcpyAsync(d.tiles, W.tiles_host[g], size_t(n_tiles) * sizeof(int2), cudaMemcpyHostToDevice, cs),
            "tiles");
      check(cudaMemcpyAsync(d.route_token, W.route_token_host, size_t(R) * 4, cudaMemcpyHostToDevice, cs), "routes");
      if (g == 0)
        check(cudaMemcpyAsync(d.route_of, W.route_of_host, size_t(R) * 4, cudaMemcpyHostToDevice, cs), "route_of");
      if (g == 1 && !batches[1].empty()) {
        check(cudaStreamWaitEvent(cs, W.x_ready, 0), "wait x");
        check(cudaMemcpyAsync(d.x, W.x_host, size_t(T) * H * 2, cudaMemcpyHostToDevice, cs), "x to gpu1");
        check(blz_wide_encode(d.x, T, H, d.x_planes, d.x_emax, cs), "encode gpu1");
        if (cfg_.wide_a8) check(blz_wide_to_a8(d.x_planes, T, H, cs), "x to a8 gpu1");
      }
    }

    // SSD-tier records in the order the loop below streams them, read ahead into the ring of pinned slots. A slot is
    // refilled only once the copy that read it out has completed (the reader waits for that copy's event).
    std::vector<std::pair<int, int>> ssd;  // (GPU, expert) per SSD-tier record
    {
      size_t nx[2] = {0, 0};
      while (nx[0] < batches[0].size() || nx[1] < batches[1].size())
        for (int g = 0; g < 2; ++g) {
          if (nx[g] >= batches[g].size()) continue;
          const Batch& b = batches[g][nx[g]++];
          if (!b.streamed) continue;
          for (int j = 0; j < b.items; ++j) {
            const int e = W.items_host[g][b.item0 + j].expert;
            if (!host_records_[size_t(layer) * E + e]) ssd.emplace_back(g, e);
          }
        }
    }
    const size_t ring = W.ahead.size();
    std::vector<std::future<void>> reads(ssd.size());
    auto start_read = [&](size_t n) {
      const size_t slot = n % ring;
      // The previous user of the slot (record n - ring) and the GPU its copy was queued on.
      const cudaEvent_t copied = n >= ring ? W.ahead_copied[ssd[n - ring].first][slot] : nullptr;
      const int prev_dev = n >= ring ? W.gpu[ssd[n - ring].first].device : -1;
      reads[n] = std::async(std::launch::async, [this, &W, layer, slot, copied, prev_dev, e = ssd[n].second] {
        if (copied) {
          check(cudaSetDevice(prev_dev), "set device");
          check(cudaEventSynchronize(copied), "read-ahead slot");
        }
        store_->read(layer, e, W.ahead[slot], cfg_.ssd_direct);
      });
    };
    for (size_t n = 0; n < std::min(ring, ssd.size()); ++n) start_read(n);
    size_t ssd_next = 0;

    // Batches: both GPUs' work is issued interleaved so their copy engines and SMs run concurrently.
    const float* mult[2] = {mult0_ + size_t(layer) * E * 3, has_gpu1 ? mult1_ + size_t(layer) * E * 3 : nullptr};
    size_t next[2] = {0, 0};
    int y_half = 0;
    bool any_import = false;
    while (next[0] < batches[0].size() || next[1] < batches[1].size()) {
      for (int g = 0; g < 2; ++g) {
        if (next[g] >= batches[g].size()) continue;
        const Batch& b = batches[g][next[g]++];
        Wide::Gpu& d = W.gpu[g];
        check(cudaSetDevice(d.device), "set device");
        cudaStream_t cs = g == 0 ? stream : d.compute;
        if (b.streamed) {
          check(cudaStreamWaitEvent(d.copy, d.freed[b.half], 0), "wait staging");
          for (int j = 0; j < b.items; ++j) {
            const blz_wide_item& it = W.items_host[g][b.item0 + j];
            const size_t i = size_t(layer) * E + it.expert;
            const uint8_t* src = host_records_[i];
            if (src) {
              check(cudaMemcpyAsync(const_cast<uint8_t*>(it.record), src, stride_, cudaMemcpyHostToDevice, d.copy),
                    "stream record");
              continue;
            }
            const size_t n = ssd_next++, slot = n % ring;  // SSD tier: the read-ahead slot of this record
            reads[n].get();
            check(cudaSetDevice(d.device), "set device");
            check(cudaMemcpyAsync(const_cast<uint8_t*>(it.record), W.ahead[slot], stride_, cudaMemcpyHostToDevice,
                                  d.copy),
                  "stream ssd record");
            check(cudaEventRecord(W.ahead_copied[g][slot], d.copy), "ssd record copied");
            if (n + ring < ssd.size()) start_read(n + ring);
          }
          check(cudaEventRecord(d.staged[b.half], d.copy), "staged");
          check(cudaStreamWaitEvent(cs, d.staged[b.half], 0), "wait staged");
        }
        blz_wide_batch bt{};
        bt.items = d.items + b.item0;
        bt.tiles = d.tiles + b.tile0;
        bt.n_tiles = b.tiles;
        bt.n_routes = b.routes;
        bt.route0 = b.route0;
        bt.route_token = d.route_token;
        bt.x_planes = d.x_planes, bt.x_emax = d.x_emax, bt.mult = mult[g];
        bt.hidden = H, bt.inter = I;
        bt.h = d.h, bt.h_planes = d.h_planes, bt.h_emax = d.h_emax;
        bt.a8 = cfg_.wide_a8;
        if (g == 0) {
          bt.y = d.y + size_t(b.route0) * H;
        } else {
          bt.y = d.y + size_t(y_half) * kBatchRoutes * H;
          check(cudaStreamWaitEvent(cs, d.y_exported[y_half], 0), "wait y half");
        }
        check(blz_wide_gate_up(&bt, cs), "gate_up");
        check(blz_wide_h_encode(&bt, cs), "h_encode");
        if (cfg_.wide_a8) check(blz_wide_to_a8(bt.h_planes, b.routes, I, cs), "h to a8");
        check(blz_wide_down(&bt, cs), "down");
        if (b.streamed) check(cudaEventRecord(d.freed[b.half], cs), "freed");
        if (g == 1) {
          // y half -> host -> GPU0's y at the batch's routes.
          Wide::Gpu& d0 = W.gpu[0];
          uint16_t* yh = W.y_host + size_t(y_half) * kBatchRoutes * H;
          const size_t bytes = size_t(b.routes) * H * 2;
          check(cudaEventRecord(d.y_ready[y_half], cs), "y ready");
          check(cudaStreamWaitEvent(d.aux, d.y_ready[y_half], 0), "wait y");
          check(cudaStreamWaitEvent(d.aux, d0.y_imported[y_half], 0), "wait host half");
          check(cudaMemcpyAsync(yh, bt.y, bytes, cudaMemcpyDeviceToHost, d.aux), "y to host");
          check(cudaEventRecord(d.y_exported[y_half], d.aux), "y exported");
          check(cudaSetDevice(cfg_.gpu0), "set gpu0");
          check(cudaStreamWaitEvent(d0.aux, d.y_exported[y_half], 0), "wait export");
          check(cudaMemcpyAsync(d0.y + size_t(b.route0) * H, yh, bytes, cudaMemcpyHostToDevice, d0.aux), "y to gpu0");
          check(cudaEventRecord(d0.y_imported[y_half], d0.aux), "y imported");
          y_half ^= 1;
          any_import = true;
        }
      }
    }
    check(cudaSetDevice(cfg_.gpu0), "set gpu0");
    if (any_import) {
      check(cudaEventRecord(W.imports_done, W.gpu[0].aux), "imports");
      check(cudaStreamWaitEvent(stream, W.imports_done, 0), "wait imports");
    }
    check(blz_wide_combine(W.gpu[0].route_of, wts, W.gpu[0].y, T, K, H, out, stream), "combine");
    // Pinned tables, staging and pool slots are reused by the next pass / the swap engine.
    check(cudaStreamSynchronize(stream), "wide pass");
    return cudaSuccess;
  } catch (const std::exception& e) {
    log("blaze: %s", e.what());
    cudaSetDevice(cfg_.gpu0);
    return cudaErrorUnknown;
  }
}

}  // namespace blaze
