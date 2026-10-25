#include "blaze/hybrid.h"
#include "blaze/log.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace blaze {

namespace {

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("blaze hybrid: ") + what + ": " + cudaGetErrorString(e));
}

// Pinned memory cannot be swapped out: an arena larger than what the system can spare starves every other process
// (the machine stops answering) instead of failing. Keep this much available after the arena.

size_t mem_available() {
  FILE* f = std::fopen("/proc/meminfo", "r");
  if (!f) return SIZE_MAX;
  char line[256];
  size_t kib = SIZE_MAX;
  while (std::fgets(line, sizeof(line), f))
    if (std::sscanf(line, "MemAvailable: %zu kB", &kib) == 1) break;
  std::fclose(f);
  return kib == SIZE_MAX ? SIZE_MAX : kib * 1024;
}

}  // namespace

HybridMoe::HybridMoe(const HybridConfig& cfg) : cfg_(cfg) {
  store_ = std::make_unique<ExpertStore>(cfg_.model_dir, "", 192 << 10, cfg_.mtp_dir);
  const int L = store_->num_layers(), E = store_->num_experts();
  const expert::Geometry geo = store_->geometry();
  const size_t n = size_t(L) * E;
  stride_ = store_->record_stride();
  if (cfg_.defer_last == -2) cfg_.defer_last = store_->main_layers() - 1;
  capacity_[0] = int(cfg_.gpu0_bytes / stride_);
  capacity_[1] = cfg_.gpu1 >= 0 ? int(cfg_.gpu1_bytes / stride_) : 0;
  int spare[3] = {0, 0, 0};
  if (cfg_.swap.enabled)
    for (int t = 0; t < 2; ++t) spare[t] = std::min(cfg_.swap.spare_slots, capacity_[t] / 2);
  if (cfg_.ram_bytes == HybridConfig::kRamAuto) {
    // RAM takes every expert the GPUs do not hold: nothing is left on SSD.
    const size_t on_gpu = std::min(n, size_t(capacity_[0] - spare[0]) + size_t(capacity_[1] - spare[1]));
    spare[2] = cfg_.swap.enabled ? cfg_.swap.spare_slots : 0;
    capacity_[2] = int(n - on_gpu) + spare[2];
  } else {
    capacity_[2] = int(cfg_.ram_bytes / stride_);
    if (cfg_.swap.enabled) spare[2] = std::min(cfg_.swap.spare_slots, capacity_[2] / 2);
  }
  std::vector<double> heat = cfg_.heat, heat_long = cfg_.heat_long;
  if (heat.empty() && !cfg_.swap.heat_path.empty()) {
    HeatProfile profile = load_heat(cfg_.swap.heat_path, L, E);
    heat = std::move(profile.recent), heat_long = std::move(profile.longterm);
  }
  if (heat_long.size() != n) heat_long = heat;
  placement_ = place_experts(L, E, heat, capacity_[0] - spare[0], capacity_[1] - spare[1], capacity_[2] - spare[2],
                             heat_long);
  log("blaze: %d layers x %d experts (%s%s) -> GPU0 %d, GPU1 %d, RAM %d%s", L, E, store_->prefix().c_str(),
      heat.empty() ? "" : ", placement from the routing profile", placement_.used[0], placement_.used[1],
      placement_.used[2], placement_.used[3] ? (", SSD " + std::to_string(placement_.used[3])).c_str() : "");
  for (int t = 0; t < 3; ++t)
    for (int s = placement_.used[t]; s < capacity_[t]; ++s) free_[t].push_back(s);
  for (int t = 0; t < 4; ++t) stat_resident_[t] = placement_.used[t];
  score_ = heat.size() == n ? heat : std::vector<double>(n, 0.0);
  score_long_ = heat_long.size() == n ? heat_long : std::vector<double>(n, 0.0);

  // Multipliers and planning tables.
  mult_host_.resize(n * 3);
  std::vector<int32_t> slot0(n, -1), slot1(n, -1);
  host_records_.assign(n, nullptr);
  for (size_t i = 0; i < n; ++i) {
    std::memcpy(&mult_host_[i * 3], store_->mult(int(i / E), int(i % E)), 3 * sizeof(float));
    if (placement_.tier[i] == Tier::kGpu0) slot0[i] = placement_.slot[i];
    if (placement_.tier[i] == Tier::kSsd) slot0[i] = kSlotSsd;
    if (placement_.tier[i] == Tier::kGpu1) slot1[i] = placement_.slot[i];
  }

  // Host side first: pinned RAM arena and the mapped mailbox (portable: both GPUs DMA from them).
  const size_t arena = size_t(capacity_[2]) * stride_, avail = mem_available();
  const size_t host_reserve = size_t(cfg_.host_reserve_gib * double(1 << 30));
  if (arena > 0 && (avail < host_reserve || arena > avail - host_reserve))
    throw std::runtime_error("blaze hybrid: RAM tier of " + std::to_string(arena >> 20) + " MiB does not fit: " +
                             std::to_string(avail >> 20) + " MiB available, " + std::to_string(host_reserve >> 20) +
                             " MiB must stay free (lower the RAM budget)");
  if (capacity_[2] > 0)
    check(cudaHostAlloc(reinterpret_cast<void**>(&ram_), size_t(capacity_[2]) * stride_, cudaHostAllocPortable),
          "RAM arena");
  for (size_t i = 0; i < n; ++i)
    if (placement_.tier[i] == Tier::kRam) host_records_[i] = ram_slot(placement_.slot[i]);
  const blz_mail_layout ml = blz_mail_layout_for(geo.hidden);
  check(cudaHostAlloc(reinterpret_cast<void**>(&mail_), ml.total, cudaHostAllocMapped | cudaHostAllocPortable),
        "mailbox");
  std::memset(mail_, 0, ml.total);
  check(cudaHostAlloc(reinterpret_cast<void**>(&heat_host_), n * 4, cudaHostAllocPortable), "heat snapshot");
  check(cudaHostAlloc(reinterpret_cast<void**>(&table_host_), kTableStaging * 4, cudaHostAllocPortable),
        "table staging");
  heat_last_.assign(n, 0);
  loading_.assign(n, 0);

  // GPU0
  check(cudaSetDevice(cfg_.gpu0), "set gpu0");
  check(cudaMalloc(&pool0_, std::max<size_t>(1, size_t(capacity_[0]) * stride_)), "GPU0 pool");
  check(cudaMalloc(&slot0_dev_, n * 4), "slot table");
  check(cudaMemcpy(slot0_dev_, slot0.data(), n * 4, cudaMemcpyHostToDevice), "slot table");
  check(cudaMalloc(&slot1_dev_, n * 4), "slot1 table");
  check(cudaMemcpy(slot1_dev_, slot1.data(), n * 4, cudaMemcpyHostToDevice), "slot1 table");
  check(cudaMalloc(&mult0_, n * 3 * 4), "mult");
  check(cudaMemcpy(mult0_, mult_host_.data(), n * 3 * 4, cudaMemcpyHostToDevice), "mult");
  check(cudaMalloc(&heat_dev_, n * 4), "heat");
  check(cudaMemset(heat_dev_, 0, n * 4), "heat");
  check(cudaMalloc(&ws_mem_, blz_moe_workspace_bytes(geo.hidden, geo.inter)), "workspace");
  blz_moe_workspace_carve(ws_mem_, geo.hidden, geo.inter, &ws_);
  check(cudaMemset(ws_.defer, 0, sizeof(blz_moe_defer)), "defer state");
  check(cudaMalloc(&counters_, 8), "counters");
  check(cudaMemset(counters_, 0, 8), "counters");
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&ws_.mail), mail_, 0), "mailbox device pointer");
  ws_.seq = counters_, ws_.blocks_done = counters_ + 1;
  check(cudaStreamCreateWithFlags(&swap_stream_[0], cudaStreamNonBlocking), "swap stream");
  check(cudaStreamCreateWithFlags(&d2h_stream_[0], cudaStreamNonBlocking), "swap stream");

  // GPU1
  if (cfg_.gpu1 >= 0) {
    check(cudaSetDevice(cfg_.gpu1), "set gpu1");
    check(cudaMalloc(&pool1_, std::max<size_t>(1, size_t(capacity_[1]) * stride_)), "GPU1 pool");
    check(cudaMalloc(&mult1_, n * 3 * 4), "mult1");
    check(cudaMemcpy(mult1_, mult_host_.data(), n * 3 * 4, cudaMemcpyHostToDevice), "mult1");
    check(cudaStreamCreateWithFlags(&swap_stream_[1], cudaStreamNonBlocking), "swap stream");
    check(cudaStreamCreateWithFlags(&d2h_stream_[1], cudaStreamNonBlocking), "swap stream");
  }

  load_tiers();

  cpu::EngineConfig ec;
  ec.geo = geo, ec.num_layers = L, ec.num_experts = E, ec.threads = cfg_.cpu_threads, ec.cores = cfg_.cpu_cores;
  ec.store = store_.get();
  ec.fast_min_cols = cfg_.cpu_fast_cols;
  ec.gu_chunk_groups = cfg_.cpu_gu_chunk;
  ec.down_chunk_groups = cfg_.cpu_down_chunk;
  ec.ssd_io_threads = cfg_.ssd_io_threads;
  ec.tune_threads = cfg_.cpu_tune;
  ec.ssd_direct = cfg_.ssd_direct;
  cpu_ = std::make_unique<cpu::Engine>(ec, mail_, host_records_.data(), mult_host_.data());
  if (cfg_.gpu1 >= 0) {
    const blz_moe_layer helper{geo.hidden, geo.inter, E, cfg_.topk, nullptr, nullptr, pool1_, int64_t(stride_), mult1_,
                               nullptr};
    gpu1_ = std::make_unique<Gpu1Service>(cfg_.gpu1, helper, mail_);
  }
  check(cudaSetDevice(cfg_.gpu0), "set gpu0");
  init_wide();
  if (cfg_.swap.enabled) {
    for (int t = 0; t < std::max(1, cfg_.swap.ssd_threads); ++t) ssd_pool_.emplace_back([this] { ssd_worker(); });
    swap_thread_ = std::thread([this] { swap_loop(); });
  }
}

HybridMoe::~HybridMoe() {
  if (swap_thread_.joinable()) {
    swap_stop_.store(true, std::memory_order_release);
    swap_thread_.join();
    save_profile();
  }
  free_wide();
  {
    std::lock_guard<std::mutex> lock(ssd_mu_);
    ssd_stop_ = true;
  }
  ssd_cv_.notify_all();
  for (auto& t : ssd_pool_) t.join();
  gpu1_.reset();
  cpu_.reset();
  cudaSetDevice(cfg_.gpu0);
  cudaFree(pool0_), cudaFree(slot0_dev_), cudaFree(slot1_dev_), cudaFree(mult0_), cudaFree(heat_dev_);
  cudaFree(ws_mem_), cudaFree(counters_);
  for (cudaStream_t st : {swap_stream_[0], d2h_stream_[0]})
    if (st) cudaStreamDestroy(st);
  if (cfg_.gpu1 >= 0) {
    cudaSetDevice(cfg_.gpu1);
    cudaFree(pool1_), cudaFree(mult1_);
    for (cudaStream_t st : {swap_stream_[1], d2h_stream_[1]})
      if (st) cudaStreamDestroy(st);
  }
  if (ram_) cudaFreeHost(ram_);
  if (mail_) cudaFreeHost(mail_);
  if (heat_host_) cudaFreeHost(heat_host_);
  if (table_host_) cudaFreeHost(table_host_);
}

HybridStats HybridMoe::stats() const {
  HybridStats st;
  for (int t = 0; t < 4; ++t) {
    st.routes[t] = stat_routes_[t].load(std::memory_order_relaxed);
    st.resident[t] = stat_resident_[t].load(std::memory_order_relaxed);
  }
  st.moves_gpu = stat_moves_gpu_.load(std::memory_order_relaxed);
  st.moves_ram = stat_moves_ram_.load(std::memory_order_relaxed);
  return st;
}

void HybridMoe::load_tiers() {
  // Experts are laid out from the checkpoint straight into their tier: RAM-tier records into the pinned arena,
  // GPU-tier records through a pinned staging buffer per loader thread.
  const int L = store_->num_layers(), E = store_->num_experts();
  const size_t n = size_t(L) * E;
  std::atomic<size_t> next{0}, done_bytes{0};
  std::atomic<bool> failed{false};
  std::string error;
  auto work = [&] {
    try {
      uint8_t* staging = nullptr;
      cudaStream_t streams[2] = {nullptr, nullptr};
      check(cudaHostAlloc(reinterpret_cast<void**>(&staging), stride_, cudaHostAllocPortable), "staging");
      for (size_t i; !failed && (i = next++) < n;) {
        const Tier t = placement_.tier[i];
        if (t == Tier::kSsd) continue;
        const int l = int(i / E), e = int(i % E);
        // O_DIRECT: the checkpoint bypasses the page cache, which would otherwise hold it next to the pinned copies
        // and push the rest of the host memory to swap.
        if (t == Tier::kRam) {
          store_->read(l, e, ram_slot(placement_.slot[i]), true);
        } else {
          store_->read(l, e, staging, true);
          const int dev = t == Tier::kGpu0 ? cfg_.gpu0 : cfg_.gpu1;
          uint8_t* dst = gpu_slot(t, placement_.slot[i]);
          check(cudaSetDevice(dev), "set device");
          cudaStream_t& s = streams[t == Tier::kGpu0 ? 0 : 1];
          if (!s) check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
          check(cudaMemcpyAsync(dst, staging, stride_, cudaMemcpyHostToDevice, s), "upload");
          check(cudaStreamSynchronize(s), "upload sync");
        }
        done_bytes += stride_;
      }
      for (int d = 0; d < 2; ++d)
        if (streams[d]) {
          cudaSetDevice(d == 0 ? cfg_.gpu0 : cfg_.gpu1);
          cudaStreamDestroy(streams[d]);
        }
      cudaFreeHost(staging);
    } catch (const std::exception& ex) {
      if (!failed.exchange(true)) error = ex.what();
    }
  };
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < std::max(1, cfg_.load_threads); ++t) pool.emplace_back(work);
  for (auto& t : pool) t.join();
  if (failed) throw std::runtime_error(error);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  log("blaze: loaded %.1f GiB of experts from the checkpoint in %.1f s (%.2f GB/s)",
              done_bytes / double(1 << 30), s, done_bytes / s / 1e9);
}

cudaError_t HybridMoe::forward(int layer, const uint16_t* x, int tokens, const int32_t* ids, const float* w,
                               uint16_t* out, cudaStream_t stream, bool allow_defer) {
  const expert::Geometry geo = store_->geometry();
  const int E = store_->num_experts();
  const blz_moe_layer l{geo.hidden,
                        geo.inter,
                        E,
                        cfg_.topk,
                        slot0_dev_ + size_t(layer) * E,
                        cfg_.gpu1 >= 0 ? slot1_dev_ + size_t(layer) * E : nullptr,
                        pool0_,
                        int64_t(stride_),
                        mult0_ + size_t(layer) * E * 3,
                        heat_dev_ + size_t(layer) * E,
                        cfg_.defer_last >= 0 && allow_defer && layer < cfg_.defer_last,
                        cfg_.defer_min_rank,
                        w,
                        cfg_.imm_heat,
                        cfg_.defer_ssd_rank >= 0 ? cfg_.defer_ssd_rank : cfg_.defer_min_rank};
  const bool deferral = cfg_.defer_last >= 0;
  cudaError_t err = deferral ? blz_moe_defer_collect(geo.hidden, tokens, &ws_, stream) : cudaSuccess;
  if (err != cudaSuccess) return err;
  err = blz_moe_gpu_route_layer(&l, layer, x, tokens, ids, &ws_, stream);
  if (err != cudaSuccess) return err;
  err = deferral ? blz_moe_wait_defer(&ws_, stream) : blz_moe_wait_cpu(&ws_, stream);
  if (err != cudaSuccess) return err;
  return blz_moe_combine(&l, tokens, w, &ws_, nullptr, out, stream);
}

}  // namespace blaze
