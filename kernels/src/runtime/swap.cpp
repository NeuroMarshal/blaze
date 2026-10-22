// Swap engine of HybridMoe (see blaze/hybrid.h): decayed route scores and exclusive moves between the tiers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "blaze/hybrid.h"
#include "blaze/log.h"

namespace blaze {

namespace {

constexpr char kHeatMagicV1[8] = {'Q', 'R', 'T', 'H', 'E', 'A', 'T', '1'};
constexpr char kHeatMagic[8] = {'Q', 'R', 'T', 'H', 'E', 'A', 'T', '2'};
// Freed slots are reused after this many further layer calls: by then every plan made with the old tables, and the
// CPU / GPU1 work it published, is complete (one call can be planned before a table write becomes visible).
constexpr uint32_t kEpochCalls = 3;
// A move must win by at least this many (decayed) routes on top of the hysteresis factor, so noise never moves.
constexpr double kMinGain = 1.0;

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("blaze swap: ") + what + ": " + cudaGetErrorString(e));
}

// The `count` indices of `idx` with the highest (hot) or lowest (cold) score, best first.
void select(std::vector<size_t>& idx, const std::vector<double>& score, size_t count, bool hot) {
  count = std::min(count, idx.size());
  auto better = [&](size_t a, size_t b) { return hot ? score[a] > score[b] : score[a] < score[b]; };
  std::partial_sort(idx.begin(), idx.begin() + ptrdiff_t(count), idx.end(), better);
  idx.resize(count);
}

}  // namespace

SwapConfig parse_swap_options(const std::string& options, SwapConfig c) {
  std::stringstream ss(options);
  std::string kv;
  while (std::getline(ss, kv, ',')) {
    if (kv.empty()) continue;
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("swap option without '=': " + kv);
    const std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
    if (key == "enabled") c.enabled = val != "0" && val != "false" && val != "off";
    else if (key == "period_ms") c.period_ms = std::stoi(val);
    else if (key == "max_moves") c.max_moves = std::stoi(val);
    else if (key == "halflife") c.halflife_tokens = std::stod(val);
    else if (key == "halflife_ssd") c.halflife_ssd_tokens = std::stod(val);
    else if (key == "hysteresis") c.hysteresis = std::stod(val);
    else if (key == "spare") c.spare_slots = std::stoi(val);
    else if (key == "ssd_threads") c.ssd_threads = std::stoi(val);
    else if (key == "prefill_weight") c.prefill_weight = std::stod(val);
    else if (key == "demote") {
      if (val != "pcie" && val != "ssd") throw std::invalid_argument("demote must be pcie or ssd");
      c.demote = val;
    }
    else if (key == "save_s") c.save_period_s = std::stoi(val);
    else if (key == "log_s") c.log_period_s = std::stoi(val);
    else throw std::invalid_argument("unknown swap option: " + key);
  }
  return c;
}

void apply_hybrid_options(const std::string& options, HybridConfig& cfg) {
  std::stringstream ss(options);
  std::string kv;
  while (std::getline(ss, kv, ',')) {
    if (kv.empty()) continue;
    const size_t eq = kv.find('=');
    if (eq == std::string::npos) throw std::invalid_argument("option without '=': " + kv);
    const std::string key = kv.substr(0, eq), val = kv.substr(eq + 1);
    if (key == "cpu_fast_cols") cfg.cpu_fast_cols = std::stoi(val);
    else if (key == "wide_tokens") cfg.wide_tokens = std::stoi(val);
    else if (key == "wide_min_tokens") cfg.wide_min_tokens = std::stoi(val);
    else if (key == "wide_slots") cfg.wide_slots = std::stoi(val);
    else if (key == "defer_last") cfg.defer_last = std::stoi(val);
    else if (key == "defer") cfg.defer_min_rank = std::stoi(val), cfg.defer_last = -2;
    else if (key == "defer_min_rank") cfg.defer_min_rank = std::stoi(val);
    else if (key == "defer_ssd") cfg.defer_ssd_rank = std::stoi(val);
    else if (key == "imm_heat") cfg.imm_heat = std::stoi(val);
    else if (key == "cpu_gu_chunk") cfg.cpu_gu_chunk = std::stoi(val);
    else if (key == "cpu_down_chunk") cfg.cpu_down_chunk = std::stoi(val);
    else if (key == "wide_a8") cfg.wide_a8 = std::stoi(val) != 0;
    else if (key == "host_reserve_gib") cfg.host_reserve_gib = std::stod(val);
    else if (key == "ssd_io") cfg.ssd_io_threads = std::stoi(val);
    else if (key == "ssd_direct") cfg.ssd_direct = std::stoi(val) != 0;
    else throw std::invalid_argument("unknown blaze option: " + key);
  }
}

HeatProfile load_heat(const std::string& path, int num_layers, int num_experts) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return {};
  char magic[8];
  uint32_t dims[2];
  const size_t n = size_t(num_layers) * num_experts;
  HeatProfile h;
  h.recent.resize(n);
  bool ok = std::fread(magic, 1, 8, f) == 8 &&
            (std::memcmp(magic, kHeatMagic, 8) == 0 || std::memcmp(magic, kHeatMagicV1, 8) == 0) &&
            std::fread(dims, 4, 2, f) == 2 && dims[0] == uint32_t(num_layers) && dims[1] == uint32_t(num_experts) &&
            std::fread(h.recent.data(), 8, n, f) == n;
  if (ok && std::memcmp(magic, kHeatMagic, 8) == 0) {
    h.longterm.resize(n);
    ok = std::fread(h.longterm.data(), 8, n, f) == n;
  } else {
    h.longterm = h.recent;
  }
  std::fclose(f);
  if (!ok) {
    log("blaze: ignoring routing profile %s (not a %d x %d profile)", path.c_str(), num_layers,
                 num_experts);
    return {};
  }
  return h;
}

void save_heat(const std::string& path, int num_layers, int num_experts, const HeatProfile& heat) {
  const std::string tmp = path + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) throw std::runtime_error("cannot write " + tmp);
  const uint32_t dims[2] = {uint32_t(num_layers), uint32_t(num_experts)};
  const size_t n = size_t(num_layers) * num_experts;
  const bool ok = heat.recent.size() == n && heat.longterm.size() == n && std::fwrite(kHeatMagic, 1, 8, f) == 8 &&
                  std::fwrite(dims, 4, 2, f) == 2 && std::fwrite(heat.recent.data(), 8, n, f) == n &&
                  std::fwrite(heat.longterm.data(), 8, n, f) == n;
  if (std::fclose(f) != 0 || !ok || std::rename(tmp.c_str(), path.c_str()) != 0)
    throw std::runtime_error("cannot write " + path);
}

void HybridMoe::save_profile() const {
  if (cfg_.swap.heat_path.empty()) return;
  try {
    save_heat(cfg_.swap.heat_path, num_layers(), num_experts(), {score_, score_long_});
  } catch (const std::exception& e) {
    log("blaze swap: %s", e.what());
  }
}

uint32_t HybridMoe::layer_calls() const {
  return blz_req_seq(__atomic_load_n(&reinterpret_cast<const blz_mail_header*>(mail_)->req, __ATOMIC_ACQUIRE));
}

uint8_t* HybridMoe::gpu_slot(Tier tier, int slot) const {
  return (tier == Tier::kGpu0 ? pool0_ : pool1_) + size_t(slot) * stride_;
}

// Both planning tables live on GPU0: one 4-byte copy per entry from its own staging word, then one sync.
void HybridMoe::write_gpu_tables(const std::vector<std::pair<int32_t*, int32_t>>& writes) {
  for (size_t i = 0; i < writes.size(); i += kTableStaging) {
    const size_t n = std::min(writes.size() - i, size_t(kTableStaging));
    for (size_t k = 0; k < n; ++k) {
      table_host_[k] = writes[i + k].second;
      check(cudaMemcpyAsync(writes[i + k].first, &table_host_[k], 4, cudaMemcpyHostToDevice, swap_stream_[0]),
            "table write");
    }
    check(cudaStreamSynchronize(swap_stream_[0]), "table write");
  }
}

void HybridMoe::recycle(bool force) {
  const uint32_t now = layer_calls();
  const int E = num_experts();
  size_t keep = 0;
  for (const PendingSlot& p : pending_) {
    if (!force && int32_t(now - p.ready) < 0) {
      pending_[keep++] = p;
      continue;
    }
    if (p.clear_record >= 0 && host_records_[size_t(p.clear_record)] == ram_slot(p.slot)) {
      host_records_[size_t(p.clear_record)] = nullptr;
      cpu_->set_record(int(p.clear_record / E), int(p.clear_record % E), nullptr);
    }
    free_[int(p.tier)].push_back(p.slot);
  }
  pending_.resize(keep);
}

// Hot RAM experts take the GPU slots of the coldest GPU experts, which move to RAM. All moves of a round run at
// once: per GPU, promotions on one stream (host -> device copy engine); the evicted experts get their RAM copies
// either on another stream (device -> host, demote=pcie) or from the SSD pool (demote=ssd, completed in later rounds;
// until then they keep serving from VRAM).
void HybridMoe::swap_gpu_ram(int max_moves) {
  if (capacity_[0] + capacity_[1] == 0 || capacity_[2] == 0) return;
  const int E = num_experts();
  const bool via_ssd = cfg_.swap.demote == "ssd";
  if (via_ssd) {
    const size_t max_inflight = size_t(std::max(1, 2 * cfg_.swap.max_moves));
    if (ssd_inflight_.size() >= max_inflight) return;
    max_moves = int(std::min<size_t>(size_t(max_moves), max_inflight - ssd_inflight_.size()));
  }
  std::vector<size_t> hot, cold;
  for (size_t i = 0; i < score_.size(); ++i) {
    if (loading_[i]) continue;
    const Tier t = placement_.tier[i];
    if (t == Tier::kRam) hot.push_back(i);
    else if (t == Tier::kGpu0 || t == Tier::kGpu1) cold.push_back(i);
  }
  select(hot, score_, size_t(max_moves), true);
  select(cold, score_, size_t(max_moves), false);
  struct Move {
    size_t a, b;
    Tier gt;
    int g_new, r_new, r_old, g_old;
  };
  std::vector<Move> moves;
  for (size_t k = 0; k < std::min(hot.size(), cold.size()); ++k) {
    const size_t a = hot[k], b = cold[k];
    if (!(score_[a] > score_[b] * (1.0 + cfg_.swap.hysteresis) + kMinGain)) break;
    const Tier gt = placement_.tier[b];
    std::vector<int>& gfree = free_[int(gt)];
    if (gfree.empty() || free_[2].empty()) break;
    moves.push_back({a, b, gt, gfree.back(), free_[2].back(), placement_.slot[a], placement_.slot[b]});
    gfree.pop_back(), free_[2].pop_back();
  }
  if (moves.empty()) return;
  for (int d = 0; d < 2; ++d) {
    const Tier gt = d == 0 ? Tier::kGpu0 : Tier::kGpu1;
    bool any = false;
    for (const Move& m : moves) any |= m.gt == gt;
    if (!any) continue;
    check(cudaSetDevice(d == 0 ? cfg_.gpu0 : cfg_.gpu1), "set device");
    for (const Move& m : moves) {
      if (m.gt != gt) continue;
      check(cudaMemcpyAsync(gpu_slot(gt, m.g_new), ram_slot(m.r_old), stride_, cudaMemcpyHostToDevice,
                            swap_stream_[d]),
            "promote");
      if (!via_ssd)
        check(cudaMemcpyAsync(ram_slot(m.r_new), gpu_slot(gt, m.g_old), stride_, cudaMemcpyDeviceToHost,
                              d2h_stream_[d]),
              "demote");
    }
  }
  for (int d = 0; d < 2; ++d) {
    if (!swap_stream_[d]) continue;
    check(cudaSetDevice(d == 0 ? cfg_.gpu0 : cfg_.gpu1), "set device");
    check(cudaStreamSynchronize(swap_stream_[d]), "promote");
    check(cudaStreamSynchronize(d2h_stream_[d]), "demote");
  }
  check(cudaSetDevice(cfg_.gpu0), "set gpu0");
  std::vector<std::pair<int32_t*, int32_t>> writes;
  std::vector<SsdLoad*> queued;
  for (const Move& m : moves) {
    int32_t* table = m.gt == Tier::kGpu0 ? slot0_dev_ : slot1_dev_;
    writes.push_back({table + m.a, m.g_new});
    if (via_ssd) {
      auto load = std::make_unique<SsdLoad>();
      load->expert = m.b, load->slot = m.r_new, load->demote = true, load->from_tier = m.gt, load->from_slot = m.g_old;
      loading_[m.b] = 1;
      queued.push_back(load.get());
      ssd_inflight_.push_back(std::move(load));
    } else {
      // The evicted expert's RAM record is valid before any plan can route it to the CPU.
      host_records_[m.b] = ram_slot(m.r_new);
      cpu_->set_record(int(m.b / E), int(m.b % E), host_records_[m.b]);
      writes.push_back({table + m.b, -1});
    }
  }
  write_gpu_tables(writes);
  const uint32_t ready = layer_calls() + kEpochCalls;
  for (const Move& m : moves) {
    placement_.tier[m.a] = m.gt, placement_.slot[m.a] = m.g_new;
    pending_.push_back({Tier::kRam, m.r_old, ready, int64_t(m.a)});
    if (!via_ssd) {
      placement_.tier[m.b] = Tier::kRam, placement_.slot[m.b] = m.r_new;
      pending_.push_back({m.gt, m.g_old, ready, -1});
    }
  }
  if (!queued.empty()) {
    {
      std::lock_guard<std::mutex> lock(ssd_mu_);
      for (SsdLoad* l : queued) ssd_queue_.push_back(l);
    }
    ssd_cv_.notify_all();
  }
  stat_moves_gpu_.fetch_add(uint64_t(moves.size()), std::memory_order_relaxed);
}

// Hot SSD experts are read into RAM by the SSD pool, asynchronously to the rounds; the coldest RAM experts fall back
// to SSD at once (the checkpoint always has them), their slots reused after the epoch.
void HybridMoe::swap_ram_ssd(int max_moves) {
  if (capacity_[2] == 0) return;
  const int E = num_experts();
  const size_t max_inflight = size_t(std::max(1, 2 * cfg_.swap.max_moves));
  if (ssd_inflight_.size() >= max_inflight) return;
  max_moves = int(std::min<size_t>(size_t(max_moves), max_inflight - ssd_inflight_.size()));
  std::vector<size_t> hot, cold;
  for (size_t i = 0; i < score_.size(); ++i) {
    const Tier t = placement_.tier[i];
    if (t == Tier::kSsd && !loading_[i]) hot.push_back(i);
    else if (t == Tier::kRam) cold.push_back(i);
  }
  select(hot, score_long_, size_t(max_moves), true);
  select(cold, score_long_, size_t(max_moves), false);
  std::vector<SsdLoad*> queued;
  std::vector<std::pair<int32_t*, int32_t>> writes;
  for (size_t k = 0; k < std::min(hot.size(), cold.size()); ++k) {
    const size_t a = hot[k], b = cold[k];
    if (!(score_long_[a] > score_long_[b] * (1.0 + cfg_.swap.hysteresis) + kMinGain)) break;
    if (free_[2].empty()) break;
    auto load = std::make_unique<SsdLoad>();
    load->expert = a;
    load->slot = free_[2].back();
    free_[2].pop_back();
    loading_[a] = 1;
    queued.push_back(load.get());
    ssd_inflight_.push_back(std::move(load));
    const int r_old = placement_.slot[b];
    host_records_[b] = nullptr;
    cpu_->set_record(int(b / E), int(b % E), nullptr);
    placement_.tier[b] = Tier::kSsd, placement_.slot[b] = -1;
    writes.push_back({slot0_dev_ + b, kSlotSsd});
    pending_.push_back({Tier::kRam, r_old, layer_calls() + kEpochCalls, -1});
  }
  if (queued.empty()) return;
  write_gpu_tables(writes);
  {
    std::lock_guard<std::mutex> lock(ssd_mu_);
    for (SsdLoad* l : queued) ssd_queue_.push_back(l);
  }
  ssd_cv_.notify_all();
}

// Publish the SSD reads that completed since the last round: promotions from SSD become RAM experts; experts evicted
// from VRAM switch to their new RAM copy, and their GPU slots are reused after the epoch.
void HybridMoe::finish_ssd_loads() {
  const int E = num_experts();
  size_t keep = 0;
  std::vector<std::pair<int32_t*, int32_t>> writes;
  std::vector<PendingSlot> freed;
  for (auto& up : ssd_inflight_) {
    SsdLoad& l = *up;
    const int state = l.state.load(std::memory_order_acquire);
    if (state == 0) {
      ssd_inflight_[keep++] = std::move(up);
      continue;
    }
    loading_[l.expert] = 0;
    if (state == 2) {
      log("blaze swap: %s", l.error.c_str());
      free_[2].push_back(l.slot);  // a failed eviction leaves the expert in VRAM
      continue;
    }
    host_records_[l.expert] = ram_slot(l.slot);
    cpu_->set_record(int(l.expert / E), int(l.expert % E), host_records_[l.expert]);
    placement_.tier[l.expert] = Tier::kRam, placement_.slot[l.expert] = l.slot;
    if (l.demote) {
      writes.push_back({(l.from_tier == Tier::kGpu0 ? slot0_dev_ : slot1_dev_) + l.expert, -1});
      freed.push_back({l.from_tier, l.from_slot, 0, -1});
    } else {
      writes.push_back({slot0_dev_ + l.expert, -1});  // SSD -> RAM: no longer marked as SSD tier
      stat_moves_ram_.fetch_add(1, std::memory_order_relaxed);
    }
  }
  ssd_inflight_.resize(keep);
  if (writes.empty()) return;
  write_gpu_tables(writes);
  const uint32_t ready = layer_calls() + kEpochCalls;
  for (PendingSlot& p : freed) {
    p.ready = ready;
    pending_.push_back(p);
  }
}

void HybridMoe::ssd_worker() {
  const int E = num_experts();
  for (;;) {
    SsdLoad* l;
    {
      std::unique_lock<std::mutex> lock(ssd_mu_);
      ssd_cv_.wait(lock, [&] { return ssd_stop_ || !ssd_queue_.empty(); });
      if (ssd_stop_) return;
      l = ssd_queue_.front();
      ssd_queue_.pop_front();
    }
    try {
      store_->read(int(l->expert / E), int(l->expert % E), ram_slot(l->slot), cfg_.ssd_direct);
      l->state.store(1, std::memory_order_release);
    } catch (const std::exception& e) {
      l->error = e.what();
      l->state.store(2, std::memory_order_release);
    }
  }
}

// One round: snapshot the route counters, decay and add to the scores, then move. False when nothing was routed.
bool HybridMoe::swap_round(std::vector<uint32_t>& delta) {
  const size_t n = score_.size();
  check(cudaMemcpyAsync(heat_host_, heat_dev_, n * 4, cudaMemcpyDeviceToHost, swap_stream_[0]), "heat snapshot");
  check(cudaStreamSynchronize(swap_stream_[0]), "heat snapshot");
  std::lock_guard<std::mutex> lock(swap_mu_);  // wide passes read placement and slots
  uint64_t routes = 0;
  const bool prefill = prefill_routes_.size() == n;
  for (size_t i = 0; i < n; ++i) {
    delta[i] = heat_host_[i] - heat_last_[i];
    heat_last_[i] = heat_host_[i];
    if (prefill && prefill_routes_[i]) {
      delta[i] += uint32_t(prefill_routes_[i] * cfg_.swap.prefill_weight + 0.5);
      prefill_routes_[i] = 0;
    }
    routes += delta[i];
  }
  recycle(false);
  finish_ssd_loads();
  if (routes == 0) return false;
  const double tokens = double(routes) / (double(num_layers()) * cfg_.topk);
  const double decay = std::exp2(-tokens / std::max(1.0, cfg_.swap.halflife_tokens));
  const double decay_long = std::exp2(-tokens / std::max(1.0, cfg_.swap.halflife_ssd_tokens));
  uint64_t per_tier[4] = {0, 0, 0, 0};
  for (size_t i = 0; i < n; ++i) {
    score_[i] = score_[i] * decay + delta[i];
    score_long_[i] = score_long_[i] * decay_long + delta[i];
    per_tier[int(placement_.tier[i])] += delta[i];
  }
  for (int t = 0; t < 4; ++t) stat_routes_[t].fetch_add(per_tier[t], std::memory_order_relaxed);
  // GPU promotions first: they remove CPU work, and SSD reads would otherwise take every spare RAM slot.
  swap_gpu_ram(cfg_.swap.max_moves);
  swap_ram_ssd(cfg_.swap.max_moves);
  return true;
}

void HybridMoe::swap_loop() {
  using clock = std::chrono::steady_clock;
  std::vector<uint32_t> delta(score_.size());
  auto next_save = clock::now() + std::chrono::seconds(cfg_.swap.save_period_s);
  auto last_log = clock::now();
  HybridStats logged = stats();
  uint64_t logged_cpu[3] = {cpu_->served(), cpu_->items(), cpu_->busy_ns()};
  bool unsaved = false;
  try {
    // The host's model captures CUDA graphs in global mode; this thread's copies must not invalidate them.
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    check(cudaThreadExchangeStreamCaptureMode(&mode), "capture mode");
    check(cudaSetDevice(cfg_.gpu0), "set gpu0");
    while (!swap_stop_.load(std::memory_order_acquire)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.swap.period_ms));
      unsaved |= swap_round(delta);
      const auto now = clock::now();
      if (unsaved && now >= next_save) {
        save_profile();
        unsaved = false;
        next_save = now + std::chrono::seconds(cfg_.swap.save_period_s);
      }
      if (cfg_.swap.log_period_s > 0 && now - last_log >= std::chrono::seconds(cfg_.swap.log_period_s)) {
        const HybridStats st = stats();
        uint64_t d[4], total = 0;
        for (int t = 0; t < 4; ++t) total += d[t] = st.routes[t] - logged.routes[t];
        if (total > 0) {
          const double sec = std::chrono::duration<double>(now - last_log).count();
          const uint64_t jobs = cpu_->served() - logged_cpu[0], items = cpu_->items() - logged_cpu[1];
          const uint64_t busy = cpu_->busy_ns() - logged_cpu[2];
          const double vram = 100.0 * (d[0] + d[1]) / total;
          char ssd[48] = "";
          if (st.resident[3] > 0) std::snprintf(ssd, sizeof(ssd), ", SSD %.2f%%", 100.0 * d[3] / total);
          log("blaze experts: %.0f tok in %.0f s, VRAM hits %.1f%% (GPU0 %.1f%%, GPU1 %.1f%%), CPU %.1f%%%s "
              "(%.1f experts/layer, %.0f us), swaps %llu",
              double(total) / (double(num_layers()) * cfg_.topk), sec, vram, 100.0 * d[0] / total,
              100.0 * d[1] / total, 100.0 * (d[2] + d[3]) / total, ssd, jobs ? double(items) / jobs : 0.0,
              jobs ? busy / 1e3 / jobs : 0.0,
              (unsigned long long)(st.moves_gpu - logged.moves_gpu + st.moves_ram - logged.moves_ram));
        }
        logged = st;
        logged_cpu[0] = cpu_->served(), logged_cpu[1] = cpu_->items(), logged_cpu[2] = cpu_->busy_ns();
        last_log = now;
      }
    }
  } catch (const std::exception& e) {
    log("blaze swap: %s; swap engine stopped", e.what());
  }
}

}  // namespace blaze
