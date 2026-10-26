// ExpertStore on synthetic NVFP4 checkpoints (compressed-tensors and ModelOpt naming, 2 shards, 2 layers x 3 experts,
// small geometry): every record read equals pack_record of the source codes / scales (whole, and chunk by chunk in
// reverse order with O_DIRECT where the file system has it), the multipliers match the exporter's convention, the
// layer prefix is found and an MTP block's experts are ignored.
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "blaze/expert.h"
#include "blaze/hybrid.h"
#include "blaze/log.h"
#include "blaze/ple.h"
#include "blaze/store.h"
#include "synthetic_checkpoint.h"

namespace fs = std::filesystem;
using namespace blaze;

static std::string callback_message;

extern "C" void test_log_callback(const char* message) { callback_message = message; }

bool check_log_callback() {
  blaze_set_log_callback(test_log_callback);
  blaze::log("formatted CPU log %s=%d", "count", 17);
  blaze_set_log_callback(nullptr);
  if (callback_message != "formatted CPU log count=17") {
    std::printf("C++ log callback received '%s'\n", callback_message.c_str());
    return false;
  }
  return true;
}

bool check_option_validation() {
  int bad = 0;
  auto rejects = [&](const char* name, auto parse) {
    try {
      parse();
      std::printf("option validation accepted %s\n", name);
      ++bad;
    } catch (const std::invalid_argument&) {
    }
  };

  rejects("spare=-1", [] { (void)parse_swap_options("spare=-1"); });
  rejects("max_moves=-1", [] { (void)parse_swap_options("max_moves=-1"); });
  rejects("max_moves overflow", [] { (void)parse_swap_options("max_moves=1073741824"); });
  rejects("period_ms=0", [] { (void)parse_swap_options("period_ms=0"); });
  rejects("halflife=nan", [] { (void)parse_swap_options("halflife=nan"); });
  rejects("cpu_gu_chunk=0", [] {
    HybridConfig cfg;
    apply_hybrid_options("cpu_gu_chunk=0", cfg);
  });
  rejects("cpu_down_chunk=-1", [] {
    HybridConfig cfg;
    apply_hybrid_options("cpu_down_chunk=-1", cfg);
  });
  rejects("host_reserve_gib=inf", [] {
    HybridConfig cfg;
    apply_hybrid_options("host_reserve_gib=inf", cfg);
  });
  rejects("ssd_io=0", [] {
    HybridConfig cfg;
    apply_hybrid_options("ssd_io=0", cfg);
  });

  const SwapConfig swap = parse_swap_options("spare=0,max_moves=1,period_ms=1");
  HybridConfig cfg;
  apply_hybrid_options("cpu_gu_chunk=1,cpu_down_chunk=1,host_reserve_gib=0,ssd_io=1", cfg);
  if (swap.spare_slots != 0 || swap.max_moves != 1 || swap.period_ms != 1 || cfg.cpu_gu_chunk != 1 ||
      cfg.cpu_down_chunk != 1 || cfg.host_reserve_gib != 0 || cfg.ssd_io_threads != 1) {
    std::printf("option validation changed valid boundary values\n");
    ++bad;
  }
  return bad == 0;
}

bool check_ple_warm_reader_lifecycle(bool start_reader_after_stop, bool stop_during_read) {
  std::atomic<uint32_t> generation{0};
  std::atomic<int> busy{0};
  std::atomic<bool> stop{false}, published{false}, in_read{false}, release_read{false};
  std::atomic<bool> dispatcher_done{false}, reader_drained{false};
  uint32_t initial_seen = generation.load(std::memory_order_acquire);

  std::thread dispatcher([&] {
    detail::publish_warm_job(busy, generation, 1);
    published.store(true, std::memory_order_release);
    published.notify_one();
    detail::wait_for_warm_readers(busy);
    dispatcher_done.store(true, std::memory_order_release);
    dispatcher_done.notify_one();
  });

  while (!published.load(std::memory_order_acquire)) published.wait(false, std::memory_order_acquire);
  if (start_reader_after_stop) stop.store(true, std::memory_order_release);

  std::thread reader([&] {
    while (detail::wait_for_warm_job(generation, initial_seen, stop, busy)) {
      in_read.store(true, std::memory_order_release);
      in_read.notify_one();
      if (stop_during_read) {
        while (!release_read.load(std::memory_order_acquire)) release_read.wait(false, std::memory_order_acquire);
      }
      detail::warm_reader_finished(busy);
    }
    reader_drained.store(true, std::memory_order_release);
    reader_drained.notify_one();
  });

  while (!in_read.load(std::memory_order_acquire)) in_read.wait(false, std::memory_order_acquire);
  if (stop_during_read) {
    stop.store(true, std::memory_order_release);
    const bool dispatcher_waited_for_reader = !dispatcher_done.load(std::memory_order_acquire);
    release_read.store(true, std::memory_order_release);
    release_read.notify_one();
    if (!dispatcher_waited_for_reader) {
      reader.join();
      dispatcher.join();
      return false;
    }
  }

  while (!dispatcher_done.load(std::memory_order_acquire)) dispatcher_done.wait(false, std::memory_order_acquire);
  // Match PleService destruction: the idle termination wake follows the dispatcher join/drain.
  dispatcher.join();
  generation.fetch_add(1, std::memory_order_release);
  generation.notify_all();
  reader.join();
  return reader_drained.load(std::memory_order_acquire) && busy.load(std::memory_order_acquire) == 0;
}

bool check_ple_warm_reader_lifecycle() {
  // A delayed reader must consume the first published generation even though stop was already requested.
  if (!check_ple_warm_reader_lifecycle(true, false)) return false;
  // Shutdown during an active read must keep the dispatcher blocked until the reader decrements busy.
  return check_ple_warm_reader_lifecycle(false, true);
}

// One synthetic checkpoint in either exporter's naming; returns the number of wrong experts.
int check(bool modelopt, const std::string& prefix) {
  const expert::Geometry geo{96, 64};
  const int L = 2, E = 3;
  const fs::path dir = fs::temp_directory_path() / "blz_test_store";
  fs::remove_all(dir);
  fs::create_directories(dir / "model");
  std::mt19937 rng(5);
  struct Expert {
    std::vector<uint8_t> c[3], s[3];
    float g[3];
  };
  std::vector<Expert> ex(size_t(L) * E);
  std::vector<Out> shard[2];
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) {
      Expert& x = ex[size_t(l) * E + e];
      const char* proj[3] = {"gate", "up", "down"};
      for (int m = 0; m < 3; ++m) {
        const int rows = m == 2 ? geo.hidden : geo.inter, k = m == 2 ? geo.inter : geo.hidden;
        x.c[m].resize(size_t(rows) * k / 2), x.s[m].resize(size_t(rows) * k / 16);
        for (auto& v : x.c[m]) v = uint8_t(rng());
        for (auto& v : x.s[m]) v = uint8_t(0x20 + rng() % 0x40);
        x.g[m] = 100.0f + float(rng() % 1000);
        const std::string base =
            prefix + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj[m] + "_proj.";
        std::vector<uint8_t> gb(4);
        std::memcpy(gb.data(), &x.g[m], 4);
        auto& sh = shard[(l + e) % 2];  // tensors of one expert spread over both shards
        sh.push_back({base + (modelopt ? "weight" : "weight_packed"), "U8", {rows, k / 2}, x.c[m]});
        sh.push_back({base + "weight_scale", "F8_E4M3", {rows, k / 16}, x.s[m]});
        sh.push_back({base + (modelopt ? "weight_scale_2" : "weight_global_scale"), "F32", {1}, gb});
      }
    }
  // An MTP block's experts must not be taken for the decoder's.
  shard[0].push_back({"mtp.layers.0.mlp.experts.0.gate_proj.weight_packed", "U8", {geo.inter, geo.hidden / 2},
                      std::vector<uint8_t>(size_t(geo.inter) * geo.hidden / 2)});
  write_safetensors(dir / "model" / "model-00001-of-00002.safetensors", shard[0]);
  write_safetensors(dir / "model" / "model-00002-of-00002.safetensors", shard[1]);
  int bad = 0;
  // 1200 bytes per chunk: one gate/up row group (16 * (48 + 6) bytes) or two down row groups (16 * (32 + 4)) each.
  ExpertStore store((dir / "model").string(), "", 1200);
  if (store.num_chunks() != 8 + 3) {
    std::printf("unexpected chunking: %d chunks\n", store.num_chunks());
    ++bad;
  }
  if (store.num_layers() != L || store.num_experts() != E || store.geometry().hidden != geo.hidden ||
      store.geometry().inter != geo.inter || store.prefix() != prefix) {
    std::printf("geometry or prefix mismatch\n");
    ++bad;
  }
  std::vector<uint8_t> want(store.record_stride()), got(store.record_stride());
  for (size_t i = 0; i < ex.size(); ++i) {
    const Expert& x = ex[i];
    const int l = int(i / E), e = int(i % E);
    std::fill(want.begin(), want.end(), 0);
    expert::pack_record(geo, x.c[0].data(), x.s[0].data(), x.c[1].data(), x.s[1].data(), x.c[2].data(), x.s[2].data(),
                        want.data());
    store.read(l, e, got.data());
    bool rec_ok = std::memcmp(want.data(), got.data(), want.size()) == 0;
    std::fill(got.begin(), got.end(), 0);
    for (int c = store.num_chunks() - 1; c >= 0; --c) store.read_chunk(l, e, c, got.data(), true);
    rec_ok &= std::memcmp(want.data(), got.data(), want.size()) == 0;
    const float* m = store.mult(l, e);
    bool mult_ok = true;
    for (int k = 0; k < 3; ++k) mult_ok &= m[k] == (modelopt ? x.g[k] : 1.0f / x.g[k]);
    if (!rec_ok || !mult_ok) {
      std::printf("expert %zu: record %d mult %d\n", i, rec_ok, mult_ok);
      ++bad;
    }
  }
  fs::remove_all(dir);
  std::printf("%s, prefix %s: %d / %zu experts wrong\n", modelopt ? "ModelOpt" : "compressed-tensors", prefix.c_str(),
              bad, ex.size());
  return bad;
}

int main() {
  const int bad = (check_log_callback() ? 0 : 1) + (check_option_validation() ? 0 : 1) +
                  (check_ple_warm_reader_lifecycle() ? 0 : 1) +
                  check(false, "model.language_model.layers.") +
                  check(true, "model.layers.");
  std::printf(bad ? "FAILED\n" : "PASSED\n");
  return bad ? 1 : 0;
}
