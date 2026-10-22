// SSD-tier cost of one expert on a real checkpoint: ExpertStore reads (checkpoint bytes + rg16 packing) from a cold
// page cache, buffered and O_DIRECT, whole records and chunks spread over a pool of reader threads (as the CPU
// engine reads SSD-tier experts), plus a warm pass (packing only). Checks the packed records against
// expert::pack_record of the raw tensors first.
//   bench-store MODEL_DIR [N=64] [THREADS=8] [CHUNK_KB=192]
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

#include "blaze/safetensors.h"
#include "blaze/store.h"

using Clock = std::chrono::steady_clock;
using namespace blaze;

static void drop_page_cache(const std::string& dir) {
  for (const auto& e : std::filesystem::directory_iterator(dir)) {
    if (e.path().extension() != ".safetensors") continue;
    const int fd = open(e.path().c_str(), O_RDONLY);
    if (fd < 0) continue;
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
  }
}

// expert::pack_record of the raw tensors (the scalar reference packing).
static std::vector<uint8_t> reference(const std::string& dir, const ExpertStore& store, int l, int e) {
  SafetensorsDir st(dir);
  const auto geo = store.geometry();
  std::vector<uint8_t> raw[6];
  const char* proj[3] = {"gate", "up", "down"};
  for (int m = 0; m < 3; ++m) {
    const std::string base = store.prefix() + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj[m] +
                             "_proj.";
    const auto* codes = st.find(base + "weight_packed");
    if (!codes) codes = st.find(base + "weight");
    const auto& scales = st.at(base + "weight_scale");
    raw[2 * m].resize(codes->bytes()), raw[2 * m + 1].resize(scales.bytes());
    st.read(*codes, raw[2 * m].data(), codes->bytes());
    st.read(scales, raw[2 * m + 1].data(), scales.bytes());
  }
  std::vector<uint8_t> rec(store.record_stride(), 0);
  expert::pack_record(geo, raw[0].data(), raw[1].data(), raw[2].data(), raw[3].data(), raw[4].data(), raw[5].data(),
                      rec.data());
  return rec;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s MODEL_DIR [N] [THREADS] [CHUNK_KB]\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const int n = argc > 2 ? std::atoi(argv[2]) : 64;
  const int threads = argc > 3 ? std::atoi(argv[3]) : 8;
  const size_t chunk_kb = argc > 4 ? size_t(std::atoi(argv[4])) : 192;
  ExpertStore store(dir, "", chunk_kb << 10);
  const auto geo = store.geometry();
  std::printf("%d layers x %d experts, H=%d I=%d, record %.2f MB, %d chunks\n", store.num_layers(),
              store.num_experts(), geo.hidden, geo.inter, geo.record_bytes() / 1e6, store.num_chunks());
  std::mt19937 rng(1);
  std::vector<std::pair<int, int>> picks(n);
  for (auto& p : picks) p = {int(rng() % store.num_layers()), int(rng() % store.num_experts())};
  uint8_t* dst = static_cast<uint8_t*>(std::aligned_alloc(4096, store.record_stride()));

  int bad = 0;
  for (int i = 0; i < 3; ++i) {
    const auto [l, e] = picks[i];
    const auto want = reference(dir, store, l, e);
    for (int direct = 0; direct < 2; ++direct) {
      std::memset(dst, 0xA5, store.record_stride());
      store.read(l, e, dst, direct);
      bad += std::memcmp(dst, want.data(), geo.record_bytes()) != 0;
    }
  }
  std::printf("packing vs pack_record: %s\n", bad ? "MISMATCH" : "identical");

  auto serial = [&](const char* what, bool direct, bool cold) {
    if (cold) drop_page_cache(dir);
    const auto t0 = Clock::now();
    for (const auto& [l, e] : picks) store.read(l, e, dst, direct);
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / n;
    std::printf("%-34s %.3f ms per expert (%.2f GB/s)\n", what, ms, geo.record_bytes() / ms / 1e6);
  };
  // One expert at a time, its chunks spread over `threads` readers (latency of a single SSD-tier miss).
  auto parallel = [&](const char* what, bool direct) {
    drop_page_cache(dir);
    std::atomic<int> next{0}, done{0};
    std::atomic<int> gen{-1};
    std::atomic<bool> stop{false};
    const int C = store.num_chunks();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&] {
        int seen = -1;
        while (!stop.load()) {
          const int g = gen.load(std::memory_order_acquire);
          if (g == seen) {
            std::this_thread::yield();
            continue;
          }
          for (int c; (c = next.fetch_add(1)) < C;) {
            store.read_chunk(picks[g].first, picks[g].second, c, dst, direct);
            done.fetch_add(1, std::memory_order_release);
          }
          seen = g;
        }
      });
    double total = 0;
    for (int i = 0; i < n; ++i) {
      done.store(0), next.store(0);
      const auto t0 = Clock::now();
      gen.store(i, std::memory_order_release);
      while (done.load(std::memory_order_acquire) < C) std::this_thread::yield();
      total += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }
    stop.store(true);
    for (auto& t : pool) t.join();
    std::printf("%-34s %.3f ms per expert (%.2f GB/s)\n", what, total / n, geo.record_bytes() / (total / n) / 1e6);
  };
  serial("cold, serial, buffered", false, true);
  serial("warm, serial (packing only)", false, false);
  serial("cold, serial, O_DIRECT", true, true);
  parallel("cold, chunks on readers, buffered", false);
  parallel("cold, chunks on readers, O_DIRECT", true);
  std::free(dst);
  return bad ? 1 : 0;
}
