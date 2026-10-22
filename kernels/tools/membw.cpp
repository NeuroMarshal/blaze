// membw: host memory read bandwidth as the CPU expert engine sees it.
//
//   membw [--gib 8] [--cpus 3-4,6,9-17] [--huge 0|1] [--mode seq|experts] [--seconds 3]
//
// seq: every thread streams its own slice. experts: every thread reads whole 2.76 MB records picked at random
// (the RAM tier's access pattern). --huge 1 backs the buffer with transparent huge pages (madvise).
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "blaze/cpu_engine.h"

int main(int argc, char** argv) {
  double gib = 8, seconds = 3;
  std::string cpus = "3-4,6,9-17,19-23", mode = "seq";
  bool huge = false;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string a = argv[i], v = argv[i + 1];
    if (a == "--gib") gib = std::stod(v);
    else if (a == "--cpus") cpus = v;
    else if (a == "--huge") huge = v == "1";
    else if (a == "--mode") mode = v;
    else if (a == "--seconds") seconds = std::stod(v);
  }
  const std::vector<int> cpu = blaze::cpu::parse_cpu_list(cpus);
  const size_t bytes = size_t(gib * double(1ull << 30)) & ~size_t((2 << 20) - 1);
  auto* buf = static_cast<uint8_t*>(mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  if (buf == MAP_FAILED) return 1;
  if (huge) madvise(buf, bytes, MADV_HUGEPAGE);
  const int T = int(cpu.size());
  const size_t slice = bytes / T / 64 * 64;
  constexpr size_t kRecord = 2764800;
  const size_t records = bytes / kRecord;
  std::atomic<int> ready{0};
  std::atomic<bool> go{false}, stop{false};
  std::vector<uint64_t> read(T, 0);
  std::vector<std::thread> th;
  for (int t = 0; t < T; ++t)
    th.emplace_back([&, t] {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu[t], &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      std::memset(buf + slice * t, 1, slice);  // first touch from the reading thread
      ++ready;
      while (!go) std::this_thread::yield();
      std::mt19937_64 rng(t);
      __m256i acc = _mm256_setzero_si256();
      uint64_t n = 0;
      while (!stop) {
        const uint8_t* p;
        size_t len;
        if (mode == "seq") {
          p = buf + slice * t, len = slice;
        } else {
          p = buf + (rng() % records) * kRecord, len = kRecord;
        }
        for (size_t o = 0; o + 128 <= len; o += 128) {
          const __m256i* q = reinterpret_cast<const __m256i*>(p + o);
          acc = _mm256_add_epi64(acc, _mm256_loadu_si256(q));
          acc = _mm256_add_epi64(acc, _mm256_loadu_si256(q + 1));
          acc = _mm256_add_epi64(acc, _mm256_loadu_si256(q + 2));
          acc = _mm256_add_epi64(acc, _mm256_loadu_si256(q + 3));
        }
        n += len / 128 * 128;
      }
      read[t] = n + uint64_t(_mm256_extract_epi64(acc, 0) & 1);
    });
  while (ready < T) std::this_thread::sleep_for(std::chrono::milliseconds(10));
  const auto t0 = std::chrono::steady_clock::now();
  go = true;
  std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
  stop = true;
  for (auto& x : th) x.join();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  uint64_t total = 0;
  for (uint64_t v : read) total += v;
  std::printf("%-7s huge=%d threads=%2d cpus=%s: %.1f GB/s\n", mode.c_str(), int(huge), T, cpus.c_str(), total / s / 1e9);
  munmap(buf, bytes);
  return 0;
}
