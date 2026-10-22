// CPU expert kernels: AVX2 == scalar == forward_ref bit for bit; bandwidth against a STREAM-like read.
#include <algorithm>
#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "blaze/cpu_engine.h"
#include "blaze/cpu_expert.h"

using namespace blaze;
static int g_fail = 0;

static std::vector<uint8_t> random_record(const expert::Geometry& geo, std::mt19937& rng) {
  auto mat = [&](int rows, int k, std::vector<uint8_t>& c, std::vector<uint8_t>& s) {
    c.resize(size_t(rows) * k / 2), s.resize(size_t(rows) * k / 16);
    for (auto& v : c) v = uint8_t(rng());
    for (auto& v : s) v = uint8_t(rng() % 0x7F);  // every finite E4M3 code incl. subnormals
  };
  std::vector<uint8_t> gc, gs, uc, us, dc, ds;
  mat(geo.inter, geo.hidden, gc, gs), mat(geo.inter, geo.hidden, uc, us), mat(geo.hidden, geo.inter, dc, ds);
  std::vector<uint8_t> rec(geo.record_bytes());
  expert::pack_record(geo, gc.data(), gs.data(), uc.data(), us.data(), dc.data(), ds.data(), rec.data());
  return rec;
}

static std::vector<uint16_t> random_column(int n, std::mt19937& rng) {
  // Values over ~25 binades, some exact zeros.
  std::vector<uint16_t> v(n);
  std::uniform_int_distribution<int> e(100, 135);
  for (auto& b : v) b = (rng() % 17 == 0) ? 0 : uint16_t(((rng() & 1) << 15) | (e(rng) << 7) | (rng() & 0x7F));
  return v;
}

static void test_exact(const expert::Geometry& geo) {
  std::mt19937 rng(31);
  const auto rec = random_record(geo, rng);
  const expert::Multipliers m{0.011f, 0.017f, 0.0093f};
  cpu::Scratch scratch(geo);
  int bad_avx = 0, bad_scalar = 0, total = 0;
  for (int ncols = 1; ncols <= 8; ++ncols) {
    std::vector<std::vector<uint16_t>> xs, ya(ncols, std::vector<uint16_t>(geo.hidden)), ys = ya;
    for (int c = 0; c < ncols; ++c) xs.push_back(random_column(geo.hidden, rng));
    std::vector<const uint16_t*> xp;
    std::vector<uint16_t*> yap, ysp;
    for (int c = 0; c < ncols; ++c) xp.push_back(xs[c].data()), yap.push_back(ya[c].data()), ysp.push_back(ys[c].data());
    cpu::expert_forward(rec.data(), m, xp.data(), ncols, yap.data(), scratch, true);
    cpu::expert_forward(rec.data(), m, xp.data(), ncols, ysp.data(), scratch, false);
    std::vector<uint16_t> ref(geo.hidden);
    for (int c = 0; c < ncols; ++c) {
      expert::forward_ref(geo, rec.data(), m, xs[c].data(), ref.data());
      for (int r = 0; r < geo.hidden; ++r) {
        bad_avx += ya[c][r] != ref[r];
        bad_scalar += ys[c][r] != ref[r];
        ++total;
      }
    }
  }
  std::printf("cpu expert H=%d I=%d: avx2 %d, scalar %d of %d outputs differ from forward_ref\n", geo.hidden,
              geo.inter, bad_avx, bad_scalar, total);
  if (bad_avx || bad_scalar) ++g_fail;
}

// Read bandwidth of `threads` threads streaming disjoint parts of a large buffer (STREAM-like, read only).
static double read_bandwidth(const std::vector<uint8_t>& buf, int threads) {
  std::atomic<uint64_t> sink{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  const size_t part = buf.size() / threads / 64 * 64;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      const uint64_t* p = reinterpret_cast<const uint64_t*>(buf.data() + part * t);
      uint64_t acc = 0;
      for (size_t i = 0; i < part / 8; i += 8) acc += p[i] ^ p[i + 1] ^ p[i + 2] ^ p[i + 3] ^ p[i + 4] ^ p[i + 5] ^ p[i + 6] ^ p[i + 7];
      sink += acc;
    });
  }
  for (auto& th : pool) th.join();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return double(part) * threads / s / 1e9;
}

static void bench(int threads, int ncols, const std::vector<int>& cpus = {}) {
  const expert::Geometry geo = expert::kFlashNext;
  std::mt19937 rng(41);
  const int n_experts = 384;  // 1 GB: well beyond the caches
  std::vector<uint8_t> bank(geo.record_bytes() * n_experts);
  {
    const auto rec = random_record(geo, rng);
    for (int e = 0; e < n_experts; ++e) std::memcpy(bank.data() + geo.record_bytes() * e, rec.data(), rec.size());
  }
  const expert::Multipliers m{0.011f, 0.017f, 0.0093f};
  std::vector<std::vector<uint16_t>> xs;
  for (int c = 0; c < ncols; ++c) xs.push_back(random_column(geo.hidden, rng));
  // Each thread computes whole experts (the engine will split experts across threads by row groups instead).
  std::atomic<int> next{0};
  auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&, t] {
      if (!cpus.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[size_t(t) % cpus.size()], &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
      }
      cpu::Scratch scratch(geo);
      std::vector<std::vector<uint16_t>> ys(ncols, std::vector<uint16_t>(geo.hidden));
      std::vector<const uint16_t*> xp;
      std::vector<uint16_t*> yp;
      for (int c = 0; c < ncols; ++c) xp.push_back(xs[c].data()), yp.push_back(ys[c].data());
      for (int e; (e = next++) < n_experts;)
        cpu::expert_forward(bank.data() + geo.record_bytes() * e, m, xp.data(), ncols, yp.data(), scratch, true);
    });
  }
  for (auto& th : pool) th.join();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double gbs = double(bank.size()) / s / 1e9;
  const double stream = read_bandwidth(bank, threads);
  std::printf("bench cpu experts: %2d threads, %d col(s): %6.1f GB/s of records (%.0f us/expert/thread); read-only "
              "stream %.1f GB/s -> %.0f%%\n",
              threads, ncols, gbs, 1e6 * s * threads / n_experts, stream, 100 * gbs / stream);
}

// Approximate Q8 row sums vs the exact ones: error relative to the largest sum, and AVX2 time for 4 columns.
static void test_q8(int ncols) {
  const expert::Geometry geo = expert::kFlashNext;
  std::mt19937 rng(77);
  const auto rec = random_record(geo, rng);
  std::vector<std::vector<uint16_t>> xs;
  std::vector<cpu::ColumnBuffer> bufs(ncols);
  std::vector<std::vector<uint8_t>> mem(ncols, std::vector<uint8_t>(4 * size_t(geo.hidden) + 64));
  std::vector<std::vector<int8_t>> q(ncols, std::vector<int8_t>(geo.hidden));
  std::vector<std::vector<uint8_t>> sh(ncols, std::vector<uint8_t>(geo.hidden / 16));
  std::vector<cpu::Column> cols(ncols);
  std::vector<cpu::QColumn> qcols(ncols);
  for (int c = 0; c < ncols; ++c) {
    xs.push_back(random_column(geo.hidden, rng));
    uint8_t* p = mem[c].data();
    bufs[c].lo = p, bufs[c].mid = p + geo.hidden, bufs[c].hi = reinterpret_cast<int8_t*>(p + 2 * geo.hidden);
    bufs[c].hi_sum = reinterpret_cast<int32_t*>(p + 3 * geo.hidden);
    cpu::encode_column(xs[c].data(), geo.hidden, bufs[c]);
    cols[c] = bufs[c].view();
    cpu::quantize_column_q8(cols[c], geo.hidden, q[c].data(), sh[c].data());
    qcols[c] = {q[c].data(), sh[c].data()};
  }
  const int G = geo.gu_groups();
  std::vector<int64_t> exact(size_t(G) * 16 * ncols), approx(exact.size());
  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < 20; ++r) cpu::row_sums_avx2(rec.data(), geo.gu_blocks(), 0, G, cols.data(), ncols, exact.data());
  auto t1 = std::chrono::steady_clock::now();
  for (int r = 0; r < 20; ++r) cpu::row_sums_q8_avx2(rec.data(), geo.gu_blocks(), 0, G, qcols.data(), ncols, approx.data());
  auto t2 = std::chrono::steady_clock::now();
  double emax = 0, amax = 0;
  for (size_t i = 0; i < exact.size(); ++i) {
    emax = std::max(emax, std::fabs(double(approx[i] - exact[i])));
    amax = std::max(amax, std::fabs(double(exact[i])));
  }
  const double te = std::chrono::duration<double>(t1 - t0).count(), tq = std::chrono::duration<double>(t2 - t1).count();
  std::printf("q8 row sums, %d col(s): max error %.2e of the largest sum, %.2fx faster than exact (1 thread, cached)\n",
              ncols, emax / amax, te / tq);
  if (emax / amax > 2e-2) ++g_fail;
}

int main(int argc, char** argv) {
  test_exact({256, 64});
  test_exact(expert::kFlashNext);
  for (int c : {1, 4, 8}) test_q8(c);
  if (argc > 1) {  // test_cpu_expert CPU_LIST: pinned workers, 1..8 columns
    const std::vector<int> cpus = cpu::parse_cpu_list(argv[1]);
    for (int c : {1, 2, 4, 8}) bench(int(cpus.size()), c, cpus);
  } else {
    const int hw = int(std::thread::hardware_concurrency());
    for (int t : {1, 4, std::max(1, hw / 2), hw}) bench(t, 1);
    bench(std::max(1, hw / 2), 2);
    bench(std::max(1, hw / 2), 4);
  }
  std::printf(g_fail ? "FAILED\n" : "PASSED\n");
  return g_fail ? 1 : 0;
}
