// blz_host_ar through the shared host buffer, checked bitwise against the sum of the inputs (bf16 / fp32, decode to MB
// sizes, zero-copy and DMA paths, mixed):
//  - a GPU rank and a CPU thread playing the other ranks (1 to 7 of them) by the buffer protocol (GPU -> RAM and
//    RAM -> GPU visibility; runs on one GPU): sequential and back-to-back calls, replays of captured graphs,
//    all-reduce + RMSNorm;
//  - with two or more GPUs, two forked processes on GPUs 0 and 1: results, replays of a captured CUDA graph, the
//    time of back-to-back reductions, and zero-copy against DMA by size.
// Two processes on one GPU are not a valid setup: each registers the shared pages separately, and the GPU caches the
// two mappings as unrelated lines.
//   test_host_allreduce [iterations]
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <vector>

#include "blaze/host_allreduce.h"

namespace {

constexpr int kWorld = 2;
constexpr int64_t kMaxBytes = 6 << 20;  // above the 4 MiB DMA chunk: chunked calls

void check(cudaError_t e, const char* w, int rank) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "rank %d: %s: %s\n", rank, w, cudaGetErrorString(e));
    std::exit(1);
  }
}

// Inputs of rank r for case c: what every rank can recompute for the reference.
std::vector<float> inputs(int r, int c, size_t n) {
  std::mt19937 rng(1000u * unsigned(c) + unsigned(r));
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

uint16_t to_bf16(float f) { return __bfloat16_as_ushort(__float2bfloat16_rn(f)); }
float from_bf16(uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }

std::vector<uint8_t> to_bytes(const std::vector<float>& v, bool bf) {
  std::vector<uint8_t> b(v.size() * (bf ? 2 : 4));
  for (size_t i = 0; i < v.size(); ++i) {
    if (bf)
      reinterpret_cast<uint16_t*>(b.data())[i] = to_bf16(v[i]);
    else
      reinterpret_cast<float*>(b.data())[i] = v[i];
  }
  return b;
}

float value(const uint8_t* p, size_t i, bool bf) {
  return bf ? from_bf16(reinterpret_cast<const uint16_t*>(p)[i]) : reinterpret_cast<const float*>(p)[i];
}

// The sum of the tensors t[0 .. world) in rank order, as the library does it: fp32 accumulation, one rounding.
void reference_sum(const std::vector<const uint8_t*>& t, int64_t bytes, bool bf, uint8_t* out) {
  const size_t n = size_t(bytes) / (bf ? 2 : 4);
  for (size_t i = 0; i < n; ++i) {
    float acc = value(t[0], i, bf);
    for (size_t r = 1; r < t.size(); ++r) acc += value(t[r], i, bf);
    if (bf)
      reinterpret_cast<uint16_t*>(out)[i] = to_bf16(acc);
    else
      reinterpret_cast<float*>(out)[i] = acc;
  }
}

// One case: reduce bytes of dtype; returns the number of mismatching elements.
size_t run_case(blz_host_ar* ar, int rank, int c, int64_t bytes, blz_ar_dtype_t dt, cudaStream_t s) {
  const bool bf = dt == BLZ_AR_BF16;
  const size_t n = size_t(bytes) / (bf ? 2 : 4);
  std::vector<std::vector<float>> in(kWorld);
  for (int r = 0; r < kWorld; ++r) in[r] = inputs(r, c, n);
  std::vector<uint8_t> host(bytes), back(bytes);
  for (size_t i = 0; i < n; ++i) {
    if (bf)
      reinterpret_cast<uint16_t*>(host.data())[i] = to_bf16(in[rank][i]);
    else
      reinterpret_cast<float*>(host.data())[i] = in[rank][i];
  }
  void* d = nullptr;
  check(cudaMalloc(&d, bytes), "malloc", rank);
  check(cudaMemcpyAsync(d, host.data(), bytes, cudaMemcpyHostToDevice, s), "upload", rank);
  check(blz_host_ar_all_reduce(ar, d, bytes, dt, s), "all_reduce", rank);
  check(cudaStreamSynchronize(s), "sync", rank);
  check(cudaMemcpy(back.data(), d, bytes, cudaMemcpyDeviceToHost), "download", rank);
  cudaFree(d);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    float acc = 0.0f;
    for (int r = 0; r < kWorld; ++r) acc += bf ? from_bf16(to_bf16(in[r][i])) : in[r][i];
    if (bf ? reinterpret_cast<uint16_t*>(back.data())[i] != to_bf16(acc)
           : reinterpret_cast<float*>(back.data())[i] != acc)
      ++bad;
  }
  return bad;
}

int child(int rank, const std::string& name, int created_fd, int opened_fd, int iters) {
  int devices = 0;
  check(cudaGetDeviceCount(&devices), "device count", rank);
  check(cudaSetDevice(rank % devices), "set device", rank);
  blz_host_ar* ar = nullptr;
  char token = 1;
  if (rank == 0) {
    check(blz_host_ar_open(name.c_str(), rank, kWorld, kMaxBytes, 1, &ar), "open (create)", rank);
    if (write(created_fd, &token, 1) != 1) return 1;
    if (read(opened_fd, &token, 1) != 1) return 1;  // every other rank has mapped it
    blz_host_ar_unlink(name.c_str());
  } else {
    if (read(created_fd, &token, 1) != 1) return 1;
    check(blz_host_ar_open(name.c_str(), rank, kWorld, kMaxBytes, 0, &ar), "open", rank);
    if (write(opened_fd, &token, 1) != 1) return 1;
  }
  cudaStream_t s;
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", rank);

  int fails = 0, c = 0;
  // Both sides of the 64 KiB DMA threshold, one chunk, and a size that splits into two chunks.
  const int64_t sizes[] = {16, 10240, 65520, 65536, 81920, 327680, 1 << 20, 4 << 20, kMaxBytes};
  for (blz_ar_dtype_t dt : {BLZ_AR_BF16, BLZ_AR_FP32})
    for (int64_t bytes : sizes) {
      const size_t bad = run_case(ar, rank, c++, bytes, dt, s);
      if (rank == 0 || bad)
        std::printf("rank %d: %s %8lld B: %s\n", rank, dt == BLZ_AR_BF16 ? "bf16" : "fp32",
                    static_cast<long long>(bytes), bad ? "MISMATCH" : "ok");
      fails += bad != 0;
    }
  if (blz_host_ar_all_reduce(ar, nullptr, 17, BLZ_AR_BF16, s) != cudaErrorInvalidValue) {
    std::printf("rank %d: a size that is not a multiple of 16 was accepted\n", rank);
    ++fails;
  }

  // Captured graph: two reductions per replay, inputs refreshed before each replay (zero-copy and DMA sizes).
  for (const int64_t bytes : {int64_t(10240), int64_t(81920)}) {
    const size_t n = bytes / 2;
    void *src = nullptr, *buf = nullptr;
    check(cudaMalloc(&src, bytes), "malloc", rank);
    check(cudaMalloc(&buf, bytes), "malloc", rank);
    cudaGraph_t g;
    cudaGraphExec_t ge;
    check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture", rank);
    check(cudaMemcpyAsync(buf, src, bytes, cudaMemcpyDeviceToDevice, s), "copy", rank);
    check(blz_host_ar_all_reduce(ar, buf, bytes, BLZ_AR_BF16, s), "all_reduce (capture)", rank);
    check(blz_host_ar_all_reduce(ar, buf, bytes, BLZ_AR_BF16, s), "all_reduce (capture)", rank);
    check(cudaStreamEndCapture(s, &g), "end capture", rank);
    check(cudaGraphInstantiate(&ge, g, 0), "instantiate", rank);
    size_t bad = 0;
    std::vector<uint16_t> host(n), back(n);
    for (int rep = 0; rep < 5; ++rep) {
      std::vector<std::vector<float>> in(kWorld);
      for (int r = 0; r < kWorld; ++r) in[r] = inputs(r, 100 + rep, n);
      for (size_t i = 0; i < n; ++i) host[i] = to_bf16(in[rank][i]);
      check(cudaMemcpyAsync(src, host.data(), bytes, cudaMemcpyHostToDevice, s), "upload", rank);
      check(cudaGraphLaunch(ge, s), "graph launch", rank);
      check(cudaStreamSynchronize(s), "graph sync", rank);
      check(cudaMemcpy(back.data(), buf, bytes, cudaMemcpyDeviceToHost), "download", rank);
      for (size_t i = 0; i < n; ++i) {
        float once = 0.0f;
        for (int r = 0; r < kWorld; ++r) once += from_bf16(to_bf16(in[r][i]));
        const float sum = from_bf16(to_bf16(once));  // the second reduction adds the (equal) results of both ranks
        if (back[i] != to_bf16(sum + sum)) ++bad;
      }
    }
    if (rank == 0 || bad) std::printf("rank %d: graph replays, %lld B: %s\n", rank, static_cast<long long>(bytes), bad ? "MISMATCH" : "ok");
    fails += bad != 0;
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaFree(src);
    cudaFree(buf);
  }

  // all-reduce + residual + RMSNorm: the DMA path against the zero-copy one (bit for bit) and the residual against the
  // sum of both ranks' inputs.
  for (const auto& sh : {std::pair<int, int>{4, 5120}, {8, 4096}, {16, 5120}, {64, 5120}, {64, 8192}}) {
    const int rows = sh.first, dim = sh.second;
    const int64_t n = int64_t(rows) * dim, bytes = n * 2;
    uint16_t *x, *res, *w, *out[2];
    check(cudaMalloc(&x, bytes), "malloc", rank);
    check(cudaMalloc(&res, bytes), "malloc", rank);
    check(cudaMalloc(&w, size_t(dim) * 2), "malloc", rank);
    check(cudaMalloc(&out[0], bytes), "malloc", rank);
    check(cudaMalloc(&out[1], bytes), "malloc", rank);
    std::vector<std::vector<uint8_t>> xs(kWorld);
    for (int r = 0; r < kWorld; ++r) xs[r] = to_bytes(inputs(r, 500, size_t(n)), true);
    const std::vector<uint8_t> r0 = to_bytes(inputs(kWorld, 500, size_t(n)), true), wv = to_bytes(inputs(kWorld + 1, 500, size_t(dim)), true);
    check(cudaMemcpyAsync(x, xs[rank].data(), bytes, cudaMemcpyHostToDevice, s), "upload", rank);
    check(cudaMemcpyAsync(w, wv.data(), size_t(dim) * 2, cudaMemcpyHostToDevice, s), "upload", rank);
    std::vector<uint8_t> got[2], res_got[2];
    for (int pass = 0; pass < 2; ++pass) {
      blz_host_ar_set_dma_min(ar, pass == 0 ? 16 : INT64_MAX);  // both ranks alike
      check(cudaMemcpyAsync(res, r0.data(), bytes, cudaMemcpyHostToDevice, s), "upload", rank);
      check(blz_host_ar_rmsnorm(ar, x, res, w, 1e-6f, rows, dim, 1, out[pass], s), "rmsnorm", rank);
      check(cudaStreamSynchronize(s), "sync", rank);
      got[pass].resize(bytes), res_got[pass].resize(bytes);
      check(cudaMemcpy(got[pass].data(), out[pass], bytes, cudaMemcpyDeviceToHost), "download", rank);
      check(cudaMemcpy(res_got[pass].data(), res, bytes, cudaMemcpyDeviceToHost), "download", rank);
    }
    blz_host_ar_set_dma_min(ar, 64 << 10);
    std::vector<uint8_t> sum(bytes);
    reference_sum({xs[0].data(), xs[1].data()}, bytes, true, sum.data());
    size_t bad_res = 0;
    for (int64_t i = 0; i < n; ++i) {
      const uint16_t want = to_bf16(from_bf16(reinterpret_cast<uint16_t*>(sum.data())[i]) + from_bf16(reinterpret_cast<const uint16_t*>(r0.data())[i]));
      for (int pass = 0; pass < 2; ++pass) bad_res += reinterpret_cast<uint16_t*>(res_got[pass].data())[i] != want;
    }
    const bool same = got[0] == got[1];
    if (rank == 0 || bad_res || !same)
      std::printf("rank %d: rmsnorm %2d x %d: residual %s, DMA == zero-copy output: %s\n", rank, rows, dim,
                  bad_res ? "MISMATCH" : "ok", same ? "ok" : "MISMATCH");
    fails += bad_res != 0 || !same;
    cudaFree(x), cudaFree(res), cudaFree(w), cudaFree(out[0]), cudaFree(out[1]);
  }

  // Back-to-back reductions (decode: one token, a 4 x 8-token verify).
  for (int64_t bytes : {int64_t(10240), int64_t(327680)}) {
    void* d = nullptr;
    check(cudaMalloc(&d, bytes), "malloc", rank);
    check(cudaMemset(d, 0, bytes), "memset", rank);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 20; ++i) check(blz_host_ar_all_reduce(ar, d, bytes, BLZ_AR_BF16, s), "warmup", rank);
    cudaEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) check(blz_host_ar_all_reduce(ar, d, bytes, BLZ_AR_BF16, s), "bench", rank);
    cudaEventRecord(e1, s);
    check(cudaStreamSynchronize(s), "bench sync", rank);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    if (rank == 0)
      std::printf("%7lld B bf16: %.2f us per all-reduce (%d back to back)\n", static_cast<long long>(bytes),
                  1000.0f * ms / iters, iters);
    cudaFree(d);
  }
  // all-reduce + residual + RMSNorm of decode / verify rows (27B: hidden 5120), against the plain all-reduce of the
  // same bytes
  {
    const int dim = 5120, max_rows = 32;
    uint16_t *x = nullptr, *res = nullptr, *w = nullptr, *out = nullptr;
    check(cudaMalloc(&x, size_t(max_rows) * dim * 2), "malloc", rank);
    check(cudaMalloc(&res, size_t(max_rows) * dim * 2), "malloc", rank);
    check(cudaMalloc(&w, size_t(dim) * 2), "malloc", rank);
    check(cudaMalloc(&out, size_t(max_rows) * dim * 2), "malloc", rank);
    check(cudaMemset(x, 0, size_t(max_rows) * dim * 2), "memset", rank);
    check(cudaMemset(res, 0, size_t(max_rows) * dim * 2), "memset", rank);
    check(cudaMemset(w, 0, size_t(dim) * 2), "memset", rank);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int n = iters / 4;
    for (int rows : {1, 2, 4, 8, 16, 32}) {
      float ms[2] = {0, 0};
      for (int kind = 0; kind < 2; ++kind) {
        auto call = [&]() {
          return kind == 0 ? blz_host_ar_rmsnorm(ar, x, res, w, 1e-6f, rows, dim, 1, out, s)
                           : blz_host_ar_all_reduce(ar, x, int64_t(rows) * dim * 2, BLZ_AR_BF16, s);
        };
        for (int i = 0; i < 20; ++i) check(call(), "warmup", rank);
        cudaEventRecord(e0, s);
        for (int i = 0; i < n; ++i) check(call(), "bench", rank);
        cudaEventRecord(e1, s);
        check(cudaStreamSynchronize(s), "bench sync", rank);
        cudaEventElapsedTime(&ms[kind], e0, e1);
      }
      if (rank == 0)
        std::printf("%2d rows x %d: ar+rmsnorm %.2f us, ar %.2f us\n", rows, dim, 1000.0f * ms[0] / n, 1000.0f * ms[1] / n);
    }
    cudaFree(x), cudaFree(res), cudaFree(w), cudaFree(out);
  }
  // Zero-copy against DMA for the same call (the setting is the same on both ranks): where the threshold belongs.
  {
    uint16_t *x = nullptr, *res = nullptr, *w = nullptr, *out = nullptr;
    const int dim = 5120;
    check(cudaMalloc(&x, 64 * size_t(dim) * 2), "malloc", rank);
    check(cudaMalloc(&res, 64 * size_t(dim) * 2), "malloc", rank);
    check(cudaMalloc(&w, size_t(dim) * 2), "malloc", rank);
    check(cudaMalloc(&out, 64 * size_t(dim) * 2), "malloc", rank);
    check(cudaMemset(x, 0, 64 * size_t(dim) * 2), "memset", rank);
    check(cudaMemset(res, 0, 64 * size_t(dim) * 2), "memset", rank);
    check(cudaMemset(w, 0, size_t(dim) * 2), "memset", rank);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    const int n = std::max(iters / 8, 20);
    if (rank == 0) std::printf("zero-copy vs DMA, us per call (%d back to back):\n", n);
    for (int64_t bytes : {int64_t(16) << 10, int64_t(32) << 10, int64_t(48) << 10, int64_t(64) << 10, int64_t(96) << 10,
                          int64_t(128) << 10, int64_t(327680), int64_t(1) << 20, int64_t(4) << 20, kMaxBytes}) {
      float us[2][2] = {{0, 0}, {0, 0}};  // [all-reduce, ar + rmsnorm][zero-copy, DMA]
      void* d = nullptr;
      check(cudaMalloc(&d, bytes), "malloc", rank);
      check(cudaMemset(d, 0, bytes), "memset", rank);
      const int rows = int(bytes / (dim * 2));
      const bool norm = bytes % (dim * 2) == 0 && rows >= 1 && rows <= 64;
      for (int kind = 0; kind < (norm ? 2 : 1); ++kind)
        for (int path = 0; path < 2; ++path) {
          blz_host_ar_set_dma_min(ar, path ? 16 : INT64_MAX);
          auto call = [&]() {
            return kind == 0 ? blz_host_ar_all_reduce(ar, d, bytes, BLZ_AR_BF16, s)
                             : blz_host_ar_rmsnorm(ar, x, res, w, 1e-6f, rows, dim, 1, out, s);
          };
          for (int i = 0; i < 10; ++i) check(call(), "warmup", rank);
          cudaEventRecord(e0, s);
          for (int i = 0; i < n; ++i) check(call(), "bench", rank);
          cudaEventRecord(e1, s);
          check(cudaStreamSynchronize(s), "bench sync", rank);
          float ms = 0.0f;
          cudaEventElapsedTime(&ms, e0, e1);
          us[kind][path] = 1000.0f * ms / n;
        }
      if (rank == 0) {
        std::printf("%8lld B: all-reduce %8.1f / %8.1f", static_cast<long long>(bytes), us[0][0], us[0][1]);
        if (norm) std::printf("   rmsnorm (%2d x %d) %8.1f / %8.1f", rows, dim, us[1][0], us[1][1]);
        std::printf("\n");
      }
      cudaFree(d);
    }
    blz_host_ar_set_dma_min(ar, 64 << 10);
    cudaFree(x), cudaFree(res), cudaFree(w), cudaFree(out);
  }
  blz_host_ar_close(ar);
  return fails ? 1 : 0;
}

// Layout of the shared buffer, mirroring src/cuda/host_allreduce.cu.
constexpr int64_t kFlagBytes = 4096, kMaxBlocksAr = 64, kBytesPerBlock = 4096;
constexpr int kReadyWord = 512, kConsumedWord = 528;
constexpr int64_t kDmaChunk = 4 << 20, kDmaMinDefault = 64 << 10;
constexpr int64_t kNever = INT64_MAX;  // dma_min that disables the DMA path
int64_t dma_offset(int world) { return (kFlagBytes + int64_t(world) * 2 * kMaxBytes + 4095) / 4096 * 4096; }

size_t count_diff(const uint8_t* a, const uint8_t* b, size_t bytes, size_t* first) {
  size_t bad = 0;
  *first = bytes;
  for (size_t i = 0; i < bytes; ++i)
    if (a[i] != b[i]) {
      if (bad++ == 0) *first = i;
    }
  return bad;
}

// The CPU side of a group whose rank `gpu` is the GPU under test: every other rank is played here by the buffer
// protocol (zero-copy or DMA, chosen by size exactly like the library). One thread: a call publishes all the played
// ranks first and then waits for the GPU. Each call also hands back what the GPU published (its slot).
struct Emu {
  uint8_t* base;
  int world, gpu;
  int64_t dma_off;
  int64_t dma_min = kDmaMinDefault;
  uint32_t seq = 0;       // the GPU's call counter
  uint32_t last_dma = 0;  // seq of the last DMA call: the GPU has read the slots of the played ranks by then

  uint32_t* flags() const { return reinterpret_cast<uint32_t*>(base); }

  static bool wait(const uint32_t* f, uint32_t want) {
    const auto t0 = std::chrono::steady_clock::now();
    while (int32_t(__atomic_load_n(f, __ATOMIC_ACQUIRE) - want) < 0)
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) return false;
    return true;
  }

  // One zero-copy call: `blocks` flags per rank. own[r] (r != gpu) is rank r's tensor of `bytes`.
  bool zero_copy(const std::vector<const uint8_t*>& own, int64_t bytes, int blocks, uint8_t* gpu_slot) {
    const uint32_t s = ++seq;
    auto slot = [&](int r) { return base + kFlagBytes + (int64_t(r) * 2 + (s & 1)) * kMaxBytes; };
    for (int r = 0; r < world; ++r)
      if (r != gpu) std::memcpy(slot(r), own[r], bytes);
    for (int r = 0; r < world; ++r)
      if (r != gpu)
        for (int b = 0; b < blocks; ++b) __atomic_store_n(flags() + r * kMaxBlocksAr + b, s, __ATOMIC_RELEASE);
    for (int b = 0; b < blocks; ++b)
      if (!wait(flags() + gpu * kMaxBlocksAr + b, s)) return false;
    std::memcpy(gpu_slot, slot(gpu), bytes);
    return true;
  }

  // One DMA call (bytes <= kDmaChunk): refill my slots once the GPU has read the last ones, signal, read the GPU's.
  bool dma(const std::vector<const uint8_t*>& own, int64_t bytes, uint8_t* gpu_slot) {
    const uint32_t s = ++seq;
    auto slot = [&](int r) { return base + dma_off + int64_t(r) * kDmaChunk; };
    if (!wait(flags() + kConsumedWord + gpu, last_dma)) return false;
    for (int r = 0; r < world; ++r)
      if (r != gpu) std::memcpy(slot(r), own[r], bytes);
    for (int r = 0; r < world; ++r)
      if (r != gpu) __atomic_store_n(flags() + kReadyWord + r, s, __ATOMIC_RELEASE);
    if (!wait(flags() + kReadyWord + gpu, s)) return false;
    std::memcpy(gpu_slot, slot(gpu), bytes);
    for (int r = 0; r < world; ++r)
      if (r != gpu) __atomic_store_n(flags() + kConsumedWord + r, s, __ATOMIC_RELEASE);
    last_dma = s;
    return true;
  }

  // blz_host_ar_all_reduce of `bytes`: the library's split into calls.
  bool all_reduce(const std::vector<const uint8_t*>& own, int64_t bytes, uint8_t* gpu_slot) {
    if (bytes < dma_min) {
      const int blocks = int(std::min<int64_t>(kMaxBlocksAr, (bytes + kBytesPerBlock - 1) / kBytesPerBlock));
      return zero_copy(own, bytes, blocks, gpu_slot);
    }
    const int64_t chunks = (bytes + kDmaChunk - 1) / kDmaChunk, per = (bytes / chunks + 15) / 16 * 16;
    for (int64_t off = 0; off < bytes; off += per) {
      std::vector<const uint8_t*> o(world, nullptr);
      for (int r = 0; r < world; ++r)
        if (r != gpu) o[r] = own[r] + off;
      if (!dma(o, std::min(per, bytes - off), gpu_slot + off)) return false;
    }
    return true;
  }

  // blz_host_ar_rmsnorm of rows x dim bf16: one call, a flag block per row on the zero-copy path.
  bool rmsnorm(const std::vector<const uint8_t*>& own, int rows, int dim, uint8_t* gpu_slot) {
    const int64_t bytes = int64_t(rows) * dim * 2;
    if (bytes >= dma_min && bytes <= kDmaChunk) return dma(own, bytes, gpu_slot);
    return zero_copy(own, bytes, rows, gpu_slot);
  }
};

void set_dma_min(Emu& e, blz_host_ar* ar, int64_t bytes) {
  e.dma_min = bytes;
  blz_host_ar_set_dma_min(ar, bytes);
}

constexpr int kTimeout = 2;  // result of a case whose CPU side gave up waiting for the GPU: the run is over

// Inputs of every rank for case c; the GPU rank's goes to the device.
struct Case {
  std::vector<std::vector<uint8_t>> host;  // per rank
  std::vector<const uint8_t*> own;
  Case(int world, int c, size_t n, bool bf) : host(world), own(world) {
    for (int r = 0; r < world; ++r) {
      host[r] = to_bytes(inputs(r, c, n), bf);
      own[r] = host[r].data();
    }
  }
};

// Sequential case: one all-reduce, the GPU result and the GPU's published slot against the reference.
int ar_case(Emu& e, blz_host_ar* ar, cudaStream_t s, int c, int64_t bytes, blz_ar_dtype_t dt) {
  const bool bf = dt == BLZ_AR_BF16;
  Case in(e.world, c, size_t(bytes) / (bf ? 2 : 4), bf);
  void* d = nullptr;
  check(cudaMalloc(&d, bytes), "malloc", e.gpu);
  check(cudaMemcpyAsync(d, in.own[e.gpu], bytes, cudaMemcpyHostToDevice, s), "upload", e.gpu);
  check(blz_host_ar_all_reduce(ar, d, bytes, dt, s), "all_reduce", e.gpu);
  std::vector<uint8_t> slot(bytes), got(bytes), ref(bytes);
  const bool ok = e.all_reduce(in.own, bytes, slot.data());
  if (!ok) {
    std::printf("cpu emulation: timed out waiting for the GPU (%lld B)\n", static_cast<long long>(bytes));
    return kTimeout;
  }
  check(cudaStreamSynchronize(s), "sync", e.gpu);
  check(cudaMemcpy(got.data(), d, bytes, cudaMemcpyDeviceToHost), "download", e.gpu);
  cudaFree(d);
  reference_sum(in.own, bytes, bf, ref.data());
  size_t first_slot, first_got;
  const size_t bad_slot = count_diff(slot.data(), in.own[e.gpu], bytes, &first_slot);
  const size_t bad_got = count_diff(got.data(), ref.data(), bytes, &first_got);
  if (bad_slot || bad_got)
    std::printf("cpu emulation: %s %lld B: %zu bytes of the GPU's slot differ (first at %zu), %zu of its result (first at %zu)\n",
                bf ? "bf16" : "fp32", static_cast<long long>(bytes), bad_slot, first_slot, bad_got, first_got);
  return bad_slot || bad_got;
}

// Back-to-back calls without a sync in between: the GPU runs ahead of the emulation as far as the protocol lets it.
int ar_pipeline(Emu& e, blz_host_ar* ar, cudaStream_t s, int c0,
                const std::vector<std::pair<int64_t, blz_ar_dtype_t>>& calls) {
  const int k = int(calls.size());
  std::vector<Case> in;
  in.reserve(k);  // `own` points into the stored cases
  std::vector<void*> d(k);
  for (int i = 0; i < k; ++i) {
    const int64_t bytes = calls[i].first;
    const bool bf = calls[i].second == BLZ_AR_BF16;
    in.emplace_back(e.world, c0 + i, size_t(bytes) / (bf ? 2 : 4), bf);
    check(cudaMalloc(&d[i], bytes), "malloc", e.gpu);
    check(cudaMemcpyAsync(d[i], in[i].own[e.gpu], bytes, cudaMemcpyHostToDevice, s), "upload", e.gpu);
  }
  for (int i = 0; i < k; ++i) check(blz_host_ar_all_reduce(ar, d[i], calls[i].first, calls[i].second, s), "all_reduce", e.gpu);
  std::vector<std::vector<uint8_t>> slot(k);
  for (int i = 0; i < k; ++i) {
    slot[i].resize(calls[i].first);
    if (!e.all_reduce(in[i].own, calls[i].first, slot[i].data())) {
      std::printf("cpu emulation: timed out waiting for the GPU (pipelined call %d, %lld B)\n", i,
                  static_cast<long long>(calls[i].first));
      return kTimeout;
    }
  }
  check(cudaStreamSynchronize(s), "sync", e.gpu);
  int bad = 0;
  for (int i = 0; i < k; ++i) {
    const int64_t bytes = calls[i].first;
    const bool bf = calls[i].second == BLZ_AR_BF16;
    std::vector<uint8_t> got(bytes), ref(bytes);
    check(cudaMemcpy(got.data(), d[i], bytes, cudaMemcpyDeviceToHost), "download", e.gpu);
    cudaFree(d[i]);
    reference_sum(in[i].own, bytes, bf, ref.data());
    size_t f1, f2;
    const size_t bad_slot = count_diff(slot[i].data(), in[i].own[e.gpu], bytes, &f1);
    const size_t bad_got = count_diff(got.data(), ref.data(), bytes, &f2);
    if (bad_slot || bad_got)
      std::printf("cpu emulation: pipelined call %d, %s %lld B: %zu slot bytes differ, %zu result bytes differ\n", i,
                  bf ? "bf16" : "fp32", static_cast<long long>(bytes), bad_slot, bad_got);
    bad += bad_slot || bad_got;
  }
  return bad;
}

// Captured graph with a mixed sequence on one buffer (DMA, zero-copy, DMA, a chunked DMA call under dma_min `mode`),
// replayed with fresh inputs. The emulated ranks keep their tensors across the calls of a replay.
int graph_case(Emu& e, blz_host_ar* ar, cudaStream_t s, int c0, int64_t mode) {
  const int64_t total = 5 << 20;  // two chunks of 2.5 MiB
  const int64_t sizes[] = {81920, 10240, 81920, total};
  set_dma_min(e, ar, mode);
  void *src = nullptr, *buf = nullptr;
  check(cudaMalloc(&src, total), "malloc", e.gpu);
  check(cudaMalloc(&buf, total), "malloc", e.gpu);
  cudaGraph_t g;
  cudaGraphExec_t ge;
  check(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "begin capture", e.gpu);
  check(cudaMemcpyAsync(buf, src, total, cudaMemcpyDeviceToDevice, s), "copy", e.gpu);
  for (int64_t bytes : sizes) check(blz_host_ar_all_reduce(ar, buf, bytes, BLZ_AR_BF16, s), "all_reduce (capture)", e.gpu);
  check(cudaStreamEndCapture(s, &g), "end capture", e.gpu);
  check(cudaGraphInstantiate(&ge, g, 0), "instantiate", e.gpu);
  int bad = 0;
  for (int rep = 0; rep < 4; ++rep) {
    Case in(e.world, c0 + rep, size_t(total) / 2, true);
    check(cudaMemcpyAsync(src, in.own[e.gpu], total, cudaMemcpyHostToDevice, s), "upload", e.gpu);
    check(cudaGraphLaunch(ge, s), "graph launch", e.gpu);
    std::vector<uint8_t> slot(total), sum(total);
    for (int64_t bytes : sizes) {
      if (!e.all_reduce(in.own, bytes, slot.data())) {
        std::printf("cpu emulation: timed out waiting for the GPU (graph, %lld B)\n", static_cast<long long>(bytes));
        return kTimeout;
      }
      size_t first;
      if (count_diff(slot.data(), in.host[e.gpu].data(), bytes, &first)) {
        std::printf("cpu emulation: graph replay %d, %lld B: the GPU's slot differs from its tensor (first at %zu)\n", rep,
                    static_cast<long long>(bytes), first);
        ++bad;
      }
      // After the call every rank holds the sum in the first `bytes`.
      reference_sum(in.own, bytes, true, sum.data());
      for (int r = 0; r < e.world; ++r) std::memcpy(in.host[r].data(), sum.data(), bytes);
    }
    check(cudaStreamSynchronize(s), "graph sync", e.gpu);
    std::vector<uint8_t> got(total);
    check(cudaMemcpy(got.data(), buf, total, cudaMemcpyDeviceToHost), "download", e.gpu);
    size_t first;
    if (count_diff(got.data(), in.host[e.gpu].data(), total, &first)) {
      std::printf("cpu emulation: graph replay %d (dma_min %lld): result differs (first at byte %zu)\n", rep,
                  static_cast<long long>(mode), first);
      ++bad;
    }
  }
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  cudaFree(src);
  cudaFree(buf);
  set_dma_min(e, ar, kDmaMinDefault);
  return bad;
}

// All-reduce + residual + RMSNorm of rows x dim: the residual against the exact reference (sum in rank order -> bf16,
// + residual -> bf16), the output against a loose reference, and the output of the DMA path against the zero-copy
// kernel's, bit for bit (the same norm code reading the peers from another place).
int rmsnorm_case(Emu& e, blz_host_ar* ar, cudaStream_t s, int c, int rows, int dim, int zero_centered) {
  const int64_t n = int64_t(rows) * dim, bytes = n * 2;
  const float eps = 1e-6f;
  Case x(e.world, c, size_t(n), true);
  const std::vector<uint8_t> res0 = to_bytes(inputs(e.world, c, size_t(n)), true), w = to_bytes(inputs(e.world + 1, c, size_t(dim)), true);
  uint16_t *dx, *dres, *dw, *dout;
  check(cudaMalloc(&dx, bytes), "malloc", e.gpu);
  check(cudaMalloc(&dres, bytes), "malloc", e.gpu);
  check(cudaMalloc(&dw, size_t(dim) * 2), "malloc", e.gpu);
  check(cudaMalloc(&dout, bytes), "malloc", e.gpu);
  check(cudaMemcpyAsync(dx, x.own[e.gpu], bytes, cudaMemcpyHostToDevice, s), "upload", e.gpu);
  check(cudaMemcpyAsync(dw, w.data(), size_t(dim) * 2, cudaMemcpyHostToDevice, s), "upload", e.gpu);

  // Pass 0: the default path of this size; pass 1: the other one.
  const bool dma_first = bytes >= kDmaMinDefault && bytes <= kDmaChunk;
  std::vector<uint8_t> res[2], out[2];
  for (int pass = 0; pass < 2; ++pass) {
    set_dma_min(e, ar, pass == 0 ? kDmaMinDefault : (dma_first ? kNever : 16));
    check(cudaMemcpyAsync(dres, res0.data(), bytes, cudaMemcpyHostToDevice, s), "upload", e.gpu);
    check(blz_host_ar_rmsnorm(ar, dx, dres, dw, eps, rows, dim, zero_centered, dout, s), "rmsnorm", e.gpu);
    std::vector<uint8_t> slot(bytes);
    if (!e.rmsnorm(x.own, rows, dim, slot.data())) {
      std::printf("cpu emulation: timed out waiting for the GPU (rmsnorm %d x %d)\n", rows, dim);
      return kTimeout;
    }
    check(cudaStreamSynchronize(s), "sync", e.gpu);
    res[pass].resize(bytes), out[pass].resize(bytes);
    check(cudaMemcpy(res[pass].data(), dres, bytes, cudaMemcpyDeviceToHost), "download", e.gpu);
    check(cudaMemcpy(out[pass].data(), dout, bytes, cudaMemcpyDeviceToHost), "download", e.gpu);
    size_t first;
    if (count_diff(slot.data(), x.own[e.gpu], bytes, &first)) {
      std::printf("cpu emulation: rmsnorm %d x %d: the GPU's slot differs from its tensor\n", rows, dim);
      return 1;
    }
  }
  set_dma_min(e, ar, kDmaMinDefault);
  cudaFree(dx), cudaFree(dres), cudaFree(dw), cudaFree(dout);

  // Exact: residual.
  std::vector<uint8_t> sum(bytes), ref(bytes);
  reference_sum(x.own, bytes, true, sum.data());
  for (int64_t i = 0; i < n; ++i)
    reinterpret_cast<uint16_t*>(ref.data())[i] = to_bf16(from_bf16(reinterpret_cast<uint16_t*>(sum.data())[i]) +
                                                         from_bf16(reinterpret_cast<const uint16_t*>(res0.data())[i]));
  size_t first;
  int bad = 0;
  for (int pass = 0; pass < 2; ++pass)
    if (size_t b = count_diff(res[pass].data(), ref.data(), size_t(bytes), &first)) {
      std::printf("cpu emulation: rmsnorm %d x %d (%s path): residual differs in %zu bytes (first at %zu)\n", rows, dim,
                  (pass == 0) == dma_first ? "DMA" : "zero-copy", b, first);
      ++bad;
    }
  // Bit for bit between the two paths.
  if (size_t b = count_diff(out[0].data(), out[1].data(), size_t(bytes), &first)) {
    std::printf("cpu emulation: rmsnorm %d x %d: DMA and zero-copy outputs differ in %zu bytes (first at %zu)\n", rows, dim, b,
                first);
    ++bad;
  }
  // Loose: the output itself (all that the rounding order can change is the last bf16 bit).
  size_t far = 0;
  for (int row = 0; row < rows; ++row) {
    double ss = 0;
    for (int i = 0; i < dim; ++i) {
      const double h = from_bf16(reinterpret_cast<const uint16_t*>(ref.data())[int64_t(row) * dim + i]);
      ss += h * h;
    }
    const double inv = 1.0 / std::sqrt(ss / dim + double(eps));
    for (int i = 0; i < dim; ++i) {
      const double h = from_bf16(reinterpret_cast<const uint16_t*>(ref.data())[int64_t(row) * dim + i]);
      const double wi = zero_centered ? 1.0 + from_bf16(reinterpret_cast<const uint16_t*>(w.data())[i])
                                      : from_bf16(reinterpret_cast<const uint16_t*>(w.data())[i]);
      const double want = h * inv * wi, got = from_bf16(reinterpret_cast<const uint16_t*>(out[0].data())[int64_t(row) * dim + i]);
      if (std::fabs(got - want) > 0.01 * std::fabs(want) + 1e-4) ++far;
    }
  }
  if (far) {
    std::printf("cpu emulation: rmsnorm %d x %d: %zu outputs far from the reference\n", rows, dim, far);
    ++bad;
  }
  return bad;
}

struct Config {
  int world, gpu, rounds;
};
const Config kConfigs[] = {{2, 0, 6}, {3, 1, 3}, {4, 3, 3}, {8, 5, 3}};

int cpu_peer_test(int idx) {
  const Config cfg = kConfigs[idx];
  const std::string name = "/blz_ar_cpu_" + std::to_string(getpid());
  check(cudaSetDevice(0), "set device", 0);
  blz_host_ar* ar = nullptr;
  check(blz_host_ar_open(name.c_str(), cfg.gpu, cfg.world, kMaxBytes, 1, &ar), "open (create)", cfg.gpu);
  const int64_t dma_off = dma_offset(cfg.world);
  const size_t total = size_t(dma_off) + size_t(cfg.world) * kDmaChunk;
  const int fd = shm_open(name.c_str(), O_RDWR, 0600);
  auto* base = static_cast<uint8_t*>(mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  close(fd);
  blz_host_ar_unlink(name.c_str());
  if (base == MAP_FAILED) return 1;
  Emu e{base, cfg.world, cfg.gpu, dma_off};
  cudaStream_t s;
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", cfg.gpu);

  int bad = 0, cases = 0;
  auto done = [&](int r) {
    ++cases;
    bad += r != 0;
    return r != kTimeout;
  };

  // Sizes on both sides of the DMA threshold in turn; rounds use dma_min = default, never, and always.
  const int64_t sizes[] = {16, 81920, 10240, 65536, 65520, 327680, 16, 1 << 20, kDmaChunk + 16, 10240, kMaxBytes, 65520, 81920, 16};
  const int64_t modes[] = {kDmaMinDefault, kNever, 16};
  int c = 1000;
  for (int round = 0; round < cfg.rounds; ++round) {
    set_dma_min(e, ar, modes[round % 3]);
    for (blz_ar_dtype_t dt : {BLZ_AR_BF16, BLZ_AR_FP32})
      for (int64_t bytes : sizes)
        if (!done(ar_case(e, ar, s, c++, bytes, dt))) return 1;
  }
  set_dma_min(e, ar, kDmaMinDefault);
  const int sequential = cases;

  const int64_t pipe_sizes[] = {16, 10240, 65520, 65536, 81920, 327680, 1 << 20, kDmaChunk + 16};
  for (int batch = 0; batch < cfg.rounds; ++batch) {
    std::mt19937 rng(7u + unsigned(batch));
    std::vector<std::pair<int64_t, blz_ar_dtype_t>> calls;
    for (int i = 0; i < 16; ++i) calls.push_back({pipe_sizes[rng() % 8], rng() % 2 ? BLZ_AR_BF16 : BLZ_AR_FP32});
    if (!done(ar_pipeline(e, ar, s, c, calls))) return 1;
    c += 16;
  }
  const int pipelined = cases - sequential;

  for (int64_t mode : {kDmaMinDefault, int64_t(16), kNever})
    if (!done(graph_case(e, ar, s, c += 10, mode))) return 1;

  const int shapes[][2] = {{1, 5120}, {4, 5120}, {7, 4096}, {8, 4096}, {8, 5120}, {16, 5120},
                           {33, 5120}, {64, 5120}, {40, 2048}, {64, 8192}};
  int zc = 0;
  for (const auto& sh : shapes) {
    if (!done(rmsnorm_case(e, ar, s, c++, sh[0], sh[1], zc ^= 1))) return 1;
  }

  std::printf("GPU rank %d of %d + CPU ranks: %d sequential, %d pipelined, 3 graph, %zu rmsnorm cases, %s\n", cfg.gpu,
              cfg.world, sequential, pipelined, sizeof(shapes) / sizeof(shapes[0]), bad ? "MISMATCH" : "ok");
  munmap(base, total);
  blz_host_ar_close(ar);
  return bad ? 1 : 0;
}

// Visible GPUs, counted in a child so that this process forks before initializing CUDA.
int device_count() {
  const pid_t p = fork();
  if (p == 0) {
    int n = 0;
    std::exit(cudaGetDeviceCount(&n) == cudaSuccess ? std::min(n, 100) : 0);
  }
  int st = 0;
  waitpid(p, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 0;
}

int in_child(int (*fn)(int), int arg) {
  const pid_t p = fork();
  if (p == 0) std::exit(fn(arg));
  int st = 0;
  waitpid(p, &st, 0);
  return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 2000;
  int fails = 0;
  for (int i = 0; i < int(sizeof(kConfigs) / sizeof(kConfigs[0])); ++i) fails += in_child(cpu_peer_test, i) != 0;
  if (device_count() < 2) {
    std::printf("two-process test skipped: fewer than two GPUs\n");
    std::printf("%s\n", fails ? "FAILED" : "all passed");
    return fails ? 1 : 0;
  }
  const std::string name = "/blz_ar_test_" + std::to_string(getpid());
  int created[2], opened[2];
  if (pipe(created) != 0 || pipe(opened) != 0) return 1;
  pid_t pids[kWorld];
  for (int r = 0; r < kWorld; ++r) {
    pids[r] = fork();
    if (pids[r] == 0) std::exit(child(r, name, r == 0 ? created[1] : created[0], r == 0 ? opened[0] : opened[1], iters));
  }
  for (pid_t p : pids) {
    int st = 0;
    waitpid(p, &st, 0);
    fails += !(WIFEXITED(st) && WEXITSTATUS(st) == 0);
  }
  std::printf("%s\n", fails ? "FAILED" : "all passed");
  return fails ? 1 : 0;
}
