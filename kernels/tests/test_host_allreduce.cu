// blz_host_ar through the shared host buffer, checked bitwise against the sum of the inputs (bf16 / fp32, decode to MB
// sizes):
//  - a GPU rank and a CPU thread playing the other rank by the buffer protocol (GPU -> RAM and RAM -> GPU visibility;
//    runs on one GPU);
//  - with two or more GPUs, two forked processes on GPUs 0 and 1: results, replays of a captured CUDA graph, and the
//    time of back-to-back decode-sized reductions.
// Two processes on one GPU are not a valid setup: each registers the shared pages separately, and the GPU caches the
// two mappings as unrelated lines.
//   test_host_allreduce [iterations]
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
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
constexpr int64_t kMaxBytes = 4 << 20;

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
  const int64_t sizes[] = {16, 10240, 81920, 327680, 1 << 20, kMaxBytes};
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

  // Captured graph: two reductions per replay, inputs refreshed before each replay.
  {
    const int64_t bytes = 10240;
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
    if (rank == 0 || bad) std::printf("rank %d: graph replays: %s\n", rank, bad ? "MISMATCH" : "ok");
    fails += bad != 0;
    cudaGraphExecDestroy(ge);
    cudaGraphDestroy(g);
    cudaFree(src);
    cudaFree(buf);
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
  blz_host_ar_close(ar);
  return fails ? 1 : 0;
}


// Layout of the shared buffer, mirroring src/cuda/host_allreduce.cu.
constexpr int64_t kFlagBytes = 4096, kMaxBlocksAr = 64, kBytesPerBlock = 4096;

// Rank 1 on the CPU: one call of `bytes` at call counter `seq`; own values in, the reduced values out.
bool cpu_rank_call(uint8_t* base, uint32_t seq, int64_t bytes, bool bf, const std::vector<uint8_t>& own,
                   std::vector<uint8_t>& out) {
  const int blocks = int(std::min<int64_t>(kMaxBlocksAr, (bytes + kBytesPerBlock - 1) / kBytesPerBlock));
  auto slot = [&](int r) { return base + kFlagBytes + (int64_t(r) * 2 + (seq & 1)) * kMaxBytes; };
  auto* flags = reinterpret_cast<uint32_t*>(base);
  std::memcpy(slot(1), own.data(), bytes);
  std::atomic_thread_fence(std::memory_order_release);
  for (int b = 0; b < blocks; ++b) __atomic_store_n(flags + kMaxBlocksAr + b, seq, __ATOMIC_RELEASE);
  for (int b = 0; b < blocks; ++b) {
    const auto t0 = std::chrono::steady_clock::now();
    while (int32_t(__atomic_load_n(flags + b, __ATOMIC_ACQUIRE) - seq) < 0)
      if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(10)) return false;
  }
  out.resize(bytes);
  const uint8_t* peer = slot(0);
  const size_t n = size_t(bytes) / (bf ? 2 : 4);
  for (size_t i = 0; i < n; ++i) {
    if (bf) {
      const float acc = from_bf16(reinterpret_cast<const uint16_t*>(peer)[i]) +
                        from_bf16(reinterpret_cast<const uint16_t*>(own.data())[i]);
      reinterpret_cast<uint16_t*>(out.data())[i] = to_bf16(acc);
    } else {
      reinterpret_cast<float*>(out.data())[i] =
          reinterpret_cast<const float*>(peer)[i] + reinterpret_cast<const float*>(own.data())[i];
    }
  }
  return true;
}

int cpu_peer_test(int rounds) {
  const std::string name = "/blz_ar_cpu_" + std::to_string(getpid());
  check(cudaSetDevice(0), "set device", 0);
  blz_host_ar* ar = nullptr;
  check(blz_host_ar_open(name.c_str(), 0, kWorld, kMaxBytes, 1, &ar), "open (create)", 0);
  const size_t total = kFlagBytes + size_t(kWorld) * 2 * kMaxBytes;
  const int fd = shm_open(name.c_str(), O_RDWR, 0600);
  auto* base = static_cast<uint8_t*>(mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
  close(fd);
  blz_host_ar_unlink(name.c_str());
  if (base == MAP_FAILED) return 1;
  cudaStream_t s;
  check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream", 0);
  const int64_t sizes[] = {16, 10240, 81920, 327680, 1 << 20, kMaxBytes, 327680, 10240, 1 << 20, 16};
  uint32_t seq = 0;
  size_t bad = 0, calls = 0;
  for (int round = 0; round < rounds; ++round)
    for (blz_ar_dtype_t dt : {BLZ_AR_BF16, BLZ_AR_FP32})
      for (int64_t bytes : sizes) {
        const bool bf = dt == BLZ_AR_BF16;
        const size_t n = size_t(bytes) / (bf ? 2 : 4);
        const int c = 1000 + int(calls++);
        std::vector<std::vector<uint8_t>> host(kWorld, std::vector<uint8_t>(bytes));
        std::vector<std::vector<float>> in(kWorld);
        for (int r = 0; r < kWorld; ++r) {
          in[r] = inputs(r, c, n);
          for (size_t i = 0; i < n; ++i) {
            if (bf)
              reinterpret_cast<uint16_t*>(host[r].data())[i] = to_bf16(in[r][i]);
            else
              reinterpret_cast<float*>(host[r].data())[i] = in[r][i];
          }
        }
        void* d = nullptr;
        check(cudaMalloc(&d, bytes), "malloc", 0);
        check(cudaMemcpyAsync(d, host[0].data(), bytes, cudaMemcpyHostToDevice, s), "upload", 0);
        check(blz_host_ar_all_reduce(ar, d, bytes, dt, s), "all_reduce", 0);
        std::vector<uint8_t> cpu_out, gpu_out(bytes);
        const bool ok = cpu_rank_call(base, ++seq, bytes, bf, host[1], cpu_out);
        check(cudaStreamSynchronize(s), "sync", 0);
        check(cudaMemcpy(gpu_out.data(), d, bytes, cudaMemcpyDeviceToHost), "download", 0);
        cudaFree(d);
        if (!ok) {
          std::printf("cpu peer: timed out waiting for the GPU (%lld B)\n", static_cast<long long>(bytes));
          return 1;
        }
        size_t case_bad = 0, gpu_bad = 0, first_gpu = n, first_cpu = n;
        for (size_t i = 0; i < n; ++i) {
          const float acc = bf ? from_bf16(to_bf16(in[0][i])) + from_bf16(to_bf16(in[1][i])) : in[0][i] + in[1][i];
          const bool g = bf ? reinterpret_cast<uint16_t*>(gpu_out.data())[i] == to_bf16(acc)
                            : reinterpret_cast<float*>(gpu_out.data())[i] == acc;
          const bool h = bf ? reinterpret_cast<uint16_t*>(cpu_out.data())[i] == to_bf16(acc)
                            : reinterpret_cast<float*>(cpu_out.data())[i] == acc;
          case_bad += !g || !h;
          gpu_bad += !g;
          if (!g && first_gpu == n) first_gpu = i;
          if (!h && first_cpu == n) first_cpu = i;
        }
        if (case_bad)
          std::printf("cpu peer: %s %lld B: %zu mismatches (GPU rank %zu, first at %zu; CPU rank first at %zu)\n",
                      bf ? "bf16" : "fp32", static_cast<long long>(bytes), case_bad, gpu_bad, first_gpu, first_cpu);
        bad += case_bad != 0;
      }
  std::printf("GPU rank + CPU rank: %zu calls, %s\n", calls, bad ? "MISMATCH" : "ok");
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
  int fails = in_child(cpu_peer_test, 5) != 0;
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
