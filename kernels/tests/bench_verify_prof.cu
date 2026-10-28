// Phase profile of the pair verify kernel (mha_verify.cu built into this file with BLZ_VERIFY_PROFILE): clock64 marks
// per warp, so it shows where a key tile's time goes on this GPU (copies issued, Q K^T, score exchange, softmax, the
// mid-tile wait and barrier, P V, the end barrier) without a profiler. The 27B's per-GPU geometry under TP2: 12 query
// heads, 2 KV heads, NVFP4 pool, one request, the runtime's one-wave splits.
#define BLZ_VERIFY_PROFILE 1
#include "../src/cuda/mha_verify.cu"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    const cudaError_t e_ = (x);                                                                    \
    if (e_ != cudaSuccess) {                                                                       \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));    \
      std::exit(1);                                                                                \
    }                                                                                              \
  } while (0)

template <typename T>
T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  CHECK(cudaMalloc(&d, h.size() * sizeof(T)));
  CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
  return d;
}

void run(int ctx, int q_len, bool f16, std::mt19937& rng) {
  const int kv_heads = 2, q_heads = 12, dim = 256, slots = ctx + 64;
  int dev = 0, sms = 0, khz = 0;
  CHECK(cudaGetDevice(&dev));
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CHECK(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev));
  int per_tile = 0, mt = 0;
  const int tiles = blaze::fa::verify_tiles(q_heads, kv_heads, q_len, true, &per_tile, &mt);
  if (tiles == 0) {
    std::printf("%d tokens: not a pair-kernel shape\n", q_len);
    return;
  }
  const int splits = std::max(1, std::min(64, sms / (tiles * kv_heads)));

  std::vector<uint8_t> codes(size_t(slots) * kv_heads * dim / 2), scales(size_t(slots) * kv_heads * dim / 16, 0x38);
  for (auto& b : codes) b = uint8_t(rng() & 0x77);
  uint8_t *kd = upload(codes), *vd = upload(codes), *ks = upload(scales), *vs = upload(scales);
  float* g = upload(std::vector<float>{1.0f});
  const blz_nvfp4_pool pool{kd, vd, ks, vs, g, g, kv_heads, dim, BLZ_KV_NVFP4};
  std::vector<int32_t> r2t(ctx + 8);
  for (int i = 0; i < ctx + 8; ++i) r2t[i] = i % ctx;
  int32_t* d_r2t = upload(r2t);
  int32_t* d_len = upload(std::vector<int32_t>{ctx});
  int64_t* d_req = upload(std::vector<int64_t>{0});
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<__nv_bfloat16> q(size_t(q_len) * q_heads * dim);
  for (auto& x : q) x = __float2bfloat16(u(rng));
  __nv_bfloat16* d_q = upload(q);
  float *po = nullptr, *pml = nullptr;
  CHECK(cudaMalloc(&po, size_t(q_len) * q_heads * splits * dim * sizeof(float)));
  CHECK(cudaMalloc(&pml, size_t(q_len) * q_heads * splits * 2 * sizeof(float)));

  auto launch = [&] {
    CHECK(blaze::fa::verify_launch(d_q, int64_t(q_heads) * dim, dim, q_heads, pool, d_r2t, ctx + 8, {d_req, BLZ_I64},
                                   {d_len, BLZ_I32}, 1, q_len, splits, 0.0625f, f16, true, po, pml, nullptr));
  };
  for (int i = 0; i < 3; ++i) launch();
  CHECK(cudaDeviceSynchronize());
  const unsigned long long zero[blaze::fa::kProfPhases + 1] = {};
  CHECK(cudaMemcpyToSymbol(blaze::fa::g_verify_prof, zero, sizeof(zero)));
  cudaEvent_t a, e;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&e));
  const int iters = 10;
  CHECK(cudaEventRecord(a));
  for (int i = 0; i < iters; ++i) launch();
  CHECK(cudaEventRecord(e));
  CHECK(cudaEventSynchronize(e));
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, a, e));
  unsigned long long prof[blaze::fa::kProfPhases + 1];
  CHECK(cudaMemcpyFromSymbol(prof, blaze::fa::g_verify_prof, sizeof(prof)));

  const double us = 1000.0 * ms / iters;
  const double bytes = double(ctx) * kv_heads * 2 * (dim / 2 + dim / 16);
  const int64_t key_tiles = (ctx + 31) / 32, per_split = (key_tiles + splits - 1) / splits;
  const double warps = double(prof[blaze::fa::kProfPhases]);
  const double per_tile_clk = 1.0 / (warps * double(per_split));
  double total = 0;
  for (int i = 0; i < blaze::fa::kProfPhases; ++i) total += double(prof[i]);
  std::printf("pair %s, %d tokens (MT %d, %d warps a CTA), %dk context, %d splits: %.1f us (%.0f GB/s), %.0f clocks a "
              "key tile and warp at %.2f GHz\n",
              f16 ? "FP16" : "FP32", q_len, mt, 2 * mt, ctx / 1000, splits, us, bytes / us / 1e3, total * per_tile_clk,
              khz / 1e6);
  static const char* names[] = {"copies", "QK", "exchange", "softmax", "wait+bar", "PV", "bar"};
  std::printf("   ");
  for (int i = 0; i < blaze::fa::kProfPhases; ++i)
    std::printf(" %s %.0f (%.0f%%)", names[i], double(prof[i]) * per_tile_clk, 100.0 * double(prof[i]) / total);
  std::printf("\n");

  for (void* p : {static_cast<void*>(kd), static_cast<void*>(vd), static_cast<void*>(ks), static_cast<void*>(vs),
                  static_cast<void*>(g), static_cast<void*>(d_r2t), static_cast<void*>(d_len),
                  static_cast<void*>(d_req), static_cast<void*>(d_q), static_cast<void*>(po), static_cast<void*>(pml)})
    CHECK(cudaFree(p));
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(e));
}

}  // namespace

int main(int argc, char** argv) {
  const int ctx = argc > 1 ? std::atoi(argv[1]) : 524000;
  std::mt19937 rng(7);
  for (bool f16 : {true, false})
    for (int q_len : {9, 8, 3}) run(ctx, q_len, f16, rng);
  return 0;
}
