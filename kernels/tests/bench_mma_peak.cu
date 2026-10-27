// Tensor-core peak of this GPU per MMA flavour (independent chains, registers only): what an attention kernel can
// reach. bf16 / f16 with FP32 accumulation, f16 with FP16 accumulation, e4m3 with FP32 accumulation (SM89+).
//   bench_mma_peak
#include <cuda_runtime.h>

#include <cstdint>

#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { \
  const cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)

namespace {

constexpr int kChains = 8, kIters = 4096;

template <int K>
__global__ void peak(float* out) {
  uint32_t a[4] = {threadIdx.x, threadIdx.x * 3u, threadIdx.x * 5u, threadIdx.x * 7u}, b[2] = {threadIdx.x * 11u, 13u};
  float c[kChains][4] = {};
  uint32_t h[kChains][2] = {};
  for (int it = 0; it < kIters; ++it) {
#pragma unroll
    for (int j = 0; j < kChains; ++j) {
      if constexpr (K == 0)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else if constexpr (K == 1)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else if constexpr (K == 2)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                     : "+r"(h[j][0]), "+r"(h[j][1])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
      else {
#if __CUDA_ARCH__ >= 890
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
      }
    }
  }
  float s = 0;
#pragma unroll
  for (int j = 0; j < kChains; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3] + float(h[j][0] & 1) + float(h[j][1] & 1);
  if (s == 12345.0f) out[threadIdx.x] = s;
}

template <int K>
void run(const char* name, double flops_per_mma) {
  int sms = 0;
  CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
  float* out = nullptr;
  CHECK(cudaMalloc(&out, 1024 * sizeof(float)));
  const dim3 grid(sms * 8), block(256);
  peak<K><<<grid, block>>>(out);
  CHECK(cudaGetLastError());
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  CHECK(cudaEventRecord(a));
  peak<K><<<grid, block>>>(out);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(b));
  CHECK(cudaEventSynchronize(b));
  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, a, b));
  const double mmas = double(grid.x) * (block.x / 32) * kChains * kIters;
  std::printf("%-26s %7.1f TFLOPS\n", name, mmas * flops_per_mma / (ms * 1e-3) / 1e12);
  CHECK(cudaEventDestroy(a));
  CHECK(cudaEventDestroy(b));
  CHECK(cudaFree(out));
}

}  // namespace

int main() {
  cudaDeviceProp p{};
  CHECK(cudaGetDeviceProperties(&p, 0));
  std::printf("%s, SM %d.%d, %d SMs\n", p.name, p.major, p.minor, p.multiProcessorCount);
  run<0>("bf16 x bf16 -> f32", 2.0 * 16 * 8 * 16);
  run<1>("f16 x f16 -> f32", 2.0 * 16 * 8 * 16);
  run<2>("f16 x f16 -> f16", 2.0 * 16 * 8 * 16);
  if (p.major * 10 + p.minor >= 89) run<3>("e4m3 x e4m3 -> f32", 2.0 * 16 * 8 * 32);
  return 0;
}
