// All-reduce through a shared buffer of pinned host memory (see blaze/host_allreduce.h).
//
// Buffer layout: flags [world][kMaxBlocks] (uint32, the call counter a block of a rank has published), then the
// slots [world][2][max_bytes]. Block b of every rank handles the same byte range, so a block only waits for block b of
// the other ranks: blocks need not be co-resident. The call counter (slot parity) is one per call, not per block: calls
// of different sizes split the slots differently, and a block must never write the half another rank may still be
// reading through a different block of the previous call.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <new>

#include "blaze/host_allreduce.h"

struct blz_host_ar {
  void* host = nullptr;
  size_t size = 0;
  uint8_t* dev = nullptr;   // device view of `host`
  uint32_t* ctl = nullptr;   // device memory: {calls completed, blocks finished in the current call}
  int rank = 0, world = 0;
  int64_t max_bytes = 0;
};

namespace blaze {
namespace {

constexpr int kMaxBlocks = 64;
constexpr int kThreads = 256;
constexpr int64_t kBytesPerBlock = 4096;  // one 16-byte pack per thread and pass
constexpr size_t kFlagBytes = 4096;       // >= 8 ranks x kMaxBlocks x 4
// Only against hanging forever: ranks drift apart by seconds outside the steady state (loading, warmup, compiles).
constexpr unsigned long long kTimeoutNs = 300ull * 1000 * 1000 * 1000;

__device__ __forceinline__ void store_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void store_sys(uint4* p, const uint4& v) {
  asm volatile("st.relaxed.sys.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
               : "memory");
}

__device__ __forceinline__ uint4 load_sys(const uint4* p) {
  uint4 v;
  asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ unsigned long long global_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Another rank is at call `seq` or already at seq + 1 (it cannot pass seq + 1 before this rank publishes seq + 1).
__device__ void wait_for(const uint32_t* flag, uint32_t seq) {
  const unsigned long long start = global_ns();
  for (int spin = 0;; ++spin) {
    if (int32_t(load_acquire_sys(flag) - seq) >= 0) return;
    if ((spin & 1023) == 1023 && global_ns() - start > kTimeoutNs) __trap();
  }
}

template <typename T>
struct Pack;
template <>
struct Pack<__nv_bfloat16> {
  static constexpr int n = 8;
  static __device__ void to_float(const uint4& v, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x = __bfloat1622float2(h[i]);
      f[2 * i] = x.x, f[2 * i + 1] = x.y;
    }
  }
  static __device__ uint4 from_float(const float* f) {
    uint4 v;
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
};
template <>
struct Pack<__half> {
  static constexpr int n = 8;
  static __device__ void to_float(const uint4& v, float* f) {
    const __half2* h = reinterpret_cast<const __half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 x = __half22float2(h[i]);
      f[2 * i] = x.x, f[2 * i + 1] = x.y;
    }
  }
  static __device__ uint4 from_float(const float* f) {
    uint4 v;
    __half2* h = reinterpret_cast<__half2*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
    return v;
  }
};
template <>
struct Pack<float> {
  static constexpr int n = 4;
  static __device__ void to_float(const uint4& v, float* f) {
    f[0] = __uint_as_float(v.x), f[1] = __uint_as_float(v.y), f[2] = __uint_as_float(v.z), f[3] = __uint_as_float(v.w);
  }
  static __device__ uint4 from_float(const float* f) {
    return make_uint4(__float_as_uint(f[0]), __float_as_uint(f[1]), __float_as_uint(f[2]), __float_as_uint(f[3]));
  }
};

template <typename T>
__global__ void __launch_bounds__(kThreads) host_ar_kernel(uint8_t* buf, uint32_t* ctl, int rank, int world,
                                                           int64_t max_bytes, uint4* data, int64_t packs) {
  const int b = blockIdx.x, tid = threadIdx.x;
  __shared__ uint32_t s_seq;
  if (tid == 0) s_seq = *reinterpret_cast<volatile uint32_t*>(ctl) + 1;
  __syncthreads();
  const uint32_t seq = s_seq;
  uint32_t* flags = reinterpret_cast<uint32_t*>(buf);
  auto slot = [&](int r) {
    return reinterpret_cast<uint4*>(buf + kFlagBytes + (int64_t(r) * 2 + (seq & 1)) * max_bytes);
  };
  const int64_t per = (packs + gridDim.x - 1) / gridDim.x;
  const int64_t p0 = min(packs, int64_t(b) * per), p1 = min(packs, p0 + per);

  uint4* mine = slot(rank);
  for (int64_t p = p0 + tid; p < p1; p += kThreads) store_sys(mine + p, data[p]);
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    store_release_sys(flags + rank * kMaxBlocks + b, seq);
    for (int r = 0; r < world; ++r)
      if (r != rank) wait_for(flags + r * kMaxBlocks + b, seq);
  }
  __syncthreads();

  constexpr int n = Pack<T>::n;
  for (int64_t p = p0 + tid; p < p1; p += kThreads) {
    float acc[n], v[n];
    const uint4 own = data[p];
    for (int r = 0; r < world; ++r) {
      // Strong (sys-scope) loads: ordered after thread 0's acquire of the flag, never a stale cached line.
      Pack<T>::to_float(r == rank ? own : load_sys(slot(r) + p), r == 0 ? acc : v);
      if (r > 0)
#pragma unroll
        for (int i = 0; i < n; ++i) acc[i] += v[i];
    }
    data[p] = Pack<T>::from_float(acc);
  }
  // The last block to finish advances the counter: every block has read it by then.
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(ctl + 1, 1u) == gridDim.x - 1) {
      ctl[1] = 0;
      *reinterpret_cast<volatile uint32_t*>(ctl) = seq;
    }
  }
}

// All-reduce of a few rows fused with the residual add and RMSNorm that follow it (decode): one block per row, the
// same rounding steps as host_ar_kernel + rmsnorm (sum in rank order -> bf16, + residual -> bf16, normalized).
// One thread per 16-byte pack of the row, so the PCIe reads of the other ranks' rows go out in one round.
constexpr int kNormMaxThreads = 1024;

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t f2bf(float v) { return __bfloat16_as_ushort(__float2bfloat16(v)); }

__global__ void __launch_bounds__(kNormMaxThreads) host_ar_rmsnorm_kernel(uint8_t* buf, uint32_t* ctl, int rank, int world,
                                                                       int64_t max_bytes, const uint16_t* x,
                                                                       uint16_t* residual, const uint16_t* w,
                                                                       float eps, int dim, int zero_centered,
                                                                       uint16_t* out) {
  extern __shared__ float h[];
  __shared__ float red[kNormMaxThreads / 32];
  __shared__ uint32_t s_seq;
  const int b = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
  if (tid == 0) s_seq = *reinterpret_cast<volatile uint32_t*>(ctl) + 1;
  __syncthreads();
  const uint32_t seq = s_seq;
  uint32_t* flags = reinterpret_cast<uint32_t*>(buf);
  const int packs = dim / 8;  // 8 bf16 per 16-byte pack
  auto slot = [&](int r) {
    return reinterpret_cast<uint4*>(buf + kFlagBytes + (int64_t(r) * 2 + (seq & 1)) * max_bytes) + int64_t(b) * packs;
  };
  const uint4* xr = reinterpret_cast<const uint4*>(x + int64_t(b) * dim);
  uint4* mine = slot(rank);
  for (int p = tid; p < packs; p += nt) store_sys(mine + p, xr[p]);
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    store_release_sys(flags + rank * kMaxBlocks + b, seq);
    for (int r = 0; r < world; ++r)
      if (r != rank) wait_for(flags + r * kMaxBlocks + b, seq);
  }
  __syncthreads();
  uint16_t* res = residual + int64_t(b) * dim;
  float ss = 0.0f;
  for (int p = tid; p < packs; p += nt) {
    float acc[8], v[8];
    const uint4 own = xr[p];
    for (int r = 0; r < world; ++r) {
      Pack<__nv_bfloat16>::to_float(r == rank ? own : load_sys(slot(r) + p), r == 0 ? acc : v);
      if (r > 0)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] += v[i];
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = p * 8 + i;
      const uint16_t sum = f2bf(bf16f(f2bf(acc[i])) + bf16f(res[e]));
      res[e] = sum;
      const float hv = bf16f(sum);
      h[e] = hv;
      ss += hv * hv;
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) red[tid >> 5] = ss;
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < kNormMaxThreads / 32; ++i)
    if (i < nt / 32) total += red[i];
  const float inv = rsqrtf(total / float(dim) + eps);
  for (int e = tid; e < dim; e += nt) {
    const float wi = zero_centered ? 1.0f + bf16f(w[e]) : bf16f(w[e]);
    out[int64_t(b) * dim + e] = f2bf(h[e] * inv * wi);
  }
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(ctl + 1, 1u) == gridDim.x - 1) {
      ctl[1] = 0;
      *reinterpret_cast<volatile uint32_t*>(ctl) = seq;
    }
  }
}

}  // namespace
}  // namespace blaze

extern "C" cudaError_t blz_host_ar_rmsnorm(blz_host_ar* ar, const uint16_t* x, uint16_t* residual, const uint16_t* w,
                                           float eps, int rows, int dim, int zero_centered, uint16_t* out,
                                           cudaStream_t stream) {
  if (ar == nullptr || rows <= 0 || rows > blaze::kMaxBlocks || dim % 8 != 0 ||
      int64_t(rows) * dim * 2 > ar->max_bytes || reinterpret_cast<uintptr_t>(x) % 16 != 0)
    return cudaErrorInvalidValue;
  const int threads = std::min(blaze::kNormMaxThreads, (dim / 8 + 31) / 32 * 32);
  blaze::host_ar_rmsnorm_kernel<<<rows, threads, size_t(dim) * sizeof(float), stream>>>(
      ar->dev, ar->ctl, ar->rank, ar->world, ar->max_bytes, x, residual, w, eps, dim, zero_centered, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_host_ar_open(const char* name, int rank, int world, int64_t max_bytes, int create,
                                        blz_host_ar** out) {
  *out = nullptr;
  if (world < 2 || world > 8 || rank < 0 || rank >= world || max_bytes <= 0 || max_bytes % 16 != 0)
    return cudaErrorInvalidValue;
  const size_t size = blaze::kFlagBytes + size_t(world) * 2 * size_t(max_bytes);
  const int fd = shm_open(name, create ? O_CREAT | O_EXCL | O_RDWR : O_RDWR, 0600);
  if (fd < 0) return cudaErrorInvalidValue;
  if (create && ftruncate(fd, off_t(size)) != 0) {
    close(fd);
    shm_unlink(name);
    return cudaErrorMemoryAllocation;
  }
  void* host = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  close(fd);
  if (host == MAP_FAILED) return cudaErrorMemoryAllocation;
  auto* ar = new (std::nothrow) blz_host_ar;
  if (ar == nullptr) {
    munmap(host, size);
    return cudaErrorMemoryAllocation;
  }
  ar->host = host, ar->size = size, ar->rank = rank, ar->world = world, ar->max_bytes = max_bytes;
  void* dev = nullptr;
  cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err == cudaSuccess) err = cudaHostGetDevicePointer(&dev, host, 0);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&ar->ctl), 2 * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMemset(ar->ctl, 0, 2 * sizeof(uint32_t));
  if (err != cudaSuccess) {
    blz_host_ar_close(ar);
    return err;
  }
  ar->dev = static_cast<uint8_t*>(dev);
  *out = ar;
  return cudaSuccess;
}

extern "C" void blz_host_ar_unlink(const char* name) { shm_unlink(name); }

extern "C" void blz_host_ar_close(blz_host_ar* ar) {
  if (ar == nullptr) return;
  if (ar->ctl) cudaFree(ar->ctl);
  if (ar->host) {
    cudaHostUnregister(ar->host);
    munmap(ar->host, ar->size);
  }
  delete ar;
}

extern "C" cudaError_t blz_host_ar_all_reduce(blz_host_ar* ar, void* data, int64_t bytes, blz_ar_dtype_t dtype,
                                              cudaStream_t stream) {
  if (ar == nullptr || bytes % 16 != 0 || bytes > ar->max_bytes || reinterpret_cast<uintptr_t>(data) % 16 != 0)
    return cudaErrorInvalidValue;
  if (bytes == 0) return cudaSuccess;
  const int64_t packs = bytes / 16;
  const int blocks = int(std::min<int64_t>(blaze::kMaxBlocks, (bytes + blaze::kBytesPerBlock - 1) / blaze::kBytesPerBlock));
  uint4* d = static_cast<uint4*>(data);
  switch (dtype) {
    case BLZ_AR_BF16:
      blaze::host_ar_kernel<__nv_bfloat16><<<blocks, blaze::kThreads, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world,
                                                                              ar->max_bytes, d, packs);
      break;
    case BLZ_AR_FP16:
      blaze::host_ar_kernel<__half><<<blocks, blaze::kThreads, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world,
                                                                       ar->max_bytes, d, packs);
      break;
    case BLZ_AR_FP32:
      blaze::host_ar_kernel<float><<<blocks, blaze::kThreads, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world,
                                                                      ar->max_bytes, d, packs);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}
