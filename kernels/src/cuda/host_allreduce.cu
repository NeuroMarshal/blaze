// All-reduce through a shared buffer of pinned host memory (see blaze/host_allreduce.h).
//
// Buffer layout: flags [world][kMaxBlocks] (uint32, the call counter a block of a rank has published), then the
// slots [world][2][max_bytes]. Block b of every rank handles the same byte range, so a block only waits for block b of
// the other ranks: blocks need not be co-resident. The call counter (slot parity) is one per call, not per block: calls
// of different sizes split the slots differently, and a block must never write the half another rank may still be
// reading through a different block of the previous call.
//
// DMA path (calls of at least dma_min bytes): SM loads over PCIe reach only ~4 GB/s whatever the block count, the copy
// engines run at link speed. A second region of the buffer holds one slot [world][kDmaChunk] per rank, filled and
// drained by cudaMemcpyAsync. Per call, in stream order (every step is a kernel or a copy: no host sync, so it is
// graph-capturable):
//   K1  wait until every peer has read my slot of my previous DMA call (consumed[r] >= ctl[2])
//   D2H my tensor -> my slot
//   K2  ready[me] = seq, wait for ready[r] >= seq of every peer
//   H2D every peer's slot -> tmp (device memory)
//   K3  consumed[me] = seq (the H2D copies are complete), sum in rank order from device memory, ctl[2] = ctl[0] = seq
// Visibility. Slot -> flag: K2 starts only after the D2H copy has completed, i.e. its writes have reached host memory;
// the flag store follows them on the same link (posted PCIe writes of one function stay in order), and the peer reads
// the slot only after it has seen the flag. Flag -> slot: the peer's K2 acquires the flag before the H2D launches, and
// the copy reads host memory, which is coherent with the other GPU's completed writes. consumed has the opposite
// direction: K3 runs after the H2D copies have finished reading, so a peer that sees consumed may overwrite the slot.
// K1 waits for ctl[2] (the seq of my last DMA call), not for seq - 1: zero-copy calls in between never set consumed.
//
// Mixing with zero-copy calls (all share the call counter ctl[0], so the parity of a zero-copy call is the counter's).
// The zero-copy rule is: a rank that has finished call n-1 knows every peer has finished call n-2, so call n may reuse
// the parity half of n-2 (it is the only reuse that happens: slot halves alternate by n & 1). That holds for both kinds
// of call, since each one waits for every peer to have *started* it (zero-copy: block flags of seq; DMA: ready of seq),
// and a peer starts call n only after its call n-1 has completed in stream order. A DMA call between two zero-copy
// calls (seq s-1, s, s+1: parity of s-1 reused at s+1) is therefore safe: finishing the DMA call s (K2) proves the
// peers have finished s-1 and so have stopped reading the zero-copy slot half that s+1 overwrites. Zero-copy flags
// are set by zero-copy calls only, so a stale flag (< seq) never satisfies a later wait; DMA flags likewise only by
// DMA calls, and waits are ">=" on wrapping counters.

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
  // Device memory: {calls completed, blocks finished in the current call, seq of the last DMA call, unused}.
  uint32_t* ctl = nullptr;
  uint8_t* tmp = nullptr;   // device memory: the peers' DMA slots of the current call, (world - 1) x kDmaChunk
  int rank = 0, world = 0;
  int64_t max_bytes = 0;
  int64_t dma_off = 0;      // start of the DMA slots in the shared buffer
  int64_t dma_min = 0;      // calls of at least this many bytes take the DMA path
};

namespace blaze {
namespace {

constexpr int kMaxBlocks = 64;
constexpr int kThreads = 256;
constexpr int64_t kBytesPerBlock = 4096;  // one 16-byte pack per thread and pass
constexpr size_t kFlagBytes = 4096;       // >= 8 ranks x kMaxBlocks x 4, + the DMA flags below
// DMA flags, in the flag region after the zero-copy block flags (8 ranks x kMaxBlocks words).
constexpr int kReadyWord = 512, kConsumedWord = 528;
static_assert(8 * kMaxBlocks <= kReadyWord && kConsumedWord + 8 <= kFlagBytes / 4, "flag words overlap");
constexpr int64_t kDmaChunk = 4 << 20;    // bytes per rank slot; larger calls loop over chunks, one DMA call each
constexpr int64_t kDmaMin = 64 << 10;     // default dma_min: below this the zero-copy path is faster (fixed copy latency)
constexpr int kDmaMaxBlocks = 256;
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

// One thread: wait for flag[r] >= seq of every other rank.
__device__ void wait_peers(const uint32_t* flag, int rank, int world, uint32_t seq) {
  for (int r = 0; r < world; ++r)
    if (r != rank) wait_for(flag + r, seq);
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

// DMA path, K1 (see the header comment): my DMA slot may be overwritten once every peer has read it.
__global__ void host_ar_dma_acquire_kernel(uint8_t* buf, const uint32_t* ctl, int rank, int world) {
  const uint32_t last = *reinterpret_cast<const volatile uint32_t*>(ctl + 2);
  wait_peers(reinterpret_cast<const uint32_t*>(buf) + kConsumedWord, rank, world, last);
}

// K2: my slot is filled (the D2H copy before this kernel is complete); wait for the peers' slots.
__global__ void host_ar_dma_publish_kernel(uint8_t* buf, const uint32_t* ctl, int rank, int world) {
  const uint32_t seq = *reinterpret_cast<const volatile uint32_t*>(ctl) + 1;
  uint32_t* flags = reinterpret_cast<uint32_t*>(buf);
  __threadfence_system();
  store_release_sys(flags + kReadyWord + rank, seq);
  wait_peers(flags + kReadyWord, rank, world, seq);
}

// K3: the peers' slots are in tmp [world - 1][kDmaChunk] (rank order, own rank skipped). Same sum as host_ar_kernel.
// Plain grid-stride loop: blocks never wait for each other here.
template <typename T>
__global__ void __launch_bounds__(kThreads) host_ar_dma_reduce_kernel(uint8_t* buf, uint32_t* ctl, int rank, int world,
                                                                      uint4* data, const uint4* tmp, int64_t packs) {
  __shared__ uint32_t s_seq;
  const int tid = threadIdx.x;
  if (tid == 0) s_seq = *reinterpret_cast<volatile uint32_t*>(ctl) + 1;
  __syncthreads();
  const uint32_t seq = s_seq;
  // The H2D copies before this kernel have finished reading the peers' slots: they may be refilled.
  if (blockIdx.x == 0 && tid == 0) store_release_sys(reinterpret_cast<uint32_t*>(buf) + kConsumedWord + rank, seq);

  constexpr int n = Pack<T>::n;
  for (int64_t p = int64_t(blockIdx.x) * kThreads + tid; p < packs; p += int64_t(gridDim.x) * kThreads) {
    float acc[n], v[n];
    const uint4 own = data[p];
    for (int r = 0; r < world; ++r) {
      Pack<T>::to_float(r == rank ? own : tmp[int64_t(r - (r > rank)) * (kDmaChunk / 16) + p], r == 0 ? acc : v);
      if (r > 0)
#pragma unroll
        for (int i = 0; i < n; ++i) acc[i] += v[i];
    }
    data[p] = Pack<T>::from_float(acc);
  }
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(ctl + 1, 1u) == gridDim.x - 1) {
      ctl[1] = 0;
      *reinterpret_cast<volatile uint32_t*>(ctl + 2) = seq;
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

// kDma: the peers' rows come from tmp (device memory, see host_ar_dma_reduce_kernel) instead of the zero-copy slots,
// and the kernel starts after the exchange (no store, no flags but consumed).
template <bool kDma>
__global__ void __launch_bounds__(kNormMaxThreads) host_ar_rmsnorm_kernel(uint8_t* buf, uint32_t* ctl, int rank, int world,
                                                                       int64_t max_bytes, const uint4* tmp,
                                                                       const uint16_t* x, uint16_t* residual,
                                                                       const uint16_t* w, float eps, int dim,
                                                                       int zero_centered, uint16_t* out) {
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
  auto peer = [&](int r, int p) -> uint4 {
    if constexpr (kDma) return tmp[int64_t(r - (r > rank)) * (kDmaChunk / 16) + int64_t(b) * packs + p];
    else return load_sys(slot(r) + p);
  };
  const uint4* xr = reinterpret_cast<const uint4*>(x + int64_t(b) * dim);
  if constexpr (kDma) {
    if (b == 0 && tid == 0) store_release_sys(flags + kConsumedWord + rank, seq);  // as in host_ar_dma_reduce_kernel
  } else {
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
  }
  uint16_t* res = residual + int64_t(b) * dim;
  float ss = 0.0f;
  for (int p = tid; p < packs; p += nt) {
    float acc[8], v[8];
    const uint4 own = xr[p];
    for (int r = 0; r < world; ++r) {
      Pack<__nv_bfloat16>::to_float(r == rank ? own : peer(r, p), r == 0 ? acc : v);
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
      if constexpr (kDma) *reinterpret_cast<volatile uint32_t*>(ctl + 2) = seq;
      *reinterpret_cast<volatile uint32_t*>(ctl) = seq;
    }
  }
}

// The exchange of the DMA path (K1, D2H, K2, H2D) for `bytes` (<= kDmaChunk) at `src`: afterwards tmp holds the peers'
// bytes. K3, the caller's kernel, follows in the same stream.
cudaError_t dma_exchange(blz_host_ar* ar, const void* src, int64_t bytes, cudaStream_t stream) {
  uint8_t* host = static_cast<uint8_t*>(ar->host) + ar->dma_off;
  host_ar_dma_acquire_kernel<<<1, 1, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world);
  cudaError_t err = cudaGetLastError();
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(host + int64_t(ar->rank) * kDmaChunk, src, size_t(bytes), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) return err;
  host_ar_dma_publish_kernel<<<1, 1, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world);
  err = cudaGetLastError();
  for (int r = 0, i = 0; r < ar->world && err == cudaSuccess; ++r)
    if (r != ar->rank)
      err = cudaMemcpyAsync(ar->tmp + int64_t(i++) * kDmaChunk, host + int64_t(r) * kDmaChunk, size_t(bytes),
                            cudaMemcpyHostToDevice, stream);
  return err;
}

template <typename T>
cudaError_t dma_all_reduce(blz_host_ar* ar, uint4* data, int64_t bytes, cudaStream_t stream) {
  cudaError_t err = dma_exchange(ar, data, bytes, stream);
  if (err != cudaSuccess) return err;
  const int64_t packs = bytes / 16;
  const int blocks = int(std::min<int64_t>(kDmaMaxBlocks, (packs + kThreads - 1) / kThreads));
  host_ar_dma_reduce_kernel<T><<<blocks, kThreads, 0, stream>>>(ar->dev, ar->ctl, ar->rank, ar->world, data,
                                                                reinterpret_cast<const uint4*>(ar->tmp), packs);
  return cudaGetLastError();
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
  const int64_t bytes = int64_t(rows) * dim * 2;
  // Beyond one chunk (not reachable with rows <= 64 at hidden sizes) the zero-copy kernel stays.
  if (bytes >= ar->dma_min && bytes <= blaze::kDmaChunk) {
    const cudaError_t err = blaze::dma_exchange(ar, x, bytes, stream);
    if (err != cudaSuccess) return err;
    blaze::host_ar_rmsnorm_kernel<true><<<rows, threads, size_t(dim) * sizeof(float), stream>>>(
        ar->dev, ar->ctl, ar->rank, ar->world, ar->max_bytes, reinterpret_cast<const uint4*>(ar->tmp), x, residual, w,
        eps, dim, zero_centered, out);
    return cudaGetLastError();
  }
  blaze::host_ar_rmsnorm_kernel<false><<<rows, threads, size_t(dim) * sizeof(float), stream>>>(
      ar->dev, ar->ctl, ar->rank, ar->world, ar->max_bytes, nullptr, x, residual, w, eps, dim, zero_centered, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_host_ar_open(const char* name, int rank, int world, int64_t max_bytes, int create,
                                        blz_host_ar** out) {
  *out = nullptr;
  if (world < 2 || world > 8 || rank < 0 || rank >= world || max_bytes <= 0 || max_bytes % 16 != 0)
    return cudaErrorInvalidValue;
  // The DMA slots start on a page boundary.
  const int64_t dma_off = (int64_t(blaze::kFlagBytes) + int64_t(world) * 2 * max_bytes + 4095) / 4096 * 4096;
  const size_t size = size_t(dma_off) + size_t(world) * size_t(blaze::kDmaChunk);
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
  ar->dma_off = dma_off, ar->dma_min = blaze::kDmaMin;
  void* dev = nullptr;
  cudaError_t err = cudaHostRegister(host, size, cudaHostRegisterMapped | cudaHostRegisterPortable);
  if (err == cudaSuccess) err = cudaHostGetDevicePointer(&dev, host, 0);
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&ar->ctl), 4 * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMemset(ar->ctl, 0, 4 * sizeof(uint32_t));
  if (err == cudaSuccess) err = cudaMalloc(reinterpret_cast<void**>(&ar->tmp), size_t(world - 1) * size_t(blaze::kDmaChunk));
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
  if (ar->tmp) cudaFree(ar->tmp);
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
  if (bytes >= ar->dma_min) {
    // Equal chunks of at most kDmaChunk, each a DMA call of its own (a rank has no use for a tiny tail call).
    const int64_t chunks = (bytes + blaze::kDmaChunk - 1) / blaze::kDmaChunk;
    const int64_t per = (bytes / chunks + 15) / 16 * 16;
    for (int64_t off = 0; off < bytes; off += per) {
      uint4* d = reinterpret_cast<uint4*>(static_cast<uint8_t*>(data) + off);
      const int64_t n = std::min(per, bytes - off);
      cudaError_t err;
      switch (dtype) {
        case BLZ_AR_BF16: err = blaze::dma_all_reduce<__nv_bfloat16>(ar, d, n, stream); break;
        case BLZ_AR_FP16: err = blaze::dma_all_reduce<__half>(ar, d, n, stream); break;
        case BLZ_AR_FP32: err = blaze::dma_all_reduce<float>(ar, d, n, stream); break;
        default: return cudaErrorInvalidValue;
      }
      if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
  }
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

extern "C" void blz_host_ar_set_dma_min(blz_host_ar* ar, int64_t bytes) {
  if (ar != nullptr) ar->dma_min = std::max<int64_t>(bytes, 16);
}
