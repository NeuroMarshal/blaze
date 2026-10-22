// GPU side of the PLE service (blaze/ple.h): publish row ids, then wait for and copy the gathered rows.
#include <cuda_runtime.h>

#include <cstdint>

#include "blaze/ple.h"

namespace blaze {
namespace {

constexpr size_t kIdsOffset = 128;

// One block: copy ids into the mailbox, fence, then release-store req = (++seq << 32) | n.
__global__ void ple_request_kernel(const int64_t* __restrict__ ids, int n, uint8_t* mail, uint32_t* seq) {
  auto* dst = reinterpret_cast<int64_t*>(mail + kIdsOffset);
  for (int i = threadIdx.x; i < n; i += blockDim.x) dst[i] = ids[i];
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t s = *seq + 1;
    *seq = s;
    auto* hdr = reinterpret_cast<PleMailHeader*>(mail);
    const uint64_t req = (uint64_t(s) << 32) | uint32_t(n);
    asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(&hdr->req), "l"(req) : "memory");
  }
}

__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// One thread waits for done == seq (a spinning kernel must not hold more than one SM slot: it may run on a side
// stream next to the model's layers).
__global__ void ple_wait_kernel(const uint8_t* mail, const uint32_t* seq) {
  const auto* hdr = reinterpret_cast<const PleMailHeader*>(mail);
  const uint32_t target = *seq;
  while (ld_acquire_sys(&hdr->done) != target) __nanosleep(64);
}

// Copy the gathered rows (16 bytes per thread) once the wait kernel has returned.
__global__ void ple_copy_kernel(uint16_t* __restrict__ out, size_t vecs, const uint8_t* mail, size_t rows_offset) {
  const auto* src = reinterpret_cast<const volatile uint4*>(mail + rows_offset);
  auto* dst = reinterpret_cast<uint4*>(out);
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < vecs; i += size_t(gridDim.x) * blockDim.x) {
    const volatile uint4& v = src[i];
    dst[i] = make_uint4(v.x, v.y, v.z, v.w);
  }
}

}  // namespace

cudaError_t ple_request_launch(const int64_t* ids, int n, uint8_t* mail, uint32_t* seq, cudaStream_t s) {
  ple_request_kernel<<<1, 256, 0, s>>>(ids, n, mail, seq);
  return cudaGetLastError();
}

cudaError_t ple_fetch_launch(uint16_t* out, int n, int row_dim, const uint8_t* mail, const uint32_t* seq,
                             size_t rows_offset, cudaStream_t s) {
  ple_wait_kernel<<<1, 1, 0, s>>>(mail, seq);
  const size_t vecs = size_t(n) * row_dim * 2 / 16;
  const int blocks = int(std::min<size_t>(64, (vecs + 255) / 256));
  if (blocks > 0) ple_copy_kernel<<<blocks, 256, 0, s>>>(out, vecs, mail, rows_offset);
  return cudaGetLastError();
}

}  // namespace blaze
