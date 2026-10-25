// Pipeline-parallel hand-off through host memory (blaze/pp_link.h).

#include "blaze/pp_link.h"

#include <algorithm>

namespace {

// header words: the last step signaled, the bytes of slot 0 and 1, from word 8 the last step each receiver took
constexpr int kFlag = 0, kBytes = 2, kAck = 8, kMaxReceivers = 16;
constexpr int64_t kHeader = 256;

__global__ void set_kernel(uint64_t* counters, uint64_t v) {
  counters[0] = v;
  counters[1] = v;
}

__global__ void signal_kernel(uint64_t* hdr, uint64_t* counter, uint64_t bytes) {
  volatile uint64_t* h = hdr;
  const uint64_t v = *counter + 1;
  *counter = v;
  // an abort (bytes ~0) after the step's rows went out already: nothing to release
  if (bytes == ~0ull && h[kFlag] >= v) return;
  h[kBytes + (v - 1) % 2] = bytes;
  __threadfence_system();  // the rows before this kernel are complete; the flag goes out after them
  h[kFlag] = v;
}

__global__ void wait_kernel(const uint64_t* hdr, uint64_t* counter) {
  const uint64_t v = *counter + 1;
  *counter = v;
  while (reinterpret_cast<const volatile uint64_t*>(hdr)[kFlag] < v) __nanosleep(200);
  __threadfence_system();
}

__global__ void ack_kernel(uint64_t* hdr, const uint64_t* counter, int receiver) {
  __threadfence_system();  // the rows are read before the slot is given back
  reinterpret_cast<volatile uint64_t*>(hdr)[kAck + receiver] = *counter;
}

__global__ void wait_ack_kernel(const uint64_t* hdr, const uint64_t* counter, int receivers) {
  // the slot of step v held step v - 2: every receiver must have taken it
  const uint64_t v = *counter + 1;
  const volatile uint64_t* h = hdr;
  for (int r = 0; r < receivers; ++r)
    while (h[kAck + r] + 2 < v) __nanosleep(200);
  __threadfence_system();
}

// copy between a device buffer and the slot of the step (`send`: the next one to signal, else the one received)
__global__ void copy_kernel(uint64_t* hdr, int64_t slot_bytes, const uint64_t* counter, bool send, int64_t offset,
                            uint8_t* dev, int64_t bytes) {
  const uint64_t c = *counter;
  const uint64_t slot = send ? c % 2 : (c - 1) % 2;
  uint8_t* data = reinterpret_cast<uint8_t*>(hdr) + kHeader + int64_t(slot) * slot_bytes + offset;
  const uint8_t* src = send ? dev : data;
  uint8_t* dst = send ? data : dev;
  const int64_t n16 = bytes / 16, stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n16; i += stride)
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src)[i];
  for (int64_t i = n16 * 16 + int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < bytes; i += stride) dst[i] = src[i];
}

}  // namespace

extern "C" cudaError_t blz_pp_set(uint64_t* counters, uint64_t v, cudaStream_t stream) {
  set_kernel<<<1, 1, 0, stream>>>(counters, v);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_signal(uint64_t* flag, uint64_t* counter, uint64_t bytes, cudaStream_t stream) {
  signal_kernel<<<1, 1, 0, stream>>>(flag, counter, bytes);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_wait(const uint64_t* flag, uint64_t* counter, cudaStream_t stream) {
  wait_kernel<<<1, 1, 0, stream>>>(flag, counter);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_ack(uint64_t* flag, const uint64_t* counter, int receiver, cudaStream_t stream) {
  if (receiver < 0 || receiver >= kMaxReceivers) return cudaErrorInvalidValue;
  ack_kernel<<<1, 1, 0, stream>>>(flag, counter, receiver);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_wait_ack(const uint64_t* flag, const uint64_t* counter, int receivers, cudaStream_t stream) {
  if (receivers < 1 || receivers > kMaxReceivers) return cudaErrorInvalidValue;
  wait_ack_kernel<<<1, 1, 0, stream>>>(flag, counter, receivers);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_copy(uint64_t* flag, int64_t slot_bytes, const uint64_t* counter, int send, int64_t offset,
                                   void* dev, int64_t bytes, cudaStream_t stream) {
  if (bytes <= 0) return cudaSuccess;
  if (offset % 16 || slot_bytes % 16 || reinterpret_cast<uintptr_t>(dev) % 16) return cudaErrorInvalidValue;
  const int64_t blocks = std::min<int64_t>((bytes / 16 + 255) / 256 + 1, 1024);
  copy_kernel<<<unsigned(blocks), 256, 0, stream>>>(flag, slot_bytes, counter, send != 0, offset, static_cast<uint8_t*>(dev),
                                                    bytes);
  return cudaGetLastError();
}
