// Pipeline-parallel hand-off through host memory (blaze/pp_link.h).

#include "blaze/pp_link.h"

namespace {

__global__ void set_kernel(uint64_t* counters, uint64_t v) {
  counters[0] = v;
  counters[1] = v;
}

__global__ void signal_kernel(uint64_t* flag, uint64_t* counter, uint64_t bytes) {
  const uint64_t v = *counter + 1;
  *counter = v;
  // an abort (bytes ~0) after the step's rows went out already: nothing to release
  if (bytes == ~0ull && *reinterpret_cast<volatile uint64_t*>(flag) >= v) return;
  reinterpret_cast<volatile uint64_t*>(flag)[1] = bytes;
  __threadfence_system();  // the copy before this kernel is complete; the flag goes out after it
  *reinterpret_cast<volatile uint64_t*>(flag) = v;
}

__global__ void wait_kernel(const uint64_t* flag, uint64_t* counter) {
  const uint64_t v = *counter + 1;
  *counter = v;
  while (*reinterpret_cast<const volatile uint64_t*>(flag) < v) __nanosleep(200);
  __threadfence_system();
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
