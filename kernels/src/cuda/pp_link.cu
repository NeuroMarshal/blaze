// Pipeline-parallel hand-off through host memory (blaze/pp_link.h).

#include "blaze/pp_link.h"

namespace {

__global__ void signal_kernel(uint64_t* flag, uint64_t* counter) {
  const uint64_t v = *counter + 1;
  *counter = v;
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

extern "C" cudaError_t blz_pp_signal(uint64_t* flag, uint64_t* counter, cudaStream_t stream) {
  signal_kernel<<<1, 1, 0, stream>>>(flag, counter);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_pp_wait(const uint64_t* flag, uint64_t* counter, cudaStream_t stream) {
  wait_kernel<<<1, 1, 0, stream>>>(flag, counter);
  return cudaGetLastError();
}
