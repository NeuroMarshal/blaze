// Pipeline-parallel hand-off between the GPUs of two stages through pinned, mapped host memory (no P2P): the sender
// copies a step's hidden rows into the link and raises its flag, the receiver waits for the flag on the GPU (a spin
// kernel: graph-capturable, no host round trip) and copies the rows in. Each side counts its steps in device memory,
// so a captured step replays with the next number.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief After the copy of `bytes` into the link (same stream): flag[1] = bytes, counter += 1, flag[0] = counter
/// (host-mapped header; a forwarder to another machine reads both).
cudaError_t blz_pp_signal(uint64_t* flag, uint64_t* counter, uint64_t bytes, cudaStream_t stream);

/// \brief counter += 1, then wait until the host-mapped flag reaches it (before the copy out of the link).
cudaError_t blz_pp_wait(const uint64_t* flag, uint64_t* counter, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
