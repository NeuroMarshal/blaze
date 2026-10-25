// Pipeline-parallel hand-off between the GPUs of two stages through pinned, mapped host memory (no P2P). The link is
// a 256-byte header (u64 words: 0 the last step signaled, 2 and 3 the bytes of slot 0 and of slot 1, from 8 the last
// step each receiver took: the tensor-parallel ranks of the next stage all read the rows) and two slots of rows: step
// v's rows go to slot (v - 1) % 2, so the sender fills the next step's slot while the receivers still read the last one
// (a split prefill chunk runs on both stages at once). The sender waits until the slot it fills was taken
// (blz_pp_wait_ack), copies the rows in (blz_pp_copy) and raises the flag (blz_pp_signal); a receiver waits for the
// flag on the GPU (a spin kernel: graph-capturable, no host round trip), copies the rows out and gives the slot back
// (blz_pp_ack). The step's number is set into the device counters before each step (blz_pp_set: the host counts the
// steps every stage is sent, so a step that failed on one stage leaves none a step behind); a captured step reads it
// there, slots included.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// \brief counters[0] = counters[1] = v (the steps before this one).
cudaError_t blz_pp_set(uint64_t* counters, uint64_t v, cudaStream_t stream);

/// \brief After the rows of the step's slot (same stream): its bytes, counter += 1, flag = counter (host-mapped header;
/// a forwarder to another machine reads them). bytes ~0: an abort, no rows (skipped when the flag is there already).
cudaError_t blz_pp_signal(uint64_t* flag, uint64_t* counter, uint64_t bytes, cudaStream_t stream);

/// \brief counter += 1, then wait until the host-mapped flag reaches it (before the copy out of the link).
cudaError_t blz_pp_wait(const uint64_t* flag, uint64_t* counter, cudaStream_t stream);

/// \brief Receiver `receiver` (< 16) took the rows of step `*counter` (after its copies).
cudaError_t blz_pp_ack(uint64_t* flag, const uint64_t* counter, int receiver, cudaStream_t stream);

/// \brief The sender waits until every one of `receivers` gave back the slot of its next step (`*counter + 1`).
cudaError_t blz_pp_wait_ack(const uint64_t* flag, const uint64_t* counter, int receivers, cudaStream_t stream);

/// \brief Copy `bytes` between `dev` and the link at `offset` within the step's slot of `slot_bytes`: send (1) into the
/// slot of the next step to signal (`*counter + 1`), else out of the slot of the step received (`*counter`). offset,
/// slot_bytes and dev 16-byte aligned.
cudaError_t blz_pp_copy(uint64_t* flag, int64_t slot_bytes, const uint64_t* counter, int send, int64_t offset, void* dev,
                        int64_t bytes, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
