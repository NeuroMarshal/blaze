// All-reduce between the tensor-parallel ranks of one host through a shared buffer of pinned host memory, for GPUs
// without peer access (where NCCL goes through its host proxy: tens of microseconds per decode-sized reduction).
// C API (raw device pointers, caller's stream; no torch dependency).
//
// Each rank copies its tensor into its own slot of the buffer (PCIe writes), publishes a flag, waits for the flags of
// the other ranks and adds their slots (PCIe reads) to its tensor, in rank order: every rank gets the same bits. The
// kernel keeps its call counter in device memory, so it can be captured in CUDA graphs; slots alternate by the parity
// of the counter (a rank is at most one call ahead of another).
//
// Reductions of at least dma_min bytes (default 64 KiB) take the copy engines instead: tensor -> own slot of the
// buffer (D2H), the peers' slots -> device memory (H2D), then the same sum on the device. The SM reads over PCIe of
// the zero-copy path stay at ~4 GB/s whatever the block count; the copy engines run at link speed, for a fixed
// latency of a few launches. Same bits either way.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct blz_host_ar blz_host_ar;
typedef enum { BLZ_AR_BF16 = 0, BLZ_AR_FP16 = 1, BLZ_AR_FP32 = 2 } blz_ar_dtype_t;

/// \brief Map the shared buffer of a rank group and register it with the current device.
/// \param name      POSIX shared memory name ("/blz_ar_..."), the same on every rank
/// \param rank      this rank, 0 .. world - 1
/// \param world     ranks in the group (2 .. 8)
/// \param max_bytes largest reduction served (bytes of one rank's tensor, multiple of 16)
/// \param create    nonzero on exactly one rank, which must return before the others open the buffer
/// \param out       the handle
cudaError_t blz_host_ar_open(const char* name, int rank, int world, int64_t max_bytes, int create,
                             blz_host_ar** out);

/// \brief Remove the shared memory name once every rank has opened it (the mappings stay valid).
void blz_host_ar_unlink(const char* name);

/// \brief Unregister and unmap the buffer.
void blz_host_ar_close(blz_host_ar* ar);

/// \brief Reductions of at least `bytes` bytes use the DMA path (default 64 KiB; INT64_MAX: never). Every rank must set
/// the same value, between calls: a captured graph keeps the choice of its capture.
void blz_host_ar_set_dma_min(blz_host_ar* ar, int64_t bytes);

/// \brief In-place sum of `data` over the group. Every rank must issue the same sequence of calls with equal sizes.
/// \param data  device tensor, `bytes` contiguous bytes (multiple of 16, at most max_bytes)
/// \return cudaErrorInvalidValue for an unsupported size, else the launch error. A rank that waits for more than
///         ~5 minutes on another one traps (sticky device error) instead of hanging. Sizes above 4 MiB on the DMA path
///         are split into equal chunks, one DMA call each.
cudaError_t blz_host_ar_all_reduce(blz_host_ar* ar, void* data, int64_t bytes, blz_ar_dtype_t dtype,
                                   cudaStream_t stream);

/// \brief Decode form of all-reduce + residual add + RMSNorm in one kernel (one block per row, rows <= 64): x [rows, dim]
/// is summed over the group, residual += sum (bf16), out = norm(residual) * (zero_centered ? 1 + w : w). Same bits as
/// blz_host_ar_all_reduce followed by blz_rmsnorm; x is left unchanged. Takes the DMA path like the all-reduce when
/// rows * dim * 2 is in [dma_min, 4 MiB].
cudaError_t blz_host_ar_rmsnorm(blz_host_ar* ar, const uint16_t* x, uint16_t* residual, const uint16_t* w, float eps,
                                int rows, int dim, int zero_centered, uint16_t* out, cudaStream_t stream);

#ifdef __cplusplus
}
#endif
