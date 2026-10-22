// Token rows from a pinned host table (blaze/embed.h): one block per token, 16-byte loads over PCIe.
#include "blaze/embed.h"

namespace {

__global__ void embed_gather_kernel(const int64_t* __restrict__ ids, const uint4* __restrict__ table, int64_t rows,
                                    int64_t vecs_per_row, uint4* __restrict__ out) {
  const int64_t i = blockIdx.x;
  const int64_t id = ids[i];
  uint4* dst = out + i * vecs_per_row;
  if (id < 0 || id >= rows) {
    for (int64_t v = threadIdx.x; v < vecs_per_row; v += blockDim.x) dst[v] = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint4* src = table + id * vecs_per_row;
  for (int64_t v = threadIdx.x; v < vecs_per_row; v += blockDim.x) dst[v] = src[v];
}

}  // namespace

extern "C" cudaError_t blz_embed_gather(const int64_t* ids, int64_t n, const uint8_t* table, int64_t rows,
                                        int64_t row_bytes, uint8_t* out, cudaStream_t stream) {
  if (n <= 0) return cudaSuccess;
  if (row_bytes % 16 != 0) return cudaErrorInvalidValue;
  embed_gather_kernel<<<static_cast<unsigned>(n), 128, 0, stream>>>(
      ids, reinterpret_cast<const uint4*>(table), rows, row_bytes / 16, reinterpret_cast<uint4*>(out));
  return cudaGetLastError();
}
