// Sampling support (blaze/sampling.h).
//
// Top-K: radix select on the bf16 keys (16 bits: two 256-bin histogram passes find the K-th largest value exactly),
// one pass collecting the elements above it and the lowest-column ties at it, a 64-element sort. Keys order by value
// descending, then column ascending, so the result is deterministic.

#include <cuda_bf16.h>

#include <cfloat>

#include "blaze/sampling.h"

namespace {

constexpr int kThreads = 1024;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t to_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16(v)); }

// a before b in the sorted (descending) order
__device__ __forceinline__ bool before(float va, int ia, float vb, int ib) {
  return va > vb || (va == vb && ia < ib);
}

// Bitonic sort of n (power of two) keys in shared memory, descending.
__device__ void bitonic_sort(float* v, int* id, int n) {
  for (int k = 2; k <= n; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int p = i ^ j;
        if (p > i) {
          const bool desc = (i & k) == 0;
          const bool swap = desc ? before(v[p], id[p], v[i], id[i]) : before(v[i], id[i], v[p], id[p]);
          if (swap) {
            const float tv = v[i];
            v[i] = v[p];
            v[p] = tv;
            const int ti = id[i];
            id[i] = id[p];
            id[p] = ti;
          }
        }
      }
      __syncthreads();
    }
  }
}

// Order-preserving 16-bit key of a bf16 value: larger value <=> larger key.
__device__ __forceinline__ uint32_t key_of(uint16_t b) { return (b & 0x8000) ? (~b & 0xffffu) : (b | 0x8000u); }

constexpr int kMaxTies = 4096;

// One block per row: radix select of the K-th largest 16-bit key (two histogram passes), then the elements above it
// plus the lowest-index ties at it, sorted by (value desc, column asc).
__global__ void __launch_bounds__(kThreads) topk_radix_kernel(const uint16_t* __restrict__ logits, int64_t ld,
                                                              int vocab, int k, int col0, float* __restrict__ vals,
                                                              int32_t* __restrict__ ids) {
  __shared__ int hist[256];
  __shared__ int sel[4];  // bin, need-in-bin, threshold key, count above threshold
  __shared__ int n_gt, n_eq;
  __shared__ float cv[BLZ_TOPK_MAX];
  __shared__ int ci[BLZ_TOPK_MAX];
  __shared__ int ties[kMaxTies];
  __shared__ float tv[kMaxTies];
  const uint16_t* row = logits + int64_t(blockIdx.x) * ld;
  const int tid = threadIdx.x;

  // pass 1: high byte
  for (int i = tid; i < 256; i += kThreads) hist[i] = 0;
  __syncthreads();
  for (int c = tid; c < vocab; c += kThreads) atomicAdd(&hist[key_of(row[c]) >> 8], 1);
  __syncthreads();
  if (tid == 0) {
    int above = 0, b = 255;
    while (b > 0 && above + hist[b] < k) above += hist[b--];
    sel[0] = b;
    sel[1] = k - above;  // elements needed from bin b
  }
  __syncthreads();
  const uint32_t hb = uint32_t(sel[0]);
  // pass 2: low byte within the bin
  for (int i = tid; i < 256; i += kThreads) hist[i] = 0;
  __syncthreads();
  for (int c = tid; c < vocab; c += kThreads) {
    const uint32_t key = key_of(row[c]);
    if ((key >> 8) == hb) atomicAdd(&hist[key & 0xff], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int need = sel[1], above = 0, lb = 255;
    while (lb > 0 && above + hist[lb] < need) above += hist[lb--];
    sel[2] = int((hb << 8) | uint32_t(lb));
    sel[3] = k - (need - above);  // elements strictly above the threshold
    n_gt = 0;
    n_eq = 0;
  }
  __syncthreads();
  const uint32_t thr = uint32_t(sel[2]);
  // pass 3: collect
  for (int c = tid; c < vocab; c += kThreads) {
    const uint16_t b = row[c];
    const uint32_t key = key_of(b);
    if (key > thr) {
      const int slot = atomicAdd(&n_gt, 1);
      if (slot < BLZ_TOPK_MAX) cv[slot] = bf(b), ci[slot] = c;
    } else if (key == thr) {
      const int slot = atomicAdd(&n_eq, 1);
      if (slot < kMaxTies) tv[slot] = bf(b), ties[slot] = c;
    }
  }
  __syncthreads();
  const int gt = min(n_gt, BLZ_TOPK_MAX), eq = min(n_eq, kMaxTies);
  // ties: lowest columns first
  int ne = 1;
  while (ne < eq) ne <<= 1;
  for (int i = eq + tid; i < ne; i += kThreads) tv[i] = -FLT_MAX, ties[i] = 0x7fffffff;
  __syncthreads();
  if (eq > 1) {
    for (int kk = 2; kk <= ne; kk <<= 1)
      for (int j = kk >> 1; j > 0; j >>= 1) {
        for (int i = tid; i < ne; i += kThreads) {
          const int p = i ^ j;
          if (p > i) {
            const bool up = (i & kk) == 0;
            if (up ? ties[i] > ties[p] : ties[i] < ties[p]) {
              const int t = ties[i];
              ties[i] = ties[p];
              ties[p] = t;
            }
          }
        }
        __syncthreads();
      }
  }
  for (int i = tid; i < k - gt && i < eq; i += kThreads) cv[gt + i] = tv[0], ci[gt + i] = ties[i];
  // tv[0] after the sort is not the tie value any more: all ties share it, take it from the threshold key
  __syncthreads();
  if (tid < k - gt && tid < eq) {
    const uint32_t key = thr;
    const uint16_t b = (key & 0x8000) ? uint16_t(key & 0x7fff) : uint16_t(~key & 0xffff);
    cv[gt + tid] = bf(b);
  }
  const int n = min(k, gt + eq);
  for (int i = n + tid; i < BLZ_TOPK_MAX; i += kThreads) cv[i] = -FLT_MAX, ci[i] = 0x7fffffff;
  __syncthreads();
  bitonic_sort(cv, ci, BLZ_TOPK_MAX);
  for (int i = tid; i < k; i += kThreads) {
    vals[int64_t(blockIdx.x) * k + i] = cv[i];
    ids[int64_t(blockIdx.x) * k + i] = ci[i] == 0x7fffffff ? -1 : ci[i] + col0;
  }
}

__global__ void pack_kernel(const float* __restrict__ vals, const int32_t* __restrict__ ids,
                            const float* __restrict__ lse, int rows, int k, float* __restrict__ pairs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * (k + 1)) return;
  const int r = i / (k + 1), j = i % (k + 1);
  if (j < k) {
    pairs[2 * i] = vals[r * k + j];
    pairs[2 * i + 1] = float(ids[r * k + j]);
  } else {
    pairs[2 * i] = lse ? lse[r] : 0.0f;
    pairs[2 * i + 1] = -2.0f;
  }
}

__global__ void penalties_kernel(uint16_t* __restrict__ logits, int64_t ld, int col0, int ncols,
                                 const int32_t* __restrict__ ids, const int32_t* __restrict__ counts,
                                 const int32_t* __restrict__ off, const float* __restrict__ pres,
                                 const float* __restrict__ freq) {
  const int row = blockIdx.x;
  const float presence = pres ? pres[row] : 0.0f, frequency = freq ? freq[row] : 0.0f;
  if (presence == 0.0f && frequency == 0.0f) return;
  for (int i = off[row] + threadIdx.x; i < off[row + 1]; i += blockDim.x) {
    const int c = ids[i] - col0;
    if (c < 0 || c >= ncols) continue;
    uint16_t* p = logits + int64_t(row) * ld + c;
    *p = to_bf(bf(*p) - presence - frequency * float(counts ? counts[i] : 1));
  }
}

__global__ void edits_kernel(uint16_t* __restrict__ logits, int64_t ld, int col0, int ncols,
                             const int32_t* __restrict__ ids, const float* __restrict__ coef,
                             const int32_t* __restrict__ off) {
  const int row = blockIdx.x;
  for (int i = off[row] + threadIdx.x; i < off[row + 1]; i += blockDim.x) {
    const int c = ids[i] - col0;
    if (c < 0 || c >= ncols) continue;
    uint16_t* p = logits + int64_t(row) * ld + c;
    const float x = bf(*p);
    *p = to_bf((x > 0.0f ? x * coef[3 * i] : x * coef[3 * i + 1]) + coef[3 * i + 2]);
  }
}

// grid (column blocks, rows): 8 columns per thread; rows without a mask leave at once
__global__ void mask_kernel(uint16_t* __restrict__ logits, int64_t ld, int col0, int ncols, int words,
                            const uint32_t* __restrict__ mask, const int32_t* __restrict__ mask_row) {
  const int row = blockIdx.y;
  const int mr = mask_row[row];
  if (mr < 0) return;
  const uint32_t* m = mask + int64_t(mr) * words;
  uint16_t* p = logits + int64_t(row) * ld;
  const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  for (int j = 0; j < 8; ++j) {
    const int c = c0 + j;
    if (c >= ncols) return;
    const int t = col0 + c;
    const bool ok = t < words * 32 && (m[t >> 5] >> (t & 31)) & 1u;
    if (!ok) p[c] = 0xFF80u;  // bf16 -inf
  }
}

__global__ void __launch_bounds__(1024) lse_kernel(const uint16_t* __restrict__ logits, int64_t ld, int ncols,
                                                    float* __restrict__ lse) {
  __shared__ float red[32];
  const uint16_t* row = logits + int64_t(blockIdx.x) * ld;
  float mx = -FLT_MAX;
  for (int c = threadIdx.x; c < ncols; c += blockDim.x) mx = fmaxf(mx, bf(row[c]));
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  mx = -FLT_MAX;
  for (int i = 0; i < int(blockDim.x >> 5); ++i) mx = fmaxf(mx, red[i]);
  __syncthreads();
  float sum = 0.0f;
  for (int c = threadIdx.x; c < ncols; c += blockDim.x) sum += __expf(bf(row[c]) - mx);
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    float t = 0.0f;
    for (int i = 0; i < int(blockDim.x >> 5); ++i) t += red[i];
    lse[blockIdx.x] = mx + logf(t);
  }
}

}  // namespace

extern "C" cudaError_t blz_logit_edits(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, const int32_t* ids,
                                       const float* coef, const int32_t* off, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  edits_kernel<<<rows, 256, 0, stream>>>(logits, ld, col0, ncols, ids, coef, off);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_token_mask(uint16_t* logits, int64_t ld, int rows, int col0, int ncols, int words,
                                      const uint32_t* mask, const int32_t* mask_row, cudaStream_t stream) {
  if (rows == 0 || ncols == 0) return cudaSuccess;
  const dim3 grid{unsigned((ncols + 2047) / 2048), unsigned(rows)};
  mask_kernel<<<grid, 256, 0, stream>>>(logits, ld, col0, ncols, words, mask, mask_row);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_row_lse(const uint16_t* logits, int64_t ld, int rows, int ncols, float* lse,
                                   cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  lse_kernel<<<rows, 1024, 0, stream>>>(logits, ld, ncols, lse);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_topk_pack(const float* vals, const int32_t* ids, const float* lse, int rows, int k,
                                     float* pairs, cudaStream_t stream) {
  const int n = rows * (k + 1);
  if (n == 0) return cudaSuccess;
  pack_kernel<<<(n + 255) / 256, 256, 0, stream>>>(vals, ids, lse, rows, k, pairs);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_apply_penalties(uint16_t* logits, int64_t ld, int rows, int col0, int ncols,
                                           const int32_t* ids, const int32_t* counts, const int32_t* off,
                                           const float* presence, const float* frequency, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  penalties_kernel<<<rows, 128, 0, stream>>>(logits, ld, col0, ncols, ids, counts, off, presence, frequency);
  return cudaGetLastError();
}

extern "C" size_t blz_topk_workspace_bytes(int rows, int vocab) {
  (void)rows;
  (void)vocab;
  return 16;
}

extern "C" cudaError_t blz_topk(const uint16_t* logits, int64_t ld, int rows, int vocab, int col0, int k, float* vals,
                                int32_t* ids, void* workspace, cudaStream_t stream) {
  (void)workspace;
  if (rows == 0) return cudaSuccess;
  if (k < 1 || k > BLZ_TOPK_MAX || k > vocab) return cudaErrorInvalidValue;
  topk_radix_kernel<<<rows, kThreads, 0, stream>>>(logits, ld, vocab, k, col0, vals, ids);
  return cudaGetLastError();
}
