#include "blaze/sampling.h"
#include <cuda_bf16.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cmath>
#include <cstdint>
#include <climits>

namespace {
constexpr int Threads = 256, K = BLZ_TOPK_MAX;
struct Offset {
  int width;
  __host__ __device__ int operator()(int i) const { return i * width; }
};
using Offsets = thrust::transform_iterator<Offset, thrust::counting_iterator<int>>;
Offsets offsets(int width, int start = 0) { return Offsets(thrust::counting_iterator<int>(start), Offset{width}); }
size_t aligned(size_t n) { return (n + 255) & ~size_t(255); }
struct Work {
  float *values, *canonical, *sorted;
  int *ids, *sorted_ids, *counts, *status;
  void* sort_space;
  size_t sort_bytes;
  Work(void* ptr, size_t bytes, int rows, int vocab) {
    auto* p = static_cast<char*>(ptr);
    auto take = [&](size_t n) { char* out = p; p += aligned(n); return out; };
    size_t n = size_t(rows) * vocab;
    values = reinterpret_cast<float*>(take(n * 4));
    canonical = reinterpret_cast<float*>(take(n * 4));
    sorted = reinterpret_cast<float*>(take(n * 4));
    ids = reinterpret_cast<int*>(take(n * 4));
    sorted_ids = reinterpret_cast<int*>(take(n * 4));
    counts = reinterpret_cast<int*>(take(rows * 4));
    status = reinterpret_cast<int*>(take(rows * 4));
    sort_space = p; sort_bytes = bytes - size_t(p - static_cast<char*>(ptr));
  }
};

__device__ float reduce_sum(float x) {
  __shared__ float scratch[Threads];
  scratch[threadIdx.x] = x; __syncthreads();
  for (int d = Threads / 2; d; d >>= 1) { if (threadIdx.x < d) scratch[threadIdx.x] += scratch[threadIdx.x + d]; __syncthreads(); }
  float result = scratch[0]; __syncthreads();
  return result;
}
__device__ double reduce_double(double x) {
  __shared__ double scratch[Threads];
  scratch[threadIdx.x] = x; __syncthreads();
  for (int d = Threads / 2; d; d >>= 1) { if (threadIdx.x < d) scratch[threadIdx.x] += scratch[threadIdx.x + d]; __syncthreads(); }
  double result = scratch[0]; __syncthreads();
  return result;
}
__device__ float reduce_max(float x) {
  __shared__ float scratch[Threads];
  scratch[threadIdx.x] = x; __syncthreads();
  for (int d = Threads / 2; d; d >>= 1) { if (threadIdx.x < d) scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + d]); __syncthreads(); }
  float result = scratch[0]; __syncthreads();
  return result;
}
__device__ int reduce_min(int x) {
  __shared__ int scratch[Threads];
  scratch[threadIdx.x] = x; __syncthreads();
  for (int d = Threads / 2; d; d >>= 1) { if (threadIdx.x < d) scratch[threadIdx.x] = min(scratch[threadIdx.x], scratch[threadIdx.x + d]); __syncthreads(); }
  int result = scratch[0]; __syncthreads();
  return result;
}
__device__ int op(const blz_sampling_row* p, int slot) { return slot < p->len ? p->stages[slot] : 0; }

__global__ void pack(const uint16_t* x, int local, int vocab, int rank, int rows, float* full) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * local) return;
  full[size_t(i / local) * vocab + rank * local + i % local] = __uint_as_float(uint32_t(x[i]) << 16);
}
__global__ void initialize(const float* raw, int vocab, int true_vocab, const uint32_t* valid_ids, const int* ids, const float* edits,
                           const int* off, const uint32_t* masks, const int* mask_rows, int words, Work w) {
  int row = blockIdx.x, tid = threadIdx.x;
  if (!tid) w.status[row] = 0;
  __syncthreads();
  int count = 0;
  for (int i = tid; i < vocab; i += Threads) {
    size_t at = size_t(row) * vocab + i;
    float x = raw[at];
    bool allowed = i < true_vocab && (!valid_ids || ((valid_ids[i / 32] >> (i % 32)) & 1)) && (mask_rows[row] < 0 || ((masks[size_t(mask_rows[row]) * words + i / 32] >> (i % 32)) & 1));
    if (allowed && (isnan(x) || x == INFINITY)) atomicExch(w.status + row, -2);
    if (!allowed || !isfinite(x)) x = -INFINITY;
    if (x == 0.0f) x = 0.0f;
    w.values[at] = w.canonical[at] = x;
    w.ids[at] = i;
  }
  __syncthreads();
  for (int e = off[row] + tid; e < off[row + 1]; e += Threads) {
    int id = ids[e]; if (id < 0 || id >= vocab) continue;
    size_t at = size_t(row) * vocab + id; const float* c = edits + 5 * e;
    if (c[4] != 0.0f) w.values[at] = w.canonical[at] = -INFINITY;
    else if (isfinite(w.values[at])) {
      float x = w.values[at]; w.canonical[at] = (x > 0.0f ? x * c[0] : x * c[1]) + (c[2] + c[3]);
      if (!isfinite(w.canonical[at])) atomicExch(w.status + row, -2);
    }
  }
  __syncthreads();
  for (int i = tid; i < vocab; i += Threads) count += isfinite(w.values[size_t(row) * vocab + i]);
  int n = int(reduce_sum(float(count)));
  if (!tid) { w.counts[row] = n; if (!n && !w.status[row]) w.status[row] = -1; }
}

__global__ void transform(int vocab, int slot, const blz_sampling_row* programs,
                          const int* ids, const float* edits, const int* off, Work w) {
  int row = blockIdx.x, tid = threadIdx.x; const auto& p = programs[row]; int kind = op(&p, slot);
  float* z = w.values + size_t(row) * vocab;
  if (kind == 1 || kind == 2) {
    for (int e = off[row] + tid; e < off[row + 1]; e += Threads) {
      int id = ids[e]; if (id < 0 || id >= vocab || !isfinite(z[id])) continue;
      const float* c = edits + e * 5;
      float x = kind == 1 ? (z[id] > 0.0f ? z[id] * c[0] : z[id] * c[1]) + c[2] : z[id] + c[3];
      if (!isfinite(x)) { atomicExch(w.status + row, -2); x = -INFINITY; }
      z[id] = x == 0.0f ? 0.0f : x;
    }
  } else if (kind == 3) {
    if (p.temperature == 0.0f) {
      float mx = -INFINITY; for (int i = tid; i < vocab; i += Threads) mx = fmaxf(mx, z[i]);
      mx = reduce_max(mx);
      int id = INT_MAX; for (int i = tid; i < vocab; i += Threads) if (z[i] == mx) id = min(id, i);
      id = reduce_min(id);
      for (int i = tid; i < vocab; i += Threads) if (i != id) z[i] = -INFINITY;
      if (!tid) w.counts[row] = isfinite(mx) ? 1 : 0;
    } else {
      for (int i = tid; i < vocab; i += Threads) if (isfinite(z[i])) {
        float x = z[i] / p.temperature;
        if (!isfinite(x)) { atomicExch(w.status + row, -2); x = -INFINITY; }
        z[i] = x == 0.0f ? 0.0f : x;
      }
    }
  }
}

// Radix-select a bounded prefix in FP32; ties are collected in token order.
__global__ void top64(int vocab, Work w) {
  __shared__ int hist[256], prefix, need, above, gt, taken;
  __shared__ float vals[K]; __shared__ int ids[K];
  int row = blockIdx.x, tid = threadIdx.x, k = min(K, vocab); const float* z = w.values + size_t(row) * vocab;
  if (!tid) { prefix = 0; need = k; above = 0; }
  __syncthreads();
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[tid] = 0; __syncthreads();
    for (int i = tid; i < vocab; i += Threads) {
      uint32_t bits = __float_as_uint(z[i]); uint32_t key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
      uint32_t highmask = shift == 24 ? 0u : ~0u << (shift + 8);
      if ((key & highmask) == uint32_t(prefix)) atomicAdd(hist + ((key >> shift) & 255), 1);
    }
    __syncthreads();
    if (!tid) {
      int bin = 255, n = 0; while (bin && n + hist[bin] < need) n += hist[bin--];
      prefix |= int(uint32_t(bin) << shift); need -= n; above += n;
    }
    __syncthreads();
  }
  if (!tid) { gt = 0; taken = 0; }
  for (int i = tid; i < K; i += Threads) { vals[i] = -INFINITY; ids[i] = INT_MAX; }
  __syncthreads();
  for (int i = tid; i < vocab; i += Threads) {
    uint32_t bits = __float_as_uint(z[i]); uint32_t key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    if (key > uint32_t(prefix)) { int at = atomicAdd(&gt, 1); if (at < K) vals[at] = z[i], ids[at] = i; }
  }
  __syncthreads();
  __shared__ int warp_n[Threads / 32];
  for (int base = 0; base < vocab; base += Threads) {
    int i = base + tid; uint32_t bits = i < vocab ? __float_as_uint(z[i]) : 0;
    uint32_t key = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    bool tie = i < vocab && key == uint32_t(prefix);
    unsigned ballot = __ballot_sync(0xffffffffu, tie);
    if ((tid & 31) == 0) warp_n[tid >> 5] = __popc(ballot);
    __syncthreads();
    int at = taken + __popc(ballot & ((1u << (tid & 31)) - 1));
    for (int j = 0; j < (tid >> 5); ++j) at += warp_n[j];
    if (tie && gt + at < k) vals[gt + at] = z[i], ids[gt + at] = i;
    __syncthreads();
    if (!tid) for (int j = 0; j < Threads / 32; ++j) taken += warp_n[j];
    __syncthreads(); if (taken >= need) break;
  }
  for (int span = 2; span <= K; span <<= 1) for (int dist = span >> 1; dist; dist >>= 1) {
    if (tid < K) {
      int j = tid ^ dist;
      if (j > tid) {
        bool better = vals[j] > vals[tid] || (vals[j] == vals[tid] && ids[j] < ids[tid]);
        if (better == ((tid & span) == 0)) { float x = vals[tid]; vals[tid] = vals[j]; vals[j] = x; int id = ids[tid]; ids[tid] = ids[j]; ids[j] = id; }
      }
    }
    __syncthreads();
  }
  if (tid < k) { size_t at = size_t(row) * vocab + tid; w.sorted[at] = vals[tid]; w.sorted_ids[at] = ids[tid]; }
}

__global__ void filter(int vocab, int slot, const blz_sampling_row* programs, Work w) {
  int row = blockIdx.x, tid = threadIdx.x; const auto& p = programs[row]; int kind = op(&p, slot);
  if (kind < 4 || kind > 8) return;
  float* z = w.values + size_t(row) * vocab;
  __shared__ int keep; __shared__ float threshold; __shared__ int last_id;
  if (kind == 7) {
    double sum = 0; int n = 0; float mx = -INFINITY;
    for (int i = tid; i < vocab; i += Threads) if (isfinite(z[i])) { sum += z[i]; ++n; mx = fmaxf(mx, z[i]); }
    sum = reduce_double(sum); n = int(reduce_sum(float(n))); mx = reduce_max(mx);
    double mean = n ? sum / n : 0, var = 0;
    for (int i = tid; i < vocab; i += Threads) if (isfinite(z[i])) { double d = z[i] - mean; var += d * d; }
    var = reduce_double(var);
    double cutoff = double(mx) - double(p.top_nsigma) * sqrt(n ? var / n : 0);
    int count = 0;
    for (int i = tid; i < vocab; i += Threads) { if (isfinite(z[i]) && double(z[i]) < cutoff) z[i] = -INFINITY; count += isfinite(z[i]); }
    count = int(reduce_sum(float(count))); if (!tid) w.counts[row] = count;
    return;
  }
  double total = 0, weighted_total = 0;
  const float* ranked = w.sorted + size_t(row) * vocab;
  if ((kind == 5 && p.top_p < 1) || (kind == 8 && p.top_h < 1)) {
    for (int i = tid; i < w.counts[row]; i += Threads) {
      double d = double(ranked[i]) - double(ranked[0]);
      double x = exp(d); total += x; weighted_total += x * d;
    }
  }
  total = reduce_double(total); weighted_total = reduce_double(weighted_total);
  if (!tid) {
    int n = w.counts[row]; keep = n;
    const float* sorted = w.sorted + size_t(row) * vocab;
    if (kind == 4 && p.top_k > 0) keep = min(n, p.top_k);
    if ((kind == 5 && p.top_p < 1) || (kind == 6 && p.min_p > 0) || (kind == 8 && p.top_h < 1)) {
      double sum = total, weighted = weighted_total; float mx = sorted[0];
      double h = sum > 0 ? log(sum) - weighted / sum : 0;
      double mass = 0, wz = 0; keep = 0;
      for (int i = 0; i < n; ++i) {
        double x = exp((double(sorted[i]) - double(mx)));
        if (kind == 6 && x < double(p.min_p)) break;
        mass += x; wz += x * (double(sorted[i]) - double(mx));
        if (kind == 8 && log(mass) - wz / mass > double(p.top_h) * h + 1e-12) break;
        ++keep;
        if (kind == 5 && mass >= double(p.top_p) * sum) break;
      }
      if (n) keep = max(1, keep);
    }
    if (keep) { threshold = sorted[keep - 1]; last_id = w.sorted_ids[size_t(row) * vocab + keep - 1]; }
    else { threshold = INFINITY; last_id = -1; }
    w.counts[row] = keep;
  }
  __syncthreads();
  for (int i = tid; i < vocab; i += Threads) if (z[i] < threshold || (z[i] == threshold && i > last_id)) z[i] = -INFINITY;
}

__global__ void draw(int vocab, const blz_sampling_row* p, Work w, float* out) {
  int r = blockIdx.x, tid = threadIdx.x;
  const float* z = w.sorted + size_t(r) * vocab; const int* ids = w.sorted_ids + size_t(r) * vocab;
  int n = w.counts[r];
  __shared__ double masses[Threads];
  int begin = int(int64_t(n) * tid / Threads), end = int(int64_t(n) * (tid + 1) / Threads);
  double mass = 0;
  if (!w.status[r] && n) for (int i = begin; i < end; ++i) mass += exp(double(z[i]) - double(z[0]));
  masses[tid] = mass; __syncthreads();
  double total = reduce_double(mass), prefix = 0;
  for (int t = 0; t < tid; ++t) prefix += masses[t];
  double target = double(p[r].draw) * total;
  int choice = INT_MAX;
  if (mass && target >= prefix && (target < prefix + mass || tid == Threads - 1)) {
    double acc = prefix; choice = ids[end - 1];
    for (int i = begin; i < end; ++i) { acc += exp(double(z[i]) - double(z[0])); if (target < acc) { choice = ids[i]; break; } }
  }
  choice = reduce_min(choice);
  if (!tid) {
    int id = w.status[r] ? w.status[r] : choice == INT_MAX ? -1 : choice;
    out[(size_t(r) * (K + 2) + K + 1) * 2 + 1] = float(id);
    out[(size_t(r) * (K + 2) + K + 1) * 2] = id >= 0 ? w.canonical[size_t(r) * vocab + id] : -INFINITY;
  }
}
__global__ void export_results(int vocab, bool logprobs, Work w, float* out) {
  int r = blockIdx.x, tid = threadIdx.x; size_t base = size_t(r) * (K + 2) * 2;
  for (int j = tid; j < K; j += Threads) {
    out[base + 2 * j] = j < vocab ? w.sorted[size_t(r) * vocab + j] : -INFINITY;
    out[base + 2 * j + 1] = j < vocab ? float(w.sorted_ids[size_t(r) * vocab + j]) : -1;
  }
  double sum = 0; float mx = -INFINITY;
  const float* z = w.canonical + size_t(r) * vocab;
  if (logprobs) for (int i = tid; i < vocab; i += Threads) mx = fmaxf(mx, z[i]);
  mx = reduce_max(mx);
  if (logprobs && isfinite(mx)) for (int i = tid; i < vocab; i += Threads) sum += exp((double(z[i]) - double(mx)));
  sum = reduce_double(sum);
  if (!tid) { out[base + K * 2] = sum > 0 ? float(double(mx) + log(sum)) : 0; out[base + K * 2 + 1] = -2; }
}
} // namespace

extern "C" size_t blz_sampling_workspace_bytes(int rows, int vocab) {
  size_t bytes = 0;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, bytes, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
      static_cast<int*>(nullptr), static_cast<int*>(nullptr), rows * vocab, rows, offsets(vocab), offsets(vocab, 1));
  return 5 * aligned(size_t(rows) * vocab * 4) + 2 * aligned(size_t(rows) * 4) + bytes;
}
extern "C" cudaError_t blz_sampling_pack(const uint16_t* local, int local_vocab, int vocab, int rank, int rows, float* full, cudaStream_t stream) {
  pack<<<(rows * local_vocab + 255) / 256, 256, 0, stream>>>(local, local_vocab, vocab, rank, rows, full);
  return cudaGetLastError();
}
extern "C" cudaError_t blz_sampling_run(const float* raw, int rows, int vocab, int true_vocab, const uint32_t* valid_ids,
    const blz_sampling_row* programs, const int32_t* edit_ids, const float* edits, const int32_t* off,
    const uint32_t* masks, const int32_t* mask_rows, int words, uint32_t execution,
    bool logprobs, void* workspace, size_t bytes, float* out, cudaStream_t stream) {
  if (!rows) return cudaSuccess;
  Work w(workspace, bytes, rows, vocab);
  initialize<<<rows, Threads, 0, stream>>>(raw, vocab, true_vocab, valid_ids, edit_ids, edits, off, masks, mask_rows, words, w);
  auto sort = [&](const float* values) {
    return cub::DeviceSegmentedRadixSort::SortPairsDescending(w.sort_space, w.sort_bytes, values, w.sorted, w.ids, w.sorted_ids,
        rows * vocab, rows, offsets(vocab), offsets(vocab, 1), 0, 32, stream);
  };
  bool ordered = false, bounded = execution & (1u << 24);
  for (int slot = 0; slot < 8; ++slot) {
    if (!(execution & (1u << slot))) continue;
    if (execution & (1u << (slot + 16))) {
      transform<<<rows, Threads, 0, stream>>>(vocab, slot, programs, edit_ids, edits, off, w);
      ordered = false;
    }
    if (!ordered && (execution & (1u << (slot + 8)))) {
      if (bounded) top64<<<rows, Threads, 0, stream>>>(vocab, w);
      else { cudaError_t e = sort(w.values); if (e != cudaSuccess) return e; }
      ordered = true;
    }
    filter<<<rows, Threads, 0, stream>>>(vocab, slot, programs, w);
  }
  if (!ordered) {
    if (bounded) top64<<<rows, Threads, 0, stream>>>(vocab, w);
    else { cudaError_t e = sort(w.values); if (e != cudaSuccess) return e; }
  }
  draw<<<rows, Threads, 0, stream>>>(vocab, programs, w, out);
  if (logprobs) { cudaError_t e = sort(w.canonical); if (e != cudaSuccess) return e; }
  export_results<<<rows, Threads, 0, stream>>>(vocab, logprobs, w, out);
  return cudaGetLastError();
}
