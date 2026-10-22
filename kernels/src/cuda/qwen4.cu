// Qwen4-Exp layer-structure kernels (blaze/qwen4.h).

#include <cuda_bf16.h>

#include "blaze/qwen4.h"

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float bf(uint16_t v) { return __uint_as_float(uint32_t(v) << 16); }
__device__ __forceinline__ uint16_t to_bf(float v) { return __bfloat16_as_ushort(__float2bfloat16(v)); }

template <int N>
__device__ __forceinline__ void block_sum(float (&v)[N], float (*red)[kThreads / 32]) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v[i] += __shfl_xor_sync(0xffffffffu, v[i], o);
    if (lane == 0) red[i][warp] = v[i];
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < N; ++i) {
    float t = 0.0f;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) t += red[i][w];
    v[i] = t;
  }
}

// One block per (row, group).
__global__ void __launch_bounds__(kThreads) grouped_rmsnorm_kernel(const uint16_t* __restrict__ x,
                                                                   const uint16_t* __restrict__ w, float eps,
                                                                   int groups, int dim, uint16_t* __restrict__ out) {
  __shared__ float red[1][kThreads / 32];
  const int64_t row = blockIdx.x / groups;
  const int g = blockIdx.x % groups;
  const uint16_t* xr = x + (row * groups + g) * dim;
  const uint16_t* wr = w + int64_t(g) * dim;
  float ss[1] = {0.0f};
  for (int i = threadIdx.x; i < dim; i += kThreads) {
    const float v = bf(xr[i]);
    ss[0] += v * v;
  }
  block_sum(ss, red);
  const float inv = rsqrtf(ss[0] / float(dim) + eps);
  uint16_t* o = out + (row * groups + g) * dim;
  for (int i = threadIdx.x; i < dim; i += kThreads) o[i] = to_bf(bf(xr[i]) * inv * (1.0f + bf(wr[i])));
}

template <int HC>
__global__ void __launch_bounds__(kThreads) hc_combine_kernel(uint16_t* __restrict__ streams,
                                                              const uint16_t* __restrict__ normed,
                                                              const uint16_t* __restrict__ inject,
                                                              const uint16_t* __restrict__ y, int64_t ld_y, int dim) {
  __shared__ float red[HC][kThreads / 32];
  const int64_t row = blockIdx.x;
  const int width = HC * dim;
  const uint16_t* nr = normed + row * width;
  float acc[HC];
#pragma unroll
  for (int c = 0; c < HC; ++c) acc[c] = 0.0f;
  for (int i = threadIdx.x; i < width; i += kThreads) {
    const float v = bf(nr[i]);
#pragma unroll
    for (int c = 0; c < HC; ++c) acc[c] = fmaf(v, bf(inject[int64_t(c) * width + i]), acc[c]);
  }
  block_sum(acc, red);
  float a[HC];
#pragma unroll
  for (int c = 0; c < HC; ++c) a[c] = 2.0f / (1.0f + __expf(-acc[c] / float(HC)));
  uint16_t* sr = streams + row * width;
  const uint16_t* yr = y + row * ld_y;
  for (int i = threadIdx.x; i < width; i += kThreads) {
    const int c = i / dim;
    sr[i] = to_bf(bf(sr[i]) + a[c] * bf(yr[i - c * dim]));
  }
}

__global__ void hc_widen_kernel(const uint16_t* __restrict__ x, int rows, int hc, int dim,
                                uint16_t* __restrict__ streams) {
  const int64_t total = int64_t(rows) * hc * dim;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = i / (int64_t(hc) * dim), col = i % dim;
    streams[i] = x[row * dim + col];
  }
}

__device__ __forceinline__ int64_t window_token(const int64_t* tokens, const int64_t* h, int start, int k) {
  return k < 2 ? h[k] : tokens[start + k - 2];
}

__global__ void ngram_hash_kernel(const int64_t* __restrict__ tokens, const int32_t* __restrict__ seq_start,
                                  const int32_t* __restrict__ slots, int seqs, const int64_t* __restrict__ hist,
                                  const int64_t* __restrict__ mult, const int64_t* __restrict__ sizes,
                                  const int64_t* __restrict__ offsets, int heads_per_ngram, int64_t eos,
                                  int64_t* __restrict__ ids) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= seq_start[seqs]) return;
  int s = 0;
  while (seq_start[s + 1] <= j) ++s;
  const int start = seq_start[s], i = j - start;
  const int64_t* h = hist + int64_t(slots[s]) * 2;
  const int64_t w0 = window_token(tokens, h, start, i), w1 = window_token(tokens, h, start, i + 1),
                w2 = window_token(tokens, h, start, i + 2);
  // An n-gram stops at an EOS before the current token: SGLang's _shift_right_ignore_eos on the last window column.
  const int reach = w1 == eos ? 0 : (w0 == eos ? 1 : 2);
  const uint64_t t0 = uint64_t(w2), t1 = uint64_t(reach >= 1 ? w1 : eos), t2 = uint64_t(reach >= 2 ? w0 : eos);
  const uint64_t mix2 = (t0 * uint64_t(mult[0])) ^ (t1 * uint64_t(mult[1]));
  const uint64_t mix3 = mix2 ^ (t2 * uint64_t(mult[2]));
  int64_t* out = ids + int64_t(j) * 2 * heads_per_ngram;
  for (int hh = 0; hh < 2 * heads_per_ngram; ++hh) {
    const int64_t mix = int64_t(hh < heads_per_ngram ? mix2 : mix3);
    int64_t r = mix % sizes[hh];
    if (r < 0) r += sizes[hh];
    out[hh] = r + offsets[hh];
  }
}

__global__ void ngram_hist_kernel(const int64_t* __restrict__ tokens, const int32_t* __restrict__ seq_start,
                                  const int32_t* __restrict__ slots, int seqs, int64_t* __restrict__ hist) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= seqs) return;
  const int start = seq_start[s], n = seq_start[s + 1] - start;
  int64_t* h = hist + int64_t(slots[s]) * 2;
  const int64_t a = window_token(tokens, h, start, n), b = window_token(tokens, h, start, n + 1);
  h[0] = a;
  h[1] = b;
}

// One block per (row, stream).
__global__ void __launch_bounds__(kThreads) ple_gate_kernel(const uint16_t* __restrict__ key_n,
                                                            const uint16_t* __restrict__ query_n,
                                                            const uint16_t* __restrict__ value, int hc, int dim,
                                                            uint16_t* __restrict__ gated) {
  __shared__ float red[1][kThreads / 32];
  const int64_t row = blockIdx.x / hc;
  const int c = blockIdx.x % hc;
  const int64_t base = (row * hc + c) * dim;
  float d[1] = {0.0f};
  for (int i = threadIdx.x; i < dim; i += kThreads) d[0] = fmaf(bf(key_n[base + i]), bf(query_n[base + i]), d[0]);
  block_sum(d, red);
  const float x = d[0] * rsqrtf(float(dim));
  const float sq = copysignf(sqrtf(fmaxf(fabsf(x), 1e-6f)), x);
  const float g = 1.0f / (1.0f + __expf(-sq));
  const uint16_t* vr = value + row * dim;
  for (int i = threadIdx.x; i < dim; i += kThreads) gated[base + i] = to_bf(g * bf(vr[i]));
}

// One thread per (sequence, channel); kernel 4, dilation 3: 9 earlier inputs.
constexpr int kConvTaps = 4, kConvDil = 3, kConvState = (kConvTaps - 1) * kConvDil;

__global__ void ple_conv_kernel(const uint16_t* __restrict__ gated_n, const uint16_t* __restrict__ gated,
                                const int32_t* __restrict__ seq_start, const int32_t* __restrict__ slots,
                                int channels, const uint16_t* __restrict__ weight, uint16_t* __restrict__ state,
                                int write_state, uint16_t* __restrict__ streams) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int s = blockIdx.y;
  if (c >= channels) return;
  uint16_t* st = state + (int64_t(slots[s]) * channels + c) * kConvState;
  float h[kConvState], w[kConvTaps];
#pragma unroll
  for (int j = 0; j < kConvState; ++j) h[j] = bf(st[j]);
#pragma unroll
  for (int j = 0; j < kConvTaps; ++j) w[j] = bf(weight[int64_t(c) * kConvTaps + j]);
  for (int t = seq_start[s]; t < seq_start[s + 1]; ++t) {
    const int64_t idx = int64_t(t) * channels + c;
    const float x = bf(gated_n[idx]);
    float acc = w[kConvTaps - 1] * x;
#pragma unroll
    for (int j = 0; j < kConvTaps - 1; ++j) acc = fmaf(w[j], h[j * kConvDil], acc);
    const float cv = bf(to_bf(acc));
    const float sl = bf(to_bf(cv / (1.0f + __expf(-cv))));
    const float o = bf(to_bf(bf(gated[idx]) + sl));
    streams[idx] = to_bf(bf(streams[idx]) + o);
#pragma unroll
    for (int j = 0; j < kConvState - 1; ++j) h[j] = h[j + 1];
    h[kConvState - 1] = x;
  }
  if (write_state)
#pragma unroll
    for (int j = 0; j < kConvState; ++j) st[j] = to_bf(h[j]);
}

int grid_for(int64_t n, int threads) {
  const int64_t b = (n + threads - 1) / threads;
  return int(b < 65535 * 32 ? b : 65535 * 32);
}

}  // namespace

extern "C" cudaError_t blz_grouped_rmsnorm(const uint16_t* x, const uint16_t* w, float eps, int rows, int groups,
                                           int dim, uint16_t* out, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  grouped_rmsnorm_kernel<<<rows * groups, kThreads, 0, stream>>>(x, w, eps, groups, dim, out);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_hc_combine(uint16_t* streams, const uint16_t* normed, const uint16_t* inject,
                                      const uint16_t* y, int64_t ld_y, int rows, int hc, int dim,
                                      cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  switch (hc) {
    case 2: hc_combine_kernel<2><<<rows, kThreads, 0, stream>>>(streams, normed, inject, y, ld_y, dim); break;
    case 4: hc_combine_kernel<4><<<rows, kThreads, 0, stream>>>(streams, normed, inject, y, ld_y, dim); break;
    case 8: hc_combine_kernel<8><<<rows, kThreads, 0, stream>>>(streams, normed, inject, y, ld_y, dim); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t blz_hc_widen(const uint16_t* x, int rows, int hc, int dim, uint16_t* streams,
                                    cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  hc_widen_kernel<<<grid_for(int64_t(rows) * hc * dim, 256), 256, 0, stream>>>(x, rows, hc, dim, streams);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_ngram_hash(const int64_t* tokens, const int32_t* seq_start, const int32_t* slots, int seqs,
                                      int max_tokens, int64_t* hist, int update_hist, const int64_t* multipliers,
                                      const int64_t* sizes, const int64_t* offsets, int heads_per_ngram, int64_t eos,
                                      int64_t* ids, cudaStream_t stream) {
  if (seqs == 0 || max_tokens == 0) return cudaSuccess;
  // The token count lives on the device (graph replays): the grid covers max_tokens, extra threads return.
  ngram_hash_kernel<<<(max_tokens + 127) / 128, 128, 0, stream>>>(tokens, seq_start, slots, seqs, hist, multipliers, sizes, offsets,
                                             heads_per_ngram, eos, ids);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess || !update_hist) return e;
  ngram_hist_kernel<<<(seqs + 127) / 128, 128, 0, stream>>>(tokens, seq_start, slots, seqs, hist);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_ple_gate(const uint16_t* key_n, const uint16_t* query_n, const uint16_t* value, int rows,
                                    int hc, int dim, uint16_t* gated, cudaStream_t stream) {
  if (rows == 0) return cudaSuccess;
  ple_gate_kernel<<<rows * hc, kThreads, 0, stream>>>(key_n, query_n, value, hc, dim, gated);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_ple_conv(const uint16_t* gated_n, const uint16_t* gated, const int32_t* seq_start,
                                    const int32_t* slots, int seqs, int channels, int kernel, int dilation,
                                    const uint16_t* weight, uint16_t* state, int write_state, uint16_t* streams,
                                    cudaStream_t stream) {
  if (seqs == 0) return cudaSuccess;
  if (kernel != kConvTaps || dilation != kConvDil) return cudaErrorNotSupported;
  const dim3 grid(unsigned((channels + 127) / 128), unsigned(seqs));
  ple_conv_kernel<<<grid, 128, 0, stream>>>(gated_n, gated, seq_start, slots, channels, weight, state, write_state,
                                            streams);
  return cudaGetLastError();
}
