#pragma once

#include <math_constants.h>
#include <mma.h>

#include "blaze/nvfp4.h"
#include "fa_tile.cuh"

namespace blaze::fa {

__device__ inline float warp_sum(float x) {
    for (int shift = 16; shift; shift /= 2) x += __shfl_xor_sync(0xffffffff, x, shift);
    return x;
}

constexpr int kPromptQueries = 16;
constexpr int kPromptKeys = 32;

template<int HeadDim, bool Packed>
__global__ void window_prefill(const __nv_bfloat16* q, const uint8_t* keys, const uint8_t* values,
        const uint8_t* ks, const uint8_t* vs, const __nv_bfloat16* relative, float* scratch, int kv_heads,
        int capacity, int window, int start, int extent, int span, int rows, float scale) {
    static_assert(HeadDim == 128);
    constexpr int kWarps = HeadDim / 32;
    namespace wmma = nvcuda::wmma;
    const int head = blockIdx.x;
    const int kv_head = head / (gridDim.x / kv_heads);
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int query_base = blockIdx.z * kPromptQueries;
    const int active = min(kPromptQueries, rows - query_base);
    const int leading = HeadDim + 8;
    q += size_t(query_base) * gridDim.x * HeadDim;
    if (relative) relative += size_t(query_base) * gridDim.x * extent;
    scratch += size_t(query_base) * gridDim.x * gridDim.y * (HeadDim + 2);
    start += query_base;
    const int first = max(0, start + 1 - window) + blockIdx.y * span;
    const int end = min(first + span, start + active);
    __align__(32) __shared__ __half q_tile[kPromptQueries][HeadDim];
    __align__(32) __shared__ __half k_tile[kPromptKeys][leading];
    __align__(32) __shared__ __half v_tile[kPromptKeys][leading];
    __align__(32) __shared__ float score[kPromptQueries][kPromptKeys];
    __align__(32) __shared__ __half probability[kPromptQueries][kPromptKeys];
    __align__(32) __shared__ float numerator[kPromptQueries][HeadDim];
    __align__(32) __shared__ float product[kPromptQueries][HeadDim];
    __shared__ float maximum[kPromptQueries], mass[kPromptQueries], keep[kPromptQueries];

    for (int i = threadIdx.x; i < kPromptQueries * HeadDim; i += blockDim.x) {
        const int row = i / HeadDim;
        const int col = i % HeadDim;
        q_tile[row][col] = row < active ? __float2half(__bfloat162float(q[(size_t(row) * gridDim.x + head) * HeadDim + col])) : __float2half(0.0f);
        numerator[row][col] = 0.0f;
    }
    if (threadIdx.x < kPromptQueries) {
        maximum[threadIdx.x] = -CUDART_INF_F;
        mass[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    for (int base = first; base < end; base += kPromptKeys) {
        if constexpr (Packed) {
            constexpr int groups = HeadDim / blaze::kNvfp4Block;
            for (int i = threadIdx.x; i < kPromptKeys * groups; i += blockDim.x) {
                const int key = i / groups;
                const int group = i % groups;
                const int pos = base + key;
                const int col = group * blaze::kNvfp4Block;
                if (pos < end) {
                    const size_t kv_row = size_t(pos % capacity) * kv_heads + kv_head;
                    const size_t bytes = kv_row * (HeadDim / 2) + col / 2;
                    blaze::fa::dequant_block(keys + bytes, ks[kv_row * groups + group], &k_tile[key][col], scale);
                    blaze::fa::dequant_block(values + bytes, vs[kv_row * groups + group], &v_tile[key][col], scale);
                } else {
                    const uint4 zero = {};
                    auto* k = reinterpret_cast<uint4*>(&k_tile[key][col]);
                    auto* v = reinterpret_cast<uint4*>(&v_tile[key][col]);
                    k[0] = k[1] = v[0] = v[1] = zero;
                }
            }
        } else {
            for (int i = threadIdx.x; i < kPromptKeys * HeadDim; i += blockDim.x) {
                const int pos = base + i / HeadDim;
                const int col = i % HeadDim;
                const size_t kv_row = size_t(pos % capacity) * kv_heads + kv_head;
                k_tile[i / HeadDim][col] = __float2half(pos < end ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(keys)[kv_row * HeadDim + col]) : 0.0f);
                v_tile[i / HeadDim][col] = __float2half(pos < end ? __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(values)[kv_row * HeadDim + col]) : 0.0f);
            }
        }
        __syncthreads();
        if (warp < kPromptKeys / 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot;
            wmma::fill_fragment(dot, 0.0f);
            for (int k = 0; k < HeadDim; k += 16) {
                wmma::load_matrix_sync(q_fragment, &q_tile[0][k], HeadDim);
                wmma::load_matrix_sync(k_fragment, &k_tile[warp * 16][k], leading);
                wmma::mma_sync(dot, q_fragment, k_fragment, dot);
            }
            wmma::store_matrix_sync(&score[0][warp * 16], dot, kPromptKeys, wmma::mem_row_major);
        }
        __syncthreads();
        for (int row = warp; row < kPromptQueries; row += kWarps) {
            const int distance = start + row - (base + lane);
            const bool visible = row < active && base + lane < end && distance >= 0 && distance < window;
            const float bias = visible && relative && distance < extent
                ? __bfloat162float(relative[(size_t(row) * gridDim.x + head) * extent + distance]) : 0.0f;
            const float logit = visible ? score[row][lane] * rsqrtf(float(HeadDim)) + bias : -CUDART_INF_F;
            float peak = logit;
            for (int shift = 16; shift; shift /= 2) peak = fmaxf(peak, __shfl_xor_sync(0xffffffff, peak, shift));
            const float next = fmaxf(maximum[row], peak);
            const float carry = next == -CUDART_INF_F ? 0.0f : __expf(maximum[row] - next);
            const float p = visible ? __expf(logit - next) : 0.0f;
            const float sum = warp_sum(p);
            probability[row][lane] = __float2half(p);
            if (lane == 0) {
                mass[row] = fmaf(mass[row], carry, sum);
                maximum[row] = next;
                keep[row] = carry;
            }
        }
        __syncthreads();
        for (int col = warp * 16; col < HeadDim; col += kWarps * 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_fragment;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum;
            wmma::fill_fragment(accum, 0.0f);
            for (int k = 0; k < kPromptKeys; k += 16) {
                wmma::load_matrix_sync(p_fragment, &probability[0][k], kPromptKeys);
                wmma::load_matrix_sync(v_fragment, &v_tile[k][col], leading);
                wmma::mma_sync(accum, p_fragment, v_fragment, accum);
            }
            wmma::store_matrix_sync(&product[0][col], accum, HeadDim, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kPromptQueries * HeadDim; i += blockDim.x) {
            const int row = i / HeadDim;
            const int col = i % HeadDim;
            numerator[row][col] = fmaf(numerator[row][col], keep[row], product[row][col]);
        }
        __syncthreads();
    }
    for (int row = 0; row < active; ++row) {
        auto* part = scratch + ((size_t(row) * gridDim.x + head) * gridDim.y + blockIdx.y) * (HeadDim + 2);
        part[threadIdx.x] = numerator[row][threadIdx.x];
        if (threadIdx.x == 0) {
            part[HeadDim] = maximum[row];
            part[HeadDim + 1] = mass[row];
        }
    }
}

}  // namespace blaze::fa
