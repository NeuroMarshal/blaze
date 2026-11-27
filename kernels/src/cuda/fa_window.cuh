#pragma once

#include <math_constants.h>

#include "blaze/nvfp4.h"
#include "fa_tile.cuh"

namespace blaze::fa {

__device__ inline float warp_sum(float x) {
    for (int shift = 16; shift; shift /= 2) x += __shfl_xor_sync(0xffffffff, x, shift);
    return x;
}

constexpr int kPromptQueries = 16;
constexpr int kPromptKeys = 32;

template<int HeadDim, bool Packed, int Group>
__global__ void window_prefill(const __nv_bfloat16* q, const uint8_t* keys, const uint8_t* values,
        const uint8_t* ks, const uint8_t* vs, const __nv_bfloat16* relative, float* scratch, int heads,
        int kv_heads, int capacity, int window, int start, int extent, int span, int rows, float scale) {
    static_assert(HeadDim == 128);
    constexpr int kLeading = HeadDim + 8;
    constexpr int kGroups = HeadDim / blaze::kNvfp4Block;
    constexpr int kRawBytes = HeadDim / 2;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int head = Group == 1 ? blockIdx.x : blockIdx.x * Group + warp;
    const int kv_head = head / (heads / kv_heads);
    const int query_base = blockIdx.z * kPromptQueries;
    const int active = min(kPromptQueries, rows - query_base);
    const int first_pos = start + query_base;
    const int first = max(0, first_pos + 1 - window) + blockIdx.y * span;
    const int end = min(first + span, first_pos + active);

    __align__(32) __shared__ __half q_tile[Group][kPromptQueries][kLeading];
    __align__(32) __shared__ __half kv_tile[2][kPromptKeys][kLeading];
    __align__(16) __shared__ uint8_t raw[2][kPromptKeys][Packed ? kRawBytes : 1];
    __align__(8) __shared__ uint8_t scales[2][kPromptKeys][Packed ? kGroups : 1];

    for (int i = lane; i < kPromptQueries * HeadDim; i += 32) {
        const int row = i / HeadDim, col = i % HeadDim;
        q_tile[warp][row][col] = row < active
            ? __float2half(__bfloat162float(q[(size_t(query_base + row) * heads + head) * HeadDim + col]))
            : __float2half(0.0f);
    }

    const auto queue_packed = [&](int base) {
        constexpr int kChunks = kRawBytes / 16;
        for (int i = threadIdx.x; i < 2 * kPromptKeys * kChunks; i += blockDim.x) {
            const int side = i / (kPromptKeys * kChunks);
            const int row = (i / kChunks) % kPromptKeys;
            const int off = (i % kChunks) * 16;
            const bool valid = base + row < end;
            const size_t slot = valid ? size_t((base + row) % capacity) * kv_heads + kv_head : 0;
            cp16(&raw[side][row][off], (side ? values : keys) + slot * kRawBytes + off, valid);
            if (off == 0) {
                *reinterpret_cast<uint64_t*>(scales[side][row]) = valid
                    ? *reinterpret_cast<const uint64_t*>((side ? vs : ks) + slot * kGroups) : 0;
            }
        }
        cp_commit();
    };
    if constexpr (Packed) {
        if (first < end) queue_packed(first);
    }
    __syncthreads();

    const int row0 = lane / 4, row1 = row0 + 8, col2 = (lane % 4) * 2;
    float maxima[2] = {-CUDART_INF_F, -CUDART_INF_F};
    float masses[2] = {};
    float numerators[HeadDim / 8][4] = {};
    for (int base = first; base < end; base += kPromptKeys) {
        if constexpr (Packed) {
            cp_wait<0>();
            __syncthreads();
            for (int i = threadIdx.x; i < 2 * kPromptKeys * kGroups; i += blockDim.x) {
                const int side = i / (kPromptKeys * kGroups);
                const int row = (i / kGroups) % kPromptKeys;
                const int group = i % kGroups;
                dequant_block(raw[side][row] + group * 8, scales[side][row][group],
                              &kv_tile[side][row][group * blaze::kNvfp4Block], scale);
            }
        } else {
            for (int i = threadIdx.x; i < 2 * kPromptKeys * HeadDim; i += blockDim.x) {
                const int side = i / (kPromptKeys * HeadDim);
                const int row = (i / HeadDim) % kPromptKeys;
                const int col = i % HeadDim;
                const auto* data = reinterpret_cast<const __nv_bfloat16*>(side ? values : keys);
                const size_t slot = size_t((base + row) % capacity) * kv_heads + kv_head;
                kv_tile[side][row][col] = __float2half(base + row < end ? __bfloat162float(data[slot * HeadDim + col]) : 0.0f);
            }
        }
        __syncthreads();
        if constexpr (Packed) {
            if (base + kPromptKeys < end) queue_packed(base + kPromptKeys);
        }

        float scores[kPromptKeys / 8][4] = {};
        const auto* qt = &q_tile[warp][0][0];
        const auto* kt = &kv_tile[0][0][0];
#pragma unroll
        for (int k = 0; k < HeadDim; k += 16) {
            uint32_t a[4];
            ldsm4(a, qt + (lane & 15) * kLeading + k + (lane >> 4) * 8);
#pragma unroll
            for (int n = 0; n < kPromptKeys; n += 16) {
                uint32_t b[4];
                ldsm4(b, kt + (n + (lane & 7) + ((lane >> 4) << 3)) * kLeading + k + ((lane >> 3) & 1) * 8);
                mma_f16(scores[n / 8], a, b[0], b[1]);
                mma_f16(scores[n / 8 + 1], a, b[2], b[3]);
            }
        }

        float peaks[2] = {-CUDART_INF_F, -CUDART_INF_F};
#pragma unroll
        for (int tile = 0; tile < kPromptKeys / 8; ++tile) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = j < 2 ? row0 : row1;
                const int key = base + tile * 8 + col2 + (j & 1);
                const int distance = first_pos + row - key;
                const bool visible = row < active && key < end && distance >= 0 && distance < window;
                const float bias = visible && relative && distance < extent
                    ? __bfloat162float(relative[(size_t(query_base + row) * heads + head) * extent + distance]) : 0.0f;
                const float score = visible ? scores[tile][j] * rsqrtf(float(HeadDim)) + bias : -CUDART_INF_F;
                scores[tile][j] = score;
                peaks[j / 2] = fmaxf(peaks[j / 2], score);
            }
        }
        float keep[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            peaks[r] = fmaxf(peaks[r], __shfl_xor_sync(0xffffffffu, peaks[r], 1));
            peaks[r] = fmaxf(peaks[r], __shfl_xor_sync(0xffffffffu, peaks[r], 2));
            const float next = fmaxf(maxima[r], peaks[r]);
            keep[r] = next == -CUDART_INF_F ? 0.0f : __expf(maxima[r] - next);
            maxima[r] = next;
        }
#pragma unroll
        for (int tile = 0; tile < HeadDim / 8; ++tile) {
            numerators[tile][0] *= keep[0], numerators[tile][1] *= keep[0];
            numerators[tile][2] *= keep[1], numerators[tile][3] *= keep[1];
        }
        float sums[2] = {};
        const auto* vt = &kv_tile[1][0][0];
#pragma unroll
        for (int k = 0; k < kPromptKeys / 16; ++k) {
            uint32_t p[4];
#pragma unroll
            for (int tile = 0; tile < 2; ++tile) {
                const auto& s = scores[2 * k + tile];
                const float ref0 = maxima[0] == -CUDART_INF_F ? 0.0f : maxima[0];
                const float ref1 = maxima[1] == -CUDART_INF_F ? 0.0f : maxima[1];
                const float p0 = __expf(s[0] - ref0), p1 = __expf(s[1] - ref0);
                const float p2 = __expf(s[2] - ref1), p3 = __expf(s[3] - ref1);
                p[2 * tile] = pack_f16(p0, p1);
                p[2 * tile + 1] = pack_f16(p2, p3);
                sums[0] += p0 + p1;
                sums[1] += p2 + p3;
            }
#pragma unroll
            for (int d = 0; d < HeadDim; d += 16) {
                uint32_t b[4];
                ldsm4t(b, vt + (16 * k + (lane & 7) + ((lane >> 3) & 1) * 8) * kLeading + d + (lane >> 4) * 8);
                mma_f16(numerators[d / 8], p, b[0], b[1]);
                mma_f16(numerators[d / 8 + 1], p, b[2], b[3]);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], 1);
            sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], 2);
            masses[r] = fmaf(masses[r], keep[r], sums[r]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = r == 0 ? row0 : row1;
        if (row >= active) continue;
        auto* part = scratch + ((size_t(query_base + row) * heads + head) * gridDim.y + blockIdx.y) * (HeadDim + 2);
#pragma unroll
        for (int d = 0; d < HeadDim; d += 8) {
            part[d + col2] = numerators[d / 8][2 * r];
            part[d + col2 + 1] = numerators[d / 8][2 * r + 1];
        }
        if (col2 == 0) {
            part[HeadDim] = maxima[r];
            part[HeadDim + 1] = masses[r];
        }
    }
}

template<int HeadDim, bool Packed>
cudaError_t launch_window_prefill(const __nv_bfloat16* q, const uint8_t* keys, const uint8_t* values,
        const uint8_t* ks, const uint8_t* vs, const __nv_bfloat16* relative, float* scratch, int heads,
        int kv_heads, int capacity, int window, int start, int extent, int span, int splits, int rows,
        float scale, cudaStream_t stream) {
    const int queries = (rows + kPromptQueries - 1) / kPromptQueries;
    if (heads / kv_heads == 3) {
        window_prefill<HeadDim, Packed, 3><<<dim3(kv_heads, splits, queries), 96, 0, stream>>>(
            q, keys, values, ks, vs, relative, scratch, heads, kv_heads, capacity, window, start, extent, span, rows, scale);
    } else {
        window_prefill<HeadDim, Packed, 1><<<dim3(heads, splits, queries), 32, 0, stream>>>(
            q, keys, values, ks, vs, relative, scratch, heads, kv_heads, capacity, window, start, extent, span, rows, scale);
    }
    return cudaGetLastError();
}

}  // namespace blaze::fa
