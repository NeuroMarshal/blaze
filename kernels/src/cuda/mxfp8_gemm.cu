#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "blaze/moe_packed.h"
#include "blaze/nvfp4_gemm.h"

namespace {

__global__ void quantize_mxfp8(const uint16_t* input, uint8_t* codes, uint8_t* scales, int inner) {
  const int lane = threadIdx.x % 32;
  const int block = blockIdx.x * 4 + threadIdx.x / 32;
  if (block >= inner / 32) return;
  const size_t at = size_t(blockIdx.y) * inner + block * 32 + lane;
  const float x = __bfloat162float(__ushort_as_bfloat16(input[at]));
  float peak = fabsf(x);
  for (int offset = 16; offset > 0; offset /= 2)
    peak = fmaxf(peak, __shfl_xor_sync(0xffffffffu, peak, offset));
  const float ratio = peak * (1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(ratio);
  const int exponent = max(-126, min(127, int((bits >> 23) & 255) - 127 + int((bits & 0x7fffff) != 0)));
  const float inverse = __uint_as_float(uint32_t(127 - exponent) << 23);
  codes[at] = __nv_fp8_e4m3(x * inverse).__x;
  if (lane == 0) scales[size_t(blockIdx.y) * (inner / 32) + block] = uint8_t(exponent + 127);
}

__global__ void mxfp8_grouped_mma(const uint8_t* input, const uint8_t* scale, const blz_moe_matrix_tile* items,
                                  uint16_t* out, int columns, int inner) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL)
  const auto item = items[blockIdx.z];
  const int tile = blockIdx.y * 16;
  if (tile >= item.rows) return;
  const int lane = threadIdx.x % 32;
  const int group = lane / 4;
  const int first_column = blockIdx.x * 32 + (threadIdx.x / 32) * 8;
  if (first_column >= columns) return;
  const int col = min(first_column + group, columns - 1);
  const int r0 = tile + group, r1 = r0 + 8, rs = r0 + (lane & 1) * 8;
  const int first_k = (lane & 3) * 4;
  const int scale_stride = inner / 32;
  input += size_t(item.begin) * inner;
  scale += size_t(item.begin) * scale_stride;
  const auto* w = item.codes + size_t(col) * inner + first_k;
  const auto* ws = item.scales + size_t(col) * scale_stride;
  const auto* a0 = input + size_t(r0 < item.rows ? r0 : 0) * inner + first_k;
  const auto* a1 = input + size_t(r1 < item.rows ? r1 : 0) * inner + first_k;
  const auto* as = scale + size_t(rs < item.rows ? rs : 0) * scale_stride;
  float acc[4] = {};
  for (int i = 0; i < scale_stride; ++i) {
    const int k = i * 32;
    const auto load = [](const uint8_t* address) { return *reinterpret_cast<const uint32_t*>(address); };
    const uint32_t a00 = r0 < item.rows ? load(a0 + k) : 0;
    const uint32_t a10 = r1 < item.rows ? load(a1 + k) : 0;
    const uint32_t a01 = r0 < item.rows ? load(a0 + k + 16) : 0;
    const uint32_t a11 = r1 < item.rows ? load(a1 + k + 16) : 0;
    const uint32_t b0 = load(w + k), b1 = load(w + k + 16);
    const uint32_t sa = as[i], sb = ws[i];
    constexpr unsigned short zero = 0;
    asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {%11, %12}, %13, {%14, %15};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a00), "r"(a10), "r"(a01), "r"(a11), "r"(b0), "r"(b1), "r"(sa), "h"(zero), "h"(zero),
        "r"(sb), "h"(zero), "h"(zero));
  }
  const int c = first_column + 2 * (lane & 3);
  for (int half = 0; half < 2; ++half) {
    const int row = r0 + 8 * half;
    if (row >= item.rows || c >= columns) continue;
    auto* dst = out + (size_t(item.begin) + row) * columns + c;
    dst[0] = __bfloat16_as_ushort(__float2bfloat16(acc[2 * half]));
    if (c + 1 < columns) dst[1] = __bfloat16_as_ushort(__float2bfloat16(acc[2 * half + 1]));
  }
#endif
}

} // namespace

extern "C" cudaError_t blz_mxfp8_quant_rows(const uint16_t* input, uint8_t* codes, uint8_t* scales,
    int rows, int inner, cudaStream_t stream) {
  if (rows < 1 || rows > 65535 || inner < 32 || inner % 32) return cudaErrorInvalidValue;
  quantize_mxfp8<<<dim3((inner + 127) / 128, rows), 128, 0, stream>>>(input, codes, scales, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_mxfp8_grouped_native(const uint8_t* input, const uint8_t* scales,
    const blz_moe_matrix_tile* tiles, uint16_t* out, int count, int n, int k, cudaStream_t stream) {
  if (count < 1 || count > 65535 || n < 1 || k < 32 || k % 32) return cudaErrorInvalidValue;
  if (!blz_nvfp4_gemm_w4a4_supported()) return cudaErrorNotSupported;
  mxfp8_grouped_mma<<<dim3((n + 31) / 32, 4, count), 128, 0, stream>>>(input, scales, tiles, out, n, k);
  return cudaGetLastError();
}
