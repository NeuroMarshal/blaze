#pragma once

#include <stdint.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200 && defined(__CUDA_ARCH_FEAT_SM120_ALL)
#define BLZ_FP4_MMA 1

namespace blaze {

__device__ __forceinline__ void nvfp4_mma(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa), "h"(uint16_t(0)),
        "h"(uint16_t(0)), "r"(sfb), "h"(uint16_t(0)), "h"(uint16_t(0)));
}

}
#endif
