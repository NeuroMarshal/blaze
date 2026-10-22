// 16-bit GEMM of linear layers through cuBLAS (blaze/ops.h: blz_gemm_bf16).

#include <cublas_v2.h>

#include "blaze/ops.h"

struct blz_gemm {
  cublasHandle_t handle = nullptr;
  void* workspace = nullptr;  // fixed workspace: cuBLAS must not allocate while a CUDA graph is captured
  cudaStream_t stream = nullptr;
  bool bound = false;
};

constexpr size_t kWorkspaceBytes = size_t(32) << 20;

extern "C" int blz_gemm_create(blz_gemm** out) {
  auto* g = new blz_gemm;
  if (cublasCreate(&g->handle) != CUBLAS_STATUS_SUCCESS) {
    delete g;
    return 1;
  }
  cublasSetMathMode(g->handle, CUBLAS_DEFAULT_MATH);
  if (cudaMalloc(&g->workspace, kWorkspaceBytes) != cudaSuccess ||
      cublasSetWorkspace(g->handle, g->workspace, kWorkspaceBytes) != CUBLAS_STATUS_SUCCESS) {
    cublasDestroy(g->handle);
    cudaFree(g->workspace);
    delete g;
    return 1;
  }
  *out = g;
  return 0;
}

extern "C" void blz_gemm_destroy(blz_gemm* g) {
  if (!g) return;
  cublasDestroy(g->handle);
  cudaFree(g->workspace);
  delete g;
}

extern "C" cudaError_t blz_gemm_bf16(blz_gemm* g, const uint16_t* x, int64_t ldx, const uint16_t* w, uint16_t* out,
                                     int64_t ldo, int m, int n, int k, cudaStream_t stream) {
  if (m == 0 || n == 0) return cudaSuccess;
  if (!g->bound || stream != g->stream) {
    // cublasSetStream resets the workspace to cuBLAS's own pool: bind ours again.
    if (cublasSetStream(g->handle, stream) != CUBLAS_STATUS_SUCCESS ||
        cublasSetWorkspace(g->handle, g->workspace, kWorkspaceBytes) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorUnknown;
    g->stream = stream;
    g->bound = true;
  }
  const float alpha = 1.0f, beta = 0.0f;
  // Column-major view: out^T [n, m] = w [n, k] (row-major, i.e. op T of a k x n matrix) * x^T [k, m].
  const cublasStatus_t st =
      cublasGemmEx(g->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF, int(ldx),
                   &beta, out, CUDA_R_16BF, int(ldo), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return st == CUBLAS_STATUS_SUCCESS ? cudaSuccess : cudaErrorUnknown;
}
