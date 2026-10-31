cmake_minimum_required(VERSION 3.28)
project(blaze_kernels LANGUAGES CXX CUDA)

# Server (RTX 5070 Ti): -DCMAKE_CUDA_ARCHITECTURES=120a; laptop (RTX 4060): 89. Default: the local GPU.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Canonical arithmetic (blaze/canon.h) needs IEEE host math: no contraction into FMAs, no fast math.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-ffp-contract=off>)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

# GPU kernels: plain C API over raw pointers. No fast math anywhere: the quantizers are checked bit for bit
# against their references.
file(GLOB BLAZE_CUDA_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu)
add_library(blaze_kernels STATIC ${BLAZE_CUDA_SOURCES})
target_include_directories(blaze_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(blaze_kernels PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
target_link_libraries(blaze_kernels PUBLIC CUDA::cudart CUDA::cublas)
target_compile_options(blaze_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --expt-relaxed-constexpr -Xcompiler=-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

# CPU side of the MoE engine (experts in RAM / on SSD) and checkpoint I/O. AVX2 + FMA is the server's ISA
# (EPYC 7K62, Zen 2); no -march=native so one build runs on both machines.
file(GLOB BLAZE_CPU_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/*.cpp)
add_library(blaze_cpu STATIC ${BLAZE_CPU_SOURCES})
target_include_directories(blaze_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(blaze_cpu PRIVATE -O3 -mavx2 -mfma -Wall -Wextra)
target_link_libraries(blaze_cpu PUBLIC Threads::Threads CUDA::cudart blaze_kernels)

option(BLAZE_BUILD_TESTS "Build the kernel tests and tools" ON)
if(BLAZE_BUILD_TESTS)
  enable_testing()
  foreach(t test_cpu_expert test_store test_cpu_ssd)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE blaze_cpu)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  add_executable(test_hybrid_graph tests/test_hybrid_graph.cu)
  target_link_libraries(test_hybrid_graph PRIVATE blaze_cpu blaze_kernels)
  add_test(NAME test_hybrid_graph COMMAND test_hybrid_graph)
  foreach(t test_nvfp4_kv test_nvfp4_qsa test_canon test_moe_gpu test_hc_mix test_moe_wide test_host_allreduce
          test_skinny_gemm test_nvfp4_mha test_ops test_nvfp4_linear test_fp8_linear test_qwen4 test_sampling test_qsa_index test_eagle test_tree test_kv_host test_nvfp4_gemm)
    add_executable(${t} tests/${t}.cu)
    target_link_libraries(${t} PRIVATE blaze_kernels)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
  add_executable(membw tools/membw.cpp)
  target_link_libraries(membw PRIVATE blaze_cpu)
  target_compile_options(membw PRIVATE -O2 -mavx2)
  add_executable(bench-store tools/bench_store.cpp)
  target_link_libraries(bench-store PRIVATE blaze_cpu)
  add_executable(bench-hybrid tools/bench_hybrid.cpp)
  target_link_libraries(bench-hybrid PRIVATE blaze_cpu blaze_kernels)
  # Benchmarks are built on demand and are never part of ctest's correctness run.
  foreach(t bench_attn_prefill bench_mma_peak)
    add_executable(${t} EXCLUDE_FROM_ALL tests/${t}.cu)
    target_link_libraries(${t} PRIVATE blaze_kernels)
  endforeach()
  # the pair verify kernel with its phase clocks: builds mha_verify.cu into itself
  add_executable(bench_verify_prof EXCLUDE_FROM_ALL tests/bench_verify_prof.cu)
  target_include_directories(bench_verify_prof PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(bench_verify_prof PRIVATE --expt-relaxed-constexpr)
  target_link_libraries(bench_verify_prof PRIVATE CUDA::cudart)
  target_link_libraries(test_nvfp4_mha PRIVATE Threads::Threads)
endif()
