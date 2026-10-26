// Host thread that serves the mailbox's GPU1 items on the helper GPU: one CUDA graph per token bucket
// (import -> gate/up -> finalize -> down -> export + done1), launched when a request with GPU1 work appears.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <thread>

#include "blaze/mailbox.h"
#include "blaze/moe.h"

namespace blaze {

class Gpu1Service {
 public:
  /// \param device     helper GPU ordinal
  /// \param layer      helper pool description (slot1 pool, mult table of every layer)
  /// \param mail_host  mailbox host address; its device address on `device` is looked up
  Gpu1Service(int device, const blz_moe_layer& layer, uint8_t* mail_host);
  ~Gpu1Service();
  Gpu1Service(const Gpu1Service&) = delete;
  Gpu1Service& operator=(const Gpu1Service&) = delete;

  uint64_t served() const { return served_.load(std::memory_order_relaxed); }

 private:
  void run();
  cudaGraphExec_t graph_for(int tokens);

  int device_;
  blz_moe_layer layer_;
  uint8_t* mail_host_;
  uint8_t* mail_dev_ = nullptr;
  void* ws_mem_ = nullptr;
  uint32_t* counters_ = nullptr;
  blz_moe_workspace ws_{};
  cudaStream_t stream_ = nullptr;
  cudaGraphExec_t graphs_[BLZ_MOE_MAX_TOKENS + 1] = {};
  uint32_t initial_seq_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> served_{0};
  std::thread thread_;
};

}  // namespace blaze
