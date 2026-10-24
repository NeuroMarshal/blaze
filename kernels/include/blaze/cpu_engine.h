// CPU expert engine of the hybrid MoE: pinned spinning workers serving the GPU's mailbox requests.
//
// Per request (one MoE layer call with n_cpu > 0): tasks are the gate/up row-group chunks of every CPU item, then the
// down chunks. The worker finishing an item's last gate/up chunk computes the item's h (SwiGLU) and encodes it; down
// chunks of the item wait for that. The worker finishing the last task release-stores done = seq.
// Items of the SSD tier are read by a separate pool of reader threads, in the store's chunks (gate/up first), queued
// as soon as the request is polled; a compute task waits only for the chunks holding its row groups, so the workers
// never block in I/O and an item is computed while the rest of it is still being read.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "blaze/cpu_expert.h"
#include "blaze/mailbox.h"
#include "blaze/store.h"

namespace blaze::cpu {

struct EngineConfig {
  expert::Geometry geo;
  int num_layers = 0;
  int num_experts = 0;
  int threads = 1;
  std::vector<int> cores;     // CPU per worker (affinity); empty: no pinning
  int gu_chunk_groups = 4;    // gate/up row groups per task
  int down_chunk_groups = 8;  // down row groups per task
  const ExpertStore* store = nullptr;  // reads records that are not in RAM (SSD tier); null: they must be in RAM
  int ssd_io_threads = 8;     // reader threads for SSD-tier items (their chunks are read in parallel)
  bool ssd_direct = true;     // O_DIRECT reads for SSD-tier items (no page cache)
  // Items with at least this many columns (MTP verify) use the approximate block-floating-point kernels
  // (row_sums_q8_avx2, ~2x less AVX2 work); 0: always exact.
  int fast_min_cols = 0;
  // Choose how many of the workers take part, by measurement while serving (the time per task of windows of jobs at a
  // few counts, the best kept, its neighbours probed now and then); false: all of them, always.
  bool tune_threads = true;
};

/// \brief Parse a CPU list such as "0-23,48-55" (ranges and single CPUs, comma separated).
std::vector<int> parse_cpu_list(const std::string& spec);

class Engine {
 public:
  /// \param mail     host address of the mapped mailbox (blz_mail_layout_for(geo.hidden).total bytes)
  /// \param records  [num_layers * num_experts] host record pointers (null: not resident in RAM)
  /// \param mult     [num_layers * num_experts * 3] gate, up, down multipliers
  Engine(const EngineConfig& cfg, uint8_t* mail, const uint8_t* const* records, const float* mult);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  /// \brief Replace a record pointer (null: read it from the store). Safe while requests are served: a request takes
  /// the pointer when it is polled, so the old record must stay valid until that request is done (see HybridMoe).
  void set_record(int layer, int expert, const uint8_t* record) {
    __atomic_store_n(&records_[size_t(layer) * cfg_.num_experts + expert], record, __ATOMIC_RELEASE);
  }

  /// \brief Requests served so far (for tests and telemetry).
  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  /// \brief Telemetry: summed request time (pickup to done) in ns, and items served.
  uint64_t busy_ns() const { return busy_ns_.load(std::memory_order_relaxed); }
  uint64_t items() const { return items_.load(std::memory_order_relaxed); }
  /// \brief Workers taking part in requests now (all of them unless tune_threads).
  int active_workers() const { return active_.load(std::memory_order_relaxed); }

 private:
  struct Job;
  struct IoTask {
    int layer, expert, chunk;
    uint8_t* record;
    std::atomic<int>* ready;
  };
  void worker(int index);
  void io_worker();
  bool poll_request(Job& job);
  void run_tasks(Job& job, int index);
  void finalize_item(Job& job, int item);
  void tune_note(double ns, int tasks);

  // The worker count tuner (tune_threads): candidate counts, the decayed ns per task measured at each (0: not yet),
  // the current window's sums. Touched only by the worker that completes a job (one at a time).
  struct Tuner {
    std::vector<int> counts;
    std::vector<double> cost;
    int cur = 0, windows = 0, jobs = 0;
    double ns = 0, tasks = 0;
  } tuner_;
  std::atomic<int> active_{1};

  EngineConfig cfg_;
  uint8_t* mail_;
  blz_mail_layout layout_;
  std::vector<const uint8_t*> records_;
  std::vector<float> mult_;
  std::unique_ptr<Job> job_;
  std::vector<void*> staging_;  // record buffers for SSD-tier items of the current job
  std::vector<std::thread> threads_;
  std::vector<std::thread> io_threads_;
  std::mutex io_mu_;
  std::condition_variable io_cv_;
  std::deque<IoTask> io_queue_;
  bool io_stop_ = false;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> job_gen_{0};
  std::atomic<int> workers_left_{0};
  std::atomic<uint64_t> served_{0};
  std::atomic<uint64_t> busy_ns_{0};
  std::atomic<uint64_t> items_{0};
};

}  // namespace blaze::cpu
