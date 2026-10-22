// PLE n-gram rows read from the checkpoint's BF16 safetensors shard on disk, through the GPU <-> CPU mailbox pattern
// (CUDA-graph capturable). Nothing is loaded up front: rows are gathered from the mmapped file on demand.
//
//   request (GPU kernel): copies the row ids [n] (int64) into the mapped mailbox and release-stores
//                         req = (seq << 32) | n
//   CPU service        : acquires req, copies the rows (BF16, as stored) into the mapped output, release-stores
//                         done = seq
//   fetch (GPU kernel)  : waits for done == seq and copies the rows into a device tensor [n][row_dim] BF16
// The request can be issued early (as soon as the ids are known) and the fetch late, so the reads overlap compute.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

namespace blaze {

struct PleMailHeader {
  uint64_t req;         // (seq << 32) | n
  uint8_t pad0[56];
  uint32_t done;        // seq of the last served request
  uint8_t pad1[60];
};

class PleService {
 public:
  /// \param path      safetensors file holding the table as BF16 tensors "...ngram_embedding.shard_N.weight"
  ///                  [rows_per_shard, dim] (orcarouter's checkpoint)
  /// \param max_ids   largest request (ids per call)
  /// \param threads   CPU gather threads
  /// \param cache_bytes  RAM for a row cache (8-way set associative, rows of row_dim BF16): the page cache keeps a
  ///                    320-byte row as a 4 KiB page, this keeps 12.8x more hot rows in the same memory; 0: none
  PleService(const std::string& path, int max_ids, int threads, size_t cache_bytes = 0);
  ~PleService();
  PleService(const PleService&) = delete;
  PleService& operator=(const PleService&) = delete;

  int row_dim() const { return row_dim_; }

  /// \brief GPU side: publish `n` row ids (device int64) to the service. Capturable.
  cudaError_t request(const int64_t* ids, int n, cudaStream_t stream);
  /// \brief GPU side: wait for the current request and copy its rows into `out` ([n][row_dim] BF16). Capturable.
  cudaError_t fetch(uint16_t* out, int n, cudaStream_t stream);

  uint64_t served() const { return served_.load(std::memory_order_relaxed); }
  uint64_t cache_hits() const { return hits_.load(std::memory_order_relaxed); }
  uint64_t cache_misses() const { return misses_.load(std::memory_order_relaxed); }

  /// \brief Host side: read these rows into the row cache ahead of their request (a prefill chunk's rows while the
  /// previous chunk computes): a pool of `threads` readers, each reading the 4 KiB block(s) of a row with O_DIRECT
  /// (no page cache; as many reads in flight as readers). Without a row cache: page-cache read-ahead hints.
  /// Returns at once.
  void warm(const int64_t* ids, int n);
  uint64_t warmed() const { return warmed_.load(std::memory_order_relaxed); }

 private:
  void run();
  void gather(uint32_t seq, int n);

  int row_dim_ = 0;
  uint64_t rows_ = 0;
  int fd_ = -1;
  const uint8_t* table_ = nullptr;  // mmap of the shard file
  size_t table_bytes_ = 0;
  int max_ids_;
  int threads_;
  uint8_t* mail_ = nullptr;         // mapped pinned: header, ids [max_ids] int64, rows [max_ids][row_dim] BF16
  uint8_t* mail_dev_ = nullptr;
  uint32_t* seq_dev_ = nullptr;     // device counter
  std::vector<const uint8_t*> shard_;  // row 0 of each table shard
  uint64_t shard_rows_ = 0;
  std::atomic<bool> stop_{false};
  std::atomic<uint64_t> served_{0};
  std::thread thread_;
  std::vector<std::thread> pool_;
  // row cache: sets x 8 ways; tag = id + 1 (0: empty); one spin lock and a round-robin victim per set
  static constexpr int kWays = 8;
  size_t cache_sets_ = 0;
  uint64_t* cache_tags_ = nullptr;
  uint8_t* cache_rows_ = nullptr;
  std::atomic<uint8_t>* cache_lock_ = nullptr;
  uint8_t* cache_next_ = nullptr;
  size_t cache_map_bytes_ = 0;
  void* cache_map_ = nullptr;
  std::atomic<uint64_t> hits_{0}, misses_{0};
  bool cache_get(int64_t id, uint16_t* dst);
  void cache_put(int64_t id, const uint16_t* src);
  // warm-up queue (row ids), its dispatcher and the O_DIRECT reader pool: the dispatcher publishes a job (ids, next
  // index) and waits until the readers have taken every id
  std::mutex warm_mu_;
  std::condition_variable warm_cv_;
  std::vector<int64_t> warm_ids_;
  std::thread warm_thread_;
  int fd_direct_ = -1;
  uint64_t file_bytes_ = 0;
  std::vector<std::thread> warm_pool_;
  std::vector<int64_t> warm_job_;
  std::atomic<size_t> warm_next_{0};
  std::atomic<int> warm_busy_{0};
  std::atomic<uint32_t> warm_gen_{0};
  std::atomic<uint64_t> warmed_{0};
  void warm_reader();
  bool read_direct(const uint8_t* src, uint16_t* dst);
  // Gather job: counter = (generation << 32) | next task, so a helper still on an older job never takes a task of
  // the current one; work_n_ is written before the counter is reset (release).
  std::atomic<uint64_t> work_ctr_{0};
  std::atomic<int> work_done_{0};
  std::atomic<uint32_t> work_gen_{0};
  std::atomic<int> work_n_{0};
  std::atomic<int> work_chunk_{1};
};

}  // namespace blaze
