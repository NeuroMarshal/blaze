#include <memory>
#include <cstdio>
#include "blaze/ple.h"

#include <fcntl.h>
#include <immintrin.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>


namespace blaze {

// Kernels (src/cuda/ple.cu).
cudaError_t ple_request_launch(const int64_t* ids, int n, uint8_t* mail, uint32_t* seq, cudaStream_t s);
cudaError_t ple_fetch_launch(uint16_t* out, int n, int row_dim, const uint8_t* mail, const uint32_t* seq,
                             size_t rows_offset, cudaStream_t s);

namespace {

constexpr size_t kIdsOffset = 128;
constexpr int kMaxChunk = 256;  // ids per gather task; decode-sized requests go one id per task (parallel page faults)

size_t rows_offset(int max_ids) { return (kIdsOffset + size_t(max_ids) * 8 + 255) & ~size_t(255); }

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string("blaze ple: ") + what + ": " + cudaGetErrorString(e));
}

}  // namespace

namespace {

// BF16 table inside a safetensors file: the "...ngram_embedding.shard_N.weight" tensors, by N.
struct Bf16Shards {
  std::vector<uint64_t> offset;  // absolute file offsets
  uint64_t rows = 0, dim = 0;
};

Bf16Shards find_bf16_shards(int fd, const std::string& path) {
  uint64_t n = 0;
  if (pread(fd, &n, 8, 0) != 8 || n > (1ull << 30)) throw std::runtime_error(path + ": not a safetensors file");
  std::string h(n, '\0');
  if (pread(fd, h.data(), n, 8) != ssize_t(n)) throw std::runtime_error(path + ": short header");
  std::vector<std::pair<int, uint64_t>> found;
  Bf16Shards t;
  const std::string tag = "ngram_embedding.shard_";
  for (size_t at = h.find(tag); at != std::string::npos; at = h.find(tag, at + 1)) {
    const int idx = std::stoi(h.substr(at + tag.size()));
    const size_t obj = h.find('{', at), end = h.find('}', obj);
    const std::string o = h.substr(obj, end - obj);
    if (o.find("\"BF16\"") == std::string::npos) throw std::runtime_error(path + ": n-gram shard is not BF16");
    const size_t s0 = o.find('[', o.find("\"shape\""));
    const uint64_t rows = std::stoull(o.substr(s0 + 1)), dim = std::stoull(o.substr(o.find(',', s0) + 1));
    const size_t f0 = o.find('[', o.find("\"data_offsets\""));
    found.push_back({idx, 8 + n + std::stoull(o.substr(f0 + 1))});
    if (t.rows && (rows != t.rows || dim != t.dim)) throw std::runtime_error(path + ": n-gram shards differ in shape");
    t.rows = rows, t.dim = dim;
  }
  if (found.empty()) throw std::runtime_error(path + ": no ngram_embedding.shard_N tensors");
  std::sort(found.begin(), found.end());
  for (size_t i = 0; i < found.size(); ++i) {
    if (found[i].first != int(i)) throw std::runtime_error(path + ": n-gram shards are not 0..N-1");
    t.offset.push_back(found[i].second);
  }
  return t;
}

}  // namespace

bool PleService::cache_get(int64_t id, uint16_t* dst) {
  if (!cache_sets_) return false;
  const size_t set = size_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 20) % cache_sets_;
  const size_t row_bytes = size_t(row_dim_) * 2;
  auto& lk = cache_lock_[set];
  uint8_t expected = 0;
  while (!lk.compare_exchange_weak(expected, 1, std::memory_order_acquire)) expected = 0, _mm_pause();
  bool hit = false;
  for (int w = 0; w < kWays; ++w)
    if (cache_tags_[set * kWays + w] == uint64_t(id) + 1) {
      std::memcpy(dst, cache_rows_ + (set * kWays + w) * row_bytes, row_bytes);
      hit = true;
      break;
    }
  lk.store(0, std::memory_order_release);
  return hit;
}

void PleService::cache_put(int64_t id, const uint16_t* src) {
  if (!cache_sets_) return;
  const size_t set = size_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 20) % cache_sets_;
  const size_t row_bytes = size_t(row_dim_) * 2;
  auto& lk = cache_lock_[set];
  uint8_t expected = 0;
  while (!lk.compare_exchange_weak(expected, 1, std::memory_order_acquire)) expected = 0, _mm_pause();
  int way = -1;
  for (int w = 0; w < kWays && way < 0; ++w)
    if (cache_tags_[set * kWays + w] == uint64_t(id) + 1) way = w;  // another thread inserted it
  if (way < 0) {
    way = cache_next_[set] % kWays;
    cache_next_[set] = uint8_t(way + 1);
    cache_tags_[set * kWays + way] = uint64_t(id) + 1;
    std::memcpy(cache_rows_ + (set * kWays + way) * row_bytes, src, row_bytes);
  }
  lk.store(0, std::memory_order_release);
}

PleService::PleService(const std::string& path, int max_ids, int threads, size_t cache_bytes)
    : max_ids_(max_ids), threads_(threads) {
  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) throw std::runtime_error("cannot open " + path);
  const Bf16Shards t = find_bf16_shards(fd_, path);
  row_dim_ = int(t.dim);
  rows_ = t.rows * t.offset.size();
  table_bytes_ = size_t(lseek(fd_, 0, SEEK_END));
  shard_rows_ = t.rows;
  void* m = mmap(nullptr, table_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
  if (m == MAP_FAILED) throw std::runtime_error("cannot map " + path);
  madvise(m, table_bytes_, MADV_RANDOM);
  table_ = static_cast<const uint8_t*>(m);
  for (uint64_t off : t.offset) shard_.push_back(table_ + off);
  if (cache_bytes > 0) {
    const size_t row_bytes = size_t(row_dim_) * 2, per_set = kWays * (row_bytes + 8) + 2;
    cache_sets_ = std::max<size_t>(1, cache_bytes / per_set);
    cache_map_bytes_ = cache_sets_ * per_set + 4096;
    cache_map_ = mmap(nullptr, cache_map_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (cache_map_ == MAP_FAILED) throw std::runtime_error("PLE row cache: out of memory");
    // 4 KiB pages: a first touch of a huge page under memory pressure (pinned experts fill most of RAM) stalls in
    // direct compaction for milliseconds, on the decode path (Flash Next decode 62.6 -> 6.1 tok/s with huge pages)
    madvise(cache_map_, cache_map_bytes_, MADV_NOHUGEPAGE);
    auto* base = static_cast<uint8_t*>(cache_map_);
    cache_tags_ = reinterpret_cast<uint64_t*>(base);
    cache_rows_ = base + cache_sets_ * kWays * 8;
    cache_lock_ = reinterpret_cast<std::atomic<uint8_t>*>(cache_rows_ + cache_sets_ * kWays * row_bytes);
    cache_next_ = reinterpret_cast<uint8_t*>(cache_lock_) + cache_sets_;
  }
  const size_t bytes = rows_offset(max_ids_) + size_t(max_ids_) * row_dim_ * 2;
  check(cudaHostAlloc(reinterpret_cast<void**>(&mail_), bytes, cudaHostAllocMapped | cudaHostAllocPortable), "mailbox");
  std::memset(mail_, 0, bytes);
  check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mail_dev_), mail_, 0), "mailbox device pointer");
  check(cudaMalloc(&seq_dev_, 4), "seq");
  check(cudaMemset(seq_dev_, 0, 4), "seq");
  for (int t = 1; t < threads_; ++t) {
    // Helpers block on the job generation (futex): the gather has slack (it is requested before layer 0 and read at
    // the PLE layer), so no core spins for it.
    pool_.emplace_back([this] {
      uint32_t seen = work_gen_.load(std::memory_order_acquire);
      while (!stop_.load(std::memory_order_acquire)) {
        work_gen_.wait(seen, std::memory_order_acquire);
        const uint32_t g = work_gen_.load(std::memory_order_acquire);
        if (g == seen) continue;
        seen = g;
        if (stop_.load(std::memory_order_acquire)) break;
        gather(g, -1);
      }
    });
  }
  thread_ = std::thread([this] { run(); });
  file_bytes_ = table_bytes_;
  if (cache_sets_) {
    fd_direct_ = open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_direct_ >= 0)
      for (int t = 0; t < threads_; ++t) warm_pool_.emplace_back([this] { warm_reader(); });
  }
  warm_thread_ = std::thread([this] {
    std::vector<int64_t> ids;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(warm_mu_);
        warm_cv_.wait(lk, [&] { return stop_.load(std::memory_order_acquire) || !warm_ids_.empty(); });
        if (stop_.load(std::memory_order_acquire)) return;
        ids.swap(warm_ids_);
      }
      if (!warm_pool_.empty()) {
        // hand the ids to the readers and wait until all are taken and read
        warm_job_.swap(ids);
        warm_next_.store(0, std::memory_order_relaxed);
        warm_busy_.store(int(warm_pool_.size()), std::memory_order_relaxed);
        warm_gen_.fetch_add(1, std::memory_order_release);
        warm_gen_.notify_all();
        while (warm_busy_.load(std::memory_order_acquire) > 0 && !stop_.load(std::memory_order_acquire))
          std::this_thread::sleep_for(std::chrono::microseconds(200));
        warm_job_.clear();
        ids.clear();
        continue;
      }
      const size_t page = 4096, row_bytes = size_t(row_dim_) * 2;
      for (int64_t id : ids) {
        if (id < 0 || uint64_t(id) >= rows_) continue;
        if (cache_sets_) {
          // a racy tag look-up is enough for a hint
          const size_t set = size_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 20) % cache_sets_;
          bool cached = false;
          for (int w = 0; w < kWays; ++w) cached |= cache_tags_[set * kWays + w] == uint64_t(id) + 1;
          if (cached) continue;
        }
        const uint8_t* p = shard_[size_t(uint64_t(id) / shard_rows_)] + size_t(uint64_t(id) % shard_rows_) * row_bytes;
        const uintptr_t a = reinterpret_cast<uintptr_t>(p) & ~(page - 1);
        const uintptr_t b = (reinterpret_cast<uintptr_t>(p) + row_bytes + page - 1) & ~(page - 1);
        madvise(reinterpret_cast<void*>(a), b - a, MADV_WILLNEED);
      }
      ids.clear();
    }
  });
}

// One O_DIRECT reader: takes ids of the current job in batches, reads each row's 4 KiB block(s) into an aligned
// buffer and puts the row into the cache.
void PleService::warm_reader() {
  const size_t blk = 4096, row_bytes = size_t(row_dim_) * 2;
  void* buf = nullptr;
  if (posix_memalign(&buf, blk, 2 * blk) != 0) return;
  auto* b = static_cast<uint8_t*>(buf);
  uint32_t seen = warm_gen_.load(std::memory_order_acquire);
  while (!stop_.load(std::memory_order_acquire)) {
    warm_gen_.wait(seen, std::memory_order_acquire);
    const uint32_t g = warm_gen_.load(std::memory_order_acquire);
    if (g == seen) continue;
    seen = g;
    if (stop_.load(std::memory_order_acquire)) break;
    const size_t n = warm_job_.size();
    for (;;) {
      const size_t i0 = warm_next_.fetch_add(16, std::memory_order_relaxed);
      if (i0 >= n) break;
      for (size_t i = i0; i < std::min(n, i0 + 16); ++i) {
        const int64_t id = warm_job_[i];
        if (id < 0 || uint64_t(id) >= rows_) continue;
        // a racy tag look-up is enough to skip a cached row
        const size_t set = size_t((uint64_t(id) * 0x9E3779B97F4A7C15ull) >> 20) % cache_sets_;
        bool cached = false;
        for (int w = 0; w < kWays; ++w) cached |= cache_tags_[set * kWays + w] == uint64_t(id) + 1;
        if (cached) continue;
        const size_t off = size_t(shard_[size_t(uint64_t(id) / shard_rows_)] - table_) +
                           size_t(uint64_t(id) % shard_rows_) * row_bytes;
        const size_t a = off & ~(blk - 1), len = ((off + row_bytes + blk - 1) & ~(blk - 1)) - a;
        if (a + len > file_bytes_ || pread(fd_direct_, b, len, off_t(a)) != ssize_t(len)) continue;
        cache_put(id, reinterpret_cast<const uint16_t*>(b + (off - a)));
        warmed_.fetch_add(1, std::memory_order_relaxed);
      }
    }
    warm_busy_.fetch_sub(1, std::memory_order_acq_rel);
  }
  free(buf);
}

void PleService::warm(const int64_t* ids, int n) {
  {
    std::lock_guard<std::mutex> lk(warm_mu_);
    warm_ids_.insert(warm_ids_.end(), ids, ids + n);
  }
  warm_cv_.notify_one();
}

PleService::~PleService() {
  stop_.store(true, std::memory_order_release);
  work_gen_.fetch_add(1, std::memory_order_release);
  work_gen_.notify_all();
  {
    std::lock_guard<std::mutex> lk(warm_mu_);
  }
  warm_cv_.notify_all();
  warm_gen_.fetch_add(1, std::memory_order_release);
  warm_gen_.notify_all();
  warm_thread_.join();
  for (auto& t : warm_pool_) t.join();
  if (fd_direct_ >= 0) close(fd_direct_);
  thread_.join();
  for (auto& t : pool_) t.join();
  cudaFree(seq_dev_);
  cudaFreeHost(mail_);
  munmap(const_cast<uint8_t*>(table_), table_bytes_);
  if (cache_map_) {
    const uint64_t h = hits_.load(), m = misses_.load();
    std::fprintf(stderr, "blaze: PLE row cache %.2f GiB: %llu hits, %llu misses (%.1f%%), %llu rows read ahead\n",
                 double(cache_map_bytes_) / double(1ull << 30), (unsigned long long)h, (unsigned long long)m,
                 h + m ? 100.0 * double(h) / double(h + m) : 0.0, (unsigned long long)warmed_.load());
    munmap(cache_map_, cache_map_bytes_);
  }
  close(fd_);
}

cudaError_t PleService::request(const int64_t* ids, int n, cudaStream_t stream) {
  if (n > max_ids_) return cudaErrorInvalidValue;
  return ple_request_launch(ids, n, mail_dev_, seq_dev_, stream);
}

cudaError_t PleService::fetch(uint16_t* out, int n, cudaStream_t stream) {
  if (n > max_ids_) return cudaErrorInvalidValue;
  return ple_fetch_launch(out, n, row_dim_, mail_dev_, seq_dev_, rows_offset(max_ids_), stream);
}

// Gather tasks of `kChunk` ids. The service thread (n >= 0) publishes the job and waits for it; helpers (n = -1) join
// the job of generation `gen` and leave as soon as the counter shows another generation.
void PleService::gather(uint32_t gen, int n) {
  if (n >= 0) {
    gen = work_gen_.load(std::memory_order_relaxed) + 1;
    work_n_.store(n, std::memory_order_relaxed);
    work_chunk_.store(std::clamp((n + 4 * threads_ - 1) / (4 * threads_), 1, kMaxChunk), std::memory_order_relaxed);
    work_done_.store(0, std::memory_order_relaxed);
    work_ctr_.store(uint64_t(gen) << 32, std::memory_order_release);
    work_gen_.store(gen, std::memory_order_release);
    work_gen_.notify_all();
  }
  const int count = work_n_.load(std::memory_order_acquire);
  const int chunk = work_chunk_.load(std::memory_order_acquire);
  const int total = (count + chunk - 1) / chunk;
  const auto* ids = reinterpret_cast<const int64_t*>(mail_ + kIdsOffset);
  auto* rows = reinterpret_cast<uint16_t*>(mail_ + rows_offset(max_ids_));
  const int dim = row_dim_;
  for (;;) {
    const uint64_t v = work_ctr_.fetch_add(1, std::memory_order_acq_rel);
    if (uint32_t(v >> 32) != gen) break;
    const int t = int(uint32_t(v));
    if (t >= total) break;
    const int end = std::min(count, (t + 1) * chunk);
    for (int i = t * chunk; i < end; ++i) {
      const int64_t id = ids[i];
      uint16_t* dst = rows + size_t(i) * dim;
      if (id < 0 || uint64_t(id) >= rows_) {
        std::memset(dst, 0, size_t(dim) * 2);
        continue;
      }
      if (cache_get(id, dst)) {
        hits_.fetch_add(1, std::memory_order_relaxed);
        continue;
      }
      misses_.fetch_add(1, std::memory_order_relaxed);
      std::memcpy(dst, shard_[size_t(uint64_t(id) / shard_rows_)] + size_t(uint64_t(id) % shard_rows_) * dim * 2,
                  size_t(dim) * 2);
      cache_put(id, dst);
    }
    work_done_.fetch_add(1, std::memory_order_acq_rel);
  }
  if (n >= 0)
    while (work_done_.load(std::memory_order_acquire) < total) _mm_pause();
}

void PleService::run() {
  auto* hdr = reinterpret_cast<PleMailHeader*>(mail_);
  uint32_t last = uint32_t(__atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE) >> 32);
  while (!stop_.load(std::memory_order_acquire)) {
    const uint64_t req = __atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE);
    const uint32_t seq = uint32_t(req >> 32);
    if (seq != last) {
      last = seq;
      gather(0, int(req & 0xFFFFFFFFu));
      __atomic_store_n(&hdr->done, seq, __ATOMIC_RELEASE);
      served_.fetch_add(1, std::memory_order_relaxed);
    } else {
      // Poll, not spin: a request is served within one timer slice (~60 us) of the prefetch, well before its use.
      std::this_thread::sleep_for(std::chrono::microseconds(20));
    }
  }
}

}  // namespace blaze

// C API (qwen4.h)
#include "blaze/qwen4.h"

struct blz_ple {
  std::unique_ptr<blaze::PleService> svc;
};

extern "C" int blz_ple_create(const char* path, int max_ids, int threads, double cache_gib, blz_ple** out, char* err,
                              size_t err_len) {
  try {
    auto* p = new blz_ple;
    p->svc = std::make_unique<blaze::PleService>(path, max_ids, threads, size_t(cache_gib * double(1ull << 30)));
    *out = p;
    return 0;
  } catch (const std::exception& e) {
    if (err && err_len) std::snprintf(err, err_len, "%s", e.what());
    return 1;
  }
}

extern "C" void blz_ple_destroy(blz_ple* p) { delete p; }

extern "C" int blz_ple_row_dim(const blz_ple* p) { return p->svc->row_dim(); }

extern "C" cudaError_t blz_ple_request(blz_ple* p, const int64_t* ids, int n, cudaStream_t stream) {
  return p->svc->request(ids, n, stream);
}

extern "C" cudaError_t blz_ple_fetch(blz_ple* p, uint16_t* out, int n, cudaStream_t stream) {
  return p->svc->fetch(out, n, stream);
}

extern "C" void blz_ple_warm(blz_ple* p, const int64_t* host_ids, int n) { p->svc->warm(host_ids, n); }
