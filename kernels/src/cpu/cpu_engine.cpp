// CPU expert engine (see blaze/cpu_engine.h).
#include "blaze/cpu_engine.h"
#include "blaze/log.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace blaze::cpu {

namespace {

constexpr int kMaxCols = BLZ_MOE_MAX_TOKENS;
// Spin while requests keep coming; after this long without one, poll with short sleeps.
constexpr auto kHotWindow = std::chrono::milliseconds(50);
constexpr auto kColdPoll = std::chrono::microseconds(100);

using Clock = std::chrono::steady_clock;

void pin_to_core(int core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(core, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

template <typename T>
T* aligned_array(size_t n) {
  void* p = std::aligned_alloc(64, (n * sizeof(T) + 63) / 64 * 64);
  if (!p) throw std::bad_alloc();
  return static_cast<T*>(p);
}

}  // namespace

struct Engine::Job {
  explicit Job(const expert::Geometry& geo) : geo(geo) {
    const int I = geo.inter;
    gu_sums = aligned_array<int64_t>(size_t(BLZ_MAIL_ITEMS) * geo.gu_rows() * kMaxCols);
    h_bits = aligned_array<uint16_t>(size_t(BLZ_MAIL_ITEMS) * kMaxCols * I);
    h_mem = aligned_array<uint8_t>(size_t(BLZ_MAIL_ITEMS) * kMaxCols * (3 * size_t(I) + 4 * size_t(I / 16) + 64));
    xq = aligned_array<int8_t>(size_t(kMaxCols) * geo.hidden);
    xq_shift = aligned_array<uint8_t>(size_t(kMaxCols) * (geo.hidden / 16));
    hq = aligned_array<int8_t>(size_t(BLZ_MAIL_ITEMS) * kMaxCols * I);
    hq_shift = aligned_array<uint8_t>(size_t(BLZ_MAIL_ITEMS) * kMaxCols * (I / 16));
    uint8_t* p = h_mem;
    for (int i = 0; i < BLZ_MAIL_ITEMS * kMaxCols; ++i) {
      ColumnBuffer& b = h_bufs[i];
      b.lo = p, b.mid = p + I, b.hi = reinterpret_cast<int8_t*>(p + 2 * I);
      b.hi_sum = reinterpret_cast<int32_t*>(p + 3 * I);
      p += 3 * size_t(I) + 4 * size_t(I / 16) + 64;
      p = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(p) + 63) & ~uintptr_t(63));
    }
  }
  ~Job() {
    std::free(gu_sums);
    std::free(h_bits);
    std::free(h_mem);
    std::free(xq), std::free(xq_shift), std::free(hq), std::free(hq_shift);
  }

  expert::Geometry geo;
  uint32_t seq = 0;
  int layer = 0, tokens = 0, n_items = 0;
  int gu_per_item = 0, down_per_item = 0, total = 0;
  blz_moe_item items[BLZ_MAIL_ITEMS];
  const uint8_t* item_record[BLZ_MAIL_ITEMS];
  int order[BLZ_MAIL_ITEMS];        // processing order: RAM items first, SSD items last
  std::atomic<int> loaded[BLZ_MAIL_ITEMS];  // 1: resident in RAM; 0: SSD tier, chunks arriving (chunk_ready)
  int n_chunks = 0;                 // store chunks per record (0: no SSD tier)
  std::unique_ptr<std::atomic<int>[]> chunk_ready;  // [item][chunk]
  Column x_cols[kMaxCols];
  std::atomic<int> next_task{1 << 30};
  std::atomic<int> tasks_done{0};
  int n_imm = 0;                    // items order[0 .. n_imm - 1] are immediate: done_imm once their down tasks finish
  std::atomic<int> imm_down_left{0};
  std::atomic<int> pending_gu[BLZ_MAIL_ITEMS];
  std::atomic<int> h_ready[BLZ_MAIL_ITEMS];
  int64_t* gu_sums;  // [item][2I][cols]
  uint16_t* h_bits;  // [item][col][I]
  uint8_t* h_mem;
  ColumnBuffer h_bufs[BLZ_MAIL_ITEMS * kMaxCols];
  bool fast = false;                // some item takes the approximate kernels
  int8_t* xq;                       // [col][H]
  uint8_t* xq_shift;                // [col][H / 16]
  int8_t* hq;                       // [item][col][I]
  uint8_t* hq_shift;                // [item][col][I / 16]
  uint32_t last_seq = 0;
  Clock::time_point t_start{};
};

Engine::Engine(const EngineConfig& cfg, uint8_t* mail, const uint8_t* const* records, const float* mult)
    : cfg_(cfg),
      mail_(mail),
      layout_(blz_mail_layout_for(cfg.geo.hidden)),
      records_(records, records + size_t(cfg.num_layers) * cfg.num_experts),
      mult_(mult, mult + size_t(cfg.num_layers) * cfg.num_experts * 3),
      job_(std::make_unique<Job>(cfg.geo)) {
  if (!cfg.geo.valid() || cfg.threads < 1) throw std::invalid_argument("blaze::cpu::Engine: bad config");
  if (cfg.store) {
    job_->n_chunks = cfg.store->num_chunks();
    job_->chunk_ready = std::make_unique<std::atomic<int>[]>(size_t(BLZ_MAIL_ITEMS) * job_->n_chunks);
    for (int i = 0; i < std::max(1, cfg.ssd_io_threads); ++i) io_threads_.emplace_back([this] { io_worker(); });
  }
  if (!cfg.cores.empty() && size_t(cfg.threads) > cfg.cores.size())
    log("blaze: %d CPU expert workers on %zu CPUs: several share a CPU (slow)", cfg.threads,
                 cfg.cores.size());
  auto* hdr = reinterpret_cast<blz_mail_header*>(mail_);
  job_->last_seq = blz_req_seq(__atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE));
  active_.store(cfg.threads, std::memory_order_relaxed);
  if (cfg.tune_threads && cfg.threads >= 4) {
    // eighths of the workers from a quarter up, all of them first
    for (int e = 8; e >= 2; --e) {
      const int n = std::max(2, (cfg.threads * e + 4) / 8);
      if (tuner_.counts.empty() || tuner_.counts.back() != n) tuner_.counts.push_back(n);
    }
    tuner_.cost.assign(tuner_.counts.size(), 0.0);
  }
  for (int i = 0; i < cfg.threads; ++i) threads_.emplace_back([this, i] { worker(i); });
}

Engine::~Engine() {
  stop_.store(true, std::memory_order_release);
  for (auto& t : threads_) t.join();
  {
    std::lock_guard<std::mutex> lock(io_mu_);
    io_stop_ = true;
  }
  io_cv_.notify_all();
  for (auto& t : io_threads_) t.join();
  for (void* p : staging_) std::free(p);
}

void Engine::io_worker() {
  for (;;) {
    IoTask t;
    {
      std::unique_lock<std::mutex> lock(io_mu_);
      io_cv_.wait(lock, [&] { return io_stop_ || !io_queue_.empty(); });
      if (io_stop_) return;
      t = io_queue_.front();
      io_queue_.pop_front();
    }
    try {
      cfg_.store->read_chunk(t.layer, t.expert, t.chunk, t.record, cfg_.ssd_direct);
    } catch (const std::exception& e) {
      log("blaze: reading layer %d expert %d from the checkpoint failed: %s", t.layer, t.expert, e.what());
      std::abort();
    }
    t.ready->store(1, std::memory_order_release);
  }
}

std::vector<int> parse_cpu_list(const std::string& spec) {
  std::vector<int> cpus;
  size_t pos = 0;
  while (pos < spec.size()) {
    const size_t comma = spec.find(',', pos);
    const std::string part = spec.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!part.empty()) {
      const size_t dash = part.find('-');
      const int lo = std::stoi(part.substr(0, dash));
      const int hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
      for (int c = lo; c <= hi; ++c) cpus.push_back(c);
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return cpus;
}

bool Engine::poll_request(Job& job) {
  auto* hdr = reinterpret_cast<blz_mail_header*>(mail_);
  const uint64_t req = __atomic_load_n(&hdr->req, __ATOMIC_ACQUIRE);
  const uint32_t seq = blz_req_seq(req), n = uint32_t(blz_req_cpu(req));
  if (seq == job.last_seq) return false;
  job.last_seq = seq;
  if (n == 0) return false;  // the GPU does not wait for requests without CPU items
  const int H = cfg_.geo.hidden;
  job.seq = seq;
  job.layer = hdr->layer;
  job.tokens = hdr->tokens;
  if (job.layer < 0 || job.layer >= cfg_.num_layers) {
    log("blaze: CPU expert request for layer %d of %d", job.layer, cfg_.num_layers);
    std::abort();
  }
  job.n_items = int(n);
  std::copy(hdr->items, hdr->items + n, job.items);
  // Records outside RAM (SSD tier) get a staging buffer; their chunks are queued for the readers below and the items
  // are processed last.
  int n_ssd = 0;
  for (int i = 0; i < int(n); ++i) {
    const uint8_t* rec = __atomic_load_n(&records_[size_t(job.layer) * cfg_.num_experts + job.items[i].expert],
                                         __ATOMIC_ACQUIRE);
    job.item_record[i] = rec;
    job.loaded[i].store(rec != nullptr ? 1 : 0, std::memory_order_relaxed);
    if (rec != nullptr) continue;
    if (cfg_.store == nullptr) {
      log("blaze: layer %d expert %d is not resident and there is no store", job.layer,
                   job.items[i].expert);
      std::abort();
    }
    if (size_t(n_ssd) == staging_.size()) {
      void* buf = std::aligned_alloc(4096, cfg_.store->record_stride());
      if (!buf) throw std::bad_alloc();
      staging_.push_back(buf);
    }
    job.item_record[i] = static_cast<const uint8_t*>(staging_[n_ssd++]);
    for (int c = 0; c < job.n_chunks; ++c) job.chunk_ready[size_t(i) * job.n_chunks + c].store(0, std::memory_order_relaxed);
  }
  // Processing order: immediate items (plan numbers them first) before deferred ones; RAM before SSD within each.
  job.n_imm = std::min(std::max(int(hdr->n_imm), 0), int(n));
  {
    int k = 0;
    for (int group = 0; group < 2; ++group)
      for (int ssd = 0; ssd < 2; ++ssd)
        for (int i = 0; i < int(n); ++i)
          if ((i < job.n_imm) == (group == 0) && (job.loaded[i].load(std::memory_order_relaxed) == 0) == (ssd == 1))
            job.order[k++] = i;
  }
  if (n_ssd > 0) {
    // Reads in processing order: the immediate items' gate/up chunks, then their down chunks, then the deferred ones.
    const int gu_chunks = cfg_.store->chunk_of(true, 0);
    {
      std::lock_guard<std::mutex> lock(io_mu_);
      for (const auto [k0, k1] : {std::pair{0, job.n_imm}, std::pair{job.n_imm, int(n)}})
        for (int down = 0; down < 2; ++down)
          for (int k = k0; k < k1; ++k) {
            const int i = job.order[k];
            if (job.loaded[i].load(std::memory_order_relaxed)) continue;
            for (int c = down ? gu_chunks : 0; c < (down ? job.n_chunks : gu_chunks); ++c)
              io_queue_.push_back({job.layer, job.items[i].expert, c, const_cast<uint8_t*>(job.item_record[i]),
                                   &job.chunk_ready[size_t(i) * job.n_chunks + c]});
          }
    }
    io_cv_.notify_all();
  }
  const uint8_t* planes = mail_ + layout_.planes;
  const int32_t* hi_sum = reinterpret_cast<const int32_t*>(mail_ + layout_.hi_sum);
  for (int t = 0; t < job.tokens; ++t) {
    const uint8_t* p = planes + size_t(t) * 3 * H;
    job.x_cols[t] = {p, p + H, reinterpret_cast<const int8_t*>(p + 2 * H), hi_sum + size_t(t) * (H / 16),
                     hdr->x_emax[t]};
  }
  job.fast = false;
  if (cfg_.fast_min_cols > 0)
    for (int i = 0; i < job.n_items; ++i) job.fast |= job.items[i].ncols >= cfg_.fast_min_cols;
  if (job.fast)
    for (int t = 0; t < job.tokens; ++t)
      quantize_column_q8(job.x_cols[t], H, job.xq + size_t(t) * H, job.xq_shift + size_t(t) * (H / 16));
  const expert::Geometry& geo = cfg_.geo;
  job.gu_per_item = (geo.gu_groups() + cfg_.gu_chunk_groups - 1) / cfg_.gu_chunk_groups;
  job.down_per_item = (geo.d_groups() + cfg_.down_chunk_groups - 1) / cfg_.down_chunk_groups;
  job.total = job.n_items * (job.gu_per_item + job.down_per_item);
  for (int i = 0; i < job.n_items; ++i) {
    job.pending_gu[i].store(job.gu_per_item, std::memory_order_relaxed);
    job.h_ready[i].store(0, std::memory_order_relaxed);
  }
  job.tasks_done.store(0, std::memory_order_relaxed);
  job.imm_down_left.store(job.n_imm < job.n_items ? job.n_imm * job.down_per_item : 0, std::memory_order_relaxed);
  job.t_start = Clock::now();
  job.next_task.store(0, std::memory_order_release);  // publishes the job to every worker
  return true;
}

void Engine::finalize_item(Job& job, int item) {
  const blz_moe_item& it = job.items[item];
  const int I = cfg_.geo.inter;
  const float* m = &mult_[(size_t(job.layer) * cfg_.num_experts + it.expert) * 3];
  const int64_t* sums = job.gu_sums + size_t(item) * cfg_.geo.gu_rows() * kMaxCols;
  for (int c = 0; c < it.ncols; ++c) {
    const int emax = job.x_cols[it.cols[c]].emax;
    uint16_t* h = job.h_bits + (size_t(item) * kMaxCols + c) * I;
    for (int i = 0; i < I; ++i) {
      if (emax < 0) {
        h[i] = canon::kCanonicalNaN;
        continue;
      }
      const uint16_t gv = canon::a16_row_output(sums[size_t(2 * i) * it.ncols + c], emax, m[0]);
      const uint16_t uv = canon::a16_row_output(sums[size_t(2 * i + 1) * it.ncols + c], emax, m[1]);
      h[i] = canon::swiglu_bf16(gv, uv);
    }
    encode_column(h, I, job.h_bufs[item * kMaxCols + c]);
    if (cfg_.fast_min_cols > 0 && it.ncols >= cfg_.fast_min_cols) {
      const size_t qi = size_t(item) * kMaxCols + c;
      quantize_column_q8(job.h_bufs[qi].view(), I, job.hq + qi * I, job.hq_shift + qi * (I / 16));
    }
  }
}

void Engine::run_tasks(Job& job, int index) {
  (void)index;
  const expert::Geometry& geo = cfg_.geo;
  const int H = geo.hidden;
  thread_local std::vector<int64_t> down_sums;
  down_sums.resize(size_t(H) * kMaxCols);
  auto* hdr = reinterpret_cast<blz_mail_header*>(mail_);
  uint16_t* y_base = reinterpret_cast<uint16_t*>(mail_ + layout_.y);
  for (;;) {
    const int task = job.next_task.fetch_add(1, std::memory_order_acq_rel);
    if (task >= job.total) break;
    // Tasks: gate/up then down of the immediate items, then the same for the deferred ones.
    int t = task, base = 0, count = job.n_imm;
    if (t >= job.n_imm * (job.gu_per_item + job.down_per_item)) {
      t -= job.n_imm * (job.gu_per_item + job.down_per_item);
      base = job.n_imm, count = job.n_items - job.n_imm;
    }
    const bool imm = base == 0 && job.n_imm < job.n_items;
    const int gu_tasks = count * job.gu_per_item;
    const bool down = t >= gu_tasks;
    const int local = down ? t - gu_tasks : t;
    const int per_item = down ? job.down_per_item : job.gu_per_item;
    const int item = job.order[base + local / per_item], chunk = local % per_item;
    const blz_moe_item& it = job.items[item];
    const int g0 = chunk * (down ? cfg_.down_chunk_groups : cfg_.gu_chunk_groups);
    const int g1 = std::min(g0 + (down ? cfg_.down_chunk_groups : cfg_.gu_chunk_groups),
                            down ? geo.d_groups() : geo.gu_groups());
    if (job.loaded[item].load(std::memory_order_relaxed) == 0) {
      // SSD tier: wait for the chunks holding row groups [g0, g1).
      const std::atomic<int>* ready = &job.chunk_ready[size_t(item) * job.n_chunks];
      for (int c = cfg_.store->chunk_of(down, g0); c <= cfg_.store->chunk_of(down, g1 - 1); ++c)
        while (ready[c].load(std::memory_order_acquire) == 0) _mm_pause();
    }
    const uint8_t* record = job.item_record[item];
    Column cols[kMaxCols];
    if (!down) {
      int64_t* gu = job.gu_sums + size_t(item) * geo.gu_rows() * kMaxCols;
      if (cfg_.fast_min_cols > 0 && it.ncols >= cfg_.fast_min_cols) {
        QColumn q[kMaxCols];
        for (int c = 0; c < it.ncols; ++c)
          q[c] = {job.xq + size_t(it.cols[c]) * H, job.xq_shift + size_t(it.cols[c]) * (H / 16)};
        row_sums_q8_avx2(record, geo.gu_blocks(), g0, g1, q, it.ncols, gu);
      } else {
        for (int c = 0; c < it.ncols; ++c) cols[c] = job.x_cols[it.cols[c]];
        row_sums_avx2(record, geo.gu_blocks(), g0, g1, cols, it.ncols, gu);
      }
      if (job.pending_gu[item].fetch_sub(1, std::memory_order_acq_rel) == 1) {
        finalize_item(job, item);
        job.h_ready[item].store(1, std::memory_order_release);
      }
    } else {
      while (job.h_ready[item].load(std::memory_order_acquire) == 0) _mm_pause();
      for (int c = 0; c < it.ncols; ++c) cols[c] = job.h_bufs[item * kMaxCols + c].view();
      if (cfg_.fast_min_cols > 0 && it.ncols >= cfg_.fast_min_cols) {
        const int I = geo.inter;
        QColumn q[kMaxCols];
        for (int c = 0; c < it.ncols; ++c) {
          const size_t qi = size_t(item) * kMaxCols + c;
          q[c] = {job.hq + qi * I, job.hq_shift + qi * (I / 16)};
        }
        row_sums_q8_avx2(record + geo.gu_bytes(), geo.d_blocks(), g0, g1, q, it.ncols, down_sums.data());
      } else {
        row_sums_avx2(record + geo.gu_bytes(), geo.d_blocks(), g0, g1, cols, it.ncols, down_sums.data());
      }
      const float md = mult_[(size_t(job.layer) * cfg_.num_experts + it.expert) * 3 + 2];
      for (int c = 0; c < it.ncols; ++c) {
        uint16_t* y = y_base + (size_t(item) * kMaxCols + c) * H;
        const int emax = cols[c].emax;
        for (int r = 16 * g0; r < 16 * g1; ++r)
          y[r] = emax < 0 ? canon::kCanonicalNaN : canon::a16_row_output(down_sums[size_t(r) * it.ncols + c], emax, md);
      }
      if (imm && job.imm_down_left.fetch_sub(1, std::memory_order_acq_rel) == 1)
        __atomic_store_n(&hdr->done_imm, job.seq, __ATOMIC_RELEASE);
    }
    if (job.tasks_done.fetch_add(1, std::memory_order_acq_rel) == job.total - 1) {
      // Park the task counter beyond any job size before the GPU (and then the next request) can move on, so a
      // straggling fetch_add never lands inside the next job while the coordinator is still filling it in.
      job.next_task.store(1 << 30, std::memory_order_relaxed);
      __atomic_store_n(&hdr->done, job.seq, __ATOMIC_RELEASE);
      served_.fetch_add(1, std::memory_order_relaxed);
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - job.t_start).count();
      busy_ns_.fetch_add(uint64_t(ns), std::memory_order_relaxed);
      items_.fetch_add(uint64_t(job.n_items), std::memory_order_relaxed);
      if (!tuner_.counts.empty()) tune_note(double(ns), job.total);
    }
  }
}

// Windows of kTuneJobs jobs (those with at least as many tasks as the larger count would use) at one worker count each:
// first every count from all workers down, then the cheapest per task, with a neighbour every kTuneProbe-th window.
// A request's time per task falls with workers until the memory bandwidth or the other host threads are the limit.
namespace {
constexpr int kTuneJobs = 192;
constexpr int kTuneProbe = 8;
}  // namespace

void Engine::tune_note(double ns, int tasks) {
  Tuner& t = tuner_;
  if (tasks < 2) return;
  t.ns += ns, t.tasks += tasks;
  if (++t.jobs < kTuneJobs) return;
  const double c = t.ns / t.tasks;
  double& m = t.cost[t.cur];
  m = m == 0 ? c : 0.7 * m + 0.3 * c;
  t.jobs = 0, t.ns = 0, t.tasks = 0;
  ++t.windows;
  const int n = int(t.counts.size());
  int next;
  const auto unmeasured = std::find(t.cost.begin(), t.cost.end(), 0.0);
  if (unmeasured != t.cost.end()) {
    next = int(unmeasured - t.cost.begin());
  } else {
    int best = 0;
    for (int i = 1; i < n; ++i)
      if (t.cost[i] < t.cost[best]) best = i;
    next = best;
    if (t.windows % kTuneProbe == 0) next = (t.windows / kTuneProbe) % 2 ? std::max(0, best - 1) : std::min(n - 1, best + 1);
    if (t.windows == n)
      log("blaze: CPU experts: %d of %d workers (%.1f us per task; all workers %.1f us)", t.counts[best], cfg_.threads,
          t.cost[best] / 1e3, t.cost[0] / 1e3);
  }
  t.cur = next;
  active_.store(t.counts[next], std::memory_order_relaxed);
}

void Engine::worker(int index) {
  if (!cfg_.cores.empty()) pin_to_core(cfg_.cores[size_t(index) % cfg_.cores.size()]);
  Job& job = *job_;
  uint64_t seen = job_gen_.load(std::memory_order_acquire);
  auto last_work = Clock::now();
  while (!stop_.load(std::memory_order_acquire)) {
    bool got = false;
    if (index == 0) {
      if (poll_request(job)) {
        job_gen_.fetch_add(1, std::memory_order_release);
        got = true;
      }
    } else {
      const uint64_t g = job_gen_.load(std::memory_order_acquire);
      if (g != seen) {
        seen = g;
        got = true;
      }
    }
    if (got && index < active_.load(std::memory_order_relaxed)) {
      run_tasks(job, index);
      last_work = Clock::now();
    } else if (got) {
      // a worker the tuner left out: the job is the others'
    } else if (Clock::now() - last_work < kHotWindow) {
      _mm_pause();
    } else {
      std::this_thread::sleep_for(kColdPoll);
    }
  }
}

}  // namespace blaze::cpu
