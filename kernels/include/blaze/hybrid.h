// HybridMoe: the routed-expert runtime of one model on GPU0 (+ GPU1 helper) + CPU + SSD.
//
// Owns: the expert store (the checkpoint), the exclusive placement, the VRAM slot pools of both GPUs, the pinned RAM
// arena, the mailbox, the CPU engine, the GPU1 service and the swap engine. forward() issues one MoE layer on a GPU0
// stream and is CUDA-graph capturable (plan -> GPU0 rows -> wait for CPU / GPU1 -> combine).
//
// Swap engine (src/runtime/swap.cpp): the plan kernel counts routes per expert on the GPU; a host thread turns the
// counts into decayed scores and moves experts between tiers so that the hottest live on the GPUs, then RAM, with
// the coldest left on SSD. Every expert is in exactly one tier. A move copies the record into a spare slot, then
// switches the tables; the vacated slot becomes spare again only after the GPU has started three more layer calls
// (mailbox seq), when no plan made with the old tables can still be running. Outputs do not depend on placement.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "blaze/cpu_engine.h"
#include "blaze/gpu1_service.h"
#include "blaze/moe.h"
#include "blaze/store.h"

namespace blaze {

struct SwapConfig {
  bool enabled = true;
  int period_ms = 50;              // score update and move round
  int max_moves = 16;              // record moves per round and tier boundary
  double halflife_tokens = 1024;   // score decay, in tokens (1k-16k measured within noise; 1k adapts fastest)
  // Decay of the long-term scores that decide RAM vs SSD (and the startup RAM tier): experts used in any recent
  // conversation stay in RAM; only the ones rare across topics fall back to SSD.
  double halflife_ssd_tokens = 262144;
  double hysteresis = 0.25;        // a candidate must be this much hotter than the expert it displaces
  int spare_slots = 32;            // per tier, kept free for moves (taken from the tier's budget)
  int ssd_threads = 4;             // parallel SSD -> RAM record reads (asynchronous to the move rounds)
  // Where an expert evicted from VRAM gets its RAM copy: "pcie" copies it back from the GPU (device -> host, the
  // direction inference barely uses), "ssd" re-reads it from the checkpoint (the GPU link stays idle). Either way the
  // expert keeps serving from VRAM until its RAM copy is complete.
  std::string demote = "pcie";
  // Weight of prefill (wide-route) routes in the scores: a long prompt pre-warms VRAM for its topic without wiping
  // what decoding has learned (its tokens also count for the decay with this weight).
  double prefill_weight = 0.25;
  std::string heat_path;           // routing profile: initial placement from it, saved periodically ("": none)
  int save_period_s = 60;
  int log_period_s = 60;           // expert hit rates and swaps in the log, at most every N s (0: off)
};

/// \brief Parse "key=value,..." (keys: enabled, period_ms, max_moves, halflife, halflife_ssd, hysteresis, spare,
/// ssd_threads, demote, prefill_weight, save_s, log_s)
/// over `base`. Throws std::invalid_argument on an unknown key.
SwapConfig parse_swap_options(const std::string& options, SwapConfig base = {});

struct HybridConfig {
  static constexpr size_t kRamAuto = ~size_t(0);

  std::string model_dir;          // checkpoint directory (compressed-tensors NVFP4 safetensors)
  int topk = 10;
  int gpu0 = 0;
  int gpu1 = -1;                  // helper GPU ordinal, -1: none
  size_t gpu0_bytes = 0;          // VRAM for expert slots on GPU0
  size_t gpu1_bytes = 0;          // VRAM for expert slots on GPU1
  size_t ram_bytes = 0;           // pinned RAM for expert slots; kRamAuto: every expert not in VRAM (no SSD tier)
  int cpu_threads = 16;
  std::vector<int> cpu_cores;     // worker affinity
  std::vector<double> heat;       // [layer * E + expert] routing profile (empty: swap.heat_path, else uniform)
  std::vector<double> heat_long;  // long-term profile for RAM vs SSD (empty: from swap.heat_path, else `heat`)
  int load_threads = 8;           // parallel record reads at startup
  SwapConfig swap;
  // Wide route (prefill): calls with at least wide_min_tokens tokens run on tensor cores over every tier, RAM experts
  // streamed to both GPUs; buffers sized for wide_tokens tokens per pass (0: no wide route).
  int wide_tokens = 4096;
  int wide_min_tokens = 256;
  int wide_slots = 8;             // streamed records per batch (double-buffered per GPU)
  bool wide_a8 = false;           // wide (prefill) route on 8-bit block-floating inputs: 1 MMA per unit instead of 3
  int cpu_fast_cols = 0;          // CPU items with >= this many columns use the approximate Q8 kernels (0: off)
  // Expert deferral of the CPU tier (decode): layers < defer_last may defer when every CPU route ranks >=
  // defer_min_rank in its token; defer_last (the model's last MoE layer) always waits. -1: off.
  int defer_last = -1;               // -2: the checkpoint's last decoder layer (option defer=N)
  int defer_min_rank = 0;
  int imm_heat = 0;                  // extra swap heat for routes ranking < defer_min_rank (blz_moe_layer::imm_heat)
  int defer_ssd_rank = -1;           // SSD-tier items: rank threshold instead of defer_min_rank (-1: the same)
  int cpu_gu_chunk = 2;              // CPU engine: gate/up row groups (16 rows) per task
  int cpu_down_chunk = 4;            // CPU engine: down row groups per task (4/8, 2/4, 1/2: same us per item)
  double host_reserve_gib = 6.0;     // MemAvailable that must stay free after the RAM tier is allocated
  int ssd_io_threads = 8;            // reader threads for SSD-tier experts of the CPU engine (chunks in parallel)
  bool ssd_direct = true;            // O_DIRECT for SSD-tier reads (decode, prefill and SSD -> RAM moves)
};

/// \brief Apply "key=value,..." runtime options (cpu_fast_cols, wide_tokens, wide_min_tokens, wide_slots,
/// host_reserve_gib, wide_a8=0|1, defer_last, defer_min_rank, defer_ssd, imm_heat, cpu_gu_chunk, cpu_down_chunk,
/// ssd_io, ssd_direct=0|1) to `cfg`.
/// Throws std::invalid_argument on an unknown key.
void apply_hybrid_options(const std::string& options, struct HybridConfig& cfg);

/// \brief Routing profile: recent scores (VRAM placement) and long-term ones (RAM vs SSD).
struct HeatProfile {
  std::vector<double> recent, longterm;
};
/// \brief Routing-profile file: "QRTHEAT2", uint32 layers, uint32 experts, then double scores [layers * experts],
/// recent then long-term. "QRTHEAT1" files (recent scores only) load with longterm = recent. Empty if absent.
HeatProfile load_heat(const std::string& path, int num_layers, int num_experts);
void save_heat(const std::string& path, int num_layers, int num_experts, const HeatProfile& heat);

struct HybridStats {
  uint64_t routes[4] = {0, 0, 0, 0};  // routes served per tier (GPU0, GPU1, RAM, SSD), as counted by the swap engine
  uint64_t moves_gpu = 0;             // RAM <-> GPU swaps
  uint64_t moves_ram = 0;             // SSD <-> RAM swaps
  int resident[4] = {0, 0, 0, 0};     // experts per tier
};

class HybridMoe {
 public:
  explicit HybridMoe(const HybridConfig& cfg);
  ~HybridMoe();
  HybridMoe(const HybridMoe&) = delete;
  HybridMoe& operator=(const HybridMoe&) = delete;

  /// \brief One MoE layer for `tokens` (<= 8) columns: out = sum_k w[t][k] * expert_{ids[t][k]}(x[t]).
  /// x, out: [T][H] BF16 bits; ids: [T][topk] int32; w: [T][topk] float. All device pointers on GPU0.
  /// allow_defer: this call is the whole batch (no other chunk of the same layer follows), so with defer_last set its
  /// CPU routes may be deferred to the next layer's call (approximate; see blz_moe_defer).
  cudaError_t forward(int layer, const uint16_t* x, int tokens, const int32_t* ids, const float* w, uint16_t* out,
                      cudaStream_t stream, bool allow_defer = false);

  /// \brief Wide route of one layer for any number of tokens (prefill): same bits as forward(). Host-synchronous
  /// (not graph-capturable); runs on `stream` (GPU0) and GPU1, returns when `out` is complete.
  cudaError_t forward_wide(int layer, const uint16_t* x, int tokens, const int32_t* ids, const float* w,
                           uint16_t* out, cudaStream_t stream);
  int wide_min_tokens() const { return wide_ ? cfg_.wide_min_tokens : 0x7fffffff; }

  /// \brief Current placement (changes while the swap engine runs; read it with the engine stopped or for logs).
  const Placement& placement() const { return placement_; }
  const ExpertStore& store() const { return *store_; }
  int num_layers() const { return store_->num_layers(); }
  int num_experts() const { return store_->num_experts(); }
  expert::Geometry geometry() const { return store_->geometry(); }
  uint64_t cpu_requests() const { return cpu_->served(); }
  uint64_t gpu1_requests() const { return gpu1_ ? gpu1_->served() : 0; }
  HybridStats stats() const;

 private:
  void load_tiers();
  // wide.cpp
  struct Wide;
  void init_wide();
  void free_wide();
  cudaError_t wide_pass(int layer, const uint16_t* x, int tokens, const int32_t* ids, const float* w, uint16_t* out,
                        cudaStream_t stream);
  // swap.cpp
  void swap_loop();
  bool swap_round(std::vector<uint32_t>& delta);
  void swap_gpu_ram(int max_moves);
  void swap_ram_ssd(int max_moves);
  void finish_ssd_loads();
  void ssd_worker();
  void recycle(bool force);
  uint32_t layer_calls() const;
  void save_profile() const;
  void write_gpu_tables(const std::vector<std::pair<int32_t*, int32_t>>& writes);
  uint8_t* gpu_slot(Tier tier, int slot) const;
  uint8_t* ram_slot(int slot) const { return ram_ + size_t(slot) * stride_; }

  static constexpr int kTableStaging = 1024;
  static constexpr int32_t kSlotSsd = -2;  // planning-table mark of SSD-tier experts (blz_moe_layer::slot_of)

  HybridConfig cfg_;
  std::unique_ptr<ExpertStore> store_;
  Placement placement_;
  size_t stride_ = 0;
  int capacity_[3] = {0, 0, 0};    // slots per tier (GPU0, GPU1, RAM)
  // GPU0
  uint8_t* pool0_ = nullptr;
  int32_t* slot0_dev_ = nullptr;   // [L][E]
  int32_t* slot1_dev_ = nullptr;   // [L][E] on GPU0 (planning view of GPU1 residency)
  float* mult0_ = nullptr;         // [L][E][3]
  uint32_t* heat_dev_ = nullptr;   // [L][E] route counters
  void* ws_mem_ = nullptr;
  uint32_t* counters_ = nullptr;
  blz_moe_workspace ws_{};
  // GPU1
  uint8_t* pool1_ = nullptr;
  float* mult1_ = nullptr;
  // host
  uint8_t* ram_ = nullptr;         // pinned arena
  uint8_t* mail_ = nullptr;        // mapped pinned mailbox
  std::vector<const uint8_t*> host_records_;  // mirror of the CPU engine's record table
  std::vector<float> mult_host_;
  std::unique_ptr<cpu::Engine> cpu_;
  std::unique_ptr<Gpu1Service> gpu1_;
  Wide* wide_ = nullptr;               // owned (wide.cpp)
  std::mutex swap_mu_;                 // placement / tables / records: swap rounds vs wide passes
  std::vector<uint32_t> prefill_routes_;  // wide-route route counts since the last swap round (under swap_mu_)
  // swap engine (owned by its thread after construction)
  struct PendingSlot {
    Tier tier;
    int slot;
    uint32_t ready;                // layer_calls() value from which the slot may be reused
    int64_t clear_record = -1;     // expert whose stale RAM record pointer is dropped on reuse
  };
  std::vector<double> score_;
  std::vector<double> score_long_;  // long-term scores: RAM vs SSD
  std::vector<uint32_t> heat_last_;
  uint32_t* heat_host_ = nullptr;  // pinned snapshot of heat_dev_
  int32_t* table_host_ = nullptr;  // pinned staging for table writes (kTableStaging values)
  struct SsdLoad {
    size_t expert;
    int slot;                      // RAM slot being filled
    bool demote = false;           // the expert is leaving VRAM (from_tier, from_slot) rather than SSD
    Tier from_tier = Tier::kSsd;
    int from_slot = -1;
    std::atomic<int> state{0};     // 0 queued / reading, 1 done, 2 failed
    std::string error;
  };
  std::vector<int> free_[3];
  std::vector<PendingSlot> pending_;
  std::vector<uint8_t> loading_;   // [L][E]: an SSD -> RAM read is in flight
  std::vector<std::unique_ptr<SsdLoad>> ssd_inflight_;
  std::deque<SsdLoad*> ssd_queue_;
  std::mutex ssd_mu_;
  std::condition_variable ssd_cv_;
  bool ssd_stop_ = false;
  std::vector<std::thread> ssd_pool_;
  cudaStream_t swap_stream_[2] = {nullptr, nullptr};  // per GPU: table writes (GPU0) and host -> device copies
  cudaStream_t d2h_stream_[2] = {nullptr, nullptr};   // per GPU: device -> host copies (the other copy engine)
  std::atomic<bool> swap_stop_{false};
  std::thread swap_thread_;
  std::atomic<uint64_t> stat_routes_[4] = {};
  std::atomic<uint64_t> stat_moves_gpu_{0}, stat_moves_ram_{0};
  std::atomic<int> stat_resident_[4] = {};
};

}  // namespace blaze
