// Routed experts read straight from the checkpoint (compressed-tensors NVFP4 safetensors, as published) and the
// initial placement of experts over the tiers (GPU0 / GPU1 / RAM).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "blaze/expert.h"
#include "blaze/safetensors.h"

namespace blaze {

/// \brief The routed experts of an NVFP4 checkpoint directory, exported by llm-compressor (compressed-tensors:
/// weight_packed / weight_scale / weight_global_scale) or ModelOpt (weight / weight_scale / weight_scale_2), one
/// tensor set per `<prefix><layer>.mlp.experts.<e>.<gate|up|down>_proj`. The layer prefix is found in the checkpoint
/// (MTP blocks excluded). read() lays one expert out in the kernels' rg16 record in memory (codes and block scales
/// copied, nothing requantized); mult() holds the per-matrix multiplier of value = e2m1 * e4m3 * mult.
class ExpertStore {
 public:
  /// \param prefix_hint  decoder layer prefix to use (e.g. "model.layers."); empty: found in the checkpoint
  /// \param chunk_bytes  checkpoint bytes per read chunk (see Chunk)
  explicit ExpertStore(const std::string& model_dir, const std::string& prefix_hint = "",
                       size_t chunk_bytes = 192 << 10);
  ExpertStore(const ExpertStore&) = delete;
  ExpertStore& operator=(const ExpertStore&) = delete;

  expert::Geometry geometry() const { return geo_; }
  int num_layers() const { return layers_; }
  int num_experts() const { return experts_; }
  const std::string& prefix() const { return prefix_; }
  /// \brief Bytes between records in the tier pools (record bytes rounded up to 256).
  size_t record_stride() const { return stride_; }
  /// \brief gate, up, down multipliers of one expert.
  const float* mult(int layer, int expert) const { return &mult_[(size_t(layer) * experts_ + expert) * 3]; }

  /// \brief One expert as an rg16 record in `dst` (record_stride bytes). Thread-safe. direct: O_DIRECT reads.
  void read(int layer, int expert, void* dst, bool direct = false) const;

  /// \brief A record read in pieces (the SSD tier): chunk c covers record row groups [g0, g1) of the gate/up matrix,
  /// or of the down matrix. Gate/up chunks come first. Each is ~chunk_bytes of checkpoint bytes in 2-4 reads.
  struct Chunk {
    bool down;
    int g0, g1;
  };
  int num_chunks() const { return gu_chunks_ + down_chunks_; }
  Chunk chunk(int c) const;
  /// \brief The chunk holding row group `g` of the gate/up (down = false) or down matrix.
  int chunk_of(bool down, int g) const { return down ? gu_chunks_ + g / down_chunk_groups_ : g / gu_chunk_groups_; }
  /// \brief Read chunk c of an expert and pack it into the record at `dst` (other chunks untouched; the padding after
  /// the record is not written). Thread-safe.
  void read_chunk(int layer, int expert, int c, void* dst, bool direct) const;

 private:
  struct Mat {
    const SafetensorsDir::Tensor* codes;
    const SafetensorsDir::Tensor* scales;
  };
  std::unique_ptr<SafetensorsDir> st_;
  expert::Geometry geo_{};
  int layers_ = 0, experts_ = 0;
  std::string prefix_;
  size_t stride_ = 0;
  int gu_chunk_groups_ = 1, down_chunk_groups_ = 1, gu_chunks_ = 0, down_chunks_ = 0;
  std::vector<Mat> mats_;     // [layer][expert][gate, up, down]
  std::vector<float> mult_;   // [layer][expert][3]
};

/// \brief Where each expert lives. GPU tiers hold a slot index; RAM holds an arena slot; SSD means neither.
enum class Tier : uint8_t { kGpu0 = 0, kGpu1 = 1, kRam = 2, kSsd = 3 };

struct Placement {
  int num_layers = 0, num_experts = 0;
  std::vector<Tier> tier;      // [layer * E + expert]
  std::vector<int32_t> slot;   // slot within the tier's pool (-1 for SSD)
  int used[4] = {0, 0, 0, 0};  // slots used per tier

  Tier tier_of(int l, int e) const { return tier[size_t(l) * num_experts + e]; }
  int32_t slot_of(int l, int e) const { return slot[size_t(l) * num_experts + e]; }
};

/// \brief Exclusive placement: hottest experts to GPU0, then GPU1, then RAM; the rest stay on SSD.
/// `heat` is [layer * E + expert] (e.g. routing counts from a profile); empty means uniform heat, in which case
/// experts are spread evenly over the layers (expert ids interleaved), so every layer gets its share of each tier.
/// `ram_heat` (same shape, optional) ranks the experts left after the GPUs for RAM vs SSD instead of `heat`.
Placement place_experts(int num_layers, int num_experts, const std::vector<double>& heat, int gpu0_slots,
                        int gpu1_slots, int ram_slots, const std::vector<double>& ram_heat = {});

}  // namespace blaze
