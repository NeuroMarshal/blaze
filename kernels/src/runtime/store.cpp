#include "blaze/store.h"

#include "blaze/repack.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace blaze {

namespace {

// Per-expert NVFP4 tensor names of the two common exporters (value = e2m1 * e4m3 * multiplier):
//   compressed-tensors (llm-compressor): <p>.weight_packed, <p>.weight_scale, <p>.weight_global_scale; m = 1 / global
//   ModelOpt:                            <p>.weight (U8),   <p>.weight_scale, <p>.weight_scale_2;      m = scale_2
struct Format {
  const char* codes;
  const char* scales;
  const char* global;
  bool reciprocal;
};
constexpr Format kFormats[] = {{"weight_packed", "weight_scale", "weight_global_scale", true},
                               {"weight", "weight_scale", "weight_scale_2", false}};

}  // namespace

ExpertStore::ExpertStore(const std::string& model_dir, const std::string& prefix_hint, size_t chunk_bytes,
                         const std::string& extra_dir)
    : st_(std::make_unique<SafetensorsDir>(model_dir)) {
  // Find "<prefix>0.mlp.experts.0.gate_proj.<codes>" with U8 codes: the prefix of the decoder layers (the MTP block's
  // "mtp." layers are not served) and the exporter's naming.
  std::string prefix;
  const Format* fmt = nullptr;
  const std::string tail = "0.mlp.experts.0.gate_proj.";
  for (const auto& [name, t] : st_->tensors()) {
    const size_t at = name.find(tail);
    if (at == std::string::npos || t.dtype != "U8" || name.rfind("mtp.", 0) == 0) continue;
    const std::string p = name.substr(0, at), part = name.substr(at + tail.size());
    if (!prefix_hint.empty() && p != prefix_hint) continue;
    if (p.size() < 7 || p.compare(p.size() - 7, 7, "layers.") != 0) continue;
    for (const Format& f : kFormats)
      if (part == f.codes) prefix = p, fmt = &f;
    if (fmt) break;
  }
  if (!fmt) throw std::runtime_error("blaze: no NVFP4 routed experts (compressed-tensors or ModelOpt) in " + model_dir);
  auto name = [&](int l, int e, const char* proj, const char* part) {
    return prefix + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj + "_proj." + part;
  };
  const auto* g0 = st_->find(name(0, 0, "gate", fmt->codes));
  const auto* d0 = st_->find(name(0, 0, "down", fmt->codes));
  if (!g0 || !d0 || g0->shape.size() != 2) throw std::runtime_error("blaze: unexpected expert tensors under " + prefix);
  const int I = int(g0->shape[0]), H = int(g0->shape[1] * 2);
  if (d0->shape.size() != 2 || d0->shape[0] != H || d0->shape[1] * 2 != I)
    throw std::runtime_error("blaze: down projection is not [H, I/2]");
  geo_ = {H, I};
  if (!geo_.valid()) throw std::runtime_error("blaze: expert geometry is not a multiple of 16");
  while (st_->find(name(0, experts_, "gate", fmt->codes))) ++experts_;
  while (st_->find(name(layers_, 0, "gate", fmt->codes))) ++layers_;
  main_layers_ = layers_;
  // the extra directory's MTP layers: ModelOpt or compressed-tensors names under "mtp.layers."
  const Format* xfmt = nullptr;
  int extra_layers = 0;
  auto xname = [&](int l, int e, const char* proj, const char* part) {
    return "mtp.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj + "_proj." + part;
  };
  if (!extra_dir.empty()) {
    extra_ = std::make_unique<SafetensorsDir>(extra_dir);
    for (const Format& f : kFormats)
      if (const auto* t = extra_->find(xname(0, 0, "gate", f.codes)); t && t->dtype == "U8") xfmt = &f;
    if (!xfmt) throw std::runtime_error("blaze: no NVFP4 MTP experts (mtp.layers.0.mlp.experts.*) in " + extra_dir);
    const auto* xg = extra_->find(xname(0, 0, "gate", xfmt->codes));
    if (xg->shape != g0->shape) throw std::runtime_error("blaze: the MTP experts of " + extra_dir + " differ in shape");
    while (extra_->find(xname(extra_layers, 0, "gate", xfmt->codes))) ++extra_layers;
    if (!extra_->find(xname(0, experts_ - 1, "gate", xfmt->codes)) || extra_->find(xname(0, experts_, "gate", xfmt->codes)))
      throw std::runtime_error("blaze: the MTP layer of " + extra_dir + " has another expert count");
    layers_ += extra_layers;
  }
  prefix_ = prefix;
  stride_ = (geo_.record_bytes() + 255) / 256 * 256;
  // Chunks of ~chunk_bytes of checkpoint bytes (codes and scales of 16 rows per group).
  gu_chunk_groups_ = std::max<int>(1, int(chunk_bytes / (16 * size_t(H / 2 + H / 16))));
  down_chunk_groups_ = std::max<int>(1, int(chunk_bytes / (16 * size_t(I / 2 + I / 16))));
  gu_chunks_ = (geo_.gu_groups() + gu_chunk_groups_ - 1) / gu_chunk_groups_;
  down_chunks_ = (geo_.d_groups() + down_chunk_groups_ - 1) / down_chunk_groups_;
  const size_t n = size_t(layers_) * experts_;
  mats_.resize(n * 3);
  mult_.resize(n * 3);
  const char* proj[3] = {"gate", "up", "down"};
  for (int l = 0; l < layers_; ++l)
    for (int e = 0; e < experts_; ++e)
      for (int m = 0; m < 3; ++m) {
        const size_t i = (size_t(l) * experts_ + e) * 3 + m;
        const size_t rows = m == 2 ? H : I, k = m == 2 ? I : H;
        const bool x = l >= main_layers_;
        const SafetensorsDir* dir = x ? extra_.get() : st_.get();
        const Format* f = x ? xfmt : fmt;
        auto nm = [&](const char* part) { return x ? xname(l - main_layers_, e, proj[m], part) : name(l, e, proj[m], part); };
        Mat& mt = mats_[i];
        mt.dir = dir;
        mt.codes = &dir->at(nm(f->codes));
        mt.scales = &dir->at(nm(f->scales));
        if (mt.codes->bytes() != rows * k / 2 || mt.scales->bytes() != rows * k / 16 || mt.scales->dtype != "F8_E4M3")
          throw std::runtime_error("blaze: unexpected NVFP4 layout at " + nm(""));
        const auto& gs = dir->at(nm(f->global));
        float g = 0;
        dir->read(gs, &g, 4);
        if (!(g > 0)) throw std::runtime_error("blaze: bad global scale at " + nm(f->global));
        mult_[i] = f->reciprocal ? 1.0f / g : g;
      }
}

void ExpertStore::read(int layer, int expert, void* dst, bool direct) const {
  auto* out = static_cast<uint8_t*>(dst);
  std::memset(out + geo_.record_bytes(), 0, stride_ - geo_.record_bytes());
  for (int c = 0; c < num_chunks(); ++c) read_chunk(layer, expert, c, out, direct);
}

ExpertStore::Chunk ExpertStore::chunk(int c) const {
  if (c < gu_chunks_) return {false, c * gu_chunk_groups_, std::min((c + 1) * gu_chunk_groups_, geo_.gu_groups())};
  c -= gu_chunks_;
  return {true, c * down_chunk_groups_, std::min((c + 1) * down_chunk_groups_, geo_.d_groups())};
}

void ExpertStore::read_chunk(int layer, int expert, int c, void* dst, bool direct) const {
  const Chunk ch = chunk(c);
  const Mat* m = &mats_[(size_t(layer) * experts_ + expert) * 3];
  const int H = geo_.hidden, I = geo_.inter;
  thread_local SafetensorsDir::Bounce bounce[4];
  expert::PackSource src{};
  auto* out = static_cast<uint8_t*>(dst);
  if (!ch.down) {
    // Record rows [16 g0, 16 g1) interleave gate and up rows [8 g0, 8 g1).
    const int r0 = 8 * ch.g0, rows = 8 * (ch.g1 - ch.g0);
    src.code_stride = size_t(H) / 2, src.scale_stride = size_t(H) / 16;
    for (int k = 0; k < 2; ++k) {
      src.codes[k] = m[k].dir->read_range(*m[k].codes, r0 * src.code_stride, rows * src.code_stride, bounce[2 * k], direct);
      src.scales[k] =
          m[k].dir->read_range(*m[k].scales, r0 * src.scale_stride, rows * src.scale_stride, bounce[2 * k + 1], direct);
    }
    src.row0 = r0, src.interleaved = true;
    expert::pack_groups(src, H, ch.g0, ch.g1, out);
  } else {
    const int r0 = 16 * ch.g0, rows = 16 * (ch.g1 - ch.g0);
    src.code_stride = size_t(I) / 2, src.scale_stride = size_t(I) / 16;
    src.codes[0] = src.codes[1] =
        m[2].dir->read_range(*m[2].codes, r0 * src.code_stride, rows * src.code_stride, bounce[0], direct);
    src.scales[0] = src.scales[1] =
        m[2].dir->read_range(*m[2].scales, r0 * src.scale_stride, rows * src.scale_stride, bounce[1], direct);
    src.row0 = r0, src.interleaved = false;
    expert::pack_groups(src, I, ch.g0, ch.g1, out + geo_.gu_bytes());
  }
}

Placement place_experts(int num_layers, int num_experts, const std::vector<double>& heat, int gpu0_slots,
                        int gpu1_slots, int ram_slots, const std::vector<double>& ram_heat, int layer_begin,
                        int layer_end) {
  Placement p;
  p.num_layers = num_layers, p.num_experts = num_experts;
  const size_t all = size_t(num_layers) * num_experts;
  p.tier.assign(all, Tier::kSsd);
  p.slot.assign(all, -1);
  const int l1 = layer_end < 0 ? num_layers : std::min(layer_end, num_layers), l0 = std::clamp(layer_begin, 0, l1);
  // the held layers' experts only (a contiguous range of the [layer * E + expert] index)
  const size_t n = size_t(l1 - l0) * num_experts;
  std::vector<size_t> order(n);
  std::iota(order.begin(), order.end(), size_t(l0) * num_experts);
  // Experts are ranked by heat (ties in expert-major order, so a uniform profile gives each tier the same expert
  // ids from every layer).
  if (heat.size() == all) {
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return heat[a] != heat[b] ? heat[a] > heat[b] : a % num_experts < b % num_experts;
    });
  } else {
    // Uniform heat: expert-major order, so each tier takes the same expert ids from every layer.
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return a % num_experts < b % num_experts; });
  }
  const int cap[3] = {gpu0_slots, gpu1_slots, ram_slots};
  if (ram_heat.size() == all) {
    const size_t on_gpu = std::min(n, size_t(std::max(0, gpu0_slots)) + size_t(std::max(0, gpu1_slots)));
    std::stable_sort(order.begin() + on_gpu, order.end(), [&](size_t a, size_t b) {
      return ram_heat[a] != ram_heat[b] ? ram_heat[a] > ram_heat[b] : a % num_experts < b % num_experts;
    });
  }
  size_t k = 0;
  for (int t = 0; t < 3; ++t) {
    for (int s = 0; s < cap[t] && k < n; ++s, ++k) {
      p.tier[order[k]] = Tier(t);
      p.slot[order[k]] = s;
      ++p.used[t];
    }
  }
  p.used[3] = int(n - k);
  return p;
}

}  // namespace blaze
