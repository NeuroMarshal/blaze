// ExpertStore on synthetic NVFP4 checkpoints (compressed-tensors and ModelOpt naming, 2 shards, 2 layers x 3 experts,
// small geometry): every record read equals pack_record of the source codes / scales (whole, and chunk by chunk in
// reverse order with O_DIRECT where the file system has it), the multipliers match the exporter's convention, the
// layer prefix is found and an MTP block's experts are ignored.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "blaze/expert.h"
#include "blaze/store.h"
#include "synthetic_checkpoint.h"

namespace fs = std::filesystem;
using namespace blaze;

// One synthetic checkpoint in either exporter's naming; returns the number of wrong experts.
int check(bool modelopt, const std::string& prefix) {
  const expert::Geometry geo{96, 64};
  const int L = 2, E = 3;
  const fs::path dir = fs::temp_directory_path() / "blz_test_store";
  fs::remove_all(dir);
  fs::create_directories(dir / "model");
  std::mt19937 rng(5);
  struct Expert {
    std::vector<uint8_t> c[3], s[3];
    float g[3];
  };
  std::vector<Expert> ex(size_t(L) * E);
  std::vector<Out> shard[2];
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) {
      Expert& x = ex[size_t(l) * E + e];
      const char* proj[3] = {"gate", "up", "down"};
      for (int m = 0; m < 3; ++m) {
        const int rows = m == 2 ? geo.hidden : geo.inter, k = m == 2 ? geo.inter : geo.hidden;
        x.c[m].resize(size_t(rows) * k / 2), x.s[m].resize(size_t(rows) * k / 16);
        for (auto& v : x.c[m]) v = uint8_t(rng());
        for (auto& v : x.s[m]) v = uint8_t(0x20 + rng() % 0x40);
        x.g[m] = 100.0f + float(rng() % 1000);
        const std::string base =
            prefix + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." + proj[m] + "_proj.";
        std::vector<uint8_t> gb(4);
        std::memcpy(gb.data(), &x.g[m], 4);
        auto& sh = shard[(l + e) % 2];  // tensors of one expert spread over both shards
        sh.push_back({base + (modelopt ? "weight" : "weight_packed"), "U8", {rows, k / 2}, x.c[m]});
        sh.push_back({base + "weight_scale", "F8_E4M3", {rows, k / 16}, x.s[m]});
        sh.push_back({base + (modelopt ? "weight_scale_2" : "weight_global_scale"), "F32", {1}, gb});
      }
    }
  // An MTP block's experts must not be taken for the decoder's.
  shard[0].push_back({"mtp.layers.0.mlp.experts.0.gate_proj.weight_packed", "U8", {geo.inter, geo.hidden / 2},
                      std::vector<uint8_t>(size_t(geo.inter) * geo.hidden / 2)});
  write_safetensors(dir / "model" / "model-00001-of-00002.safetensors", shard[0]);
  write_safetensors(dir / "model" / "model-00002-of-00002.safetensors", shard[1]);
  int bad = 0;
  // 1200 bytes per chunk: one gate/up row group (16 * (48 + 6) bytes) or two down row groups (16 * (32 + 4)) each.
  ExpertStore store((dir / "model").string(), "", 1200);
  if (store.num_chunks() != 8 + 3) {
    std::printf("unexpected chunking: %d chunks\n", store.num_chunks());
    ++bad;
  }
  if (store.num_layers() != L || store.num_experts() != E || store.geometry().hidden != geo.hidden ||
      store.geometry().inter != geo.inter || store.prefix() != prefix) {
    std::printf("geometry or prefix mismatch\n");
    ++bad;
  }
  std::vector<uint8_t> want(store.record_stride()), got(store.record_stride());
  for (size_t i = 0; i < ex.size(); ++i) {
    const Expert& x = ex[i];
    const int l = int(i / E), e = int(i % E);
    std::fill(want.begin(), want.end(), 0);
    expert::pack_record(geo, x.c[0].data(), x.s[0].data(), x.c[1].data(), x.s[1].data(), x.c[2].data(), x.s[2].data(),
                        want.data());
    store.read(l, e, got.data());
    bool rec_ok = std::memcmp(want.data(), got.data(), want.size()) == 0;
    std::fill(got.begin(), got.end(), 0);
    for (int c = store.num_chunks() - 1; c >= 0; --c) store.read_chunk(l, e, c, got.data(), true);
    rec_ok &= std::memcmp(want.data(), got.data(), want.size()) == 0;
    const float* m = store.mult(l, e);
    bool mult_ok = true;
    for (int k = 0; k < 3; ++k) mult_ok &= m[k] == (modelopt ? x.g[k] : 1.0f / x.g[k]);
    if (!rec_ok || !mult_ok) {
      std::printf("expert %zu: record %d mult %d\n", i, rec_ok, mult_ok);
      ++bad;
    }
  }
  fs::remove_all(dir);
  std::printf("%s, prefix %s: %d / %zu experts wrong\n", modelopt ? "ModelOpt" : "compressed-tensors", prefix.c_str(),
              bad, ex.size());
  return bad;
}

int main() {
  const int bad = check(false, "model.language_model.layers.") + check(true, "model.layers.");
  std::printf(bad ? "FAILED\n" : "PASSED\n");
  return bad ? 1 : 0;
}
