// CPU engine with an SSD tier, driven through the mailbox from the host: experts outside RAM are read from a synthetic
// checkpoint in chunks by the reader threads (O_DIRECT where the file system has it) while the workers compute, and
// every output equals forward_ref bit for bit: RAM and SSD items mixed, immediate and deferred items, compute chunks
// that straddle read chunks, over many requests.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

#include "blaze/cpu_engine.h"
#include "blaze/cpu_expert.h"
#include "blaze/expert.h"
#include "blaze/mailbox.h"
#include "blaze/store.h"
#include "synthetic_checkpoint.h"

namespace fs = std::filesystem;
using namespace blaze;

static std::vector<uint16_t> random_column(int n, std::mt19937& rng) {
  std::vector<uint16_t> v(n);
  std::uniform_int_distribution<int> e(110, 130);
  for (auto& b : v) b = (rng() % 17 == 0) ? 0 : uint16_t(((rng() & 1) << 15) | (e(rng) << 7) | (rng() & 0x7F));
  return v;
}

int main() {
  const expert::Geometry geo{512, 128};
  const int L = 3, E = 12, H = geo.hidden;
  // On the build directory's file system, not a tmpfs, so the O_DIRECT path is the one exercised.
  const fs::path dir = fs::current_path() / "blz_test_cpu_ssd";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::mt19937 rng(9);
  std::vector<std::vector<uint8_t>> recs(size_t(L) * E);
  std::vector<Out> tensors;
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) {
      std::vector<uint8_t> c[3], s[3];
      const char* proj[3] = {"gate", "up", "down"};
      for (int m = 0; m < 3; ++m) {
        const int rows = m == 2 ? H : geo.inter, k = m == 2 ? geo.inter : H;
        c[m].resize(size_t(rows) * k / 2), s[m].resize(size_t(rows) * k / 16);
        for (auto& v : c[m]) v = uint8_t(rng());
        for (auto& v : s[m]) v = uint8_t(0x28 + rng() % 0x30);
        const std::string base = "model.layers." + std::to_string(l) + ".mlp.experts." + std::to_string(e) + "." +
                                 proj[m] + "_proj.";
        const float g = 200.0f + float(rng() % 300);
        std::vector<uint8_t> gb(4);
        std::memcpy(gb.data(), &g, 4);
        tensors.push_back({base + "weight_packed", "U8", {rows, k / 2}, c[m]});
        tensors.push_back({base + "weight_scale", "F8_E4M3", {rows, k / 16}, s[m]});
        tensors.push_back({base + "weight_global_scale", "F32", {1}, gb});
      }
      auto& rec = recs[size_t(l) * E + e];
      rec.assign(geo.record_bytes(), 0);
      expert::pack_record(geo, c[0].data(), s[0].data(), c[1].data(), s[1].data(), c[2].data(), s[2].data(), rec.data());
    }
  write_safetensors(dir / "model.safetensors", tensors);
  // Read chunks: 2 gate/up row groups (16 * (256 + 32) bytes each) or 8 down row groups (16 * (64 + 8)).
  ExpertStore store(dir.string(), "", 9216);
  // Even experts in RAM, odd ones on the SSD tier.
  std::vector<const uint8_t*> records(size_t(L) * E, nullptr);
  for (size_t i = 0; i < records.size(); ++i)
    if (i % 2 == 0) records[i] = recs[i].data();
  std::vector<float> mult;
  for (int l = 0; l < L; ++l)
    for (int e = 0; e < E; ++e) mult.insert(mult.end(), store.mult(l, e), store.mult(l, e) + 3);

  const blz_mail_layout lay = blz_mail_layout_for(H);
  auto* mail = static_cast<uint8_t*>(std::aligned_alloc(256, (lay.total + 255) / 256 * 256));
  std::memset(mail, 0, lay.total);
  auto* hdr = reinterpret_cast<blz_mail_header*>(mail);
  cpu::EngineConfig ec;
  ec.geo = geo, ec.num_layers = L, ec.num_experts = E, ec.threads = 4;
  ec.gu_chunk_groups = 1, ec.down_chunk_groups = 3;  // down tasks straddle the 8-group read chunks
  ec.store = &store, ec.ssd_io_threads = 3, ec.ssd_direct = true;
  int bad = 0, outputs = 0, ssd_items = 0;
  {
    cpu::Engine engine(ec, mail, records.data(), mult.data());
    for (uint32_t r = 1; r <= 200; ++r) {
      const int T = 1 + int(rng() % 4), layer = int(rng() % L), K = 3;
      std::vector<std::vector<uint16_t>> x(T);
      std::vector<blz_moe_item> items;
      for (int t = 0; t < T; ++t) {
        x[t] = random_column(H, rng);
        uint8_t* p = mail + lay.planes + size_t(t) * 3 * H;
        cpu::ColumnBuffer b;
        b.lo = p, b.mid = p + H, b.hi = reinterpret_cast<int8_t*>(p + 2 * H);
        b.hi_sum = reinterpret_cast<int32_t*>(mail + lay.hi_sum) + size_t(t) * (H / 16);
        cpu::encode_column(x[t].data(), H, b);
        hdr->x_emax[t] = b.emax;
        std::vector<int> picked;
        while (int(picked.size()) < K) {
          const int e = int(rng() % E);
          if (std::find(picked.begin(), picked.end(), e) == picked.end()) picked.push_back(e);
        }
        for (int e : picked) {
          auto it = std::find_if(items.begin(), items.end(), [&](const blz_moe_item& m) { return m.expert == e; });
          if (it == items.end()) items.push_back({e, -1, 0, {}}), it = items.end() - 1;
          it->cols[it->ncols++] = t;
        }
      }
      const int n = int(items.size());
      hdr->tokens = T, hdr->layer = layer, hdr->n_cpu = n, hdr->n_gpu1 = 0;
      hdr->n_imm = int(rng() % (n + 1));
      std::copy(items.begin(), items.end(), hdr->items);
      __atomic_store_n(&hdr->req, (uint64_t(r) << 32) | uint64_t(n), __ATOMIC_RELEASE);
      while (__atomic_load_n(&hdr->done, __ATOMIC_ACQUIRE) != r) std::this_thread::yield();
      std::vector<uint16_t> ref(H);
      for (int i = 0; i < n; ++i) {
        const size_t id = size_t(layer) * E + items[i].expert;
        ssd_items += records[id] == nullptr;
        const expert::Multipliers m{mult[id * 3], mult[id * 3 + 1], mult[id * 3 + 2]};
        for (int c = 0; c < items[i].ncols; ++c) {
          expert::forward_ref(geo, recs[id].data(), m, x[items[i].cols[c]].data(), ref.data());
          const auto* y = reinterpret_cast<const uint16_t*>(mail + lay.y) + (size_t(i) * BLZ_MOE_MAX_TOKENS + c) * H;
          for (int k = 0; k < H; ++k) bad += y[k] != ref[k];
          outputs += H;
        }
      }
    }
  }
  std::free(mail);
  fs::remove_all(dir);
  std::printf("cpu engine with SSD tier: %d of %d outputs differ from forward_ref (%d SSD-tier items)\n", bad, outputs,
              ssd_items);
  std::printf(bad || ssd_items == 0 ? "FAILED\n" : "PASSED\n");
  return bad || ssd_items == 0 ? 1 : 0;
}
