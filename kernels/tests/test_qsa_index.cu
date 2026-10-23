// QSA indexer kernels (blaze/qsa_index.h) against a CPU reference written from SGLang's QSAIndexer: a 3000-token
// sequence prefilled in two steps (the second starts mid-block, so its first block mixes ring and step keys) plus a
// decode row of another sequence; compressed keys, index queries, block scores and the selected KV slots.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "blaze/qsa_index.h"

namespace {

void check(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
    std::exit(1);
  }
}

uint16_t bf(float v) { return __bfloat16_as_ushort(__float2bfloat16_rn(v)); }
float fb(uint16_t v) { return __bfloat162float(__ushort_as_bfloat16(v)); }
float rb(float v) { return fb(bf(v)); }

template <typename T>
T* dev(const std::vector<T>& h) {
  T* d;
  check(cudaMalloc(&d, h.size() * sizeof(T) + 16), "malloc");
  check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "copy");
  return d;
}

template <typename T>
std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "copy back");
  return h;
}

const int D = 128, HEADS = 4, RATIO = 4, ROT = 64, TOPK_BLOCKS = 512;
const float EPS = 1e-6f;

std::vector<float> inv_freq() {
  std::vector<float> f(ROT / 2);
  for (int i = 0; i < ROT / 2; ++i) f[i] = 1.0f / std::pow(1e7f, 2.0f * i / ROT);
  return f;
}

// Gemma norm + neox rope, rounded as the kernel does
void norm_rope(float* v, const std::vector<uint16_t>& w, int64_t pos, const std::vector<float>& f) {
  double ss = 0;
  for (int i = 0; i < D; ++i) ss += double(v[i]) * v[i];
  const float inv = 1.0f / std::sqrt(float(ss / D) + EPS);
  for (int i = 0; i < D; ++i) v[i] = rb(v[i] * inv * (1.0f + fb(w[i])));
  float o[D];
  std::copy(v, v + D, o);
  for (int i = 0; i < ROT / 2; ++i) {
    const float c = std::cos(float(pos) * f[i]), s = std::sin(float(pos) * f[i]);
    o[i] = v[i] * c - v[i + ROT / 2] * s;
    o[i + ROT / 2] = v[i + ROT / 2] * c + v[i] * s;
  }
  for (int i = 0; i < D; ++i) v[i] = rb(o[i]);
}


// Reference order of one row's table: blocks above the k-th best score in block order, then the lowest ties of the k-th
// score in block order, each expanded to its slots, then the slots of the incomplete block.
std::vector<int32_t> ref_table(const std::vector<float>& sc, int nvis, int64_t pos, const std::vector<int32_t>& rt, int topk) {
  std::vector<float> sorted(sc.begin(), sc.begin() + nvis);
  std::nth_element(sorted.begin(), sorted.begin() + (topk - 1), sorted.end(), std::greater<float>());
  const float thr = sorted[topk - 1];
  int n_gt = 0;
  for (int i = 0; i < nvis; ++i) n_gt += sc[i] > thr;
  std::vector<int> sel;
  for (int i = 0; i < nvis; ++i)
    if (sc[i] > thr) sel.push_back(i);
  int ties = topk - n_gt;
  for (int i = 0; i < nvis && ties > 0; ++i)
    if (sc[i] == thr) sel.push_back(i), --ties;
  std::vector<int32_t> out;
  for (int b : sel)
    for (int m = 0; m < RATIO; ++m) out.push_back(rt[b * RATIO + m]);
  for (int64_t t = int64_t(nvis) * RATIO; t <= pos; ++t) out.push_back(rt[t]);
  return out;
}

// One decode row against `nvis` random compressed keys (the scores buffer is laid out for max_blocks, as in the model).
bool decode_case(int nvis, int max_blocks) {
  std::mt19937 rng(nvis + 5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int64_t pos = int64_t(nvis) * RATIO + 2;
  const int64_t stride = int64_t(max_blocks) * RATIO + 16;
  std::vector<uint16_t> comp(size_t(nvis + 4) * D), q(HEADS * D);
  for (auto& x : comp) x = bf(nd(rng));
  for (auto& x : q) x = bf(nd(rng));
  std::vector<int32_t> rt(stride, 0);
  for (int64_t i = 0; i <= pos; ++i) rt[i] = int32_t(i);  // page = block
  uint16_t* dcomp = dev(comp);
  uint16_t* dq = dev(q);
  int32_t* drt = dev(rt);
  std::vector<int64_t> hp = {pos};
  std::vector<int32_t> trow = {0};
  int64_t* dp = dev(hp);
  int32_t* dtr = dev(trow);
  float* scores;
  check(cudaMalloc(&scores, size_t(max_blocks) * 4), "malloc");
  int32_t *table, *counts;
  check(cudaMalloc(&table, 2056 * 4), "malloc");
  check(cudaMalloc(&counts, 4), "malloc");
  cudaStream_t st;
  cudaStreamCreate(&st);
  auto run_scores = [&] {
    check(blz_qsa_scores(dq, HEADS, D, dp, dtr, 0, 1, 0, RATIO, drt, stride, dcomp, max_blocks, scores, max_blocks, st), "scores");
  };
  auto run_select = [&] {
    check(blz_qsa_select(scores, max_blocks, dp, dtr, 0, 1, RATIO, TOPK_BLOCKS, drt, stride, table, 2056, counts, st), "select");
  };
  run_scores();
  run_select();
  check(cudaStreamSynchronize(st), "run");
  auto hs = host(scores, nvis);
  auto ht = host(table, 2056);
  auto hc = host(counts, 1);
  const auto want = ref_table(hs, nvis, pos, rt, TOPK_BLOCKS);
  const bool same = hc[0] == int(want.size()) && std::equal(want.begin(), want.end(), ht.begin());
  // time: 40 calls captured in a graph, so launch overhead does not count
  auto time_us = [&](auto&& body) {
    cudaGraph_t g;
    cudaGraphExec_t ge;
    check(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal), "capture");
    for (int i = 0; i < 40; ++i) body();
    check(cudaStreamEndCapture(st, &g), "capture end");
    check(cudaGraphInstantiate(&ge, g, 0), "instantiate");
    cudaGraphLaunch(ge, st);
    cudaStreamSynchronize(st);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0), cudaEventCreate(&e1);
    cudaEventRecord(e0, st);
    for (int rep = 0; rep < 3; ++rep) cudaGraphLaunch(ge, st);
    cudaEventRecord(e1, st);
    check(cudaEventSynchronize(e1), "time");
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaGraphExecDestroy(ge), cudaGraphDestroy(g);
    return ms * 1000.0f / 120;
  };
  const float t_sc = time_us(run_scores), t_sel = time_us(run_select), t_both = time_us([&] { run_scores(); run_select(); });
  std::printf("decode row, %d blocks (scores laid out for %d): %s; scores %.1f us, select %.1f us, both %.1f us\n", nvis,
              max_blocks, same ? "exact in order" : "MISMATCH", t_sc, t_sel, t_both);
  cudaFree(dcomp), cudaFree(dq), cudaFree(drt), cudaFree(dp), cudaFree(dtr), cudaFree(scores), cudaFree(table), cudaFree(counts);
  cudaStreamDestroy(st);
  return same;
}

}  // namespace

int main() {
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int L = 3000, rows_rt = 2, stride = 4096;
  const int ld = HEADS * D + D;  // index q then k
  std::vector<uint16_t> qw(D), kw(D);
  for (auto& x : qw) x = bf(0.1f * nd(rng));
  for (auto& x : kw) x = bf(0.1f * nd(rng));
  const auto f = inv_freq();
  // sequence 0: positions 0..L-1 (row 0); its slots: pages in shuffled order
  const int pages0 = (L + RATIO - 1) / RATIO;
  std::vector<int> page_of(pages0);
  std::iota(page_of.begin(), page_of.end(), 10);
  std::shuffle(page_of.begin(), page_of.end(), rng);
  std::vector<int32_t> rt(size_t(rows_rt) * stride, 0);
  for (int p = 0; p < L; ++p) rt[p] = page_of[p / RATIO] * RATIO + p % RATIO;
  // sequence 1 (row 1): 37 positions in pages 2000+
  const int L1 = 37;
  for (int p = 0; p < L1; ++p) rt[stride + p] = (2000 + p / RATIO) * RATIO + p % RATIO;
  // raw projections of every position of both sequences
  std::vector<std::vector<uint16_t>> proj0(L, std::vector<uint16_t>(ld)), proj1(L1, std::vector<uint16_t>(ld));
  for (auto& r : proj0)
    for (auto& x : r) x = bf(nd(rng));
  for (auto& r : proj1)
    for (auto& x : r) x = bf(nd(rng));

  // reference: compressed keys of all complete blocks, queries of the step's rows
  auto comp_ref = [&](const std::vector<std::vector<uint16_t>>& P, int b, std::vector<float>& out) {
    out.assign(D, 0.0f);
    for (int m = 0; m < RATIO; ++m)
      for (int i = 0; i < D; ++i) out[i] += fb(P[b * RATIO + m][HEADS * D + i]);
    for (int i = 0; i < D; ++i) out[i] = rb(out[i] / RATIO);
    norm_rope(out.data(), kw, int64_t(b) * RATIO, f);
  };
  const int pages_total = 2100;
  uint16_t *comp, *ring;
  check(cudaMalloc(&comp, size_t(pages_total) * D * 2), "malloc");
  check(cudaMalloc(&ring, size_t(rows_rt) * RATIO * D * 2), "malloc");
  check(cudaMemset(comp, 0, size_t(pages_total) * D * 2), "memset");
  int32_t* drt = dev(rt);
  float* dfreq = dev(f);
  uint16_t *dqw = dev(qw), *dkw = dev(kw);

  // step 1: sequence 0 positions [0, 1501) (ends mid-block)
  // step 2: sequence 1 positions [0, 37) decode-like rows? no: sequence 0 positions [1501, 3000) + one row of seq 1
  auto run_step = [&](std::vector<int> seqs, std::vector<int> starts, std::vector<int> lens, bool select) {
    std::vector<uint16_t> proj;
    std::vector<int64_t> pos;
    std::vector<int32_t> trow, tkv;
    for (size_t s = 0; s < seqs.size(); ++s) {
      for (int p = starts[s]; p < starts[s] + lens[s]; ++p) {
        const auto& src = seqs[s] == 0 ? proj0[p] : proj1[p];
        proj.insert(proj.end(), src.begin(), src.end());
        pos.push_back(p);
        trow.push_back(seqs[s]);
        tkv.push_back(starts[s] + lens[s]);
      }
    }
    const int T = int(pos.size());
    uint16_t* dproj = dev(proj);
    int64_t* dpos = dev(pos);
    int32_t *dtrow = dev(trow), *dtkv = dev(tkv);
    check(blz_qsa_compress(dproj, ld, HEADS * D, dpos, dtrow, dtkv, T, D, RATIO, dkw, EPS, ROT, dfreq, 1.0f, drt, stride,
                           ring, comp, 0),
          "compress");
    uint16_t* dq;
    check(cudaMalloc(&dq, size_t(T) * HEADS * D * 2), "malloc");
    check(blz_qsa_index_q(dproj, ld, 0, dpos, dqw, EPS, T, HEADS, D, ROT, dfreq, 1.0f, dq, 0), "index q");
    bool ok = true;
    if (select) {
      // the last row of each sequence of this step: score + select
      const int max_blocks = 1024;
      float* scores;
      check(cudaMalloc(&scores, size_t(T) * max_blocks * 4), "malloc");
      int32_t *table, *counts;
      check(cudaMalloc(&table, size_t(T) * 2056 * 4), "malloc");
      check(cudaMalloc(&counts, size_t(T) * 4), "malloc");
      int row = 0;
      for (size_t s = 0; s < seqs.size(); ++s) {
        // seq 0 rows share keys (same_seq), seq 1 one row
        check(blz_qsa_scores(dq, HEADS, D, dpos, dtrow, row, lens[s], seqs[s] == 0, RATIO, drt, stride, comp, max_blocks,
                             scores + size_t(row) * max_blocks, max_blocks, 0),
              "scores");
        check(blz_qsa_select(scores + size_t(row) * max_blocks, max_blocks, dpos, dtrow, row, lens[s], RATIO, TOPK_BLOCKS,
                             drt, stride, table, 2056, counts, 0),
              "select");
        row += lens[s];
      }
      auto hq = host(dq, size_t(T) * HEADS * D);
      auto hs = host(scores, size_t(T) * max_blocks);
      auto ht = host(table, size_t(T) * 2056);
      auto hc = host(counts, T);
      // reference for a few rows: the last row of each sequence and one in the middle
      std::vector<int> probe;
      int r0 = 0;
      for (size_t s = 0; s < seqs.size(); ++s) {
        probe.push_back(r0 + lens[s] - 1);
        if (lens[s] > 10) probe.push_back(r0 + lens[s] / 2);
        r0 += lens[s];
      }
      double max_q = 0, max_s = 0;
      int bad_sel = 0;
      for (int r : probe) {
        const int s = trow[r];
        const auto& P = s == 0 ? proj0 : proj1;
        const int64_t p = pos[r];
        float qv[HEADS][D];
        for (int h = 0; h < HEADS; ++h) {
          for (int i = 0; i < D; ++i) qv[h][i] = fb(P[p][h * D + i]);
          norm_rope(qv[h], qw, p, f);
          for (int i = 0; i < D; ++i) max_q = std::fmax(max_q, std::fabs(qv[h][i] - fb(hq[(size_t(r) * HEADS + h) * D + i])));
        }
        const int nvis = int((p + 1) / RATIO);
        std::vector<float> sref(nvis);
        std::vector<float> kb;
        for (int b = 0; b < nvis; ++b) {
          comp_ref(P, b, kb);
          float sum = 0;
          for (int h = 0; h < HEADS; ++h) {
            float d = 0;
            for (int i = 0; i < D; ++i) d += fb(hq[(size_t(r) * HEADS + h) * D + i]) * kb[i];
            sum += std::fmax(d, 0.0f);
          }
          sref[b] = sum / std::sqrt(float(D));
          max_s = std::fmax(max_s, std::fabs(sref[b] - hs[size_t(r) * max_blocks + b]));
        }
        // selected set: top blocks by the kernel's own scores (ties make float sets ambiguous), expanded + tail
        std::vector<int> idx(nvis);
        std::iota(idx.begin(), idx.end(), 0);
        const int k = std::min(nvis, TOPK_BLOCKS);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
          const float va = hs[size_t(r) * max_blocks + a], vb = hs[size_t(r) * max_blocks + b];
          return va > vb || (va == vb && a < b);
        });
        std::multiset<int> want;
        for (int j = 0; j < k; ++j)
          for (int m = 0; m < RATIO; ++m) want.insert(rt[size_t(s) * stride + idx[j] * RATIO + m]);
        for (int64_t t = int64_t(nvis) * RATIO; t <= p; ++t) want.insert(rt[size_t(s) * stride + t]);
        std::multiset<int> got(ht.begin() + size_t(r) * 2056, ht.begin() + size_t(r) * 2056 + hc[r]);
        if (got != want) {
          ++bad_sel;
          std::printf("  row %d (pos %lld): %d selected, want %zu\n", r, (long long)p, hc[r], want.size());
        }
      }
      std::printf("  index q max abs err %.2e, scores max abs err %.2e, selection mismatches %d of %zu rows\n", max_q,
                  max_s, bad_sel, probe.size());
      ok = max_q < 2e-2 && max_s < 5e-2 && bad_sel == 0;
    }
    return ok;
  };
  bool ok = true;
  std::printf("step 1: sequence 0 [0, 1501)\n");
  ok &= run_step({0}, {0}, {1501}, true);
  std::printf("step 2: sequence 0 [1501, 3000), sequence 1 [0, 37)\n");
  ok &= run_step({0, 1}, {1501, 0}, {1499, 37}, true);
  // compressed keys of every complete block of sequence 0 against the reference
  auto hcomp = host(comp, size_t(pages_total) * D);
  double max_c = 0;
  std::vector<float> kb;
  for (int b = 0; b < L / RATIO; ++b) {
    comp_ref(proj0, b, kb);
    for (int i = 0; i < D; ++i) max_c = std::fmax(max_c, std::fabs(kb[i] - fb(hcomp[size_t(page_of[b]) * D + i])));
  }
  std::printf("compressed keys of sequence 0 (%d blocks): max abs err %.2e\n", L / RATIO, max_c);
  ok &= max_c < 2e-2;
  // block selection alone on long rows: exact top-512 (ties: lower block) against the CPU, and its time
  {
    std::mt19937 rng(9);
    std::normal_distribution<float> n01;
    for (const int nvis : {63000, 127000})
      for (int kind = 0; kind < 3; ++kind) {
        std::vector<float> sc(nvis);
        for (int i = 0; i < nvis; ++i) {
          const float x = n01(rng);
          sc[i] = kind == 0 ? std::max(0.0f, x)                                 // relu sums: many zeros
                : kind == 1 ? float(int(std::max(0.0f, x) * 8.0f)) / 8.0f       // coarse: ties at the threshold
                            : (i % 211 == 0 ? std::fabs(x) + 1.0f : 0.0f);      // few positive: threshold at zero
        }
        const int64_t pos = int64_t(nvis) * RATIO + 2;
        std::vector<int64_t> hp = {pos};
        std::vector<int32_t> trow = {0};
        const int64_t stride = pos + 8;
        std::vector<int32_t> rt(stride);
        for (int64_t i = 0; i < stride; ++i) rt[i] = int32_t(i);
        float* dsc = dev(sc);
        int64_t* dp = dev(hp);
        int32_t* dtr = dev(trow);
        int32_t* drt = dev(rt);
        int32_t *table, *counts;
        check(cudaMalloc(&table, 2056 * 4), "malloc");
        check(cudaMalloc(&counts, 4), "malloc");
        check(blz_qsa_select(dsc, nvis, dp, dtr, 0, 1, RATIO, TOPK_BLOCKS, drt, stride, table, 2056, counts, 0), "select");
        auto ht = host(table, 2056);
        auto hc = host(counts, 1);
        const auto want = ref_table(sc, nvis, pos, rt, TOPK_BLOCKS);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0), cudaEventCreate(&e1);
        cudaEventRecord(e0);
        for (int it = 0; it < 50; ++it)
          blz_qsa_select(dsc, nvis, dp, dtr, 0, 1, RATIO, TOPK_BLOCKS, drt, stride, table, 2056, counts, 0);
        cudaEventRecord(e1);
        check(cudaEventSynchronize(e1), "time");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        const bool same = hc[0] == int(want.size()) && std::equal(want.begin(), want.end(), ht.begin());
        std::printf("select %d blocks, %s: %s, %.1f us per row\n", nvis,
                    kind == 0 ? "relu scores" : kind == 1 ? "coarse ties" : "threshold at zero", same ? "exact" : "MISMATCH",
                    ms * 1000.0f / 50);
        ok &= same;
        cudaFree(dsc), cudaFree(dp), cudaFree(dtr), cudaFree(drt), cudaFree(table), cudaFree(counts);
      }
  }
  // decode-shaped call as the model makes it: one row, scores over the compressed keys then select, exact in order and timed
  ok &= decode_case(63000, 135001);
  ok &= decode_case(127000, 135001);
  ok &= decode_case(63000, 63001);
  ok &= decode_case(127000, 127001);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
