// Draft-tree kernels (blaze/tree.h, blz_nvfp4_mha_tree, blz_mha_decode_tail): every tree result against the chain
// kernels run along each row's path (the GDN convolution and recurrence bit for bit, the attention within bf16
// rounding), the row moves, the candidate top-k and the draft tree's bookkeeping against CPU references.
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "blaze/mha.h"
#include "blaze/nvfp4.h"
#include "blaze/ops.h"
#include "blaze/tree.h"

#define CHECK_CUDA(x)                                                                              \
  do {                                                                                             \
    cudaError_t err__ = (x);                                                                       \
    if (err__ != cudaSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err__)); \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

namespace {

int g_failures = 0;

void expect(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL %s\n", what);
    ++g_failures;
  }
}

template <typename T>
struct Dev {
  T* ptr = nullptr;
  size_t n;
  explicit Dev(size_t count) : n(count) {
    CHECK_CUDA(cudaMalloc(&ptr, std::max<size_t>(1, n) * sizeof(T)));
    CHECK_CUDA(cudaMemset(ptr, 0, std::max<size_t>(1, n) * sizeof(T)));
  }
  explicit Dev(const std::vector<T>& h) : Dev(h.size()) {
    CHECK_CUDA(cudaMemcpy(ptr, h.data(), n * sizeof(T), cudaMemcpyHostToDevice));
  }
  ~Dev() { cudaFree(ptr); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    CHECK_CUDA(cudaMemcpy(h.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
  }
};

uint16_t bf(float x) { return __bfloat16_as_ushort(__float2bfloat16(x)); }
float fb(uint16_t b) { return __bfloat162float(__ushort_as_bfloat16(b)); }

std::vector<uint16_t> random_bf16(size_t n, std::mt19937& rng, float scale = 1.0f) {
  std::normal_distribution<float> d;
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = bf(d(rng) * scale);
  return v;
}

// A random tree of n rows (row 0 the root; parent of row r < r), as the drafter's verify rows are.
std::vector<int> random_tree(int n, std::mt19937& rng) {
  std::vector<int> parent(n, -1);
  for (int r = 1; r < n; ++r) parent[r] = std::uniform_int_distribution<int>(std::max(0, r - 4), r - 1)(rng);
  return parent;
}

std::vector<int> path_of(const std::vector<int>& parent, int r) {
  std::vector<int> p;
  for (int x = r; x >= 0; x = parent[x]) p.push_back(x);
  std::reverse(p.begin(), p.end());
  return p;
}

// Depth-first program (row, load, save) of a tree: a row continues from the previous one when that is its parent, else
// restores its parent's saved state; rows with several children save, in the level of the branching rows above them.
std::vector<int> dfs_program(const std::vector<int>& parent, int row0, int* levels_used) {
  const int n = int(parent.size());
  std::vector<std::vector<int>> kids(n);
  for (int r = 1; r < n; ++r) kids[parent[r]].push_back(r);
  std::vector<int> level(n, -1), prog;
  int prev = -1, used = 0;
  // stack of (row, branching rows above it)
  std::vector<std::pair<int, int>> st = {{0, 0}};
  while (!st.empty()) {
    auto [r, above] = st.back();
    st.pop_back();
    const int load = (r == 0 || prev == parent[r]) ? -1 : level[parent[r]];
    int save = -1, below = above;
    if (kids[r].size() >= 2) {
      save = level[r] = above;
      below = above + 1;
      used = std::max(used, above + 1);
    }
    prog.insert(prog.end(), {row0 + r, load, save});
    prev = r;
    for (int i = int(kids[r].size()) - 1; i >= 0; --i) st.push_back({kids[r][i], below});
  }
  *levels_used = used;
  return prog;
}

void test_gdn_tree(std::mt19937& rng) {
  const int k_heads = 2, v_heads = 4, dk = 128, dv = 64, kernel = 4;
  const int channels = 2 * k_heads * dk + v_heads * dv;
  const int seqs = 2, rows_per = 13;
  const int T = seqs * rows_per;
  auto qkv = random_bf16(size_t(T) * channels, rng, 0.5f);
  auto weight = random_bf16(size_t(channels) * kernel, rng, 0.5f);
  auto conv_state = random_bf16(size_t(seqs) * channels * (kernel - 1), rng);
  std::vector<float> g(T * v_heads), beta(T * v_heads), rec_state(size_t(seqs) * v_heads * dk * dv);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  for (auto& x : g) x = -u(rng);
  for (auto& x : beta) x = u(rng);
  for (auto& x : rec_state) x = (u(rng) - 0.5f) * 0.2f;
  std::vector<int> parent(T), seq_start = {0}, slots = {1, 0};
  std::vector<std::vector<int>> trees;
  std::vector<int> prog;
  int levels = 1;
  for (int s = 0; s < seqs; ++s) {
    auto t = random_tree(rows_per, rng);
    trees.push_back(t);
    for (int r = 0; r < rows_per; ++r) parent[s * rows_per + r] = t[r] < 0 ? -1 : s * rows_per + t[r];
    int used = 0;
    auto p = dfs_program(t, s * rows_per, &used);
    prog.insert(prog.end(), p.begin(), p.end());
    levels = std::max(levels, used);
    seq_start.push_back((s + 1) * rows_per);
  }
  Dev<uint16_t> dqkv(qkv), dw(weight), dconv(conv_state), dout(size_t(T) * channels);
  Dev<int32_t> dparent(parent), dstart(seq_start), dslots(slots), dprog(prog);
  Dev<uint16_t> dqkv_rec(size_t(T) * channels);
  CHECK_CUDA(blz_gdn_conv_tree(dqkv.ptr, channels, dparent.ptr, dstart.ptr, dslots.ptr, seqs, channels, kernel, dw.ptr,
                               dconv.ptr, dout.ptr, dqkv_rec.ptr, 0));
  expect(dqkv_rec.get() == qkv, "gdn tree conv record == input rows");
  Dev<float> dg(g), dbeta(beta), drec(rec_state);
  Dev<float> scratch(blz_gdn_tree_scratch_floats(seqs, v_heads, dv, levels));
  Dev<uint16_t> dcore(size_t(T) * v_heads * dv);
  CHECK_CUDA(blz_gdn_recurrent_tree(dout.ptr, dg.ptr, dbeta.ptr, dprog.ptr, dstart.ptr, dslots.ptr, seqs, k_heads,
                                    v_heads, dk, dv, drec.ptr, scratch.ptr, levels, dcore.ptr, 0));
  const auto conv = dout.get(), core = dcore.get();
  // reference: the chain kernels along every row's path (its last row)
  int conv_bad = 0, core_bad = 0;
  for (int s = 0; s < seqs; ++s)
    for (int r = 0; r < rows_per; ++r) {
      const auto path = path_of(trees[s], r);
      const int n = int(path.size());
      std::vector<uint16_t> pq(size_t(n) * channels);
      std::vector<float> pg(n * v_heads), pb(n * v_heads);
      for (int i = 0; i < n; ++i) {
        const int t = s * rows_per + path[i];
        std::copy_n(&qkv[size_t(t) * channels], channels, &pq[size_t(i) * channels]);
        std::copy_n(&g[t * v_heads], v_heads, &pg[i * v_heads]);
        std::copy_n(&beta[t * v_heads], v_heads, &pb[i * v_heads]);
      }
      Dev<uint16_t> a(pq), c(size_t(n) * channels), o(size_t(n) * v_heads * dv);
      Dev<int32_t> st(std::vector<int32_t>{0, n}), sl(std::vector<int32_t>{slots[s]});
      CHECK_CUDA(blz_gdn_conv_ex(a.ptr, channels, st.ptr, sl.ptr, 1, channels, kernel, dw.ptr, dconv.ptr, c.ptr, 0, 0));
      Dev<float> gg(pg), bb(pb);
      CHECK_CUDA(blz_gdn_recurrent_ex(c.ptr, gg.ptr, bb.ptr, st.ptr, sl.ptr, 1, n, k_heads, v_heads, dk, dv, drec.ptr,
                                      o.ptr, 0, 0));
      const auto rc = c.get(), ro = o.get();
      const int t = s * rows_per + r;
      conv_bad += std::memcmp(&rc[size_t(n - 1) * channels], &conv[size_t(t) * channels], channels * 2) != 0;
      core_bad += std::memcmp(&ro[size_t(n - 1) * v_heads * dv], &core[size_t(t) * v_heads * dv], v_heads * dv * 2) != 0;
    }
  std::printf("gdn tree: %d rows in %d trees (%d state levels): conv %d / recurrent %d rows differ from the path chains\n",
              T, seqs, levels, conv_bad, core_bad);
  expect(conv_bad == 0 && core_bad == 0, "gdn tree == chains along the paths");
}

void test_move_rows(std::mt19937& rng) {
  const int rows = 20;
  std::vector<int64_t> bytes = {16, 4 * 37, 10240};
  std::vector<std::vector<uint32_t>> host;
  std::vector<Dev<uint32_t>*> dev;
  std::vector<blz_row_buf> bufs;
  for (auto b : bytes) {
    std::vector<uint32_t> h(rows * b / 4);
    for (auto& x : h) x = rng();
    host.push_back(h);
    dev.push_back(new Dev<uint32_t>(h));
    bufs.push_back({dev.back()->ptr, b});
  }
  // a path compaction: rows 3, 7, 8, 12 -> 1, 2, 3, 4 (row 0 stays)
  std::vector<int32_t> pairs = {3, 1, 7, 2, 8, 3, 12, 4};
  Dev<blz_row_buf> dbufs(bufs);
  Dev<int32_t> dpairs(pairs);
  CHECK_CUDA(blz_move_rows(dbufs.ptr, int(bufs.size()), dpairs.ptr, 4, 0));
  bool ok = true;
  for (size_t b = 0; b < bytes.size(); ++b) {
    auto ref = host[b];
    const int64_t w = bytes[b] / 4;
    for (size_t i = 0; i < pairs.size(); i += 2)
      std::copy_n(&ref[pairs[i] * w], w, &ref[pairs[i + 1] * w]);
    ok &= dev[b]->get() == ref;
    delete dev[b];
  }
  std::printf("move rows: %s\n", ok ? "ok" : "MISMATCH");
  expect(ok, "move rows");
}

void test_candidate_topk(std::mt19937& rng) {
  const int rows = 9, c = 128, k = 8;
  std::vector<float> logits(rows * c);
  std::vector<int32_t> ids(rows * c);
  std::normal_distribution<float> d;
  for (auto& x : logits) x = d(rng) * 3.0f;
  logits[5] = logits[7];  // a tie
  for (int i = 0; i < rows * c; ++i) ids[i] = (i * 7919) % 248320;
  Dev<float> dl(logits), dp(rows * k);
  Dev<int32_t> di(ids);
  Dev<int64_t> dt(rows * k);
  CHECK_CUDA(blz_candidate_topk(dl.ptr, di.ptr, rows, c, k, dp.ptr, dt.ptr, 0));
  const auto p = dp.get();
  const auto t = dt.get();
  double err = 0;
  bool ok = true;
  for (int r = 0; r < rows; ++r) {
    std::vector<int> order(c);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return logits[r * c + a] > logits[r * c + b]; });
    double mx = logits[r * c + order[0]], sum = 0;
    for (int i = 0; i < c; ++i) sum += std::exp(logits[r * c + i] - mx);
    for (int j = 0; j < k; ++j) {
      ok &= t[r * k + j] == ids[r * c + order[j]];
      err = std::max(err, std::fabs(p[r * k + j] - std::exp(logits[r * c + order[j]] - mx) / sum));
    }
  }
  std::printf("candidate top-k: ids %s, prob max |err| %.2e\n", ok ? "ok" : "MISMATCH", err);
  expect(ok && err < 1e-5, "candidate top-k");
}

void test_draft_tree(std::mt19937& rng) {
  const int items = 3, K = 4, D = 5, dim = 64, cap = K + (D - 1) * K * K;
  Dev<float> scores(items * cap), front_scores(items * K);
  Dev<int64_t> tokens(items * cap), front_tokens(items * K);
  Dev<int32_t> parents(items * cap), front_node(items * K), anc(items * K * D);
  blz_draft_tree t{scores.ptr, tokens.ptr, parents.ptr, front_scores.ptr, front_tokens.ptr, front_node.ptr, anc.ptr,
                   1000, K, D, cap};
  // CPU mirror
  std::vector<float> s(items * cap), fs(items * K);
  std::vector<int64_t> tk(items * cap), ft(items * K);
  std::vector<int> par(items * cap), fn(items * K), fa(items * K * D), lane_hid(items * K);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<float> rp(items * K);
  std::vector<int64_t> rt(items * K);
  for (int i = 0; i < items; ++i) {
    float left = 1.0f;
    for (int l = 0; l < K; ++l) rp[i * K + l] = left *= 0.5f, rt[i * K + l] = 100 * i + l;
  }
  auto root_hidden = random_bf16(size_t(items) * dim, rng);
  Dev<float> drp(rp);
  Dev<int64_t> drt(rt);
  Dev<uint16_t> droot(root_hidden), front_hidden(size_t(items) * K * dim);
  CHECK_CUDA(blz_tree_begin(t, items, drp.ptr, drt.ptr, droot.ptr, dim, front_hidden.ptr, 0));
  std::vector<uint16_t> fh(size_t(items) * K * dim);
  for (int i = 0; i < items; ++i)
    for (int l = 0; l < K; ++l) {
      s[i * cap + l] = fs[i * K + l] = rp[i * K + l];
      tk[i * cap + l] = ft[i * K + l] = rt[i * K + l];
      par[i * cap + l] = -1, fn[i * K + l] = l;
      fa[(i * K + l) * D] = 1000 + (i * D) * K + l;
      std::copy_n(&root_hidden[size_t(i) * dim], dim, &fh[size_t(i * K + l) * dim]);
    }
  bool ok = front_hidden.get() == fh;
  for (int step = 1; step < D; ++step) {
    std::vector<float> probs(items * K * K);
    std::vector<int64_t> toks(items * K * K);
    for (auto& x : probs) x = u(rng);
    probs[3] = probs[5];  // a tie
    for (size_t j = 0; j < toks.size(); ++j) toks[j] = 7 * step + int64_t(j);
    auto hidden = random_bf16(size_t(items) * K * dim, rng);
    Dev<float> dp(probs);
    Dev<int64_t> dt(toks);
    Dev<uint16_t> dh(hidden);
    CHECK_CUDA(blz_tree_step(t, items, step, dp.ptr, dt.ptr, dh.ptr, dim, front_hidden.ptr, 0));
    for (int i = 0; i < items; ++i) {
      std::vector<float> layer(K * K);
      const int base = K + (step - 1) * K * K;
      for (int e = 0; e < K * K; ++e) {
        const int l = e / K;
        layer[e] = fs[i * K + l] * probs[(i * K + l) * K + e % K];
        s[i * cap + base + e] = layer[e];
        tk[i * cap + base + e] = toks[(i * K + l) * K + e % K];
        par[i * cap + base + e] = fn[i * K + l];
      }
      std::vector<int> order(K * K);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return layer[a] > layer[b]; });
      std::vector<int> old_anc(fa.begin() + i * K * D, fa.begin() + (i + 1) * K * D);
      for (int r = 0; r < K; ++r) {
        const int e = order[r], l = e / K;
        fs[i * K + r] = layer[e], ft[i * K + r] = toks[(i * K + l) * K + e % K], fn[i * K + r] = base + e;
        for (int d = 0; d < D; ++d) fa[(i * K + r) * D + d] = d < step ? old_anc[l * D + d] : d == step ? 1000 + (i * D + step) * K + r : 0;
        std::copy_n(&hidden[size_t(i * K + l) * dim], dim, &fh[size_t(i * K + r) * dim]);
      }
    }
  }
  ok &= scores.get() == s && tokens.get() == tk && front_hidden.get() == fh;
  auto gp = parents.get();
  auto ga = anc.get();
  ok &= std::equal(gp.begin(), gp.end(), par.begin()) && std::equal(ga.begin(), ga.end(), fa.begin());
  ok &= front_tokens.get() == ft;
  std::printf("draft tree: %d items, top-%d, depth %d: %s\n", items, K, D, ok ? "ok" : "MISMATCH");
  expect(ok, "draft tree bookkeeping");
}

// Sharded candidates: pack per rank, sum (the all-reduce), merge to the top c, softmax, top k.
void test_cand_merge(std::mt19937& rng) {
  const int world = 2, rows = 5, c = 128, k = 8, vocab_shard = 1000;
  std::normal_distribution<float> d;
  std::vector<float> sum(size_t(world) * rows * c * 2, 0.0f);
  std::vector<std::vector<float>> logits(world, std::vector<float>(rows * c));
  std::vector<std::vector<int32_t>> ids(world, std::vector<int32_t>(rows * c));
  for (int w = 0; w < world; ++w) {
    for (int i = 0; i < rows * c; ++i) logits[w][i] = d(rng) * 3.0f, ids[w][i] = (i * 37 + w) % vocab_shard;
    ids[w][3] = -1;  // an empty candidate
    if (w == 1) logits[w][9] = logits[0][9];  // a cross-rank tie
    Dev<float> dl(logits[w]), dp(sum.size());
    Dev<int32_t> di(ids[w]);
    CHECK_CUDA(blz_cand_pack(dl.ptr, di.ptr, w * vocab_shard, rows, c, w, world, dp.ptr, 0));
    const auto part = dp.get();
    for (size_t i = 0; i < sum.size(); ++i) sum[i] += part[i];
  }
  Dev<float> dsum(sum), dprobs(rows * k);
  Dev<int64_t> dtok(rows * k), drec(rows);
  CHECK_CUDA(blz_cand_merge_topk(dsum.ptr, world, rows, c, k, dprobs.ptr, dtok.ptr, drec.ptr, 0));
  const auto probs = dprobs.get();
  const auto tok = dtok.get();
  const auto rec = drec.get();
  bool ok = true;
  double err = 0;
  for (int r = 0; r < rows; ++r) {
    std::vector<std::pair<float, int>> all;
    for (int w = 0; w < world; ++w)
      for (int j = 0; j < c; ++j)
        if (ids[w][r * c + j] >= 0) all.push_back({logits[w][r * c + j], ids[w][r * c + j] + w * vocab_shard});
    std::sort(all.begin(), all.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
    double z = 0;
    for (int j = 0; j < c; ++j) z += std::exp(double(all[j].first - all[0].first));
    for (int j = 0; j < k; ++j) {
      ok &= tok[r * k + j] == all[j].second;
      err = std::max(err, std::fabs(probs[r * k + j] - std::exp(double(all[j].first - all[0].first)) / z));
    }
    ok &= rec[r] == all[0].second;
  }
  std::printf("candidate merge over %d ranks: ids %s, prob max |err| %.2e\n", world, ok ? "ok" : "MISMATCH", err);
  expect(ok && err < 1e-5, "candidate merge");
}

// Tree verify attention: row i of the tree against a plain decode of its path; tail rows against an explicit table.
void test_mha_tree(std::mt19937& rng) {
  const int dim = 256, kvh = 2, q_heads = 12, prefix = 300, W = 11, slots = 512;
  std::normal_distribution<float> n01;
  std::vector<uint8_t> kd(size_t(slots) * kvh * dim / 2), vd(kd.size()), ks(size_t(slots) * kvh * dim / 16), vs(ks.size());
  std::vector<float> row(dim);
  for (int sl = 0; sl < slots * kvh; ++sl)
    for (int side = 0; side < 2; ++side) {
      for (auto& x : row) x = n01(rng) * 2.0f;
      blaze::nvfp4_quantize_row_ref(row.data(), dim, 1.0f, true, &(side ? vd : kd)[size_t(sl) * dim / 2],
                                  &(side ? vs : ks)[size_t(sl) * dim / 16]);
    }
  Dev<uint8_t> dkd(kd), dvd(vd), dks(ks), dvs(vs);
  Dev<float> one(std::vector<float>{1.0f});
  blz_nvfp4_pool pool{dkd.ptr, dvd.ptr, dks.ptr, dvs.ptr, one.ptr, one.ptr, kvh, dim, BLZ_KV_NVFP4};
  // the sequence: prefix positions then the W tree rows, slots shuffled
  std::vector<int32_t> perm(slots);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  const int stride = 1024;
  std::vector<int32_t> r2t(stride * (1 + W), 0);
  for (int p = 0; p < prefix + W; ++p) r2t[p] = perm[p];
  auto parent = random_tree(W, rng);
  std::vector<uint64_t> mask(W);
  for (int r = 0; r < W; ++r)
    for (int x = r; x >= 0; x = parent[x]) mask[r] |= 1ull << x;
  // reference requests 1..W: the prefix, then the row's path
  std::vector<int32_t> kv_lens = {prefix + W}, req = {0};
  for (int r = 0; r < W; ++r) {
    const auto path = path_of(parent, r);
    for (int p = 0; p < prefix; ++p) r2t[(1 + r) * stride + p] = perm[p];
    for (size_t j = 0; j < path.size(); ++j) r2t[(1 + r) * stride + prefix + j] = perm[prefix + path[j]];
    kv_lens.push_back(prefix + int(path.size()));
    req.push_back(1 + r);
  }
  auto q = random_bf16(size_t(W) * q_heads * dim, rng);
  Dev<uint16_t> dq(q), out_tree(size_t(W) * q_heads * dim), out_ref(size_t(W) * q_heads * dim);
  Dev<int32_t> dr2t(r2t), dlens(kv_lens), dreq(req);
  Dev<uint64_t> dmask(mask);
  const int splits = 2;
  // workspaces for the widest use: W one-row requests
  const int parts = std::max(blz_nvfp4_mha_parts(q_heads, kvh, W, splits), blz_nvfp4_mha_parts(q_heads, kvh, 1, splits));
  Dev<float> po(size_t(W) * q_heads * parts * dim), pml(size_t(W) * q_heads * parts * 2);
  const float scale = 1.0f / 16.0f;
  CHECK_CUDA(blz_nvfp4_mha_tree(dq.ptr, q_heads * dim, dim, q_heads, pool, dr2t.ptr, stride, {dreq.ptr, BLZ_I32},
                                {dlens.ptr, BLZ_I32}, 1, W, splits, scale, dmask.ptr, po.ptr, pml.ptr, BLZ_BF16,
                                out_tree.ptr, q_heads * dim, dim, 0));
  // one decode row per reference request
  CHECK_CUDA(blz_nvfp4_mha_decode(dq.ptr, q_heads * dim, dim, q_heads, pool, dr2t.ptr, stride, {dreq.ptr + 1, BLZ_I32},
                                  {dlens.ptr + 1, BLZ_I32}, W, 1, splits, scale, po.ptr, pml.ptr, BLZ_BF16, out_ref.ptr,
                                  q_heads * dim, dim, 0));
  const auto a = out_tree.get(), b = out_ref.get();
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, double(std::fabs(fb(a[i]) - fb(b[i]))) / (std::fabs(fb(b[i])) + 0.05));
  std::printf("mha tree: %d rows over a %d-position prefix: worst rel diff vs path decodes %.3g\n", W, prefix, worst);
  expect(worst < 0.02, "mha tree == path decodes");

  // tail: W rows, each its path's slots as the tail (window 0 and a window of 64)
  for (int window : {0, 64}) {
    const int depth = 4;
    std::vector<int32_t> tail(W * 8), lens(W), reqs(W, 0), ref_lens(W), ref_req(W);
    std::vector<int32_t> r2t2(stride * (1 + W), 0);
    for (int p = 0; p < prefix; ++p) r2t2[p] = perm[p];
    for (int r = 0; r < W; ++r) {
      lens[r] = prefix + depth;
      for (int p = 0; p < prefix; ++p) r2t2[(1 + r) * stride + p] = perm[p];
      for (int j = 0; j < depth; ++j) {
        tail[r * 8 + j] = perm[prefix + std::uniform_int_distribution<int>(0, W - 1)(rng)];
        r2t2[(1 + r) * stride + prefix + j] = tail[r * 8 + j];
      }
      ref_lens[r] = prefix + depth, ref_req[r] = 1 + r;
    }
    Dev<int32_t> dt(tail), dl(lens), dq0(reqs), dr2(r2t2), drl(ref_lens), drr(ref_req);
    const int win = window ? window + depth : 0;
    CHECK_CUDA(blz_mha_decode_tail(dq.ptr, q_heads * dim, dim, q_heads, pool, dr2.ptr, stride, {dq0.ptr, BLZ_I32},
                                   {dl.ptr, BLZ_I32}, W, splits, scale, win, dt.ptr, depth, 8, po.ptr, pml.ptr, BLZ_BF16,
                                   out_tree.ptr, q_heads * dim, dim, 0));
    CHECK_CUDA(blz_mha_decode_window(dq.ptr, q_heads * dim, dim, q_heads, pool, dr2.ptr, stride, {drr.ptr, BLZ_I32},
                                     {drl.ptr, BLZ_I32}, W, 1, splits, scale, win, 1, po.ptr, pml.ptr, BLZ_BF16,
                                     out_ref.ptr, q_heads * dim, dim, 0));
    const auto x = out_tree.get(), y = out_ref.get();
    const bool same = x == y;
    std::printf("mha tail (window %d): %s\n", window, same ? "identical to the explicit tables" : "DIFFERS");
    expect(same, "mha tail");
  }
}

}  // namespace

int main() {
  std::mt19937 rng(7);
  test_gdn_tree(rng);
  test_move_rows(rng);
  test_candidate_topk(rng);
  test_cand_merge(rng);
  test_draft_tree(rng);
  test_mha_tree(rng);
  CHECK_CUDA(cudaDeviceSynchronize());
  std::printf(g_failures ? "%d FAILED\n" : "all passed\n", g_failures);
  return g_failures ? 1 : 0;
}
