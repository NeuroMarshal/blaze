// GPU narrow route of the hybrid MoE (see blaze/moe.h): planning, exact W4A16 gate/up and down, fixed-order combine.
//
// rg16 unit (16 rows x 16 k, 128 code bytes + 16 scales): the 32-bit code word at byte 4L holds k = 16b + 4q .. +3
// (q = L / 8) of row i = L % 8 (low nibbles) and of row i + 8 (high nibbles). The rows kernel computes each unit's
// exact dot products on int8 tensor cores and accumulates P * S^ in int64 (exact, so partial sums of K ranges can be
// added in any order).

#include <cuda_runtime.h>

#include "blaze/canon.h"
#include "blaze/expert.h"
#include "blaze/mailbox.h"
#include "blaze/moe.h"

namespace blaze::moe {
namespace {

constexpr int kMaxItems = BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK;
constexpr int kMaxCols = BLZ_MOE_MAX_TOKENS;
constexpr int kWarpsPerBlock = 4;

struct Sizes {
  int hidden, inter;
  size_t plan, x_planes, x_emax, gu_sums, h, h_emax, y, h_planes, defer, defer_acc;
  __host__ __device__ Sizes(int H, int I) : hidden(H), inter(I) {
    plan = sizeof(blz_moe_plan);
    x_planes = size_t(kMaxCols) * 3 * H;
    x_emax = kMaxCols * sizeof(int32_t);
    gu_sums = size_t(kMaxItems) * kMaxCols * 2 * I * sizeof(int64_t);
    h = size_t(kMaxItems) * kMaxCols * I * sizeof(uint16_t);
    h_emax = size_t(kMaxItems) * kMaxCols * sizeof(int32_t);
    y = size_t(kMaxItems) * kMaxCols * H * sizeof(uint16_t);
    h_planes = size_t(kMaxItems) * kMaxCols * 3 * I;
    defer = sizeof(blz_moe_defer);
    defer_acc = size_t(kMaxCols) * H * sizeof(float);
  }
};

__host__ __device__ inline size_t align256(size_t n) { return (n + 255) & ~size_t(255); }
__device__ __forceinline__ int dp4a_su(int a, unsigned b, int c) {
  int d;
  asm("dp4a.s32.u32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

__device__ __forceinline__ int dp4a_ss(int a, int b, int c) { return __dp4a(a, b, c); }

// Four 4-bit codes (one per byte, 0..15) -> four signed bytes c2(code).
__device__ __forceinline__ uint32_t c2x4(uint32_t codes) {
  const uint32_t idx = codes & 0x07070707u;
  const uint32_t t = idx | (idx >> 4);
  const uint32_t sel = (t & 0xFFu) | ((t >> 8) & 0xFF00u);
  const uint32_t mag = __byte_perm(0x03020100u, 0x0C080604u, sel);  // {0,1,2,3,4,6,8,12}
  const uint32_t neg = ((codes >> 3) & 0x01010101u) * 0xFFu;
  return (mag & ~neg) | (__vneg4(mag) & neg);
}

// P = sum of 4 products c2 * X with X given as byte limbs (lo, mid unsigned; hi signed): exact in wrapping int32.
__device__ __forceinline__ int32_t limb_dot(uint32_t c, uint32_t lo, uint32_t mid, uint32_t hi) {
  const int pl = dp4a_su(int(c), lo, 0);
  const int pm = dp4a_su(int(c), mid, 0);
  const int ph = dp4a_ss(int(c), int(hi), 0);
  return int32_t(uint32_t(pl) + (uint32_t(pm) << 8) + (uint32_t(ph) << 16));
}

__device__ __forceinline__ int64_t shfl_xor_i64(int64_t v, int m) {
  const int lo = __shfl_xor_sync(0xffffffffu, int(v & 0xffffffff), m);
  const int hi = __shfl_xor_sync(0xffffffffu, int(v >> 32), m);
  return (int64_t(hi) << 32) | uint32_t(lo);
}

__device__ __forceinline__ uint32_t ld_nc_u32(const uint8_t* p) {
  return __ldg(reinterpret_cast<const unsigned int*>(p));
}

// plan

// A16 limb planes of token column t: [3][H] bytes (lo, mid, hi) and its exponent (-1: non-finite column).
// A thread handles chunks of 8 values (16-byte load, 8-byte limb stores); with a mailbox it also writes the planes,
// the per-16 hi sums (chunk pairs, one shuffle) and the exponent there for the CPU engine.
__device__ void encode_token(const blz_moe_layer& layer, const uint16_t* __restrict__ x, int t,
                             const blz_moe_workspace& ws, int* s_emax) {
  const int H = layer.hidden, tid = threadIdx.x, chunks = H / 8;
  const uint4* xv = reinterpret_cast<const uint4*>(x + size_t(t) * H);
  int e = 1;
  bool nonfinite = false;
  for (int c = tid; c < chunks; c += blockDim.x) {
    const uint4 v = xv[c];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t b = uint16_t(w[i / 2] >> (16 * (i % 2)));
      nonfinite |= canon::bf16_nonfinite(b);
      e = max(e, canon::a16_exp(b));
    }
  }
  e = __reduce_max_sync(0xffffffffu, e);
  nonfinite = __any_sync(0xffffffffu, nonfinite);
  if (tid == 0) s_emax[0] = 0;
  __syncthreads();
  if ((tid & 31) == 0) atomicMax(&s_emax[0], nonfinite ? 0x7fffffff : e);
  __syncthreads();
  const int emax = s_emax[0];
  uint8_t* planes = ws.x_planes + size_t(t) * 3 * H;
  blz_mail_layout ml{};
  if (ws.mail != nullptr) ml = blz_mail_layout_for(H);
  uint8_t* mp = ws.mail != nullptr ? ws.mail + ml.planes + size_t(t) * 3 * H : nullptr;
  int32_t* hsum_out = ws.mail != nullptr ? reinterpret_cast<int32_t*>(ws.mail + ml.hi_sum) + size_t(t) * (H / 16) : nullptr;
  for (int c = tid; c < chunks; c += blockDim.x) {
    const uint4 v = xv[c];
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    uint32_t lo[2] = {0, 0}, mid[2] = {0, 0}, hi[2] = {0, 0};
    int32_t hsum = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint16_t b = uint16_t(w[i / 2] >> (16 * (i % 2)));
      const int32_t q = emax == 0x7fffffff ? 0 : canon::a16_encode(b, emax);
      const int sh = 8 * (i % 4);
      lo[i / 4] |= uint32_t(q & 0xFF) << sh;
      mid[i / 4] |= uint32_t((q >> 8) & 0xFF) << sh;
      hi[i / 4] |= uint32_t((q >> 16) & 0xFF) << sh;
      hsum += int8_t((q >> 16) & 0xFF);
    }
    const size_t o = size_t(c) * 8;
    *reinterpret_cast<uint2*>(planes + o) = make_uint2(lo[0], lo[1]);
    *reinterpret_cast<uint2*>(planes + H + o) = make_uint2(mid[0], mid[1]);
    *reinterpret_cast<uint2*>(planes + 2 * H + o) = make_uint2(hi[0], hi[1]);
    if (mp != nullptr) {
      *reinterpret_cast<uint2*>(mp + o) = make_uint2(lo[0], lo[1]);
      *reinterpret_cast<uint2*>(mp + H + o) = make_uint2(mid[0], mid[1]);
      *reinterpret_cast<uint2*>(mp + 2 * H + o) = make_uint2(hi[0], hi[1]);
      // Chunks c and c ^ 1 (adjacent lanes) form one 16-value group.
      hsum += __shfl_xor_sync(__activemask(), hsum, 1);
      if ((c & 1) == 0) hsum_out[c / 2] = hsum;
    }
  }
  const int out_emax = emax == 0x7fffffff ? -1 : emax;
  if (tid == 0) {
    ws.x_emax[t] = out_emax;
    if (ws.mail != nullptr) reinterpret_cast<blz_mail_header*>(ws.mail)->x_emax[t] = out_emax;
  }
}

__device__ __forceinline__ void st_release_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}

__device__ __forceinline__ uint32_t ld_acquire_sys_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Last block of the plan kernel: publish the request to the CPU engine (all blocks' mailbox writes are fenced).
__device__ void plan_block_done(const blz_moe_workspace& ws, int nblocks) {
  if (ws.mail == nullptr) return;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x != 0) return;
  if (atomicAdd(ws.blocks_done, 1u) != unsigned(nblocks - 1)) return;
  *ws.blocks_done = 0;
  __threadfence();
  const blz_moe_plan* plan = ws.plan;
  auto* hdr = reinterpret_cast<blz_mail_header*>(ws.mail);
  const uint64_t req = (uint64_t(*(volatile uint32_t*)&plan->seq) << 32) |
                       (uint64_t(*(volatile int32_t*)&plan->n_gpu1 & 0xFFFF) << 16) |
                       uint64_t(*(volatile int32_t*)&plan->n_cpu & 0xFFFF);
  st_release_sys_u64(&hdr->req, req);
}

__global__ void plan_kernel(blz_moe_layer layer, int layer_id, const uint16_t* __restrict__ x, int tokens,
                            const int32_t* __restrict__ topk_ids, blz_moe_workspace ws) {
  __shared__ int s_expert[kMaxItems];
  __shared__ int s_leader[kMaxItems];
  __shared__ int s_item[kMaxItems];
  __shared__ int s_tier[kMaxItems];
  __shared__ int s_slot[kMaxItems];
  __shared__ int s_col[kMaxItems];
  __shared__ int s_ncols[kMaxItems];
  __shared__ int s_imm[kMaxItems];
  __shared__ int s_ssd[kMaxItems];
  __shared__ int s_emax[kMaxCols];
  blz_moe_plan* plan = ws.plan;
  const int K = layer.topk;
  const int routes = tokens * K;
  const int tid = threadIdx.x;
  // Blocks 1..T encode token columns (block t+1 -> token t) while block 0 builds the routing plan.
  if (blockIdx.x > 0) {
    encode_token(layer, x, blockIdx.x - 1, ws, s_emax);
    plan_block_done(ws, gridDim.x);
    return;
  }

  // Residency of every route's expert, loaded in parallel (the tables are usually out of L2 by now).
  for (int r = tid; r < routes; r += blockDim.x) {
    const int e = topk_ids[r];
    const bool valid = e >= 0 && e < layer.num_experts;
    const int slot = valid ? layer.slot_of[e] : -1;
    const int slot1 = (valid && slot < 0 && layer.slot1_of != nullptr) ? layer.slot1_of[e] : -1;
    const int tier = slot >= 0 ? 0 : (slot1 >= 0 ? 2 : 1);  // 0 this GPU, 1 CPU, 2 GPU1
    s_expert[r] = e;
    s_tier[r] = tier;
    s_imm[r] = 0;
    s_ssd[r] = slot == -2 && slot1 < 0;
    s_slot[r] = tier == 0 ? slot : (tier == 2 ? slot1 : -1);
    if (valid && layer.heat != nullptr) {
      unsigned add = 1u;
      if (layer.imm_heat > 0 && layer.route_w != nullptr && layer.defer_min_rank > 0) {
        const int t = r / K, k = r % K;
        const float* w = layer.route_w + t * K;
        int rank = 0;
        for (int j = 0; j < K; ++j) rank += w[j] > w[k] || (w[j] == w[k] && j < k);
        if (rank < layer.defer_min_rank) add += unsigned(layer.imm_heat);
      }
      atomicAdd(&layer.heat[e], add);
    }
  }
  __syncthreads();
  // Leader = first route of the same expert; column = number of earlier routes of the same expert.
  for (int r = tid; r < routes; r += blockDim.x) {
    int leader = r, col = 0, later = 0;
    for (int p = r - 1; p >= 0; --p) {
      if (s_expert[p] == s_expert[r]) {
        leader = p;
        ++col;
      }
    }
    for (int p = r + 1; p < routes; ++p) later += s_expert[p] == s_expert[r];
    s_leader[r] = leader;
    s_col[r] = col;
    s_ncols[r] = 1 + later;  // meaningful for leaders only
  }
  __syncthreads();
  // Immediate CPU items: deferral off, or some route of the expert ranks < defer_min_rank (SSD tier:
  // defer_ssd_rank) by weight in its token.
  for (int r = tid; r < routes; r += blockDim.x) {
    if (s_tier[r] != 1) continue;
    bool imm = !layer.defer;
    if (!imm) {
      const int t = r / K, k = r % K;
      const float* w = layer.route_w + t * K;
      int rank = 0;
      for (int j = 0; j < K; ++j) rank += w[j] > w[k] || (w[j] == w[k] && j < k);
      imm = rank < (s_ssd[r] ? layer.defer_ssd_rank : layer.defer_min_rank);
    }
    if (imm) atomicOr(&s_imm[s_leader[r]], 1);
  }
  __syncthreads();
  if (tid == 0) {
    // Deterministic item numbering in route order (routes <= 128: a short serial pass over shared memory).
    // Immediate CPU items are numbered before the deferred ones.
    int n[3] = {0, 0, 0}, n_imm = 0;
    for (int pass = 0; pass < 2; ++pass) {
      for (int r = 0; r < routes; ++r) {
        if (s_leader[r] != r) continue;
        const int tier = s_tier[r];
        if ((pass == 0) != (tier != 1 || s_imm[r] != 0)) continue;
        blz_moe_item* it = tier == 0 ? &plan->gpu[n[0]] : (tier == 1 ? &plan->cpu[n[1]] : &plan->gpu1[n[2]]);
        s_item[r] = n[tier]++;
        it->expert = s_expert[r];
        it->slot = s_slot[r];
        it->ncols = s_ncols[r];
      }
      if (pass == 0) n_imm = n[1];
    }
    plan->n_gpu = n[0];
    plan->n_cpu = n[1];
    plan->n_gpu1 = n[2];
    plan->n_imm = n_imm;
    plan->tokens = tokens;
    if (ws.mail != nullptr) plan->seq = atomicAdd(ws.seq, 1u) + 1u;
  }
  __syncthreads();
  for (int r = tid; r < routes; r += blockDim.x) {
    const int lead = s_leader[r];
    const int tier = s_tier[lead], item = s_item[lead];
    const int t = r / K, k = r % K;
    const int col = s_col[r];
    blz_moe_item* it = tier == 0 ? &plan->gpu[item] : (tier == 1 ? &plan->cpu[item] : &plan->gpu1[item]);
    it->cols[col] = t;
    plan->col[t][k] = int16_t(col);
    plan->src[t][k] = int16_t(tier);
    plan->item[t][k] = int16_t(item);
  }
  if (ws.mail != nullptr) {
    __syncthreads();
    auto* hdr = reinterpret_cast<blz_mail_header*>(ws.mail);
    const int n_cpu = plan->n_cpu, n_gpu1 = plan->n_gpu1;
    if (tid == 0) {
      hdr->tokens = tokens;
      hdr->n_cpu = n_cpu;
      hdr->n_gpu1 = n_gpu1;
      hdr->n_imm = plan->n_imm;
      hdr->layer = layer_id;
    }
    // Item records are small: copy them word by word.
    const int words = n_cpu * int(sizeof(blz_moe_item) / 4);
    const auto* src = reinterpret_cast<const int32_t*>(plan->cpu);
    auto* dst = reinterpret_cast<int32_t*>(hdr->items);
    for (int i = tid; i < words; i += blockDim.x) dst[i] = src[i];
    const int words1 = n_gpu1 * int(sizeof(blz_moe_item) / 4);
    const auto* src1 = reinterpret_cast<const int32_t*>(plan->gpu1);
    auto* dst1 = reinterpret_cast<int32_t*>(hdr->items1);
    for (int i = tid; i < words1; i += blockDim.x) dst1[i] = src1[i];
  }
  plan_block_done(ws, gridDim.x);
}
__global__ void finalize_kernel(blz_moe_layer layer, blz_moe_workspace ws) {
  const blz_moe_plan* plan = ws.plan;
  const int item_idx = blockIdx.x;
  if (item_idx >= plan->n_gpu) return;
  const blz_moe_item& it = plan->gpu[item_idx];
  const float* m = layer.mult + size_t(it.expert) * 3;
  const float mg = m[0], mu = m[1];
  const int I = layer.inter;
  __shared__ int s_emax;
  for (int c = 0; c < it.ncols; ++c) {
    const int xe = ws.x_emax[it.cols[c]];
    const int64_t* sums = ws.gu_sums + (size_t(item_idx) * kMaxCols + c) * 2 * I;
    uint16_t* h = ws.h + (size_t(item_idx) * kMaxCols + c) * I;
    int e = 1;
    bool nonfinite = false;
    for (int p = threadIdx.x; p < I; p += blockDim.x) {
      uint16_t v = canon::kCanonicalNaN;
      if (xe >= 0) {
        const uint16_t gv = canon::a16_row_output(sums[2 * p], xe, mg);
        const uint16_t uv = canon::a16_row_output(sums[2 * p + 1], xe, mu);
        v = canon::swiglu_bf16(gv, uv);
      }
      h[p] = v;
      nonfinite |= canon::bf16_nonfinite(v);
      e = max(e, canon::a16_exp(v));
    }
    e = __reduce_max_sync(0xffffffffu, e);
    nonfinite = __any_sync(0xffffffffu, nonfinite);
    if (threadIdx.x == 0) s_emax = 0;
    __syncthreads();
    if ((threadIdx.x & 31) == 0) atomicMax(&s_emax, nonfinite ? 0x7fffffff : e);
    __syncthreads();
    const int emax = s_emax == 0x7fffffff ? -1 : s_emax;
    if (threadIdx.x == 0) ws.h_emax[size_t(item_idx) * kMaxCols + c] = emax;
    uint8_t* planes = ws.h_planes + (size_t(item_idx) * kMaxCols + c) * 3 * I;
    for (int p = threadIdx.x; p < I; p += blockDim.x) {
      const int32_t v = emax < 0 ? 0 : canon::a16_encode(h[p], emax);
      planes[p] = uint8_t(v & 0xFF);
      planes[I + p] = uint8_t((v >> 8) & 0xFF);
      planes[2 * I + p] = uint8_t((v >> 16) & 0xFF);
    }
    __syncthreads();
  }
}

// wait

__device__ __forceinline__ uint4 ld_volatile_v4(const void* p) {
  uint4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}


__global__ void wait_kernel(blz_moe_workspace ws) {
  const blz_moe_plan* plan = ws.plan;
  const uint32_t target = plan->seq;
  const auto* hdr = reinterpret_cast<const blz_mail_header*>(ws.mail);
  if (plan->n_cpu > 0)
    while (ld_acquire_sys_u32(&hdr->done) != target) __nanosleep(32);
  if (plan->n_gpu1 > 0)
    while (ld_acquire_sys_u32(&hdr->done1) != target) __nanosleep(32);
}

__global__ void wait_defer_kernel(blz_moe_workspace ws) {
  const blz_moe_plan* plan = ws.plan;
  const auto* hdr = reinterpret_cast<const blz_mail_header*>(ws.mail);
  if (plan->n_gpu1 > 0)
    while (ld_acquire_sys_u32(&hdr->done1) != plan->seq) __nanosleep(32);
  if (plan->n_imm > 0) {
    const uint32_t* done = plan->n_imm == plan->n_cpu ? &hdr->done : &hdr->done_imm;
    while (ld_acquire_sys_u32(done) != plan->seq) __nanosleep(32);
  }
}

// grid (H / 8 / 64, T): one thread per (token, 8 outputs). Runs before the next plan, so ws.plan still describes the
// pending call and the CPU engine's y is still in the mailbox.
__global__ void defer_collect_kernel(int H, blz_moe_workspace ws, const uint16_t* y_cpu) {
  const uint32_t pending = ws.defer->pending_seq, cur = *reinterpret_cast<volatile uint32_t*>(ws.seq);
  if (pending == 0 || pending != cur) return;
  const auto* hdr = reinterpret_cast<const blz_mail_header*>(ws.mail);
  if (threadIdx.x == 0)
    while (ld_acquire_sys_u32(&hdr->done) != pending) __nanosleep(32);
  __syncthreads();
  const blz_moe_plan* plan = ws.plan;
  const int t = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
  if (blockIdx.x == 0 && t == 0 && threadIdx.x == 0) ws.defer->acc_seq = cur + 1;
  if (c >= H / 8) return;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int k = 0; k < BLZ_MOE_MAX_TOPK; ++k) {
    if (plan->src[t][k] != 1 || plan->item[t][k] < plan->n_imm) continue;
    const size_t off = (size_t(plan->item[t][k]) * kMaxCols + plan->col[t][k]) * H + size_t(c) * 8;
    const uint4 v = ld_volatile_v4(y_cpu + off);
    const uint32_t yw[4] = {v.x, v.y, v.z, v.w};
    const float w = ws.defer->w[t][k];
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[i] = __fadd_rn(acc[i], __fmul_rn(w, canon::bf16_to_f32(uint16_t(yw[i / 2] >> (16 * (i % 2))))));
  }
  float4* dst = reinterpret_cast<float4*>(ws.defer_acc + size_t(t) * H + size_t(c) * 8);
  dst[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
  dst[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
}

// GPU1 tier

// Copy the current request's GPU1 items and the token planes from the mailbox into this GPU's plan/workspace.
// Items get a global expert index (layer * E + expert) so the multipliers of every layer live in one table.
__global__ void gpu1_import_kernel(int hidden, int num_experts, blz_moe_workspace ws, const uint8_t* mail) {
  const auto* hdr = reinterpret_cast<const blz_mail_header*>(mail);
  blz_moe_plan* plan = ws.plan;
  const int tid = threadIdx.x;
  const int tokens = *reinterpret_cast<const volatile int32_t*>(&hdr->tokens);
  const int n = *reinterpret_cast<const volatile int32_t*>(&hdr->n_gpu1);
  const int layer = *reinterpret_cast<const volatile int32_t*>(&hdr->layer);
  if (tid == 0) {
    plan->n_gpu = n;
    plan->n_cpu = 0;
    plan->n_gpu1 = 0;
    plan->tokens = tokens;
    plan->seq = uint32_t(*reinterpret_cast<const volatile uint64_t*>(&hdr->req) >> 32);
  }
  for (int i = tid; i < n; i += blockDim.x) {
    blz_moe_item it;
    const volatile int32_t* src = reinterpret_cast<const volatile int32_t*>(&hdr->items1[i]);
    int32_t* dst = reinterpret_cast<int32_t*>(&it);
    for (int w = 0; w < int(sizeof(it) / 4); ++w) dst[w] = src[w];
    it.expert += layer * num_experts;
    plan->gpu[i] = it;
  }
  for (int t = 0; t < tokens; ++t) {
    if (tid == 0) ws.x_emax[t] = *reinterpret_cast<const volatile int32_t*>(&hdr->x_emax[t]);
  }
  const blz_mail_layout ml = blz_mail_layout_for(hidden);
  const auto* src = reinterpret_cast<const volatile uint4*>(mail + ml.planes);
  auto* dst = reinterpret_cast<uint4*>(ws.x_planes);
  const int vecs = tokens * 3 * hidden / 16;
  for (int i = tid; i < vecs; i += blockDim.x) {
    const uint4 v = const_cast<const uint4&>(src[i]);
    dst[i] = v;
  }
}

// Write the GPU1 items' outputs into the mailbox, then publish done1 = seq.
__global__ void gpu1_export_kernel(int hidden, blz_moe_workspace ws, uint8_t* mail) {
  const blz_moe_plan* plan = ws.plan;
  const blz_mail_layout ml = blz_mail_layout_for(hidden);
  auto* y1 = reinterpret_cast<uint4*>(mail + ml.y1);
  const auto* y = reinterpret_cast<const uint4*>(ws.y);
  const int n = plan->n_gpu;
  const int per_col = hidden * 2 / 16;  // uint4 per column
  for (int item = blockIdx.x; item < n; item += gridDim.x) {
    const int ncols = plan->gpu[item].ncols;
    for (int c = 0; c < ncols; ++c) {
      const size_t base = (size_t(item) * kMaxCols + c) * per_col;
      for (int i = threadIdx.x; i < per_col; i += blockDim.x) y1[base + i] = y[base + i];
    }
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    // the last block to finish publishes done1
    if (atomicAdd(ws.blocks_done, 1u) == gridDim.x - 1) {
      *ws.blocks_done = 0;
      auto* hdr = reinterpret_cast<blz_mail_header*>(mail);
      asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(&hdr->done1), "r"(plan->seq) : "memory");
    }
  }
}

// combine

// One thread per (token, 8 consecutive outputs): the K expert outputs are loaded first (16 bytes each, so the
// mailbox reads over PCIe are all in flight together), then summed in the fixed k order.
__global__ void combine_kernel(blz_moe_layer layer, int tokens, const float* __restrict__ topk_w,
                               blz_moe_workspace ws, const uint16_t* y_cpu, const uint16_t* y_gpu1,
                               uint16_t* __restrict__ out) {
  __shared__ int s_item[BLZ_MOE_MAX_TOPK], s_col[BLZ_MOE_MAX_TOPK], s_src[BLZ_MOE_MAX_TOPK];
  __shared__ float s_w[BLZ_MOE_MAX_TOPK];
  const blz_moe_plan* plan = ws.plan;
  const int t = blockIdx.y;
  const int H = layer.hidden, K = layer.topk;
  const bool deferred = plan->n_cpu > plan->n_imm && ws.defer != nullptr;
  if (threadIdx.x < K) {
    s_item[threadIdx.x] = plan->item[t][threadIdx.x];
    s_col[threadIdx.x] = plan->col[t][threadIdx.x];
    // Deferred CPU routes are left out here and added by the next call (blz_moe_defer_collect).
    const int src = plan->src[t][threadIdx.x];
    s_src[threadIdx.x] = deferred && src == 1 && s_item[threadIdx.x] >= plan->n_imm ? 4 : src;
    s_w[threadIdx.x] = topk_w[t * K + threadIdx.x];
    if (deferred && blockIdx.x == 0) ws.defer->w[t][threadIdx.x] = s_w[threadIdx.x];
  }
  if (deferred && blockIdx.x == 0 && t == 0 && threadIdx.x == 0) ws.defer->pending_seq = plan->seq;
  __syncthreads();
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= tokens || c >= H / 8) return;
  uint4 yv[BLZ_MOE_MAX_TOPK];
#pragma unroll
  for (int k = 0; k < BLZ_MOE_MAX_TOPK; ++k) {
    if (k >= K) break;
    const size_t off = (size_t(s_item[k]) * kMaxCols + s_col[k]) * H + size_t(c) * 8;
    const int src = s_src[k];
    yv[k] = src == 4   ? make_uint4(0, 0, 0, 0)
            : src == 0 ? *reinterpret_cast<const uint4*>(ws.y + off)
                       : ld_volatile_v4((src == 1 ? y_cpu : y_gpu1) + off);
  }
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int k = 0; k < BLZ_MOE_MAX_TOPK; ++k) {
    if (k >= K) break;
    if (s_src[k] == 4) continue;  // deferred route
    const float w = s_w[k];
    const uint32_t yw[4] = {yv[k].x, yv[k].y, yv[k].z, yv[k].w};
#pragma unroll
    for (int i = 0; i < 8; ++i)
      acc[i] = __fadd_rn(acc[i], __fmul_rn(w, canon::bf16_to_f32(uint16_t(yw[i / 2] >> (16 * (i % 2))))));
  }
  if (ws.defer != nullptr && ws.defer->acc_seq == plan->seq) {
    // The previous call's deferred CPU routes.
    const float4* d = reinterpret_cast<const float4*>(ws.defer_acc + size_t(t) * H + size_t(c) * 8);
    const float4 a = d[0], b = d[1];
    const float dv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = __fadd_rn(acc[i], dv[i]);
  }
  uint32_t o[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) o[i] = uint32_t(canon::bf16_rn(acc[2 * i])) | (uint32_t(canon::bf16_rn(acc[2 * i + 1])) << 16);
  *reinterpret_cast<uint4*>(out + size_t(t) * H + size_t(c) * 8) = make_uint4(o[0], o[1], o[2], o[3]);
}

// Tensor-core rows: block = one 16-row group of one item, its 4 warps split K; three int8 mma.m16n8k16 per unit
// (lo / mid / hi limbs of up to 8 columns) give the exact P, P * s^ is summed in int64 (order-free), the warps'
// partial sums are added in shared memory. Same results as rows_kernel; more warps in flight per row group.
// Fragment mapping as in moe_wide.cu: lane l reads unit word (l % 4) * 8 + l / 4, column l / 4.
constexpr int kKSplit = 4;

__device__ __forceinline__ void mma_s8u8(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.u8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

__device__ __forceinline__ void mma_s8s8(int32_t (&d)[4], uint32_t a0, uint32_t a1, uint32_t b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(a0), "r"(a1), "r"(b), "r"(0));
}

template <bool kDown>
__global__ void __launch_bounds__(32 * kKSplit) rows_mma_kernel(blz_moe_layer layer, blz_moe_workspace ws) {
  __shared__ int64_t red[kKSplit - 1][32][4];
  const blz_moe_plan* plan = ws.plan;
  const int item_idx = blockIdx.y;
  if (item_idx >= plan->n_gpu) return;
  const blz_moe_item& it = plan->gpu[item_idx];
  const expert::Geometry geo{layer.hidden, layer.inter};
  const int g = blockIdx.x;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int ncols = it.ncols;
  const int blocks = kDown ? geo.d_blocks() : geo.gu_blocks();
  const int k_dim = kDown ? layer.inter : layer.hidden;
  const uint8_t* group = layer.pool + int64_t(it.slot) * layer.slot_stride + (kDown ? geo.gu_bytes() : 0) +
                         expert::unit_offset(g, 0, blocks);
  const int col = min(lane >> 2, ncols - 1);  // padding columns repeat the last one (not stored)
  const uint8_t* planes = kDown ? ws.h_planes + (size_t(item_idx) * kMaxCols + col) * 3 * layer.inter
                                : ws.x_planes + size_t(it.cols[col]) * 3 * layer.hidden;
  const int word = (lane & 3) * 8 + (lane >> 2), r = lane >> 2, koff = (lane & 3) * 4;
  const int per = blocks / kKSplit, b_begin = warp * per, b_end = warp == kKSplit - 1 ? blocks : b_begin + per;
  int64_t acc[4] = {0, 0, 0, 0};
  constexpr int kU = 4;
  for (int b0 = b_begin; b0 < b_end; b0 += kU) {
    uint32_t w[kU], lo[kU], mid[kU], hi[kU];
    int32_t s0[kU], s1[kU];
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (b0 + u < b_end) {
        const uint8_t* un = group + size_t(b0 + u) * expert::kUnitBytes;
        w[u] = ld_nc_u32(un + 4 * word);
        s0[u] = canon::s_hat(un[expert::kUnitCodeBytes + r]);
        s1[u] = canon::s_hat(un[expert::kUnitCodeBytes + r + 8]);
        const uint8_t* p = planes + 16 * (b0 + u) + koff;
        lo[u] = ld_nc_u32(p), mid[u] = ld_nc_u32(p + k_dim), hi[u] = ld_nc_u32(p + 2 * k_dim);
      }
    }
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      if (b0 + u >= b_end) break;
      const uint32_t a0 = c2x4(w[u] & 0x0F0F0F0Fu), a1 = c2x4((w[u] >> 4) & 0x0F0F0F0Fu);
      int32_t dl[4], dm[4], dh[4];
      mma_s8u8(dl, a0, a1, lo[u]);
      mma_s8u8(dm, a0, a1, mid[u]);
      mma_s8s8(dh, a0, a1, hi[u]);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int32_t pv = int32_t(uint32_t(dl[j]) + (uint32_t(dm[j]) << 8) + (uint32_t(dh[j]) << 16));
        acc[j] += int64_t(pv) * (j < 2 ? s0[u] : s1[u]);
      }
    }
  }
  if (warp > 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) red[warp - 1][lane][j] = acc[j];
  }
  __syncthreads();
  if (warp != 0) return;
#pragma unroll
  for (int v = 0; v < kKSplit - 1; ++v)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[j] += red[v][lane][j];
  const float md = kDown ? layer.mult[size_t(it.expert) * 3 + 2] : 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = 2 * (lane & 3) + (j & 1);
    if (c >= ncols) continue;
    const int row = 16 * g + r + (j < 2 ? 0 : 8);
    if constexpr (kDown) {
      const int e = ws.h_emax[size_t(item_idx) * kMaxCols + c];
      ws.y[(size_t(item_idx) * kMaxCols + c) * layer.hidden + row] =
          e < 0 ? canon::kCanonicalNaN : canon::a16_row_output(acc[j], e, md);
    } else {
      ws.gu_sums[(size_t(item_idx) * kMaxCols + c) * geo.gu_rows() + row] = acc[j];
    }
  }
}

// Columns per item never exceed the token count: specialize on its power-of-two bound.
template <bool kDown>
void launch_rows(int groups, int max_items, int tokens, const blz_moe_layer& layer, const blz_moe_workspace& ws,
                 cudaStream_t stream) {
  (void)tokens;
  rows_mma_kernel<kDown><<<dim3(groups, max_items), 32 * kKSplit, 0, stream>>>(layer, ws);
}

}  // namespace
}  // namespace blaze::moe

using namespace blaze::moe;

extern "C" size_t blz_moe_workspace_bytes(int hidden, int inter) {
  const Sizes s(hidden, inter);
  return align256(s.plan) + align256(s.x_planes) + align256(s.x_emax) + align256(s.gu_sums) + align256(s.h) +
         align256(s.h_emax) + align256(s.y) + align256(s.h_planes) + align256(s.defer) + align256(s.defer_acc);
}

extern "C" void blz_moe_workspace_carve(void* base, int hidden, int inter, blz_moe_workspace* ws) {
  const Sizes s(hidden, inter);
  auto* p = static_cast<uint8_t*>(base);
  auto take = [&](size_t n) {
    uint8_t* r = p;
    p += align256(n);
    return r;
  };
  ws->plan = reinterpret_cast<blz_moe_plan*>(take(s.plan));
  ws->x_planes = take(s.x_planes);
  ws->x_emax = reinterpret_cast<int32_t*>(take(s.x_emax));
  ws->gu_sums = reinterpret_cast<int64_t*>(take(s.gu_sums));
  ws->h = reinterpret_cast<uint16_t*>(take(s.h));
  ws->h_emax = reinterpret_cast<int32_t*>(take(s.h_emax));
  ws->y = reinterpret_cast<uint16_t*>(take(s.y));
  ws->h_planes = take(s.h_planes);
  ws->defer = reinterpret_cast<blz_moe_defer*>(take(s.defer));
  ws->defer_acc = reinterpret_cast<float*>(take(s.defer_acc));
  ws->mail = nullptr;
  ws->seq = nullptr;
  ws->blocks_done = nullptr;
}

extern "C" cudaError_t blz_moe_defer_collect(int hidden, int tokens, const blz_moe_workspace* ws,
                                             cudaStream_t stream) {
  if (ws->mail == nullptr || ws->defer == nullptr || tokens < 1) return cudaSuccess;
  if (hidden % 8 != 0 || tokens > BLZ_MOE_MAX_TOKENS) return cudaErrorInvalidValue;
  constexpr int kBlock = 64;
  const blz_mail_layout ml = blz_mail_layout_for(hidden);
  defer_collect_kernel<<<dim3((hidden / 8 + kBlock - 1) / kBlock, tokens), kBlock, 0, stream>>>(
      hidden, *ws, reinterpret_cast<const uint16_t*>(ws->mail + ml.y));
  return cudaGetLastError();
}

extern "C" cudaError_t blz_moe_wait_defer(const blz_moe_workspace* ws, cudaStream_t stream) {
  if (ws->mail == nullptr) return cudaSuccess;
  wait_defer_kernel<<<1, 1, 0, stream>>>(*ws);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_moe_wait_cpu(const blz_moe_workspace* ws, cudaStream_t stream) {
  if (ws->mail == nullptr) return cudaSuccess;
  wait_kernel<<<1, 1, 0, stream>>>(*ws);
  return cudaGetLastError();
}

static cudaError_t route_impl(const blz_moe_layer* layer, int layer_id, const uint16_t* x, int tokens,
                              const int32_t* topk_ids, const blz_moe_workspace* ws, cudaStream_t stream,
                              float* kernel_ms) {
  if (tokens < 1 || tokens > BLZ_MOE_MAX_TOKENS || layer->topk < 1 || layer->topk > BLZ_MOE_MAX_TOPK ||
      !blaze::expert::Geometry{layer->hidden, layer->inter}.valid()) {
    return cudaErrorInvalidValue;
  }
  const blaze::expert::Geometry geo{layer->hidden, layer->inter};
  cudaEvent_t ev[5];
  if (kernel_ms) {
    for (auto& e : ev) cudaEventCreate(&e);
    cudaEventRecord(ev[0], stream);
  }
  auto mark = [&](int i) {
    if (kernel_ms) cudaEventRecord(ev[i], stream);
  };
  plan_kernel<<<1 + tokens, 256, 0, stream>>>(*layer, layer_id, x, tokens, topk_ids, *ws);
  mark(1);
  const int max_items = tokens * layer->topk;
  launch_rows<false>(geo.gu_groups(), max_items, tokens, *layer, *ws, stream);
  mark(2);
  finalize_kernel<<<max_items, 256, 0, stream>>>(*layer, *ws);
  mark(3);
  launch_rows<true>(geo.d_groups(), max_items, tokens, *layer, *ws, stream);
  mark(4);
  const cudaError_t err = cudaGetLastError();
  if (kernel_ms) {
    cudaEventSynchronize(ev[4]);
    for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&kernel_ms[i], ev[i], ev[i + 1]);
    for (auto& e : ev) cudaEventDestroy(e);
  }
  return err;
}

extern "C" cudaError_t blz_moe_gpu_route(const blz_moe_layer* layer, const uint16_t* x, int tokens,
                                         const int32_t* topk_ids, const blz_moe_workspace* ws, cudaStream_t stream) {
  return route_impl(layer, 0, x, tokens, topk_ids, ws, stream, nullptr);
}

extern "C" cudaError_t blz_moe_gpu_route_layer(const blz_moe_layer* layer, int layer_id, const uint16_t* x, int tokens,
                                               const int32_t* topk_ids, const blz_moe_workspace* ws,
                                               cudaStream_t stream) {
  return route_impl(layer, layer_id, x, tokens, topk_ids, ws, stream, nullptr);
}

extern "C" cudaError_t blz_moe_gpu_route_timed(const blz_moe_layer* layer, const uint16_t* x, int tokens,
                                               const int32_t* topk_ids, const blz_moe_workspace* ws,
                                               cudaStream_t stream, float* kernel_ms) {
  return route_impl(layer, 0, x, tokens, topk_ids, ws, stream, kernel_ms);
}

extern "C" cudaError_t blz_moe_gpu1_route(const blz_moe_layer* layer, const blz_moe_workspace* ws, uint8_t* mail,
                                          int tokens, cudaStream_t stream) {
  if (tokens < 1 || tokens > BLZ_MOE_MAX_TOKENS) return cudaErrorInvalidValue;
  const blaze::expert::Geometry geo{layer->hidden, layer->inter};
  const int max_items = tokens * layer->topk;
  gpu1_import_kernel<<<1, 256, 0, stream>>>(layer->hidden, layer->num_experts, *ws, mail);
  launch_rows<false>(geo.gu_groups(), max_items, tokens, *layer, *ws, stream);
  finalize_kernel<<<max_items, 256, 0, stream>>>(*layer, *ws);
  launch_rows<true>(geo.d_groups(), max_items, tokens, *layer, *ws, stream);
  gpu1_export_kernel<<<16, 256, 0, stream>>>(layer->hidden, *ws, mail);
  return cudaGetLastError();
}

extern "C" cudaError_t blz_moe_combine(const blz_moe_layer* layer, int tokens, const float* topk_w,
                                       const blz_moe_workspace* ws, const uint16_t* y_cpu, uint16_t* out,
                                       cudaStream_t stream) {
  if (layer->hidden % 8 != 0) return cudaErrorInvalidValue;
  constexpr int kBlock = 64;
  const dim3 grid((layer->hidden / 8 + kBlock - 1) / kBlock, tokens);
  const uint16_t* y1 = y_cpu;
  if (ws->mail != nullptr) {
    const blz_mail_layout ml = blz_mail_layout_for(layer->hidden);
    y_cpu = reinterpret_cast<const uint16_t*>(ws->mail + ml.y);
    y1 = reinterpret_cast<const uint16_t*>(ws->mail + ml.y1);
  }
  combine_kernel<<<grid, kBlock, 0, stream>>>(*layer, tokens, topk_w, *ws, y_cpu, y1, out);
  return cudaGetLastError();
}
