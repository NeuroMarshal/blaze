// GPU <-> CPU mailbox of the hybrid MoE: one region of mapped pinned host memory per engine.
//
// Protocol for one MoE layer call (one request in flight: the GPU waits for a request's result before the next
// layer can plan, and only requests with CPU work are waited for):
//   GPU0 plan kernel: writes tokens, the CPU items, the GPU1 items and each token's A16 planes (lo, mid, hi, hi sums,
//                     emax) into the mailbox, then __threadfence_system() and a release store of
//                     req = (seq << 32) | (n_gpu1 << 16) | n_cpu (seq counts layer calls, from a device counter).
//   CPU engine      : acquires req; if n_cpu > 0 computes the CPU items (the immediate ones first, then done_imm = seq),
//                     writes y, then release-stores done = seq.
//   GPU1 service    : acquires req; if n_gpu1 > 0 runs the GPU1 items on GPU1, writes y1, then done1 = seq.
//   GPU0 wait kernel: spins on acquire loads of done / done1 for the tiers that have work.
//   GPU0 combine    : reads y / y1 of the CPU / GPU1 items from the mailbox with volatile loads.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "blaze/moe.h"

#define BLZ_MAIL_ITEMS (BLZ_MOE_MAX_TOKENS * BLZ_MOE_MAX_TOPK)

typedef struct {
  uint64_t req;                       // (seq << 32) | (n_gpu1 << 16) | n_cpu, written last by GPU0
  uint8_t pad0[56];
  uint32_t done;                      // seq of the last request served by the CPU engine
  uint8_t pad1[60];
  uint32_t done1;                     // seq of the last request served by GPU1
  uint8_t pad2[60];
  uint32_t done_imm;                  // seq of the last request whose immediate CPU items (the first n_imm) are done
  uint8_t pad3[60];
  int32_t tokens;
  int32_t n_cpu;
  int32_t n_gpu1;
  int32_t layer;
  int32_t n_imm;                      // CPU items 0 .. n_imm - 1 are waited for now; the rest are deferred
  int32_t pad4;
  int32_t x_emax[BLZ_MOE_MAX_TOKENS];
  blz_moe_item items[BLZ_MAIL_ITEMS];   // CPU items
  blz_moe_item items1[BLZ_MAIL_ITEMS];  // GPU1 items (slot = GPU1 slot)
} blz_mail_header;

static inline uint32_t blz_req_seq(uint64_t req) { return (uint32_t)(req >> 32); }
static inline int blz_req_cpu(uint64_t req) { return (int)(req & 0xFFFF); }
static inline int blz_req_gpu1(uint64_t req) { return (int)((req >> 16) & 0xFFFF); }

/// \brief Byte offsets of the variable-size parts that follow the header (H = hidden size).
typedef struct {
  size_t planes;   // [T][3][H] bytes: lo, mid, hi (two's complement byte) limbs of each token's x
  size_t hi_sum;   // [T][H / 16] int32
  size_t y;        // [BLZ_MAIL_ITEMS][BLZ_MOE_MAX_TOKENS][H] BF16 bits of the CPU items
  size_t y1;       // [BLZ_MAIL_ITEMS][BLZ_MOE_MAX_TOKENS][H] BF16 bits of the GPU1 items
  size_t total;
} blz_mail_layout;

#if defined(__CUDACC__)
#define BLZ_MAIL_HD __host__ __device__
#else
#define BLZ_MAIL_HD
#endif

static inline BLZ_MAIL_HD size_t blz_mail_align(size_t n) { return (n + 255) & ~(size_t)255; }

static inline BLZ_MAIL_HD blz_mail_layout blz_mail_layout_for(int hidden) {
  blz_mail_layout l;
  l.planes = blz_mail_align(sizeof(blz_mail_header));
  l.hi_sum = l.planes + blz_mail_align((size_t)BLZ_MOE_MAX_TOKENS * 3 * hidden);
  l.y = l.hi_sum + blz_mail_align((size_t)BLZ_MOE_MAX_TOKENS * (hidden / 16) * 4);
  l.y1 = l.y + blz_mail_align((size_t)BLZ_MAIL_ITEMS * BLZ_MOE_MAX_TOKENS * hidden * 2);
  l.total = l.y1 + blz_mail_align((size_t)BLZ_MAIL_ITEMS * BLZ_MOE_MAX_TOKENS * hidden * 2);
  return l;
}
