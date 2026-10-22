// CPU side of the hybrid MoE: exact W4A16 expert kernels (AVX2) over rg16 records, bit-identical to the GPU route.
#pragma once

#include <cstddef>
#include <cstdint>

#include "blaze/expert.h"

namespace blaze::cpu {

/// \brief A16 limb planes of one column (K values): lo, mid unsigned bytes, hi signed byte, per-block hi sums.
struct Column {
  const uint8_t* lo;
  const uint8_t* mid;
  const int8_t* hi;
  const int32_t* hi_sum;  // [K / 16]: sum of hi over each 16-wide block (for the biased-code correction)
  int emax;               // -1: the column holds Inf or NaN
};

/// \brief Storage for one column's planes (K <= max_k).
struct ColumnBuffer {
  uint8_t* lo = nullptr;
  uint8_t* mid = nullptr;
  int8_t* hi = nullptr;
  int32_t* hi_sum = nullptr;
  int emax = 1;
  Column view() const { return {lo, mid, hi, hi_sum, emax}; }
};

/// \brief Encode a BF16 column (bits) into `buf` (arrays of >= n bytes, n / 16 sums).
void encode_column(const uint16_t* v, int n, ColumnBuffer& buf);

/// \brief Exact row sums of row groups [g0, g1): out[(16 * g + r) * ncols + c], int64. `blocks` = K / 16.
/// ncols <= 8. Requires AVX2.
void row_sums_avx2(const uint8_t* matrix, int blocks, int g0, int g1, const Column* cols, int ncols, int64_t* out);

/// \brief Block-floating-point view of a column for the approximate CPU path: per 16-value block the A16 integer is
/// kept to its top 8 bits, q = round(X >> shift) in [-127, 127]. Sums computed from it are P' << shift (in the same
/// fixed point as the exact sums), so the exact output conversion applies unchanged.
struct QColumn {
  const int8_t* q;       // [K]
  const uint8_t* shift;  // [K / 16]
};

/// \brief Fill q / shift (arrays of >= n and n / 16) from an encoded column.
void quantize_column_q8(const Column& col, int n, int8_t* q, uint8_t* shift);

/// \brief Approximate row sums from block-floating-point columns: half the AVX2 work of row_sums_avx2 (one byte
/// product per weight instead of three). Same output layout. ncols <= 8.
void row_sums_q8_avx2(const uint8_t* matrix, int blocks, int g0, int g1, const QColumn* cols, int ncols, int64_t* out);

/// \brief Portable reference of row_sums_avx2 (same output).
void row_sums_scalar(const uint8_t* matrix, int blocks, int g0, int g1, const Column* cols, int ncols,
                     int64_t* out);

/// \brief Per-thread scratch for expert_forward (sized for one geometry and up to 8 columns).
class Scratch {
 public:
  explicit Scratch(const expert::Geometry& geo);
  ~Scratch();
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;

  expert::Geometry geo;
  ColumnBuffer x[8], h[8];
  uint16_t* h_bits;  // [8][I]
  int64_t* sums;     // [max(2I, H)][8]

 private:
  void* mem_;
};

/// \brief One expert for `ncols` columns on the calling thread: y[c] = expert(x[c]) (BF16 bits, H values each).
void expert_forward(const uint8_t* record, const expert::Multipliers& m, const uint16_t* const* x, int ncols,
                    uint16_t* const* y, Scratch& scratch, bool use_avx2 = true);

}  // namespace blaze::cpu
