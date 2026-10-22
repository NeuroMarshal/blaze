// Routed-expert record (layout rg16 v1) and the scalar reference forward pass.
//
// One record holds one expert's gate, up and down NVFP4 matrices; the same bytes live on NVMe, in host RAM and in
// VRAM slots, read in place by the GPU kernels and the CPU engine (layout after Infernix's nvfp4_expert_rg16_v1):
//   - gate/up matrix: 2*I rows of K = H, rows interleaved (row 2i = gate_i, row 2i+1 = up_i); then down: H rows,
//     K = I.
//   - rows form groups of 16; for row group g and 16-wide k block b there is one 144-byte unit: 128 code bytes,
//     then 16 E4M3 scale bytes (row i of the group at byte i). Units are g-major, then b ascending.
//   - codes of a unit: four 32-byte quads q = 0..3 covering k = 16b + 4q .. 16b + 4q + 3; in quad q, byte j holds
//     row (j / 4) at k = 16b + 4q + (j % 4) in its low nibble and row 8 + (j / 4) at the same k in its high nibble.
//     So the 32-bit word at 32q + 4i holds four consecutive k of row i (low nibbles) and of row i + 8 (high).
// For H = 2560, I = 640: 1,843,200 + 921,600 = 2,764,800 bytes (675 x 4 KiB).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "blaze/canon.h"

namespace blaze::expert {

constexpr int kRowGroup = 16;
constexpr int kUnitBytes = 144;
constexpr int kUnitCodeBytes = 128;

/// \brief Shape of one expert: hidden size H and intermediate size I (both multiples of 16).
struct Geometry {
  int hidden;
  int inter;

  BLZ_HD constexpr int gu_rows() const { return 2 * inter; }
  BLZ_HD constexpr int gu_groups() const { return gu_rows() / kRowGroup; }
  BLZ_HD constexpr int gu_blocks() const { return hidden / 16; }
  BLZ_HD constexpr int d_groups() const { return hidden / kRowGroup; }
  BLZ_HD constexpr int d_blocks() const { return inter / 16; }
  BLZ_HD constexpr size_t gu_bytes() const { return size_t(gu_groups()) * gu_blocks() * kUnitBytes; }
  BLZ_HD constexpr size_t d_bytes() const { return size_t(d_groups()) * d_blocks() * kUnitBytes; }
  BLZ_HD constexpr size_t record_bytes() const { return gu_bytes() + d_bytes(); }
  BLZ_HD constexpr bool valid() const { return hidden > 0 && inter > 0 && hidden % 16 == 0 && inter % 16 == 0; }
};

constexpr Geometry kFlashNext{2560, 640};
static_assert(kFlashNext.record_bytes() == 2764800);

/// \brief Per-matrix FP32 multipliers m = fl32(1 / global weight scale).
struct Multipliers {
  float gate, up, down;
};

/// \brief Byte offset of unit (g, b) in a matrix with `blocks` k-blocks.
BLZ_HD size_t unit_offset(int g, int b, int blocks) { return (size_t(g) * blocks + b) * kUnitBytes; }

/// \brief Code (0..15) of row r, column k inside a matrix region of a record.
BLZ_HD int unit_code(const uint8_t* matrix, int blocks, int r, int k) {
  const uint8_t* u = matrix + unit_offset(r / kRowGroup, k / 16, blocks);
  const int ri = r % kRowGroup, kk = k % 16, q = kk / 4;
  const uint8_t byte = u[32 * q + 4 * (ri % 8) + kk % 4];
  return ri < 8 ? (byte & 15) : (byte >> 4);
}

/// \brief E4M3 scale byte of row r, block b.
BLZ_HD uint8_t unit_scale(const uint8_t* matrix, int blocks, int r, int b) {
  return matrix[unit_offset(r / kRowGroup, b, blocks) + kUnitCodeBytes + r % kRowGroup];
}

/// \brief Write one row-major NVFP4 matrix (codes [rows][K/2], even k in the low nibble; scales [rows][K/16]) into
/// a record matrix region, placing source row `src_row` at record row `dst_row(src_row)`.
template <typename RowMap>
inline void pack_matrix(const uint8_t* codes, const uint8_t* scales, int rows, int k_dim, uint8_t* matrix,
                        RowMap dst_row) {
  const int blocks = k_dim / 16;
  for (int r = 0; r < rows; ++r) {
    const int d = dst_row(r);
    const int ri = d % kRowGroup;
    for (int b = 0; b < blocks; ++b) {
      uint8_t* u = matrix + unit_offset(d / kRowGroup, b, blocks);
      u[kUnitCodeBytes + ri] = scales[size_t(r) * blocks + b];
      for (int kk = 0; kk < 16; ++kk) {
        const int k = 16 * b + kk;
        const int code = (codes[size_t(r) * (k_dim / 2) + k / 2] >> (4 * (k & 1))) & 15;
        uint8_t& byte = u[32 * (kk / 4) + 4 * (ri % 8) + kk % 4];
        byte = ri < 8 ? uint8_t((byte & 0xF0) | code) : uint8_t((byte & 0x0F) | (code << 4));
      }
    }
  }
}

/// \brief Pack an expert from row-major NVFP4 gate [I][H], up [I][H] and down [H][I] matrices.
inline void pack_record(const Geometry& geo, const uint8_t* gate_codes, const uint8_t* gate_scales,
                        const uint8_t* up_codes, const uint8_t* up_scales, const uint8_t* down_codes,
                        const uint8_t* down_scales, uint8_t* record) {
  uint8_t* gu = record;
  uint8_t* dn = record + geo.gu_bytes();
  pack_matrix(gate_codes, gate_scales, geo.inter, geo.hidden, gu, [](int r) { return 2 * r; });
  pack_matrix(up_codes, up_scales, geo.inter, geo.hidden, gu, [](int r) { return 2 * r + 1; });
  pack_matrix(down_codes, down_scales, geo.hidden, geo.inter, dn, [](int r) { return r; });
}

/// \brief Column encoding: emax of a BF16 column, or -1 if it holds Inf/NaN.
inline int a16_emax(const uint16_t* v, int n) {
  int emax = 1;
  for (int j = 0; j < n; ++j) {
    if (canon::bf16_nonfinite(v[j])) return -1;
    const int e = canon::a16_exp(v[j]);
    emax = e > emax ? e : emax;
  }
  return emax;
}

/// \brief Exact row sum S = sum_b (sum_j c2 * X) * S^(scale) of matrix row r.
inline int64_t row_sum_ref(const uint8_t* matrix, int k_dim, int r, const int32_t* x) {
  const int blocks = k_dim / 16;
  int64_t s = 0;
  for (int b = 0; b < blocks; ++b) {
    int32_t p = 0;
    for (int j = 0; j < 16; ++j) p += canon::c2(unit_code(matrix, blocks, r, 16 * b + j)) * x[16 * b + j];
    s += int64_t(p) * canon::s_hat(unit_scale(matrix, blocks, r, b));
  }
  return s;
}

/// \brief Scalar reference: y = expert(x), BF16 in and out (bits), H values each.
inline void forward_ref(const Geometry& geo, const uint8_t* record, const Multipliers& m, const uint16_t* x,
                        uint16_t* y) {
  const uint8_t* gu = record;
  const uint8_t* dn = record + geo.gu_bytes();
  int emax = a16_emax(x, geo.hidden);
  if (emax < 0) {
    for (int r = 0; r < geo.hidden; ++r) y[r] = canon::kCanonicalNaN;
    return;
  }
  std::vector<int32_t> xi(geo.hidden);
  for (int j = 0; j < geo.hidden; ++j) xi[j] = canon::a16_encode(x[j], emax);
  std::vector<uint16_t> h(geo.inter);
  for (int i = 0; i < geo.inter; ++i) {
    const uint16_t g = canon::a16_row_output(row_sum_ref(gu, geo.hidden, 2 * i, xi.data()), emax, m.gate);
    const uint16_t u = canon::a16_row_output(row_sum_ref(gu, geo.hidden, 2 * i + 1, xi.data()), emax, m.up);
    h[i] = canon::swiglu_bf16(g, u);
  }
  emax = a16_emax(h.data(), geo.inter);
  if (emax < 0) {
    for (int r = 0; r < geo.hidden; ++r) y[r] = canon::kCanonicalNaN;
    return;
  }
  std::vector<int32_t> hi(geo.inter);
  for (int j = 0; j < geo.inter; ++j) hi[j] = canon::a16_encode(h[j], emax);
  for (int r = 0; r < geo.hidden; ++r) y[r] = canon::a16_row_output(row_sum_ref(dn, geo.inter, r, hi.data()), emax, m.down);
}

}  // namespace blaze::expert
