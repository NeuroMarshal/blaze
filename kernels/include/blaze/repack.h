// Host-side rg16 packing of row-major NVFP4 rows (SSE): the same bytes as expert::pack_matrix, ~10x faster, and by
// row groups so an expert read from the checkpoint in pieces can be packed piece by piece.
#pragma once

#include <cstddef>
#include <cstdint>

namespace blaze::expert {

/// \brief Source rows of one record matrix region. Record row d comes from matrix m = (interleaved ? d & 1 : 0),
/// source row s = (interleaved ? d >> 1 : d); codes[m] / scales[m] point at source row `row0` of matrix m, rows
/// code_stride / scale_stride bytes apart (codes: K/2 bytes, even k in the low nibble; scales: K/16 E4M3 bytes).
struct PackSource {
  const uint8_t* codes[2];
  const uint8_t* scales[2];
  size_t code_stride;
  size_t scale_stride;
  int row0;
  bool interleaved;
};

/// \brief Pack record row groups [g0, g1) of a matrix region with K = k_dim into `matrix` (the region's base).
/// Writes exactly the units of those groups.
void pack_groups(const PackSource& src, int k_dim, int g0, int g1, uint8_t* matrix);

}  // namespace blaze::expert
