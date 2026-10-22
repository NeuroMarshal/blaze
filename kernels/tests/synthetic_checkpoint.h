// Synthetic safetensors files for the store tests.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

struct Out {
  std::string name, dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> data;
};

inline void write_safetensors(const std::filesystem::path& path, const std::vector<Out>& ts) {
  std::string h = "{\"__metadata__\":{\"format\":\"pt\"}";
  size_t off = 0;
  for (const Out& t : ts) {
    std::string shape;
    for (size_t d = 0; d < t.shape.size(); ++d) shape += (d ? "," : "") + std::to_string(t.shape[d]);
    h += ",\"" + t.name + "\":{\"dtype\":\"" + t.dtype + "\",\"shape\":[" + shape + "],\"data_offsets\":[" +
         std::to_string(off) + "," + std::to_string(off + t.data.size()) + "]}";
    off += t.data.size();
  }
  h += "}";
  while (h.size() % 8) h += ' ';
  FILE* f = std::fopen(path.c_str(), "wb");
  const uint64_t n = h.size();
  std::fwrite(&n, 8, 1, f);
  std::fwrite(h.data(), 1, h.size(), f);
  for (const Out& t : ts) std::fwrite(t.data.data(), 1, t.data.size(), f);
  std::fclose(f);
}
