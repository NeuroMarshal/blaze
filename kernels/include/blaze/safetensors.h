// Read-only view of a directory of *.safetensors files: header parsing and positioned reads of raw tensor bytes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace blaze {

class SafetensorsDir {
 public:
  struct Tensor {
    int file = -1;
    std::string dtype;            // safetensors dtype string ("U8", "F8_E4M3", "F32", "BF16", ...)
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;  // absolute byte offsets in the file
    size_t bytes() const { return size_t(end - begin); }
  };

  /// \brief Index every *.safetensors file of `dir` (headers only). Throws if there is none.
  explicit SafetensorsDir(const std::string& dir);
  ~SafetensorsDir();
  SafetensorsDir(const SafetensorsDir&) = delete;
  SafetensorsDir& operator=(const SafetensorsDir&) = delete;

  const Tensor* find(const std::string& name) const;
  /// \brief Like find, but throws with the tensor name when it is missing.
  const Tensor& at(const std::string& name) const;
  /// \brief Read all bytes of `t` into `dst` (`bytes` must equal t.bytes()). Thread-safe (pread).
  void read(const Tensor& t, void* dst, size_t bytes) const;

  /// \brief Grow-only 4 KiB-aligned buffer for read_range (one per thread).
  struct Bounce {
    uint8_t* data = nullptr;
    size_t size = 0;
    ~Bounce();
    void reserve(size_t n);
  };
  /// \brief Bytes [off, off + n) of `t` into `b`; returns where they start inside b.data. direct: O_DIRECT (no page
  /// cache: the aligned span around the range is read), falling back to a buffered read where it is not supported.
  /// Thread-safe.
  const uint8_t* read_range(const Tensor& t, uint64_t off, size_t n, Bounce& b, bool direct) const;
  size_t files() const { return fds_.size(); }
  /// \brief Every tensor, by name.
  const std::unordered_map<std::string, Tensor>& tensors() const { return map_; }

 private:
  void add(const std::string& path);

  std::vector<int> fds_;
  std::vector<int> direct_fds_;  // O_DIRECT descriptors (-1: not supported)
  std::unordered_map<std::string, Tensor> map_;
};

}  // namespace blaze
