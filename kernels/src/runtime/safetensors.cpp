#include "blaze/safetensors.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace blaze {

namespace {

void pread_all(int fd, void* dst, size_t n, uint64_t off) {
  size_t done = 0;
  while (done < n) {
    const ssize_t r = pread(fd, static_cast<uint8_t*>(dst) + done, n - done, off_t(off + done));
    if (r <= 0) throw std::runtime_error("safetensors: read failed");
    done += size_t(r);
  }
}

constexpr uint64_t kDirectAlign = 4096;
/// Largest header read (the format's own limit is 100 MB).
constexpr uint64_t kMaxHeader = 100ull << 20;

/// A JSON reader just big enough for a safetensors header: strings with escapes, nested values skipped whole.
struct Json {
  const std::string& s;
  size_t i = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("safetensors header: ") + what + " at byte " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) return ++i, true;
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail("unexpected character");
  }
  std::string str() {
    expect('"');
    std::string out;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\') {
        if (++i >= s.size()) break;
        // tensor names and dtypes are ASCII: an escape keeps its character (\u sequences are skipped as text)
      }
      out += s[i++];
    }
    if (i >= s.size()) fail("unterminated string");
    ++i;
    return out;
  }
  uint64_t num() {
    ws();
    const size_t b = i;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i == b) fail("a number expected");
    return std::stoull(s.substr(b, i - b));
  }
  std::vector<uint64_t> nums() {
    std::vector<uint64_t> v;
    expect('[');
    if (eat(']')) return v;
    do v.push_back(num());
    while (eat(','));
    expect(']');
    return v;
  }
  void skip() {
    ws();
    if (i >= s.size()) fail("a value expected");
    const char c = s[i];
    if (c == '"') {
      str();
    } else if (c == '{' || c == '[') {
      const char close = c == '{' ? '}' : ']';
      ++i;
      if (eat(close)) return;
      do {
        if (c == '{') {
          str();
          expect(':');
        }
        skip();
      } while (eat(','));
      expect(close);
    } else {
      while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
    }
  }
};

}  // namespace

SafetensorsDir::Bounce::~Bounce() { std::free(data); }

void SafetensorsDir::Bounce::reserve(size_t n) {
  if (n <= size) return;
  std::free(data);
  size = (n + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
  data = static_cast<uint8_t*>(std::aligned_alloc(kDirectAlign, size));
  if (!data) throw std::bad_alloc();
}

const uint8_t* SafetensorsDir::read_range(const Tensor& t, uint64_t off, size_t n, Bounce& b, bool direct) const {
  if (off + n > t.bytes()) throw std::runtime_error("safetensors: range outside the tensor");
  const uint64_t at = t.begin + off;
  const int dfd = direct ? direct_fds_[size_t(t.file)] : -1;
  if (dfd >= 0) {
    const uint64_t a0 = at / kDirectAlign * kDirectAlign;
    const uint64_t need = at + n - a0;
    const size_t len = (need + kDirectAlign - 1) / kDirectAlign * kDirectAlign;
    b.reserve(len);
    uint64_t done = 0;
    while (done < need) {
      const ssize_t r = pread(dfd, b.data + done, len - done, off_t(a0 + done));
      if (r < 0 && errno == EINTR) continue;
      if (r <= 0) throw std::runtime_error("safetensors: direct read failed");
      done += uint64_t(r);
    }
    return b.data + (at - a0);
  }
  b.reserve(n);
  pread_all(fds_[size_t(t.file)], b.data, n, at);
  return b.data;
}

SafetensorsDir::SafetensorsDir(const std::string& dir) {
  std::vector<std::filesystem::path> files;
  for (const auto& e : std::filesystem::directory_iterator(dir))
    if (e.path().extension() == ".safetensors") files.push_back(e.path());
  std::sort(files.begin(), files.end());
  for (const auto& f : files) add(f.string());
  if (fds_.empty()) throw std::runtime_error("no *.safetensors in " + dir);
}

SafetensorsDir::~SafetensorsDir() {
  for (int fd : fds_) close(fd);
  for (int fd : direct_fds_)
    if (fd >= 0) close(fd);
}

const SafetensorsDir::Tensor* SafetensorsDir::find(const std::string& name) const {
  auto it = map_.find(name);
  return it == map_.end() ? nullptr : &it->second;
}

const SafetensorsDir::Tensor& SafetensorsDir::at(const std::string& name) const {
  const Tensor* t = find(name);
  if (!t) throw std::runtime_error("safetensors: missing tensor " + name);
  return *t;
}

void SafetensorsDir::read(const Tensor& t, void* dst, size_t bytes) const {
  if (bytes != t.bytes()) throw std::runtime_error("safetensors: tensor size mismatch");
  pread_all(fds_[size_t(t.file)], dst, bytes, t.begin);
}

void SafetensorsDir::add(const std::string& path) {
  const int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  uint64_t n = 0;
  pread_all(fd, &n, 8, 0);
  if (n == 0 || n > kMaxHeader) {
    close(fd);
    throw std::runtime_error(path + ": a safetensors header of " + std::to_string(n) + " bytes");
  }
  std::string h(n, '\0');
  pread_all(fd, h.data(), n, 8);
  const int file = int(fds_.size());
  fds_.push_back(fd);
  direct_fds_.push_back(open(path.c_str(), O_RDONLY | O_DIRECT));
  const uint64_t base = 8 + n;
  // {"name": {"dtype": "...", "shape": [...], "data_offsets": [a, b]}, ..., "__metadata__": {...}}
  try {
    Json j{h};
    j.expect('{');
    if (j.eat('}')) return;
    do {
      const std::string key = j.str();
      j.expect(':');
      if (key == "__metadata__") {
        j.skip();
        continue;
      }
      Tensor t;
      t.file = file;
      bool offsets = false;
      j.expect('{');
      if (!j.eat('}')) {
        do {
          const std::string field = j.str();
          j.expect(':');
          if (field == "dtype") {
            t.dtype = j.str();
          } else if (field == "shape") {
            for (uint64_t d : j.nums()) t.shape.push_back(int64_t(d));
          } else if (field == "data_offsets") {
            const std::vector<uint64_t> o = j.nums();
            if (o.size() != 2 || o[1] < o[0]) j.fail("data_offsets of two ascending numbers expected");
            t.begin = base + o[0];
            t.end = base + o[1];
            offsets = true;
          } else {
            j.skip();
          }
        } while (j.eat(','));
        j.expect('}');
      }
      if (t.dtype.empty() || !offsets) j.fail(("tensor " + key + " without dtype or data_offsets").c_str());
      map_[key] = std::move(t);
    } while (j.eat(','));
    j.expect('}');
  } catch (const std::exception& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

}  // namespace blaze
