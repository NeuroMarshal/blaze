#include "blaze/safetensors.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <stdexcept>

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
  std::string h(n, '\0');
  pread_all(fd, h.data(), n, 8);
  const int file = int(fds_.size());
  fds_.push_back(fd);
  direct_fds_.push_back(open(path.c_str(), O_RDONLY | O_DIRECT));
  const uint64_t base = 8 + n;
  // Flat object: "name": {"dtype": "...", "shape": [...], "data_offsets": [a, b]}, ... plus "__metadata__".
  size_t i = h.find('{') + 1;
  while (true) {
    const size_t k0 = h.find('"', i);
    if (k0 == std::string::npos) break;
    const size_t k1 = h.find('"', k0 + 1);
    const std::string key = h.substr(k0 + 1, k1 - k0 - 1);
    const size_t o0 = h.find('{', k1);
    const size_t o1 = h.find('}', o0);
    const std::string obj = h.substr(o0, o1 - o0 + 1);
    i = o1 + 1;
    if (key == "__metadata__") continue;
    Tensor t;
    t.file = file;
    const size_t d = obj.find("\"dtype\"");
    const size_t d0 = obj.find('"', obj.find(':', d) + 1);
    t.dtype = obj.substr(d0 + 1, obj.find('"', d0 + 1) - d0 - 1);
    const size_t s0 = obj.find('[', obj.find("\"shape\""));
    const std::string shape = obj.substr(s0 + 1, obj.find(']', s0) - s0 - 1);
    for (size_t p = 0; p < shape.size();) {
      const size_t c = shape.find(',', p);
      const std::string num = shape.substr(p, c == std::string::npos ? std::string::npos : c - p);
      if (num.find_first_of("0123456789") != std::string::npos) t.shape.push_back(std::stoll(num));
      if (c == std::string::npos) break;
      p = c + 1;
    }
    const size_t f0 = obj.find('[', obj.find("\"data_offsets\""));
    const std::string offs = obj.substr(f0 + 1, obj.find(']', f0) - f0 - 1);
    const size_t comma = offs.find(',');
    t.begin = base + std::stoull(offs.substr(0, comma));
    t.end = base + std::stoull(offs.substr(comma + 1));
    map_[key] = std::move(t);
  }
}

}  // namespace blaze
