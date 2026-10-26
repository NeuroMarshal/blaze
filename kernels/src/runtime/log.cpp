#include "blaze/log.h"

#include <atomic>
#include <cstdio>
#include <ctime>

namespace {

std::atomic<blaze_log_callback_fn*> g_callback{nullptr};

}  // namespace

extern "C" void blaze_set_log_callback(blaze_log_callback_fn* callback) {
  g_callback.store(callback, std::memory_order_release);
}

namespace blaze {

void log(const char* fmt, ...) {
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);

  if (blaze_log_callback_fn* callback = g_callback.load(std::memory_order_acquire)) {
    callback(msg);
    return;
  }

  char stamp[32];
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
  std::fprintf(stderr, "[%s] %s\n", stamp, msg);
  std::fflush(stderr);
}

}  // namespace blaze
