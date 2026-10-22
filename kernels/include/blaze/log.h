// Console messages in SGLang's format ("[YYYY-MM-DD HH:MM:SS] message", stderr), so blaze's lines read like the
// server's own. Info messages are emitted once at load; per-interval statistics only when asked for (swap log_s).
#pragma once

#include <cstdarg>
#include <cstdio>
#include <ctime>

namespace blaze {

__attribute__((format(printf, 1, 2))) inline void log(const char* fmt, ...) {
  char stamp[32];
  const std::time_t now = std::time(nullptr);
  std::tm tm{};
  localtime_r(&now, &tm);
  std::strftime(stamp, sizeof(stamp), "%Y-%m-%d %H:%M:%S", &tm);
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);
  std::fprintf(stderr, "[%s] %s\n", stamp, msg);
  std::fflush(stderr);
}

}  // namespace blaze
