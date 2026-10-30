#include "blaze/log.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>

namespace {

std::atomic<blaze_log_callback_fn*> g_callback{nullptr};
std::atomic<blaze_stage_callback_fn*> g_stage{nullptr};

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

}  // namespace

extern "C" void blaze_set_log_callback(blaze_log_callback_fn* callback) {
  g_callback.store(callback, std::memory_order_release);
}

extern "C" void blaze_set_stage_callback(blaze_stage_callback_fn* callback) {
  g_stage.store(callback, std::memory_order_release);
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

Stage::Stage(const char* label, double total, const char* unit) : t0_(now_s()) {
  std::snprintf(label_, sizeof(label_), "%s", label);
  if (blaze_stage_callback_fn* cb = g_stage.load(std::memory_order_acquire))
    id_ = cb(BLAZE_STAGE_BEGIN, -1, label_, 0, total, unit, 1);
  else
    log("blaze: %s ...", label_);
}

void Stage::progress(double done, double total) {
  if (id_ < 0) return;
  if (blaze_stage_callback_fn* cb = g_stage.load(std::memory_order_acquire))
    cb(BLAZE_STAGE_PROGRESS, id_, nullptr, done, total, nullptr, 1);
}

double Stage::seconds() const { return now_s() - t0_; }

void Stage::end(const char* fmt, ...) {
  if (ended_) return;
  ended_ = true;
  char note[512] = {};
  if (fmt) {
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(note, sizeof(note), fmt, ap);
    va_end(ap);
  }
  blaze_stage_callback_fn* cb = g_stage.load(std::memory_order_acquire);
  if (cb && id_ >= 0)
    cb(BLAZE_STAGE_END, id_, note, 0, 0, nullptr, 1);
  else
    log("blaze: %s: %.1f s%s%s", label_, now_s() - t0_, note[0] ? "; " : "", note);
}

Stage::~Stage() {
  if (ended_) return;
  blaze_stage_callback_fn* cb = g_stage.load(std::memory_order_acquire);
  if (cb && id_ >= 0)
    cb(BLAZE_STAGE_END, id_, "", 0, 0, nullptr, 0);
  else
    log("blaze: %s failed after %.1f s", label_, now_s() - t0_);
}

}  // namespace blaze
