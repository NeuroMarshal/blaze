// Formatted C++ runtime logs. When installed, the callback receives the message without a timestamp; otherwise the
// original timestamped stderr output is used (including in standalone C++ tests/tools).
#pragma once

#include <cstdarg>
#include <cstdint>

extern "C" {
typedef void blaze_log_callback_fn(const char* message);
void blaze_set_log_callback(blaze_log_callback_fn* callback);

// Load stages of the host's log (begin: returns the stage's id; progress: done of total in `unit`, "B" for bytes; end:
// the note after the time, ok = 0 when the stage failed).
enum { BLAZE_STAGE_BEGIN = 0, BLAZE_STAGE_PROGRESS = 1, BLAZE_STAGE_END = 2 };
typedef int64_t blaze_stage_callback_fn(int op, int64_t id, const char* text, double done, double total,
                                        const char* unit, int ok);
void blaze_set_stage_callback(blaze_stage_callback_fn* callback);
}

namespace blaze {

__attribute__((format(printf, 1, 2))) void log(const char* fmt, ...);

// A stage of the load: "▶ label" when it begins, a progress bar while it runs (`progress`), "✔ label  time  note"
// at `end`, "✖" when it is destroyed before (an exception). Without a host callback: plain log lines.
class Stage {
 public:
  explicit Stage(const char* label, double total = 0, const char* unit = "");
  Stage(const Stage&) = delete;
  Stage& operator=(const Stage&) = delete;
  ~Stage();
  void progress(double done, double total = -1);
  __attribute__((format(printf, 2, 3))) void end(const char* fmt = nullptr, ...);
  double seconds() const;

 private:
  int64_t id_ = -1;
  bool ended_ = false;
  double t0_ = 0;
  char label_[160] = {};
};

}  // namespace blaze
