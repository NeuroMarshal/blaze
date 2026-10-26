// Formatted C++ runtime logs. When installed, the callback receives the message without a timestamp; otherwise the
// original timestamped stderr output is used (including in standalone C++ tests/tools).
#pragma once

#include <cstdarg>

extern "C" {
typedef void blaze_log_callback_fn(const char* message);
void blaze_set_log_callback(blaze_log_callback_fn* callback);
}

namespace blaze {

__attribute__((format(printf, 1, 2))) void log(const char* fmt, ...);

}  // namespace blaze
