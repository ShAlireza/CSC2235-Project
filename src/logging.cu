#include "logging.h"
#include <sys/select.h>

static struct {
  log_level_t level;
  bool quiet;
} LogConfig;

void set_log_level(log_level_t level) {
  LogConfig.level = level;
  return;
}

void set_quiet(bool enable) {
  LogConfig.quiet = enable;
  return;
}

void log_message(log_level_t level, const char *format, ...) {

  if (LogConfig.level > level) {
    return;
  }

  const char *color;
  const char *level_str;

  switch (level) {
  case DEBUG:
    color = BLUE;
    level_str = "DEBUG";
    break;
  case INFO:
    color = GREEN;
    level_str = "INFO";
    break;
  case WARN:
    color = YELLOW;
    level_str = "WARN";
    break;
  case ERR:
    color = RED;
    level_str = "ERR";
    break;
  default:
    color = RESET;
    level_str = "UNKNOWN";
  }

  // Get current time
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double epoch_time = tv.tv_sec + tv.tv_usec / 1000000.0;

  // Print log message
  printf("%s[%.6f] [%s]: ", color, epoch_time, level_str);

  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);

  printf("%s\n", RESET);
}
