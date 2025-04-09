#ifndef LOGGING_H
#define LOGGING_H

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// ANSI color codes
#define RESET "\x1B[0m"
#define GREEN "\x1B[32m"
#define YELLOW "\x1B[33m"
#define RED "\x1B[31m"
#define BLUE "\x1B[36m"

typedef enum {
  DEBUG,
  INFO,
  WARN,
  ERR,
} log_level_t;

#define LOG(level, format, ...)                                                \
  log_message(level, "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_DEBUG(format, ...)                                                  \
  log_message(DEBUG, "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_INFO(format, ...)                                                  \
  log_message(INFO, "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_WARN(format, ...)                                                  \
  log_message(WARN, "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)

#define LOG_ERR(format, ...)                                                   \
  log_message(ERR, "[%s:%d] " format, __FILE__, __LINE__, ##__VA_ARGS__)

void set_log_level(log_level_t level);
void set_quiet(bool enable);
void log_message(log_level_t level, const char *format, ...);

#endif /* LOGGING_H */
