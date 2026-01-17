#ifndef DBGHELPER_H
#define DBGHELPER_H

#ifndef NDEBUG
#define debug_print(fmt, ...)                                                  \
  do {                                                                         \
    fprintf(stderr, "%s:%d:%s(): " fmt, __FILE__, __LINE__,                    \
            __func__ __VA_OPT__(, ) __VA_ARGS__);                              \
  } while (0)

#else
#define dbg_print(...)
#endif

#endif