#ifndef DEBUG_H
#define DEBUG_H

#include <stdio.h>
#include <stdarg.h>

enum debug_category {
    DEBUG_CAT_NORMAL,
    DEBUG_CAT_NVMON,
    DEBUG_CAT_ROCMON,
    DEBUG_CAT_COUNT,
};

enum debug_level {
    DEBUG_LEV_ERROR,
    DEBUG_LEV_WARN,
    DEBUG_LEV_INFO,
    DEBUG_LEV_DEBUG,
    DEBUG_LEV_DEVELOP,
    DEBUG_LEV_COUNT,
};

/* Common printing macros. */
#define MANUAL_PRINT(cat, lev, fmt, ...) \
    debug_print_inline(stderr, __FILE__, __func__, __LINE__, (cat), (level), (fmt), ##__VA_ARGS__)
#define ERROR_PRINT(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_ERROR, fmt, ##__VA_ARGS__)
#define WARN_PRINT(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_WARN, fmt, ##__VA_ARGS__)
#define INFO_PRINT(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_INFO, fmt, ##__VA_ARGS__)
#define DEBUG_PRINT(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_DEBUG, fmt, ##__VA_ARGS__)
#define DEVELOP_PRINT(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_DEVELOP, fmt, ##__VA_ARGS__)

/* NVMON printing macros. */
#define ERROR_PRINT_NVMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NVMON, DEBUG_LEV_ERROR, (fmt), ##__VA_ARGS__)
#define WARN_PRINT_NVMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NVMON, DEBUG_LEV_WARN, (fmt), ##__VA_ARGS__)
#define INFO_PRINT_NVMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NVMON, DEBUG_LEV_INFO, (fmt), ##__VA_ARGS__)
#define DEBUG_PRINT_NVMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NVMON, DEBUG_LEV_DEBUG, (fmt), ##__VA_ARGS__)
#define DEVELOP_PRINT_NVMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NVMON, DEBUG_LEV_DEVELOP, (fmt), ##__VA_ARGS__)

/* ROCMON printing macros. */
#define ERROR_PRINT_ROCMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_ROCMON, DEBUG_LEV_ERROR, (fmt), ##__VA_ARGS__)
#define WARN_PRINT_ROCMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_ROCMON, DEBUG_LEV_WARN, (fmt), ##__VA_ARGS__)
#define INFO_PRINT_ROCMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_ROCMON, DEBUG_LEV_INFO, (fmt), ##__VA_ARGS__)
#define DEBUG_PRINT_ROCMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_ROCMON, DEBUG_LEV_DEBUG, (fmt), ##__VA_ARGS__)
#define DEVELOP_PRINT_ROCMON(fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_ROCMON, DEBUG_LEV_DEVELOP, (fmt), ##__VA_ARGS__)

/* Helper printing macros. */
// formerly called CHECK_ERROR
#define ERROR_PRINT_IF_FAIL(expr, fmt, ...) \
    do { \
        if ((expr) < 0) \
            MANUAL_PRINT(DEBUG_CAT_NORMAL, DBEUG_LEV_ERROR, (fmt), ##__VA_ARGS__); \
    } while (0)

// formerly called VERBOSEPRINTREG
#define DEBUG_PRINT_REG(cpuid, reg, flags, fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_DEBUG, fmt " [%d] Register 0x%llX, Flags: 0x%llX", ##__VA_ARGS__, (cpuid), (unsigned long long)(reg), (unsigned long long)(flags))

// formerly called VERBOSEPRINTPCIREG
#define DEBUG_PRINT_PCIREG(cpuid, dev, reg, flags, fmt, ...) \
    MANUAL_PRINT(DEBUG_CAT_NORMAL, DEBUG_LEV_DEBUG, fmt " [%d] Device %d Register 0x%llX, Flags: 0x%llX", ##__VA_ARGS__, (cpuid), (dev), (unsigned long long)(reg), (unsigned long long)(flags))

extern int debug_levels[DEBUG_CAT_COUNT];

void debug_level_set(int category, int level);
void debug_print(FILE *handle, const char *file, const char *func, int line, int category, int level, const char *fmt, va_list args);

static inline void debug_print_inline(FILE *handle, const char *file, const char *func, int line, int category, int level, const char *fmt, ...) {
    /* Handle invalid category or level */
    if (level < 0)
        level = 0;
    else if (level >= DEBUG_LEV_COUNT)
        level = DEBUG_LEV_DEVELOP;

    if (category < 0 || category >= DEBUG_CAT_COUNT)
        category = 0;

    /* Do not print errors if the level is not high enough. */
    if (level > debug_levels[category])
        return;

    va_list args;
    va_start(args, fmt);

    debug_print(handle, file, func, line, category, level, fmt, args);

    va_end(args);
}

#endif // DEBUG_H
