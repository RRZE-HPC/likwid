#include <debug.h>

#include <stdarg.h>
#include <unistd.h>

#include "error_ng.h"

static const char * const LOG_LEVEL_PREFIX_NORMAL[DEBUG_LEV_COUNT] = {
    "ERROR",
    "WARN",
    "INFO",
    "DEBUG",
    "DEVELOP",
};

static const char * const LOG_LEVEL_PREFIX_COLOR[DEBUG_LEV_COUNT] = {
    "\e[1;31mERROR\e[0m",   // bold red
    "\e[1;33mWARN\e[0m",    // bold yellow
    "\e[1;36mINFO\e[0m",    // bold turquoise
    "\e[1;34mDEBUG\e[0m",   // bold blue
    "\e[1;35mDEVELOP\e[0m", // bold magenta
};

static const char * const LOG_CAT_PREFIX_NORMAL[DEBUG_CAT_COUNT] = {
    "CORE",
    "NVMON",
    "ROCMON",
};

static const char * const LOG_CAT_PREFIX_COLOR[DEBUG_CAT_COUNT] = {
    "\e[1;34mCORE\e[0m",    // bold blue
    "\e[1;32mNVMON\e[0m",   // bold green
    "\e[1;31mROCMON\e[0m",  // bold red
};

int debug_levels[DEBUG_CAT_COUNT] = {
    DEBUG_LEV_WARN,
    DEBUG_LEV_WARN,
    DEBUG_LEV_WARN,
};

void debug_level_set(int category, int level) {
    /* Handle invalid category or level */
    if (level < 0)
        level = 0;
    else if (level >= DEBUG_LEV_COUNT)
        level = DEBUG_LEV_DEVELOP;

    if (category < 0 || category >= DEBUG_CAT_COUNT)
        category = 0;

    debug_levels[category] = level;
}

void debug_print(FILE *handle, const char *file, const char *func, int line, int category, int level, const char *fmt, va_list args) {
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

    /* Init text coloring (if TTY) */
    const char *tty_boldyel = "";
    const char *tty_reset = "";
    const char *tty_underline = "";
    const char *category_str = LOG_CAT_PREFIX_NORMAL[category];
    const char *level_str = LOG_LEVEL_PREFIX_NORMAL[level];

    if (isatty(fileno(handle))) {
        tty_boldyel = "\e[1;36m";
        tty_reset = "\e[0m";
        tty_underline = "\e[33;4m";
        category_str = LOG_CAT_PREFIX_COLOR[category];
        level_str = LOG_LEVEL_PREFIX_COLOR[level];
    }

    /* Print actual message.
     * The format looks like this:
     * WARN: MSG [file: FILE, function: FUNCTION, line: LINE] */

    fprintf(handle, "[%s/%s] ", category_str, level_str);

    fprintf(handle, "%s", tty_boldyel);
    vfprintf(handle, fmt, args);
    fprintf(handle, "%s", tty_reset);

    // clang-format off
    fprintf(handle, " [file: %s%s%s, function: %s%s%s, line: %s%d%s]\n",
            tty_underline, file, tty_reset,
            tty_underline, func, tty_reset,
            tty_underline, line, tty_reset);
    // clang-format on
}

void debug_print_error(FILE *handle, int category, int level) {
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

    lw_error_print(handle);
}
