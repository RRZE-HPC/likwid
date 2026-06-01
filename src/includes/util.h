#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#include <stdarg.h>

char *xvasprintf(const char *__restrict__ fmt, va_list ap);
char *xasprintf(const char *__restrict__ fmt, ...);

int search_path_env(const char *binary, char **pathFinal);

#endif // INCLUDE_UTIL_H
