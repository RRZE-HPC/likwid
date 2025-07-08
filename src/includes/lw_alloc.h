#ifndef ALLOC_H
#define ALLOC_H

#include <stddef.h>
#include <stdarg.h>

void *lw_malloc(size_t size);
void *lw_calloc(size_t nmemb, size_t size);
void *lw_realloc(void *ptr, size_t size);
char *lw_strdup(const char *s);
char *lw_strndup(const char *s, size_t size);
char *lw_asprintf(const char *__restrict__ fmt, ...);
char *lw_vasprintf(const char *__restrict__ fmt, va_list ap);

#endif // ALLOC_H
