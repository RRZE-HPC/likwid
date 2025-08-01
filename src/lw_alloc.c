#include "lw_alloc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *lw_malloc(size_t size)
{
    void *retval = malloc(size);
    if (!retval) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    return retval;
}

void *lw_calloc(size_t nmemb, size_t size)
{
    void *retval = calloc(nmemb, size);
    if (!retval) {
        perror("calloc");
        exit(EXIT_FAILURE);
    }
    return retval;
}

void *lw_realloc(void *ptr, size_t size)
{
    void *retval = realloc(ptr, size);
    if (!retval) {
        perror("realloc");
        exit(EXIT_FAILURE);
    }
    return retval;
}

char *lw_strdup(const char *s)
{
    char *retval = strdup(s);
    if (!retval) {
        perror("strdup");
        exit(EXIT_FAILURE);
    }
    return retval;
}

char *lw_strndup(const char *s, size_t size)
{
    char *retval = strndup(s, size);
    if (!retval) {
        perror("strndup");
        exit(EXIT_FAILURE);
    }
    return retval;
}

char *lw_asprintf(const char *__restrict__ fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    char *retval = lw_vasprintf(fmt, ap);
    va_end(ap);
    return retval;
}

char *lw_vasprintf(const char *__restrict__ fmt, va_list ap)
{
    va_list ap2;
    va_copy(ap2, ap);

    int len = vsnprintf(NULL, 0, fmt, ap);
    if (len < 0) {
        perror("vsnprintf");
        exit(EXIT_FAILURE);
    }

    const size_t bytes = (size_t)len + 1;

    char *retval       = lw_malloc(bytes);
    vsnprintf(retval, bytes, fmt, ap2);
    return retval;
}
