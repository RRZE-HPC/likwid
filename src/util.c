#include <util.h>

#include <stdlib.h>
#include <stdio.h>

char *xvasprintf(const char *__restrict__ fmt, va_list ap) {
    va_list ap2;
    va_copy(ap2, ap);

    int len = vsnprintf(NULL, 0, fmt, ap);
    if (len < 0) {
        perror("vsnprintf");
        exit(EXIT_FAILURE);
    }

    const size_t bytes = (size_t)len + 1;

    char *retval = malloc(bytes);
    if (!retval)
        return NULL;

    vsnprintf(retval, bytes, fmt, ap2);
    return retval;
}

char *xasprintf(const char *__restrict__ fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    char *retval = xvasprintf(fmt, ap);
    va_end(ap);
    return retval;
}
