#include <lw_util.h>

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

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

int search_path_env(const char *binary, char **pathFinal)
{
    const char *pathEnv = getenv("PATH");
    if (!pathEnv)
        return -ENOENT;

    char *path = strdup(pathEnv);
    if (!path)
        return -errno;

    int err = 0;
    char *saveptr;
    for (char *tok = strtok_r(path, ":", &saveptr); tok; tok = strtok_r(NULL, ":", &saveptr)) {
        char *pathCandidate;
        err = asprintf(&pathCandidate, "%s/%s", tok, binary);
        if (err < 0) {
            err = -errno;
            goto ret;
        }

        struct stat s;
        err = stat(pathCandidate, &s);
        if (err < 0) {
            err = -errno;
            free(pathCandidate);
            continue;
        }

        if ((s.st_mode & S_IFMT) == S_IFREG) {
            err = -ENOENT;
            free(pathCandidate);
            continue;
        }

        *pathFinal = pathCandidate;
        goto ret;
    }

ret:
    free(path);
    return err;
}
