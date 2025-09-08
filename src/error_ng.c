#include "error_ng.h"

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// The OUT_OF_MEMORY instance error_scope doesn't contain any information itself,
// but we use it to identify out of memory errors. For out of memory errors we
// cannot allocate a new error_scope, so we use this global instance instead.
static struct LwErrorScope OUT_OF_MEMORY;
static struct LwErrorScope INIT_ERROR;

static pthread_mutex_t init_mutex;
static pthread_key_t tsd;
static bool tsd_init = false;

static void error_free_pthread(void *scope);

__attribute__((constructor)) static void error_lib_init(void) {
    // According to man page, pthread_mutex_init cannot not fail.
    pthread_mutex_init(&init_mutex, NULL);
}

__attribute__((destructor)) static void error_lib_fini(void) {
    // This can only fail for error checking mutexes, but this mutex should not be locked during library unload.
    pthread_mutex_destroy(&init_mutex);
}

static cerr_t error_init(void) {
    if (tsd_init)
        return NULL;

    // The lock shouldn't fail if our library is correct.
    int err = pthread_mutex_lock(&init_mutex);
    assert(err == 0);

    cerr_t retval = NULL;

    if (!tsd_init) {
        if (pthread_key_create(&tsd, error_free_pthread) != 0) {
            fprintf(stderr, "pthread_key_create: Unable to create thread local error state: %s\n", strerror(err));
            retval = &INIT_ERROR;
        } else if (atexit(lw_error_clear)) {
            // The deleter function of pthread_key_create only affects launched threads, not the main thread.
            // Use atexit to delete the error from the main thread.
            // TODO: ^This is possibly unsafe. If the library is unloaded before the main thread terminates,
            // this will cause a segmentation fault, since the deleter code is no longer there.
            // How should we solve this?
            fprintf(stderr, "atexit: Unable to register error cleanup function");
            retval = &INIT_ERROR;
        } else {
            tsd_init = true;
        }
    }

    // The unlock shouldn't fail if our library is correct.
    err = pthread_mutex_unlock(&init_mutex);
    assert(err == 0);

    return retval;
}

static cerr_t error_append_valist(const char *file, const char *func, int line, int error_val, lw_error_val_to_str_t error_val_to_str, const char *fmt, va_list args);

cerr_t lw_error_set(const char *file, const char *func, int line, int error_val, lw_error_val_to_str_t error_val_to_str, const char *fmt, ...) {
    cerr_t retval = error_init();
    if (retval)
        return retval;

    lw_error_clear();

    va_list args;
    va_start(args, fmt);

    retval = error_append_valist(file, func, line, error_val, error_val_to_str, fmt, args);

    va_end(args);

    return retval;
}

cerr_t lw_error_append(const char *file, const char *func, int line, int error_val, lw_error_val_to_str_t error_val_to_str, const char *fmt, ...) {
    cerr_t retval = error_init();
    if (retval)
        return retval;

    va_list args;
    va_start(args, fmt);

    retval = error_append_valist(file, func, line, error_val, error_val_to_str, fmt, args);

    va_end(args);

    return retval;
}

cerr_t lw_error_get(void) {
    cerr_t retval = error_init();
    if (retval)
        return retval;

    return pthread_getspecific(tsd);
}

void lw_error_clear(void) {
    if (!tsd_init)
        return;

    lw_error_free_scope(pthread_getspecific(tsd));
    pthread_setspecific(tsd, NULL);
}

static cerr_t error_append_valist(const char *file, const char *func, int line, int error_val, lw_error_val_to_str_t error_val_to_str, const char *fmt, va_list args) {
    err_t new_scope = calloc(1, sizeof(*new_scope));
    if (!new_scope) {
        lw_error_clear();
        int err = pthread_setspecific(tsd, &OUT_OF_MEMORY);
        assert(err == 0);
        return &OUT_OF_MEMORY;
    }

    va_list args2;
    va_copy(args2, args);

    int len = vsnprintf(NULL, 0, fmt, args);
    if (len < 0)
        len = 0;

    const size_t message_buf_len = (size_t)(len + 1);
    char *message = calloc(message_buf_len, sizeof(*message));
    if (!message) {
        free(new_scope);
        lw_error_clear();
        int err = pthread_setspecific(tsd, &OUT_OF_MEMORY);
        assert(err == 0);
        return &OUT_OF_MEMORY;
    }

    vsnprintf(message, message_buf_len, fmt, args2);

    new_scope->message = message;
    new_scope->file = file;
    new_scope->func = func;
    new_scope->line = line;
    new_scope->error_val = error_val;
    new_scope->error_val_to_str = error_val_to_str;
    new_scope->prev = pthread_getspecific(tsd);

    pthread_setspecific(tsd, new_scope);

    return new_scope;
}

err_t lw_error_copy(void) {
    return lw_error_copy_scope(lw_error_get());
}

err_t lw_error_copy_scope(cerr_t scope) {
    if (!scope)
        return NULL;

    err_t new_scope = calloc(1, sizeof(*new_scope));
    if (!new_scope)
        goto out_of_memory;

    new_scope->message = strdup(scope->message);
    if (!new_scope->message)
        goto out_of_memory;

    new_scope->file = scope->file;
    new_scope->func = scope->func;
    new_scope->line = scope->line;
    new_scope->error_val = scope->error_val;
    new_scope->error_val_to_str = scope->error_val_to_str;
    new_scope->prev = lw_error_copy_scope(scope->prev);
    return new_scope;

out_of_memory:
    if (new_scope)
        free(new_scope->message);
    free(new_scope);
    return &OUT_OF_MEMORY;
}

void lw_error_free_scope(err_t scope) {
    if (!scope || scope == &OUT_OF_MEMORY || scope == &INIT_ERROR)
        return;

    if (scope->prev)
        lw_error_free_scope(scope->prev);

    free(scope->message);
    free(scope);
}

static void error_free_pthread(void *scope) {
    lw_error_free_scope(scope);
}

static void error_print_scope_recurse(FILE *handle, cerr_t scope, int depth);

void lw_error_print(FILE *handle) {
    if (error_init()) {
        fprintf(handle, "Unable to print error: Library/Error init failure\n");
        return;
    }
    error_print_scope_recurse(handle, pthread_getspecific(tsd), 0);
}

void lw_error_print_stdout(void) {
    lw_error_print(stdout);
}

void lw_error_print_stderr(void) {
    lw_error_print(stderr);
}

void lw_error_print_scope(FILE *handle, cerr_t scope) {
    error_print_scope_recurse(handle, scope, 0);
}

void lw_error_print_scope_stdout(cerr_t scope) {
    lw_error_print_scope(stdout, scope);
}

void lw_error_print_scope_stderr(cerr_t scope) {
    lw_error_print_scope(stderr, scope);
}

static void error_print_scope_recurse(FILE *handle, cerr_t scope, int depth) {
    if (!scope)
        return;

    /* Determine which strings to output. */
    const char *msg = "?";
    const char *file = "?";
    const char *func = "?";
    int line = 0;
    int error_val = 0;
    lw_error_val_to_str_t error_val_to_str = (lw_error_val_to_str_t)strerror; // cast due to const return value

    if (scope == &OUT_OF_MEMORY) {
        msg = "Out of memory";
        error_val = ENOMEM;
    } else if (scope == &INIT_ERROR) {
        msg = "Error library init error";
    } else {
        msg = scope->message;
        file = scope->file;
        func = scope->func;
        line = scope->line;
        error_val = scope->error_val;
        error_val_to_str = scope->error_val_to_str;
    }

    /* Init text formatters (if TTY) */
    const char *tty_boldred = "";
    const char *tty_boldtur = "";
    const char *tty_reset = "";
    const char *tty_underline = "";

    if (isatty(fileno(handle))) {
        tty_boldred = "\e[1;31m";
        tty_boldtur = "\e[1;36m";
        tty_reset = "\e[0m";
        tty_underline = "\e[33;4m";
    }

    /* Print actual error.
     * The format looks like this:
     * [Error 1] MSG1 [ERRVAL: ERRSTR, file: FILE, function: FUNCTION, line: LINE].
     * [      2] MSG2 [ERRVAL: ERRSTR, file: FILE, function: FUNCTION, line: LINE].
     *     .....
     *
     * We conditionally add colors/formatting if a TTY is available. */
    if (depth <= 0) {
        /* If this is the only error we print, do not print the depth as number. */
        if (scope->prev)
            fprintf(handle, "[%sError%s 1] ", tty_boldred, tty_reset);
        else
            fprintf(handle, "[%sError%s] ", tty_boldred, tty_reset);
    } else {
        fprintf(handle, "[%7d] ", depth + 1);
    }

    fprintf(handle, "%s%s%s [", tty_boldtur, msg, tty_reset);

    if (error_val != 0) {
        if (error_val_to_str) {
            const char *errstr = error_val_to_str(error_val);
            if (!errstr)
                errstr = "<null>";
            fprintf(handle, "errval: %s%s (%d)%s, ", tty_underline, errstr, error_val, tty_reset);
        } else {
            fprintf(handle, "%s%d%s, ", tty_underline, error_val, tty_reset);
        }
    }

    fprintf(handle, "file: %s%s%s, function: %s%s%s, line: %s%d%s]\n",
            tty_underline, file, tty_reset,
            tty_underline, func, tty_reset,
            tty_underline, line, tty_reset);

    error_print_scope_recurse(handle, scope->prev, depth + 1);
}
