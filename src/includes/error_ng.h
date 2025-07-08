#ifndef ERROR_NG_H
#define ERROR_NG_H

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

typedef const struct error_scope cerr_t;
typedef struct error_scope err_t;
typedef const char *(*error_val_to_str_t)(int error_val);

struct error_scope {
    char *message;
    const char *file;
    const char *func;
    int line;
    int error_val;
    error_val_to_str_t error_val_to_str;
    struct error_scope *prev;
};

cerr_t *error_set(const char *file, const char *func, int line, int error_val, error_val_to_str_t error_val_to_str, const char *fmt, ...);
cerr_t *error_append(const char *file, const char *func, int line, int error_val, error_val_to_str_t error_val_to_str, const char *fmt, ...);
cerr_t *error_get(void);
void error_clear(void);

err_t *error_copy(void);
err_t *error_copy_scope(cerr_t *scope);
void error_free_scope(err_t *scope);

void error_print(FILE *file);
void error_print_stdout(void);
void error_print_stderr(void);
void error_print_scope(FILE *file, cerr_t *scope);
void error_print_scope_stdout(cerr_t *scope);
void error_print_scope_stderr(cerr_t *scope);

// should we rename this? It's a bit confusing this is the same name as error_set.
#define ERROR_SET(fmt, ...) error_set(__FILE__, __func__, __LINE__, 0, NULL, (fmt), __VA_ARGS__)
#define ERROR_SET_ERRNO(fmt, ...) error_set(__FILE__, __func__, __LINE__, errno, (error_val_to_str_t)strerror, (fmt), __VA_ARGS__)
#define ERROR_APPEND(fmt, ...) error_append(__FILE__, __func__, __LINE__, 0, NULL, (fmt), __VA_ARGS__)
#define ERROR_APPEND_ERRNO(fmt, ...) error_append(__FILE__, __func__, __LINE__, errno, (error_val_to_str_t)strerror, (fmt), __VA_ARGS__)

#endif // ERROR_NG_H
