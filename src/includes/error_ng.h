#ifndef ERROR_NG_H
#define ERROR_NG_H

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "likwid.h"

typedef const struct LwErrorScope cerr_t;
typedef struct LwErrorScope err_t;

#define ERROR_SET(fmt, ...) lw_error_set(__FILE__, __func__, __LINE__, 0, NULL, (fmt), __VA_ARGS__)
#define ERROR_SET_ERRNO(fmt, ...) lw_error_set(__FILE__, __func__, __LINE__, errno, (error_val_to_str_t)strerror, (fmt), __VA_ARGS__)
#define ERROR_APPEND(fmt, ...) lw_error_append(__FILE__, __func__, __LINE__, 0, NULL, (fmt), __VA_ARGS__)
#define ERROR_APPEND_ERRNO(fmt, ...) lw_error_append(__FILE__, __func__, __LINE__, errno, (error_val_to_str_t)strerror, (fmt), __VA_ARGS__)

#endif // ERROR_NG_H
