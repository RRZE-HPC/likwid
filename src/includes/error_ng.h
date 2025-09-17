#ifndef ERROR_NG_H
#define ERROR_NG_H

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include "likwid.h"

typedef const struct LwErrorScope *cerr_t;
typedef struct LwErrorScope *err_t;

#define ERROR_SET_MANUAL(errval, strfunc, fmt, ...) \
    lw_error_set(__FILE__, __func__, __LINE__, (errval), (strfunc), (fmt), ##__VA_ARGS__)
#define ERROR_SET(fmt, ...) \
    ERROR_SET_MANUAL(0, NULL, (fmt), ##__VA_ARGS__)
#define ERROR_SET_ERRNO(fmt, ...) \
    ERROR_SET_MANUAL(errno, (lw_error_val_to_str_t)strerror, (fmt), ##__VA_ARGS__)
#define ERROR_SET_LWERR(errval, fmt, ...) \
    ERROR_SET_MANUAL((errval < 0) ? -errval : errval, (lw_error_val_to_str_t)strerror, (fmt), ##__VA_ARGS__)

#define ERROR_WRAP_MSG(fmt, ...) lw_error_wrap(__FILE__, __func__, __LINE__, 0, NULL, (fmt), ##__VA_ARGS__)
#define ERROR_WRAP() lw_error_wrap(__FILE__, __func__, __LINE__, 0, NULL, "<no description>")
#define ERROR_WRAP_CALL_MSG(scope, fmt, ...) ((scope) ? ERROR_WRAP_MSG((fmt), ##__VA_ARGS__) : NULL)
#define ERROR_WRAP_CALL(scope) ((scope) ? ERROR_WRAP() : NULL)

#endif // ERROR_NG_H
