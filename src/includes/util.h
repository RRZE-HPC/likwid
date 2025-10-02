#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#include <pthread.h>

#define DECLARE_STATIC_PTMUTEX(mutex_name) \
    static pthread_mutex_t mutex_name; \
    __attribute__((constructor)) static void name##_constructor(void) { \
        pthread_mutex_init(&mutex_name, NULL); \
    } \
    __attribute__((destructor)) static void name##_destructor(void) { \
        pthread_mutex_destroy(&mutex_name); \
    }

#endif // INCLUDE_UTIL_H
