/*
This file is part of GOTCHA.  For copyright information see the COPYRIGHT
file in the top level directory, or at
https://github.com/LLNL/gotcha/blob/master/COPYRIGHT
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License (as published by the Free
Software Foundation) version 2.1 dated February 1999.  This program is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the terms and conditions of the GNU Lesser General Public License
for more details.  You should have received a copy of the GNU Lesser General
Public License along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
*/

#if !defined(LIBC_WRAPPERS_H_)
#define LIBC_WRAPPERS_H_

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/mman.h>

#ifndef FORCE_NO_LIBC
#define GOTCHA_USE_LIBC
#endif

#if defined(GOTCHA_USE_LIBC) && !defined(BUILDING_LIBC_WRAPPERS)

#define gotcha_malloc             malloc
#define gotcha_realloc            realloc
#define gotcha_free               free
#define gotcha_memcpy             memcpy
#define gotcha_strncmp            strncmp
#define gotcha_strstr             strstr
#define gotcha_assert             assert
#define gotcha_strcmp             strcmp
#define gotcha_getenv             getenv
#define gotcha_getpid             getpid
#define gotcha_getpagesize        getpagesize
#define gotcha_open               open
#define gotcha_mmap               mmap
#define gotcha_atoi               atoi
#define gotcha_close              close
#define gotcha_mprotect           mprotect
#define gotcha_read               read
#define gotcha_memset             memset
#define gotcha_write              write
#define gotcha_strlen             strlen
#define gotcha_strnlen            strnlen
#define gotcha_strtok             strtok
#define gotcha_strncat            strncat
#define gotcha_dbg_printf(A, ...) fprintf(stderr, A, ##__VA_ARGS__)
pid_t gotcha_gettid();            //No libc gettid, always use gotcha version

#else

void *gotcha_malloc(size_t size);
void *gotcha_realloc(void* buffer, size_t size);
void gotcha_free(void* free_me);
void gotcha_memcpy(void* dest, void* src, size_t size);
int gotcha_strncmp(const char* in_one, const char* in_two, int max_length);
char *gotcha_strstr(const char* searchIn,const char* searchFor);
int gotcha_strcmp(const char* in_one, const char* in_two);
char *gotcha_getenv(const char *env);
pid_t gotcha_getpid();
pid_t gotcha_gettid();
unsigned int gotcha_getpagesize();
int gotcha_open(const char *pathname, int flags, ...);
void *gotcha_mmap(void *addr, size_t length, int prot, int flags,
                  int fd, off_t offset);
int gotcha_atoi(const char *nptr);
int gotcha_close(int fd);
int gotcha_mprotect(void *addr, size_t len, int prot);
ssize_t gotcha_read(int fd, void *buf, size_t count);
ssize_t gotcha_write(int fd, const void *buf, size_t count);
void gotcha_assert_fail(const char *s, const char *file, unsigned int line, const char *function);
void *gotcha_memset(void *s, int c, size_t n);
size_t gotcha_strlen(const char* str);
size_t gotcha_strnlen(const char* str, size_t max_length);
char* gotcha_strncat(char* dest, const char* src, size_t n);
char* gotcha_strtok(char* dest, const char* src, size_t n);

#define gotcha_dbg_printf(FORMAT, ...) gotcha_int_printf(2, FORMAT, ##__VA_ARGS__)

#define gotcha_assert(A)                                          \
   do {                                                           \
      if (! (A) )                                                 \
         gotcha_assert_fail("" #A, __FILE__, __LINE__, __func__); \
   } while (0); 

#endif

int gotcha_int_printf(int fd, const char *format, ...);

#endif
