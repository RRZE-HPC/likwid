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

/**
 * autotee -
 * Using gotcha, wrap the major IO writing routines with functions that 
 * "tee" any stdout output to another file.  Init by calling
 *    init_autotee(filename)
 * finish by calling:
 *    close_autotee()
 *
 * Note, this is a demonstration program for gotcha and does not handle
 * cases like stdout's file descriptor being dup'd or more esoteric 
 * IO routines.
 **/


#include "gotcha/gotcha_types.h"
#include "gotcha/gotcha.h"
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static int tee_fd = -1;
static FILE *tee_FILE = NULL;

static int printf_wrapper(const char *format, ...);
static int fprintf_wrapper(FILE *stream, const char *format, ...);
static int vfprintf_wrapper(FILE *stream, const char *str, va_list args);
static int vprintf_wrapper(const char *str, va_list args);
static ssize_t write_wrapper(int fd, const void *buffer, size_t size);
static int puts_wrapper(const char *str);
static int fputs_wrapper(const char *str, FILE *f);
static int fwrite_wrapper(const void *ptr, size_t size, size_t nmemb, FILE *stream);

static gotcha_wrappee_handle_t orig_printf_handle;
static gotcha_wrappee_handle_t orig_fprintf_handle;
static gotcha_wrappee_handle_t orig_vfprintf_handle;
static gotcha_wrappee_handle_t orig_vprintf_handle;
static gotcha_wrappee_handle_t orig_write_handle;
static gotcha_wrappee_handle_t orig_puts_handle;
static gotcha_wrappee_handle_t orig_fputs_handle;
static gotcha_wrappee_handle_t orig_fwrite_handle;


#define NUM_IOFUNCS 8
struct gotcha_binding_t iofuncs[] = {
   { "printf", printf_wrapper, &orig_printf_handle },
   { "fprintf", fprintf_wrapper, &orig_fprintf_handle },
   { "vfprintf", vfprintf_wrapper, &orig_vfprintf_handle },
   { "vprintf", vprintf_wrapper, &orig_vprintf_handle },
   { "write", write_wrapper, &orig_write_handle },
   { "puts", puts_wrapper, &orig_puts_handle },
   { "fputs", fputs_wrapper, &orig_fputs_handle },
   { "fwrite", fwrite_wrapper, &orig_fwrite_handle }
};

int init_autotee(const char *teefile)
{
   enum gotcha_error_t result;
   gotcha_set_priority("testing/whether/this/works", 1);
   tee_FILE = fopen(teefile, "w");
   if (!tee_FILE) {
      perror("Failed to open tee file");
      return -1;
   }
   tee_fd = fileno(tee_FILE);

   result = gotcha_wrap(iofuncs, NUM_IOFUNCS, "testing/whether");
   if (result != GOTCHA_SUCCESS) {
      fprintf(stderr, "gotcha_wrap returned %d\n", (int) result);
      return -1;
   }

   return 0;
}

int close_autotee()
{
   if (tee_FILE) {
      fclose(tee_FILE);
      tee_fd = -1;
   }
   return 0;
}

static int printf_wrapper(const char *format, ...)
{
   typeof(&vfprintf) orig_vfprintf = gotcha_get_wrappee(orig_vfprintf_handle);
   typeof(&vprintf) orig_vprintf = gotcha_get_wrappee(orig_vprintf_handle);
   int result;
   va_list args, args2;
   va_start(args, format);

   if (tee_FILE) {
      va_copy(args2, args);   
      orig_vfprintf(tee_FILE, format, args2);
      va_end(args2);
   }

   result = orig_vprintf(format, args);   
   va_end(args);

   return result;
}

static int fprintf_wrapper(FILE *stream, const char *format, ...)
{
   typeof(&vfprintf) orig_vfprintf = gotcha_get_wrappee(orig_vfprintf_handle);
   int result;
   va_list args, args2;
   va_start(args, format);
   
   if (stream != stdout) {
      result = orig_vfprintf(stream, format, args);
   }
   else {
      if (tee_FILE) {
         va_copy(args2, args);
         orig_vfprintf(tee_FILE, format, args2);
         va_end(args2);
      }
      result = orig_vfprintf(stdout, format, args);
   }

   va_end(args);
   return result;
}

static int vfprintf_wrapper(FILE *stream, const char *str, va_list args)
{
   typeof(&vfprintf) orig_vfprintf = gotcha_get_wrappee(orig_vfprintf_handle);
   va_list args2;
   if (stream != stdout) {
      return orig_vfprintf(stream, str, args);
   }
   if (tee_FILE) {
      va_copy(args2, args);
      orig_vfprintf(tee_FILE, str, args2);
      va_end(args2);
   }
   return orig_vfprintf(stream, str, args);
}

static int vprintf_wrapper(const char *str, va_list args)
{
   typeof(&vfprintf) orig_vfprintf = gotcha_get_wrappee(orig_vfprintf_handle);
   typeof(&vprintf) orig_vprintf = gotcha_get_wrappee(orig_vprintf_handle);
   va_list args2;
   if (tee_FILE) {
      va_copy(args2, args);
      orig_vfprintf(tee_FILE, str, args2);
      va_end(args2);
   }
   return orig_vprintf(str, args);
}

static ssize_t write_wrapper(int fd, const void *buffer, size_t size)
{
   typeof(&write) orig_write = gotcha_get_wrappee(orig_write_handle);
   if (fd != 1)
      return orig_write(fd, buffer, size);
   
   if (tee_fd != -1) 
      orig_write(tee_fd, buffer, size);
   return orig_write(fd, buffer, size);
}

static int puts_wrapper(const char *str)
{
   typeof(&fputs) orig_fputs = gotcha_get_wrappee(orig_fputs_handle);
   typeof(&puts) orig_puts = gotcha_get_wrappee(orig_puts_handle);
   if (tee_FILE) {
      orig_fputs(str, tee_FILE);
      orig_fputs("\n", tee_FILE);
   }
   return orig_puts(str);
}

static int fputs_wrapper(const char *str, FILE *f)
{
   typeof(&fputs) orig_fputs = gotcha_get_wrappee(orig_fputs_handle);
   if (f != stdout)
      return orig_fputs(str, f);
   if (tee_FILE)
      orig_fputs(str, tee_FILE);
   return orig_fputs(str, f);
}

static int fwrite_wrapper(const void *ptr, size_t size, size_t nmemb, FILE *stream)
{
   typeof(&fwrite) orig_fwrite = gotcha_get_wrappee(orig_fwrite_handle);
   if (stream != stdout) 
      return orig_fwrite(ptr, size, nmemb, stream);
   if (tee_FILE)
      orig_fwrite(ptr, size, nmemb, tee_FILE);
   return orig_fwrite(ptr, size, nmemb, stream);
}
