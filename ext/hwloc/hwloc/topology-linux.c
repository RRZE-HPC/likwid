/*
 * Copyright © 2009 CNRS
 * Copyright © 2009-2019 Inria.  All rights reserved.
 * Copyright © 2009-2013, 2015 Université Bordeaux
 * Copyright © 2009-2018 Cisco Systems, Inc.  All rights reserved.
 * Copyright © 2015 Intel, Inc.  All rights reserved.
 * Copyright © 2010 IBM
 * See COPYING in top-level directory.
 */

#include "private/autogen/config.h"
#include "hwloc.h"
#include "hwloc/linux.h"
#include "private/misc.h"
#include "private/private.h"
#include "private/misc.h"
#include "private/debug.h"

#include <limits.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <assert.h>
#ifdef HAVE_DIRENT_H
#include <dirent.h>
#endif
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HWLOC_HAVE_LIBUDEV
#include <libudev.h>
#endif
#include <sys/types.h>
#include <sys/stat.h>
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <mntent.h>

struct hwloc_linux_backend_data_s {
  char *root_path; /* NULL if unused */
  int root_fd; /* The file descriptor for the file system root, used when browsing, e.g., Linux' sysfs and procfs. */
  int is_real_fsroot; /* Boolean saying whether root_fd points to the real filesystem root of the system */
#ifdef HWLOC_HAVE_LIBUDEV
  struct udev *udev; /* Global udev context */
#endif
  char *dumped_hwdata_dirname;
  enum {
    HWLOC_LINUX_ARCH_X86, /* x86 32 or 64bits, including k1om (KNC) */
    HWLOC_LINUX_ARCH_IA64,
    HWLOC_LINUX_ARCH_ARM,
    HWLOC_LINUX_ARCH_POWER,
    HWLOC_LINUX_ARCH_S390,
    HWLOC_LINUX_ARCH_UNKNOWN
  } arch;
  int is_knl;
  int is_amd_with_CU;
  int use_dt;
  int use_numa_distances;
  int use_numa_distances_for_cpuless;
  int use_numa_initiators;
  struct utsname utsname; /* fields contain \0 when unknown */
  int fallback_nbprocessors; /* only used in hwloc_linux_fallback_pu_level(), maybe be <= 0 (error) earlier */
  unsigned pagesize;
};



/***************************
 * Misc Abstraction layers *
 ***************************/

#if !(defined HWLOC_HAVE_SCHED_SETAFFINITY) && (defined HWLOC_HAVE_SYSCALL)
/* libc doesn't have support for sched_setaffinity, make system call
 * ourselves: */
#    ifndef __NR_sched_setaffinity
#       ifdef __i386__
#         define __NR_sched_setaffinity 241
#       elif defined(__x86_64__)
#         define __NR_sched_setaffinity 203
#       elif defined(__ia64__)
#         define __NR_sched_setaffinity 1231
#       elif defined(__hppa__)
#         define __NR_sched_setaffinity 211
#       elif defined(__alpha__)
#         define __NR_sched_setaffinity 395
#       elif defined(__s390__)
#         define __NR_sched_setaffinity 239
#       elif defined(__sparc__)
#         define __NR_sched_setaffinity 261
#       elif defined(__m68k__)
#         define __NR_sched_setaffinity 311
#       elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#         define __NR_sched_setaffinity 222
#       elif defined(__aarch64__)
#         define __NR_sched_setaffinity 122
#       elif defined(__arm__)
#         define __NR_sched_setaffinity 241
#       elif defined(__cris__)
#         define __NR_sched_setaffinity 241
/*#       elif defined(__mips__)
  #         define __NR_sched_setaffinity TODO (32/64/nabi) */
#       else
#         warning "don't know the syscall number for sched_setaffinity on this architecture, will not support binding"
#         define sched_setaffinity(pid, lg, mask) (errno = ENOSYS, -1)
#       endif
#    endif
#    ifndef sched_setaffinity
#      define sched_setaffinity(pid, lg, mask) syscall(__NR_sched_setaffinity, pid, lg, mask)
#    endif
#    ifndef __NR_sched_getaffinity
#       ifdef __i386__
#         define __NR_sched_getaffinity 242
#       elif defined(__x86_64__)
#         define __NR_sched_getaffinity 204
#       elif defined(__ia64__)
#         define __NR_sched_getaffinity 1232
#       elif defined(__hppa__)
#         define __NR_sched_getaffinity 212
#       elif defined(__alpha__)
#         define __NR_sched_getaffinity 396
#       elif defined(__s390__)
#         define __NR_sched_getaffinity 240
#       elif defined(__sparc__)
#         define __NR_sched_getaffinity 260
#       elif defined(__m68k__)
#         define __NR_sched_getaffinity 312
#       elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#         define __NR_sched_getaffinity 223
#       elif defined(__aarch64__)
#         define __NR_sched_getaffinity 123
#       elif defined(__arm__)
#         define __NR_sched_getaffinity 242
#       elif defined(__cris__)
#         define __NR_sched_getaffinity 242
/*#       elif defined(__mips__)
  #         define __NR_sched_getaffinity TODO (32/64/nabi) */
#       else
#         warning "don't know the syscall number for sched_getaffinity on this architecture, will not support getting binding"
#         define sched_getaffinity(pid, lg, mask) (errno = ENOSYS, -1)
#       endif
#    endif
#    ifndef sched_getaffinity
#      define sched_getaffinity(pid, lg, mask) (syscall(__NR_sched_getaffinity, pid, lg, mask) < 0 ? -1 : 0)
#    endif
#endif

/* numa syscalls are only in libnuma, but libnuma devel headers aren't widely installed.
 * just redefine these syscalls to avoid requiring libnuma devel headers just because of these missing syscalls.
 * __NR_foo should be defined in headers in all modern platforms.
 * Just redefine the basic ones on important platform when not to hard to detect/define.
 */

#ifndef MPOL_DEFAULT
# define MPOL_DEFAULT 0
#endif
#ifndef MPOL_PREFERRED
# define MPOL_PREFERRED 1
#endif
#ifndef MPOL_BIND
# define MPOL_BIND 2
#endif
#ifndef MPOL_INTERLEAVE
# define MPOL_INTERLEAVE 3
#endif
#ifndef MPOL_LOCAL
# define MPOL_LOCAL 4
#endif
#ifndef MPOL_F_ADDR
# define  MPOL_F_ADDR (1<<1)
#endif
#ifndef MPOL_MF_STRICT
# define MPOL_MF_STRICT (1<<0)
#endif
#ifndef MPOL_MF_MOVE
# define MPOL_MF_MOVE (1<<1)
#endif

#ifndef __NR_mbind
# ifdef __i386__
#  define __NR_mbind 274
# elif defined(__x86_64__)
#  define __NR_mbind 237
# elif defined(__ia64__)
#  define __NR_mbind 1259
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_mbind 259
# elif defined(__sparc__)
#  define __NR_mbind 353
# elif defined(__aarch64__)
#  define __NR_mbind 235
# elif defined(__arm__)
#  define __NR_mbind 319
# endif
#endif
static __hwloc_inline long hwloc_mbind(void *addr __hwloc_attribute_unused,
				       unsigned long len __hwloc_attribute_unused,
				       int mode __hwloc_attribute_unused,
				       const unsigned long *nodemask __hwloc_attribute_unused,
				       unsigned long maxnode __hwloc_attribute_unused,
				       unsigned flags __hwloc_attribute_unused)
{
#if (defined __NR_mbind) && (defined HWLOC_HAVE_SYSCALL)
  return syscall(__NR_mbind, (long) addr, len, mode, (long)nodemask, maxnode, flags);
#else
#warning Couldn't find __NR_mbind syscall number, memory binding won't be supported
  errno = ENOSYS;
  return -1;
#endif
}

#ifndef __NR_set_mempolicy
# ifdef __i386__
#  define __NR_set_mempolicy 276
# elif defined(__x86_64__)
#  define __NR_set_mempolicy 239
# elif defined(__ia64__)
#  define __NR_set_mempolicy 1261
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_set_mempolicy 261
# elif defined(__sparc__)
#  define __NR_set_mempolicy 305
# elif defined(__aarch64__)
#  define __NR_set_mempolicy 237
# elif defined(__arm__)
#  define __NR_set_mempolicy 321
# endif
#endif
static __hwloc_inline long hwloc_set_mempolicy(int mode __hwloc_attribute_unused,
					       const unsigned long *nodemask __hwloc_attribute_unused,
					       unsigned long maxnode __hwloc_attribute_unused)
{
#if (defined __NR_set_mempolicy) && (defined HWLOC_HAVE_SYSCALL)
  return syscall(__NR_set_mempolicy, mode, nodemask, maxnode);
#else
#warning Couldn't find __NR_set_mempolicy syscall number, memory binding won't be supported
  errno = ENOSYS;
  return -1;
#endif
}

#ifndef __NR_get_mempolicy
# ifdef __i386__
#  define __NR_get_mempolicy 275
# elif defined(__x86_64__)
#  define __NR_get_mempolicy 238
# elif defined(__ia64__)
#  define __NR_get_mempolicy 1260
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_get_mempolicy 260
# elif defined(__sparc__)
#  define __NR_get_mempolicy 304
# elif defined(__aarch64__)
#  define __NR_get_mempolicy 236
# elif defined(__arm__)
#  define __NR_get_mempolicy 320
# endif
#endif
static __hwloc_inline long hwloc_get_mempolicy(int *mode __hwloc_attribute_unused,
					       const unsigned long *nodemask __hwloc_attribute_unused,
					       unsigned long maxnode __hwloc_attribute_unused,
					       void *addr __hwloc_attribute_unused,
					       int flags __hwloc_attribute_unused)
{
#if (defined __NR_get_mempolicy) && (defined HWLOC_HAVE_SYSCALL)
  return syscall(__NR_get_mempolicy, mode, nodemask, maxnode, addr, flags);
#else
#warning Couldn't find __NR_get_mempolicy syscall number, memory binding won't be supported
  errno = ENOSYS;
  return -1;
#endif
}

#ifndef __NR_migrate_pages
# ifdef __i386__
#  define __NR_migrate_pages 204
# elif defined(__x86_64__)
#  define __NR_migrate_pages 256
# elif defined(__ia64__)
#  define __NR_migrate_pages 1280
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_migrate_pages 258
# elif defined(__sparc__)
#  define __NR_migrate_pages 302
# elif defined(__aarch64__)
#  define __NR_migrate_pages 238
# elif defined(__arm__)
#  define __NR_migrate_pages 400
# endif
#endif
static __hwloc_inline long hwloc_migrate_pages(int pid __hwloc_attribute_unused,
					       unsigned long maxnode __hwloc_attribute_unused,
					       const unsigned long *oldnodes __hwloc_attribute_unused,
					       const unsigned long *newnodes __hwloc_attribute_unused)
{
#if (defined __NR_migrate_pages) && (defined HWLOC_HAVE_SYSCALL)
  return syscall(__NR_migrate_pages, pid, maxnode, oldnodes, newnodes);
#else
#warning Couldn't find __NR_migrate_pages syscall number, memory migration won't be supported
  errno = ENOSYS;
  return -1;
#endif
}

#ifndef __NR_move_pages
# ifdef __i386__
#  define __NR_move_pages 317
# elif defined(__x86_64__)
#  define __NR_move_pages 279
# elif defined(__ia64__)
#  define __NR_move_pages 1276
# elif defined(__powerpc__) || defined(__ppc__) || defined(__PPC__) || defined(__powerpc64__) || defined(__ppc64__)
#  define __NR_move_pages 301
# elif defined(__sparc__)
#  define __NR_move_pages 307
# elif defined(__aarch64__)
#  define __NR_move_pages 239
# elif defined(__arm__)
#  define __NR_move_pages 344
# endif
#endif
static __hwloc_inline long hwloc_move_pages(int pid __hwloc_attribute_unused,
					    unsigned long count __hwloc_attribute_unused,
					    void **pages __hwloc_attribute_unused,
					    const int *nodes __hwloc_attribute_unused,
					    int *status __hwloc_attribute_unused,
					    int flags __hwloc_attribute_unused)
{
#if (defined __NR_move_pages) && (defined HWLOC_HAVE_SYSCALL)
  return syscall(__NR_move_pages, pid, count, pages, nodes, status, flags);
#else
#warning Couldn't find __NR_move_pages syscall number, getting memory location won't be supported
  errno = ENOSYS;
  return -1;
#endif
}


/* Added for ntohl() */
#include <arpa/inet.h>

#ifdef HAVE_OPENAT
/* Use our own filesystem functions if we have openat */

static const char *
hwloc_checkat(const char *path, int fsroot_fd)
{
  const char *relative_path = path;

  if (fsroot_fd >= 0)
    /* Skip leading slashes.  */
    for (; *relative_path == '/'; relative_path++);

  return relative_path;
}

static int
hwloc_openat(const char *path, int fsroot_fd)
{
  const char *relative_path;

  relative_path = hwloc_checkat(path, fsroot_fd);
  if (!relative_path)
    return -1;

  return openat (fsroot_fd, relative_path, O_RDONLY);
}

static FILE *
hwloc_fopenat(const char *path, const char *mode, int fsroot_fd)
{
  int fd;

  if (strcmp(mode, "r")) {
    errno = ENOTSUP;
    return NULL;
  }

  fd = hwloc_openat (path, fsroot_fd);
  if (fd == -1)
    return NULL;

  return fdopen(fd, mode);
}

static int
hwloc_accessat(const char *path, int mode, int fsroot_fd)
{
  const char *relative_path;

  relative_path = hwloc_checkat(path, fsroot_fd);
  if (!relative_path)
    return -1;

  return faccessat(fsroot_fd, relative_path, mode, 0);
}

static int
hwloc_fstatat(const char *path, struct stat *st, int flags, int fsroot_fd)
{
  const char *relative_path;

  relative_path = hwloc_checkat(path, fsroot_fd);
  if (!relative_path)
    return -1;

  return fstatat(fsroot_fd, relative_path, st, flags);
}

static DIR*
hwloc_opendirat(const char *path, int fsroot_fd)
{
  int dir_fd;
  const char *relative_path;

  relative_path = hwloc_checkat(path, fsroot_fd);
  if (!relative_path)
    return NULL;

  dir_fd = openat(fsroot_fd, relative_path, O_RDONLY | O_DIRECTORY);
  if (dir_fd < 0)
    return NULL;

  return fdopendir(dir_fd);
}

static int
hwloc_readlinkat(const char *path, char *buf, size_t buflen, int fsroot_fd)
{
  const char *relative_path;

  relative_path = hwloc_checkat(path, fsroot_fd);
  if (!relative_path)
    return -1;

  return readlinkat(fsroot_fd, relative_path, buf, buflen);
}

#endif /* HAVE_OPENAT */

/* Static inline version of fopen so that we can use openat if we have
   it, but still preserve compiler parameter checking */
static __hwloc_inline int
hwloc_open(const char *p, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_openat(p, d);
#else
    return open(p, O_RDONLY);
#endif
}

static __hwloc_inline FILE *
hwloc_fopen(const char *p, const char *m, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_fopenat(p, m, d);
#else
    return fopen(p, m);
#endif
}

/* Static inline version of access so that we can use openat if we have
   it, but still preserve compiler parameter checking */
static __hwloc_inline int
hwloc_access(const char *p, int m, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_accessat(p, m, d);
#else
    return access(p, m);
#endif
}

static __hwloc_inline int
hwloc_stat(const char *p, struct stat *st, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_fstatat(p, st, 0, d);
#else
    return stat(p, st);
#endif
}

static __hwloc_inline int
hwloc_lstat(const char *p, struct stat *st, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_fstatat(p, st, AT_SYMLINK_NOFOLLOW, d);
#else
    return lstat(p, st);
#endif
}

/* Static inline version of opendir so that we can use openat if we have
   it, but still preserve compiler parameter checking */
static __hwloc_inline DIR *
hwloc_opendir(const char *p, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
    return hwloc_opendirat(p, d);
#else
    return opendir(p);
#endif
}

static __hwloc_inline int
hwloc_readlink(const char *p, char *l, size_t ll, int d __hwloc_attribute_unused)
{
#ifdef HAVE_OPENAT
  return hwloc_readlinkat(p, l, ll, d);
#else
  return readlink(p, l, ll);
#endif
}


/*****************************************
 ******* Helpers for reading files *******
 *****************************************/

static __hwloc_inline int
hwloc_read_path_by_length(const char *path, char *string, size_t length, int fsroot_fd)
{
  int fd, ret;

  fd = hwloc_open(path, fsroot_fd);
  if (fd < 0)
    return -1;

  ret = read(fd, string, length-1); /* read -1 to put the ending \0 */
  close(fd);

  if (ret <= 0)
    return -1;

  string[ret] = 0;

  return 0;
}

static __hwloc_inline int
hwloc_read_path_as_int(const char *path, int *value, int fsroot_fd)
{
  char string[11];
  if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
    return -1;
  *value = atoi(string);
  return 0;
}

static __hwloc_inline int
hwloc_read_path_as_uint(const char *path, unsigned *value, int fsroot_fd)
{
  char string[11];
  if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
    return -1;
  *value = (unsigned) strtoul(string, NULL, 10);
  return 0;
}

static __hwloc_inline int
hwloc_read_path_as_uint64(const char *path, uint64_t *value, int fsroot_fd)
{
  char string[22];
  if (hwloc_read_path_by_length(path, string, sizeof(string), fsroot_fd) < 0)
    return -1;
  *value = (uint64_t) strtoull(string, NULL, 10);
  return 0;
}

/* Read everything from fd and save it into a newly allocated buffer
 * returned in bufferp. Use sizep as a default buffer size, and returned
 * the actually needed size in sizep.
 */
static __hwloc_inline int
hwloc__read_fd(int fd, char **bufferp, size_t *sizep)
{
  char *buffer;
  size_t toread, filesize, totalread;
  ssize_t ret;

  toread = filesize = *sizep;

  /* Alloc and read +1 so that we get EOF on 2^n without reading once more */
  buffer = malloc(filesize+1);
  if (!buffer)
    return -1;

  ret = read(fd, buffer, toread+1);
  if (ret < 0) {
    free(buffer);
    return -1;
  }

  totalread = (size_t) ret;

  if (totalread < toread + 1)
    /* Normal case, a single read got EOF */
    goto done;

  /* Unexpected case, must extend the buffer and read again.
   * Only occurs on first invocation and if the kernel ever uses multiple page for a single mask.
   */
  do {
    char *tmp;

    toread = filesize;
    filesize *= 2;

    tmp = realloc(buffer, filesize+1);
    if (!tmp) {
      free(buffer);
      return -1;
    }
    buffer = tmp;

    ret = read(fd, buffer+toread+1, toread);
    if (ret < 0) {
      free(buffer);
      return -1;
    }

    totalread += ret;
  } while ((size_t) ret == toread);

 done:
  buffer[totalread] = '\0';
  *bufferp = buffer;
  *sizep = filesize;
  return 0;
}

/* kernel cpumaps are composed of an array of 32bits cpumasks */
#define KERNEL_CPU_MASK_BITS 32
#define KERNEL_CPU_MAP_LEN (KERNEL_CPU_MASK_BITS/4+2)

static __hwloc_inline int
hwloc__read_fd_as_cpumask(int fd, hwloc_bitmap_t set)
{
  static size_t _filesize = 0; /* will be dynamically initialized to hwloc_get_pagesize(), and increased later if needed */
  size_t filesize;
  unsigned long *maps;
  unsigned long map;
  int nr_maps = 0;
  static int _nr_maps_allocated = 8; /* Only compute the power-of-two above the kernel cpumask size once.
				      * Actually, it may increase multiple times if first read cpumaps start with zeroes.
				      */
  int nr_maps_allocated = _nr_maps_allocated;
  char *buffer, *tmpbuf;
  int i;

  /* Kernel sysfs files are usually at most one page. 4kB may contain 455 32-bit
   * masks (followed by comma), enough for 14k PUs. So allocate a page by default for now.
   *
   * If we ever need a larger buffer, we'll realloc() the buffer during the first
   * invocation of this function so that others directly allocate the right size
   * (all cpumask files have the exact same size).
   */
  filesize = _filesize;
  if (!filesize)
    filesize = hwloc_getpagesize();
  if (hwloc__read_fd(fd, &buffer, &filesize) < 0)
    return -1;
  /* Only update the static value with the final one,
   * to avoid sharing intermediate values that we modify,
   * in case there's ever multiple concurrent calls.
   */
  _filesize = filesize;

  maps = malloc(nr_maps_allocated * sizeof(*maps));
  if (!maps) {
    free(buffer);
    return -1;
  }

  /* reset to zero first */
  hwloc_bitmap_zero(set);

  /* parse the whole mask */
  tmpbuf = buffer;
  while (sscanf(tmpbuf, "%lx", &map) == 1) {
    /* read one kernel cpu mask and the ending comma */
    if (nr_maps == nr_maps_allocated) {
      unsigned long *tmp = realloc(maps, 2*nr_maps_allocated * sizeof(*maps));
      if (!tmp) {
	free(buffer);
	free(maps);
	return -1;
      }
      maps = tmp;
      nr_maps_allocated *= 2;
    }

    tmpbuf = strchr(tmpbuf, ',');
    if (!tmpbuf) {
      maps[nr_maps++] = map;
      break;
    } else
      tmpbuf++;

    if (!map && !nr_maps)
      /* ignore the first map if it's empty */
      continue;

    maps[nr_maps++] = map;
  }

  free(buffer);

  /* convert into a set */
#if KERNEL_CPU_MASK_BITS == HWLOC_BITS_PER_LONG
  for(i=0; i<nr_maps; i++)
    hwloc_bitmap_set_ith_ulong(set, i, maps[nr_maps-1-i]);
#else
  for(i=0; i<(nr_maps+1)/2; i++) {
    unsigned long mask;
    mask = maps[nr_maps-2*i-1];
    if (2*i+1<nr_maps)
      mask |= maps[nr_maps-2*i-2] << KERNEL_CPU_MASK_BITS;
    hwloc_bitmap_set_ith_ulong(set, i, mask);
  }
#endif

  free(maps);

  /* Only update the static value with the final one,
   * to avoid sharing intermediate values that we modify,
   * in case there's ever multiple concurrent calls.
   */
  if (nr_maps_allocated > _nr_maps_allocated)
    _nr_maps_allocated = nr_maps_allocated;
  return 0;
}

static __hwloc_inline int
hwloc__read_path_as_cpumask(const char *maskpath, hwloc_bitmap_t set, int fsroot_fd)
{
  int fd, err;
  fd = hwloc_open(maskpath, fsroot_fd);
  if (fd < 0)
    return -1;
  err = hwloc__read_fd_as_cpumask(fd, set);
  close(fd);
  return err;
}

static __hwloc_inline hwloc_bitmap_t
hwloc__alloc_read_path_as_cpumask(const char *maskpath, int fsroot_fd)
{
  hwloc_bitmap_t set;
  int err;
  set = hwloc_bitmap_alloc();
  if (!set)
    return NULL;
  err = hwloc__read_path_as_cpumask(maskpath, set, fsroot_fd);
  if (err < 0) {
    hwloc_bitmap_free(set);
    return NULL;
  } else
    return set;
}

int
hwloc_linux_read_path_as_cpumask(const char *maskpath, hwloc_bitmap_t set)
{
  int fd, err;
  fd = open(maskpath, O_RDONLY);
  if (fd < 0)
    return -1;
  err = hwloc__read_fd_as_cpumask(fd, set);
  close(fd);
  return err;
}

/* on failure, the content of set is undefined */
static __hwloc_inline int
hwloc__read_fd_as_cpulist(int fd, hwloc_bitmap_t set)
{
  /* Kernel sysfs files are usually at most one page.
   * But cpulists can be of very different sizes depending on the fragmentation,
   * so don't bother remember the actual read size between invocations.
   * We don't have many invocations anyway.
   */
  size_t filesize = hwloc_getpagesize();
  char *buffer, *current, *comma, *tmp;
  int prevlast, nextfirst, nextlast; /* beginning/end of enabled-segments */

  if (hwloc__read_fd(fd, &buffer, &filesize) < 0)
    return -1;

  hwloc_bitmap_fill(set);

  current = buffer;
  prevlast = -1;

  while (1) {
    /* save a pointer to the next comma and erase it to simplify things */
    comma = strchr(current, ',');
    if (comma)
      *comma = '\0';

    /* find current enabled-segment bounds */
    nextfirst = strtoul(current, &tmp, 0);
    if (*tmp == '-')
      nextlast = strtoul(tmp+1, NULL, 0);
    else
      nextlast = nextfirst;
    if (prevlast+1 <= nextfirst-1)
      hwloc_bitmap_clr_range(set, prevlast+1, nextfirst-1);

    /* switch to next enabled-segment */
    prevlast = nextlast;
    if (!comma)
      break;
    current = comma+1;
  }

  hwloc_bitmap_clr_range(set, prevlast+1, -1);
  free(buffer);
  return 0;
}

/* on failure, the content of set is undefined */
static __hwloc_inline int
hwloc__read_path_as_cpulist(const char *maskpath, hwloc_bitmap_t set, int fsroot_fd)
{
  int fd, err;
  fd = hwloc_open(maskpath, fsroot_fd);
  if (fd < 0)
    return -1;
  err = hwloc__read_fd_as_cpulist(fd, set);
  close(fd);
  return err;
}

/* on failure, the content of set is undefined */
static __hwloc_inline hwloc_bitmap_t
hwloc__alloc_read_path_as_cpulist(const char *maskpath, int fsroot_fd)
{
  hwloc_bitmap_t set;
  int err;
  set = hwloc_bitmap_alloc_full();
  if (!set)
    return NULL;
  err = hwloc__read_path_as_cpulist(maskpath, set, fsroot_fd);
  if (err < 0) {
    hwloc_bitmap_free(set);
    return NULL;
  } else
    return set;
}


/*****************************
 ******* CpuBind Hooks *******
 *****************************/

int
hwloc_linux_set_tid_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid __hwloc_attribute_unused, hwloc_const_bitmap_t hwloc_set __hwloc_attribute_unused)
{
  /* The resulting binding is always strict */

#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
  cpu_set_t *plinux_set;
  unsigned cpu;
  int last;
  size_t setsize;
  int err;

  last = hwloc_bitmap_last(hwloc_set);
  if (last == -1) {
    errno = EINVAL;
    return -1;
  }

  setsize = CPU_ALLOC_SIZE(last+1);
  plinux_set = CPU_ALLOC(last+1);

  CPU_ZERO_S(setsize, plinux_set);
  hwloc_bitmap_foreach_begin(cpu, hwloc_set)
    CPU_SET_S(cpu, setsize, plinux_set);
  hwloc_bitmap_foreach_end();

  err = sched_setaffinity(tid, setsize, plinux_set);

  CPU_FREE(plinux_set);
  return err;
#elif defined(HWLOC_HAVE_CPU_SET)
  cpu_set_t linux_set;
  unsigned cpu;

  CPU_ZERO(&linux_set);
  hwloc_bitmap_foreach_begin(cpu, hwloc_set)
    CPU_SET(cpu, &linux_set);
  hwloc_bitmap_foreach_end();

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
  return sched_setaffinity(tid, &linux_set);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  return sched_setaffinity(tid, sizeof(linux_set), &linux_set);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
#elif defined(HWLOC_HAVE_SYSCALL)
  unsigned long mask = hwloc_bitmap_to_ulong(hwloc_set);

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
  return sched_setaffinity(tid, (void*) &mask);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  return sched_setaffinity(tid, sizeof(mask), (void*) &mask);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
#else /* !SYSCALL */
  errno = ENOSYS;
  return -1;
#endif /* !SYSCALL */
}

#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
/*
 * On some kernels, sched_getaffinity requires the output size to be larger
 * than the kernel cpu_set size (defined by CONFIG_NR_CPUS).
 * Try sched_affinity on ourself until we find a nr_cpus value that makes
 * the kernel happy.
 */
static int
hwloc_linux_find_kernel_nr_cpus(hwloc_topology_t topology)
{
  static int _nr_cpus = -1;
  int nr_cpus = _nr_cpus;
  int fd;

  if (nr_cpus != -1)
    /* already computed */
    return nr_cpus;

  if (topology->levels[0][0]->complete_cpuset)
    /* start with a nr_cpus that may contain the whole topology */
    nr_cpus = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset) + 1;
  if (nr_cpus <= 0)
    /* start from scratch, the topology isn't ready yet (complete_cpuset is missing (-1) or empty (0))*/
    nr_cpus = 1;

  /* reading /sys/devices/system/cpu/kernel_max would be easier (single value to parse instead of a list),
   * but its value may be way too large (5119 on CentOS7).
   * /sys/devices/system/cpu/possible is better because it matches the current hardware.
   */

  fd = open("/sys/devices/system/cpu/possible", O_RDONLY); /* binding only supported in real fsroot, no need for data->root_fd */
  if (fd >= 0) {
    hwloc_bitmap_t possible_bitmap = hwloc_bitmap_alloc();
    if (hwloc__read_fd_as_cpulist(fd, possible_bitmap) == 0) {
      int max_possible = hwloc_bitmap_last(possible_bitmap);
      hwloc_debug_bitmap("possible CPUs are %s\n", possible_bitmap);

      if (nr_cpus < max_possible + 1)
        nr_cpus = max_possible + 1;
    }
    close(fd);
    hwloc_bitmap_free(possible_bitmap);
  }

  while (1) {
    cpu_set_t *set = CPU_ALLOC(nr_cpus);
    size_t setsize = CPU_ALLOC_SIZE(nr_cpus);
    int err = sched_getaffinity(0, setsize, set); /* always works, unless setsize is too small */
    CPU_FREE(set);
    nr_cpus = setsize * 8; /* that's the value that was actually tested */
    if (!err)
      /* Found it. Only update the static value with the final one,
       * to avoid sharing intermediate values that we modify,
       * in case there's ever multiple concurrent calls.
       */
      return _nr_cpus = nr_cpus;
    nr_cpus *= 2;
  }
}
#endif

int
hwloc_linux_get_tid_cpubind(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid __hwloc_attribute_unused, hwloc_bitmap_t hwloc_set __hwloc_attribute_unused)
{
  int err __hwloc_attribute_unused;

#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
  cpu_set_t *plinux_set;
  unsigned cpu;
  int last;
  size_t setsize;
  int kernel_nr_cpus;

  /* find the kernel nr_cpus so as to use a large enough cpu_set size */
  kernel_nr_cpus = hwloc_linux_find_kernel_nr_cpus(topology);
  setsize = CPU_ALLOC_SIZE(kernel_nr_cpus);
  plinux_set = CPU_ALLOC(kernel_nr_cpus);

  err = sched_getaffinity(tid, setsize, plinux_set);

  if (err < 0) {
    CPU_FREE(plinux_set);
    return -1;
  }

  last = -1;
  if (topology->levels[0][0]->complete_cpuset)
    last = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset);
  if (last == -1)
    /* round the maximal support number, the topology isn't ready yet (complete_cpuset is missing or empty)*/
    last = kernel_nr_cpus-1;

  hwloc_bitmap_zero(hwloc_set);
  for(cpu=0; cpu<=(unsigned) last; cpu++)
    if (CPU_ISSET_S(cpu, setsize, plinux_set))
      hwloc_bitmap_set(hwloc_set, cpu);

  CPU_FREE(plinux_set);
#elif defined(HWLOC_HAVE_CPU_SET)
  cpu_set_t linux_set;
  unsigned cpu;

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
  err = sched_getaffinity(tid, &linux_set);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  err = sched_getaffinity(tid, sizeof(linux_set), &linux_set);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  if (err < 0)
    return -1;

  hwloc_bitmap_zero(hwloc_set);
  for(cpu=0; cpu<CPU_SETSIZE; cpu++)
    if (CPU_ISSET(cpu, &linux_set))
      hwloc_bitmap_set(hwloc_set, cpu);
#elif defined(HWLOC_HAVE_SYSCALL)
  unsigned long mask;

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
  err = sched_getaffinity(tid, (void*) &mask);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  err = sched_getaffinity(tid, sizeof(mask), (void*) &mask);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  if (err < 0)
    return -1;

  hwloc_bitmap_from_ulong(hwloc_set, mask);
#else /* !SYSCALL */
  errno = ENOSYS;
  return -1;
#endif /* !SYSCALL */

  return 0;
}

/* Get the array of tids of a process from the task directory in /proc */
static int
hwloc_linux_get_proc_tids(DIR *taskdir, unsigned *nr_tidsp, pid_t ** tidsp)
{
  struct dirent *dirent;
  unsigned nr_tids = 0;
  unsigned max_tids = 32;
  pid_t *tids;
  struct stat sb;

  /* take the number of links as a good estimate for the number of tids */
  if (fstat(dirfd(taskdir), &sb) == 0)
    max_tids = sb.st_nlink;

  tids = malloc(max_tids*sizeof(pid_t));
  if (!tids) {
    errno = ENOMEM;
    return -1;
  }

  rewinddir(taskdir);

  while ((dirent = readdir(taskdir)) != NULL) {
    if (nr_tids == max_tids) {
      pid_t *newtids;
      max_tids += 8;
      newtids = realloc(tids, max_tids*sizeof(pid_t));
      if (!newtids) {
        free(tids);
        errno = ENOMEM;
        return -1;
      }
      tids = newtids;
    }
    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;
    tids[nr_tids++] = atoi(dirent->d_name);
  }

  *nr_tidsp = nr_tids;
  *tidsp = tids;
  return 0;
}

/* Per-tid callbacks */
typedef int (*hwloc_linux_foreach_proc_tid_cb_t)(hwloc_topology_t topology, pid_t tid, void *data, int idx);

static int
hwloc_linux_foreach_proc_tid(hwloc_topology_t topology,
			     pid_t pid, hwloc_linux_foreach_proc_tid_cb_t cb,
			     void *data)
{
  char taskdir_path[128];
  DIR *taskdir;
  pid_t *tids, *newtids;
  unsigned i, nr, newnr, failed = 0, failed_errno = 0;
  unsigned retrynr = 0;
  int err;

  if (pid)
    snprintf(taskdir_path, sizeof(taskdir_path), "/proc/%u/task", (unsigned) pid);
  else
    snprintf(taskdir_path, sizeof(taskdir_path), "/proc/self/task");

  taskdir = opendir(taskdir_path);
  if (!taskdir) {
    if (errno == ENOENT)
      errno = EINVAL;
    err = -1;
    goto out;
  }

  /* read the current list of threads */
  err = hwloc_linux_get_proc_tids(taskdir, &nr, &tids);
  if (err < 0)
    goto out_with_dir;

 retry:
  /* apply the callback to all threads */
  failed=0;
  for(i=0; i<nr; i++) {
    err = cb(topology, tids[i], data, i);
    if (err < 0) {
      failed++;
      failed_errno = errno;
    }
  }

  /* re-read the list of thread */
  err = hwloc_linux_get_proc_tids(taskdir, &newnr, &newtids);
  if (err < 0)
    goto out_with_tids;
  /* retry if the list changed in the meantime, or we failed for *some* threads only.
   * if we're really unlucky, all threads changed but we got the same set of tids. no way to support this.
   */
  if (newnr != nr || memcmp(newtids, tids, nr*sizeof(pid_t)) || (failed && failed != nr)) {
    free(tids);
    tids = newtids;
    nr = newnr;
    if (++retrynr > 10) {
      /* we tried 10 times, it didn't work, the application is probably creating/destroying many threads, stop trying */
      errno = EAGAIN;
      err = -1;
      goto out_with_tids;
    }
    goto retry;
  } else {
    free(newtids);
  }

  /* if all threads failed, return the last errno. */
  if (failed) {
    err = -1;
    errno = failed_errno;
    goto out_with_tids;
  }

  err = 0;
 out_with_tids:
  free(tids);
 out_with_dir:
  closedir(taskdir);
 out:
  return err;
}

/* Per-tid proc_set_cpubind callback and caller.
 * Callback data is a hwloc_bitmap_t. */
static int
hwloc_linux_foreach_proc_tid_set_cpubind_cb(hwloc_topology_t topology, pid_t tid, void *data, int idx __hwloc_attribute_unused)
{
  return hwloc_linux_set_tid_cpubind(topology, tid, (hwloc_bitmap_t) data);
}

static int
hwloc_linux_set_pid_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  return hwloc_linux_foreach_proc_tid(topology, pid,
				      hwloc_linux_foreach_proc_tid_set_cpubind_cb,
				      (void*) hwloc_set);
}

/* Per-tid proc_get_cpubind callback data, callback function and caller */
struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s {
  hwloc_bitmap_t cpuset;
  hwloc_bitmap_t tidset;
  int flags;
};

static int
hwloc_linux_foreach_proc_tid_get_cpubind_cb(hwloc_topology_t topology, pid_t tid, void *_data, int idx)
{
  struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s *data = _data;
  hwloc_bitmap_t cpuset = data->cpuset;
  hwloc_bitmap_t tidset = data->tidset;
  int flags = data->flags;

  if (hwloc_linux_get_tid_cpubind(topology, tid, tidset))
    return -1;

  /* reset the cpuset on first iteration */
  if (!idx)
    hwloc_bitmap_zero(cpuset);

  if (flags & HWLOC_CPUBIND_STRICT) {
    /* if STRICT, we want all threads to have the same binding */
    if (!idx) {
      /* this is the first thread, copy its binding */
      hwloc_bitmap_copy(cpuset, tidset);
    } else if (!hwloc_bitmap_isequal(cpuset, tidset)) {
      /* this is not the first thread, and it's binding is different */
      errno = EXDEV;
      return -1;
    }
  } else {
    /* if not STRICT, just OR all thread bindings */
    hwloc_bitmap_or(cpuset, cpuset, tidset);
  }
  return 0;
}

static int
hwloc_linux_get_pid_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
  struct hwloc_linux_foreach_proc_tid_get_cpubind_cb_data_s data;
  hwloc_bitmap_t tidset = hwloc_bitmap_alloc();
  int ret;

  data.cpuset = hwloc_set;
  data.tidset = tidset;
  data.flags = flags;
  ret = hwloc_linux_foreach_proc_tid(topology, pid,
				     hwloc_linux_foreach_proc_tid_get_cpubind_cb,
				     (void*) &data);
  hwloc_bitmap_free(tidset);
  return ret;
}

static int
hwloc_linux_set_proc_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_const_bitmap_t hwloc_set, int flags)
{
  if (pid == 0)
    pid = topology->pid;
  if (flags & HWLOC_CPUBIND_THREAD)
    return hwloc_linux_set_tid_cpubind(topology, pid, hwloc_set);
  else
    return hwloc_linux_set_pid_cpubind(topology, pid, hwloc_set, flags);
}

static int
hwloc_linux_get_proc_cpubind(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
  if (pid == 0)
    pid = topology->pid;
  if (flags & HWLOC_CPUBIND_THREAD)
    return hwloc_linux_get_tid_cpubind(topology, pid, hwloc_set);
  else
    return hwloc_linux_get_pid_cpubind(topology, pid, hwloc_set, flags);
}

static int
hwloc_linux_set_thisproc_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags)
{
  return hwloc_linux_set_pid_cpubind(topology, topology->pid, hwloc_set, flags);
}

static int
hwloc_linux_get_thisproc_cpubind(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags)
{
  return hwloc_linux_get_pid_cpubind(topology, topology->pid, hwloc_set, flags);
}

static int
hwloc_linux_set_thisthread_cpubind(hwloc_topology_t topology, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  if (topology->pid) {
    errno = ENOSYS;
    return -1;
  }
  return hwloc_linux_set_tid_cpubind(topology, 0, hwloc_set);
}

static int
hwloc_linux_get_thisthread_cpubind(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  if (topology->pid) {
    errno = ENOSYS;
    return -1;
  }
  return hwloc_linux_get_tid_cpubind(topology, 0, hwloc_set);
}

#if HAVE_DECL_PTHREAD_SETAFFINITY_NP
#pragma weak pthread_setaffinity_np
#pragma weak pthread_self

static int
hwloc_linux_set_thread_cpubind(hwloc_topology_t topology, pthread_t tid, hwloc_const_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  int err;

  if (topology->pid) {
    errno = ENOSYS;
    return -1;
  }

  if (!pthread_self) {
    /* ?! Application uses set_thread_cpubind, but doesn't link against libpthread ?! */
    errno = ENOSYS;
    return -1;
  }
  if (tid == pthread_self())
    return hwloc_linux_set_tid_cpubind(topology, 0, hwloc_set);

  if (!pthread_setaffinity_np) {
    errno = ENOSYS;
    return -1;
  }

#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
  /* Use a separate block so that we can define specific variable
     types here */
  {
     cpu_set_t *plinux_set;
     unsigned cpu;
     int last;
     size_t setsize;

     last = hwloc_bitmap_last(hwloc_set);
     if (last == -1) {
       errno = EINVAL;
       return -1;
     }

     setsize = CPU_ALLOC_SIZE(last+1);
     plinux_set = CPU_ALLOC(last+1);

     CPU_ZERO_S(setsize, plinux_set);
     hwloc_bitmap_foreach_begin(cpu, hwloc_set)
         CPU_SET_S(cpu, setsize, plinux_set);
     hwloc_bitmap_foreach_end();

     err = pthread_setaffinity_np(tid, setsize, plinux_set);

     CPU_FREE(plinux_set);
  }
#elif defined(HWLOC_HAVE_CPU_SET)
  /* Use a separate block so that we can define specific variable
     types here */
  {
     cpu_set_t linux_set;
     unsigned cpu;

     CPU_ZERO(&linux_set);
     hwloc_bitmap_foreach_begin(cpu, hwloc_set)
         CPU_SET(cpu, &linux_set);
     hwloc_bitmap_foreach_end();

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
     err = pthread_setaffinity_np(tid, &linux_set);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
     err = pthread_setaffinity_np(tid, sizeof(linux_set), &linux_set);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  }
#else /* CPU_SET */
  /* Use a separate block so that we can define specific variable
     types here */
  {
      unsigned long mask = hwloc_bitmap_to_ulong(hwloc_set);

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
      err = pthread_setaffinity_np(tid, (void*) &mask);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
      err = pthread_setaffinity_np(tid, sizeof(mask), (void*) &mask);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
  }
#endif /* CPU_SET */

  if (err) {
    errno = err;
    return -1;
  }
  return 0;
}
#endif /* HAVE_DECL_PTHREAD_SETAFFINITY_NP */

#if HAVE_DECL_PTHREAD_GETAFFINITY_NP
#pragma weak pthread_getaffinity_np
#pragma weak pthread_self

static int
hwloc_linux_get_thread_cpubind(hwloc_topology_t topology, pthread_t tid, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  int err;

  if (topology->pid) {
    errno = ENOSYS;
    return -1;
  }

  if (!pthread_self) {
    /* ?! Application uses set_thread_cpubind, but doesn't link against libpthread ?! */
    errno = ENOSYS;
    return -1;
  }
  if (tid == pthread_self())
    return hwloc_linux_get_tid_cpubind(topology, 0, hwloc_set);

  if (!pthread_getaffinity_np) {
    errno = ENOSYS;
    return -1;
  }

#if defined(HWLOC_HAVE_CPU_SET_S) && !defined(HWLOC_HAVE_OLD_SCHED_SETAFFINITY)
  /* Use a separate block so that we can define specific variable
     types here */
  {
     cpu_set_t *plinux_set;
     unsigned cpu;
     int last;
     size_t setsize;

     last = hwloc_bitmap_last(topology->levels[0][0]->complete_cpuset);
     assert (last != -1);

     setsize = CPU_ALLOC_SIZE(last+1);
     plinux_set = CPU_ALLOC(last+1);

     err = pthread_getaffinity_np(tid, setsize, plinux_set);
     if (err) {
        CPU_FREE(plinux_set);
        errno = err;
        return -1;
     }

     hwloc_bitmap_zero(hwloc_set);
     for(cpu=0; cpu<=(unsigned) last; cpu++)
       if (CPU_ISSET_S(cpu, setsize, plinux_set))
	 hwloc_bitmap_set(hwloc_set, cpu);

     CPU_FREE(plinux_set);
  }
#elif defined(HWLOC_HAVE_CPU_SET)
  /* Use a separate block so that we can define specific variable
     types here */
  {
     cpu_set_t linux_set;
     unsigned cpu;

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
     err = pthread_getaffinity_np(tid, &linux_set);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
     err = pthread_getaffinity_np(tid, sizeof(linux_set), &linux_set);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
     if (err) {
        errno = err;
        return -1;
     }

     hwloc_bitmap_zero(hwloc_set);
     for(cpu=0; cpu<CPU_SETSIZE; cpu++)
       if (CPU_ISSET(cpu, &linux_set))
	 hwloc_bitmap_set(hwloc_set, cpu);
  }
#else /* CPU_SET */
  /* Use a separate block so that we can define specific variable
     types here */
  {
      unsigned long mask;

#ifdef HWLOC_HAVE_OLD_SCHED_SETAFFINITY
      err = pthread_getaffinity_np(tid, (void*) &mask);
#else /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
      err = pthread_getaffinity_np(tid, sizeof(mask), (void*) &mask);
#endif /* HWLOC_HAVE_OLD_SCHED_SETAFFINITY */
      if (err) {
        errno = err;
        return -1;
      }

     hwloc_bitmap_from_ulong(hwloc_set, mask);
  }
#endif /* CPU_SET */

  return 0;
}
#endif /* HAVE_DECL_PTHREAD_GETAFFINITY_NP */

int
hwloc_linux_get_tid_last_cpu_location(hwloc_topology_t topology __hwloc_attribute_unused, pid_t tid, hwloc_bitmap_t set)
{
  /* read /proc/pid/stat.
   * its second field contains the command name between parentheses,
   * and the command itself may contain parentheses,
   * so read the whole line and find the last closing parenthesis to find the third field.
   */
  char buf[1024] = "";
  char name[64];
  char *tmp;
  int fd, i, err;

  /* TODO: find a way to use sched_getcpu().
   * either compare tid with gettid() in all callbacks.
   * or pass gettid() in the callback data.
   */

  if (!tid) {
#ifdef SYS_gettid
    tid = syscall(SYS_gettid);
#else
    errno = ENOSYS;
    return -1;
#endif
  }

  snprintf(name, sizeof(name), "/proc/%lu/stat", (unsigned long) tid);
  fd = open(name, O_RDONLY); /* no fsroot for real /proc */
  if (fd < 0) {
    errno = ENOSYS;
    return -1;
  }
  err = read(fd, buf, sizeof(buf)-1); /* read -1 to put the ending \0 */
  close(fd);
  if (err <= 0) {
    errno = ENOSYS;
    return -1;
  }
  buf[err-1] = '\0';

  tmp = strrchr(buf, ')');
  if (!tmp) {
    errno = ENOSYS;
    return -1;
  }
  /* skip ') ' to find the actual third argument */
  tmp += 2;

  /* skip 35 fields */
  for(i=0; i<36; i++) {
    tmp = strchr(tmp, ' ');
    if (!tmp) {
      errno = ENOSYS;
      return -1;
    }
    /* skip the ' ' itself */
    tmp++;
  }

  /* read the last cpu in the 38th field now */
  if (sscanf(tmp, "%d ", &i) != 1) {
    errno = ENOSYS;
    return -1;
  }

  hwloc_bitmap_only(set, i);
  return 0;
}

/* Per-tid proc_get_last_cpu_location callback data, callback function and caller */
struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s {
  hwloc_bitmap_t cpuset;
  hwloc_bitmap_t tidset;
};

static int
hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb(hwloc_topology_t topology, pid_t tid, void *_data, int idx)
{
  struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s *data = _data;
  hwloc_bitmap_t cpuset = data->cpuset;
  hwloc_bitmap_t tidset = data->tidset;

  if (hwloc_linux_get_tid_last_cpu_location(topology, tid, tidset))
    return -1;

  /* reset the cpuset on first iteration */
  if (!idx)
    hwloc_bitmap_zero(cpuset);

  hwloc_bitmap_or(cpuset, cpuset, tidset);
  return 0;
}

static int
hwloc_linux_get_pid_last_cpu_location(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  struct hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb_data_s data;
  hwloc_bitmap_t tidset = hwloc_bitmap_alloc();
  int ret;

  data.cpuset = hwloc_set;
  data.tidset = tidset;
  ret = hwloc_linux_foreach_proc_tid(topology, pid,
				     hwloc_linux_foreach_proc_tid_get_last_cpu_location_cb,
				     &data);
  hwloc_bitmap_free(tidset);
  return ret;
}

static int
hwloc_linux_get_proc_last_cpu_location(hwloc_topology_t topology, pid_t pid, hwloc_bitmap_t hwloc_set, int flags)
{
  if (pid == 0)
    pid = topology->pid;
  if (flags & HWLOC_CPUBIND_THREAD)
    return hwloc_linux_get_tid_last_cpu_location(topology, pid, hwloc_set);
  else
    return hwloc_linux_get_pid_last_cpu_location(topology, pid, hwloc_set, flags);
}

static int
hwloc_linux_get_thisproc_last_cpu_location(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags)
{
  return hwloc_linux_get_pid_last_cpu_location(topology, topology->pid, hwloc_set, flags);
}

static int
hwloc_linux_get_thisthread_last_cpu_location(hwloc_topology_t topology, hwloc_bitmap_t hwloc_set, int flags __hwloc_attribute_unused)
{
  if (topology->pid) {
    errno = ENOSYS;
    return -1;
  }

#if HAVE_DECL_SCHED_GETCPU
  {
    int pu = sched_getcpu();
    if (pu >= 0) {
      hwloc_bitmap_only(hwloc_set, pu);
      return 0;
    }
  }
#endif

  return hwloc_linux_get_tid_last_cpu_location(topology, 0, hwloc_set);
}



/***************************
 ****** Membind hooks ******
 ***************************/

static int
hwloc_linux_membind_policy_from_hwloc(int *linuxpolicy, hwloc_membind_policy_t policy, int flags)
{
  switch (policy) {
  case HWLOC_MEMBIND_DEFAULT:
    *linuxpolicy = MPOL_DEFAULT;
    break;
  case HWLOC_MEMBIND_FIRSTTOUCH:
    *linuxpolicy = MPOL_LOCAL;
    break;
  case HWLOC_MEMBIND_BIND:
    if (flags & HWLOC_MEMBIND_STRICT)
      *linuxpolicy = MPOL_BIND;
    else
      *linuxpolicy = MPOL_PREFERRED;
    break;
  case HWLOC_MEMBIND_INTERLEAVE:
    *linuxpolicy = MPOL_INTERLEAVE;
    break;
  /* TODO: next-touch when (if?) patch applied upstream */
  default:
    errno = ENOSYS;
    return -1;
  }
  return 0;
}

static int
hwloc_linux_membind_mask_from_nodeset(hwloc_topology_t topology __hwloc_attribute_unused,
				      hwloc_const_nodeset_t nodeset,
				      unsigned *max_os_index_p, unsigned long **linuxmaskp)
{
  unsigned max_os_index = 0; /* highest os_index + 1 */
  unsigned long *linuxmask;
  unsigned i;
  hwloc_nodeset_t linux_nodeset = NULL;

  if (hwloc_bitmap_isfull(nodeset)) {
    linux_nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_only(linux_nodeset, 0);
    nodeset = linux_nodeset;
  }

  max_os_index = hwloc_bitmap_last(nodeset);
  if (max_os_index == (unsigned) -1)
    max_os_index = 0;
  /* add 1 to convert the last os_index into a max_os_index,
   * and round up to the nearest multiple of BITS_PER_LONG */
  max_os_index = (max_os_index + 1 + HWLOC_BITS_PER_LONG - 1) & ~(HWLOC_BITS_PER_LONG - 1);

  linuxmask = calloc(max_os_index/HWLOC_BITS_PER_LONG, sizeof(unsigned long));
  if (!linuxmask) {
    hwloc_bitmap_free(linux_nodeset);
    errno = ENOMEM;
    return -1;
  }

  for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
    linuxmask[i] = hwloc_bitmap_to_ith_ulong(nodeset, i);

  if (linux_nodeset)
    hwloc_bitmap_free(linux_nodeset);

  *max_os_index_p = max_os_index;
  *linuxmaskp = linuxmask;
  return 0;
}

static void
hwloc_linux_membind_mask_to_nodeset(hwloc_topology_t topology __hwloc_attribute_unused,
				    hwloc_nodeset_t nodeset,
				    unsigned max_os_index, const unsigned long *linuxmask)
{
  unsigned i;

#ifdef HWLOC_DEBUG
  /* max_os_index comes from hwloc_linux_find_kernel_max_numnodes() so it's a multiple of HWLOC_BITS_PER_LONG */
  assert(!(max_os_index%HWLOC_BITS_PER_LONG));
#endif

  hwloc_bitmap_zero(nodeset);
  for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
    hwloc_bitmap_set_ith_ulong(nodeset, i, linuxmask[i]);
}

static int
hwloc_linux_set_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  unsigned max_os_index; /* highest os_index + 1 */
  unsigned long *linuxmask;
  size_t remainder;
  int linuxpolicy;
  unsigned linuxflags = 0;
  int err;

  remainder = (uintptr_t) addr & (hwloc_getpagesize()-1);
  addr = (char*) addr - remainder;
  len += remainder;

  err = hwloc_linux_membind_policy_from_hwloc(&linuxpolicy, policy, flags);
  if (err < 0)
    return err;

  if (linuxpolicy == MPOL_DEFAULT) {
    /* Some Linux kernels don't like being passed a set */
    return hwloc_mbind((void *) addr, len, linuxpolicy, NULL, 0, 0);

  } else if (linuxpolicy == MPOL_LOCAL) {
    if (!hwloc_bitmap_isequal(nodeset, hwloc_topology_get_complete_nodeset(topology))) {
      errno = EXDEV;
      return -1;
    }
    /* MPOL_LOCAL isn't supported before 3.8, and it's identical to PREFERRED with no nodeset, which was supported way before */
    return hwloc_mbind((void *) addr, len, MPOL_PREFERRED, NULL, 0, 0);
  }

  err = hwloc_linux_membind_mask_from_nodeset(topology, nodeset, &max_os_index, &linuxmask);
  if (err < 0)
    goto out;

  if (flags & HWLOC_MEMBIND_MIGRATE) {
    linuxflags = MPOL_MF_MOVE;
    if (flags & HWLOC_MEMBIND_STRICT)
      linuxflags |= MPOL_MF_STRICT;
  }

  err = hwloc_mbind((void *) addr, len, linuxpolicy, linuxmask, max_os_index+1, linuxflags);
  if (err < 0)
    goto out_with_mask;

  free(linuxmask);
  return 0;

 out_with_mask:
  free(linuxmask);
 out:
  return -1;
}

static void *
hwloc_linux_alloc_membind(hwloc_topology_t topology, size_t len, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  void *buffer;
  int err;

  buffer = hwloc_alloc_mmap(topology, len);
  if (!buffer)
    return NULL;

  err = hwloc_linux_set_area_membind(topology, buffer, len, nodeset, policy, flags);
  if (err < 0 && (flags & HWLOC_MEMBIND_STRICT)) {
    munmap(buffer, len);
    return NULL;
  }

  return buffer;
}

static int
hwloc_linux_set_thisthread_membind(hwloc_topology_t topology, hwloc_const_nodeset_t nodeset, hwloc_membind_policy_t policy, int flags)
{
  unsigned max_os_index; /* highest os_index + 1 */
  unsigned long *linuxmask;
  int linuxpolicy;
  int err;

  err = hwloc_linux_membind_policy_from_hwloc(&linuxpolicy, policy, flags);
  if (err < 0)
    return err;

  if (linuxpolicy == MPOL_DEFAULT) {
    /* Some Linux kernels don't like being passed a set */
    return hwloc_set_mempolicy(linuxpolicy, NULL, 0);

  } else if (linuxpolicy == MPOL_LOCAL) {
    if (!hwloc_bitmap_isequal(nodeset, hwloc_topology_get_complete_nodeset(topology))) {
      errno = EXDEV;
      return -1;
    }
    /* MPOL_LOCAL isn't supported before 3.8, and it's identical to PREFERRED with no nodeset, which was supported way before */
    return hwloc_set_mempolicy(MPOL_PREFERRED, NULL, 0);
  }

  err = hwloc_linux_membind_mask_from_nodeset(topology, nodeset, &max_os_index, &linuxmask);
  if (err < 0)
    goto out;

  if (flags & HWLOC_MEMBIND_MIGRATE) {
    unsigned long *fullmask;
    fullmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*fullmask));
    if (!fullmask)
      goto out_with_mask;
    memset(fullmask, 0xf, max_os_index/HWLOC_BITS_PER_LONG * sizeof(unsigned long));
    err = hwloc_migrate_pages(0, max_os_index+1, fullmask, linuxmask); /* returns the (positive) number of non-migrated pages on success */
    free(fullmask);
    if (err < 0 && (flags & HWLOC_MEMBIND_STRICT))
      goto out_with_mask;
  }

  err = hwloc_set_mempolicy(linuxpolicy, linuxmask, max_os_index+1);
  if (err < 0)
    goto out_with_mask;

  free(linuxmask);
  return 0;

 out_with_mask:
  free(linuxmask);
 out:
  return -1;
}

/*
 * On some kernels, get_mempolicy requires the output size to be larger
 * than the kernel MAX_NUMNODES (defined by CONFIG_NODES_SHIFT).
 * Try get_mempolicy on ourself until we find a max_os_index value that
 * makes the kernel happy.
 */
static int
hwloc_linux_find_kernel_max_numnodes(hwloc_topology_t topology __hwloc_attribute_unused)
{
  static int _max_numnodes = -1, max_numnodes;
  int linuxpolicy;
  int fd;

  if (_max_numnodes != -1)
    /* already computed */
    return _max_numnodes;

  /* start with a single ulong, it's the minimal and it's enough for most machines */
  max_numnodes = HWLOC_BITS_PER_LONG;

  /* try to get the max from sysfs */
  fd = open("/sys/devices/system/node/possible", O_RDONLY); /* binding only supported in real fsroot, no need for data->root_fd */
  if (fd >= 0) {
    hwloc_bitmap_t possible_bitmap = hwloc_bitmap_alloc();
    if (hwloc__read_fd_as_cpulist(fd, possible_bitmap) == 0) {
      int max_possible = hwloc_bitmap_last(possible_bitmap);
      hwloc_debug_bitmap("possible NUMA nodes are %s\n", possible_bitmap);

      if (max_numnodes < max_possible + 1)
        max_numnodes = max_possible + 1;
    }
    close(fd);
    hwloc_bitmap_free(possible_bitmap);
  }

  while (1) {
    unsigned long *mask;
    int err;
    mask = malloc(max_numnodes / HWLOC_BITS_PER_LONG * sizeof(*mask));
    if (!mask)
      /* we can't return anything sane, assume the default size will work */
      return _max_numnodes = max_numnodes;

    err = hwloc_get_mempolicy(&linuxpolicy, mask, max_numnodes, 0, 0);
    free(mask);
    if (!err || errno != EINVAL)
      /* Found it. Only update the static value with the final one,
       * to avoid sharing intermediate values that we modify,
       * in case there's ever multiple concurrent calls.
       */
      return _max_numnodes = max_numnodes;
    max_numnodes *= 2;
  }
}

static int
hwloc_linux_membind_policy_to_hwloc(int linuxpolicy, hwloc_membind_policy_t *policy)
{
  switch (linuxpolicy) {
  case MPOL_DEFAULT:
  case MPOL_LOCAL: /* converted from MPOL_PREFERRED + empty nodeset by the caller */
    *policy = HWLOC_MEMBIND_FIRSTTOUCH;
    return 0;
  case MPOL_PREFERRED:
  case MPOL_BIND:
    *policy = HWLOC_MEMBIND_BIND;
    return 0;
  case MPOL_INTERLEAVE:
    *policy = HWLOC_MEMBIND_INTERLEAVE;
    return 0;
  default:
    errno = EINVAL;
    return -1;
  }
}

static int hwloc_linux_mask_is_empty(unsigned max_os_index, unsigned long *linuxmask)
{
  unsigned i;
  for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
    if (linuxmask[i])
      return 0;
  return 1;
}

static int
hwloc_linux_get_thisthread_membind(hwloc_topology_t topology, hwloc_nodeset_t nodeset, hwloc_membind_policy_t *policy, int flags __hwloc_attribute_unused)
{
  unsigned max_os_index;
  unsigned long *linuxmask;
  int linuxpolicy;
  int err;

  max_os_index = hwloc_linux_find_kernel_max_numnodes(topology);

  linuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*linuxmask));;
  if (!linuxmask)
    goto out;

  err = hwloc_get_mempolicy(&linuxpolicy, linuxmask, max_os_index, 0, 0);
  if (err < 0)
    goto out_with_linuxmask;

  /* MPOL_PREFERRED with empty mask is MPOL_LOCAL */
  if (linuxpolicy == MPOL_PREFERRED && hwloc_linux_mask_is_empty(max_os_index, linuxmask))
    linuxpolicy = MPOL_LOCAL;

  if (linuxpolicy == MPOL_DEFAULT || linuxpolicy == MPOL_LOCAL) {
    hwloc_bitmap_copy(nodeset, hwloc_topology_get_topology_nodeset(topology));
  } else {
    hwloc_linux_membind_mask_to_nodeset(topology, nodeset, max_os_index, linuxmask);
  }

  err = hwloc_linux_membind_policy_to_hwloc(linuxpolicy, policy);
  if (err < 0)
    goto out_with_linuxmask;

  free(linuxmask);
  return 0;

 out_with_linuxmask:
  free(linuxmask);
 out:
  return -1;
}

static int
hwloc_linux_get_area_membind(hwloc_topology_t topology, const void *addr, size_t len, hwloc_nodeset_t nodeset, hwloc_membind_policy_t *policy, int flags __hwloc_attribute_unused)
{
  unsigned max_os_index;
  unsigned long *linuxmask;
  unsigned long *globallinuxmask;
  int linuxpolicy = 0, globallinuxpolicy = 0; /* shut-up the compiler */
  int mixed = 0;
  int full = 0;
  int first = 1;
  int pagesize = hwloc_getpagesize();
  char *tmpaddr;
  int err;
  unsigned i;

  max_os_index = hwloc_linux_find_kernel_max_numnodes(topology);

  linuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*linuxmask));
  globallinuxmask = malloc(max_os_index/HWLOC_BITS_PER_LONG * sizeof(*globallinuxmask));
  if (!linuxmask || !globallinuxmask)
    goto out_with_linuxmasks;

  memset(globallinuxmask, 0, sizeof(*globallinuxmask));

  for(tmpaddr = (char *)((unsigned long)addr & ~(pagesize-1));
      tmpaddr < (char *)addr + len;
      tmpaddr += pagesize) {
    err = hwloc_get_mempolicy(&linuxpolicy, linuxmask, max_os_index, tmpaddr, MPOL_F_ADDR);
    if (err < 0)
      goto out_with_linuxmasks;

    /* MPOL_PREFERRED with empty mask is MPOL_LOCAL */
    if (linuxpolicy == MPOL_PREFERRED && hwloc_linux_mask_is_empty(max_os_index, linuxmask))
      linuxpolicy = MPOL_LOCAL;

    /* use the first found policy. if we find a different one later, set mixed to 1 */
    if (first)
      globallinuxpolicy = linuxpolicy;
    else if (globallinuxpolicy != linuxpolicy)
      mixed = 1;

    /* agregate masks, and set full to 1 if we ever find DEFAULT or LOCAL */
    if (full || linuxpolicy == MPOL_DEFAULT || linuxpolicy == MPOL_LOCAL) {
      full = 1;
    } else {
      for(i=0; i<max_os_index/HWLOC_BITS_PER_LONG; i++)
        globallinuxmask[i] |= linuxmask[i];
    }

    first = 0;
  }

  if (mixed) {
    *policy = HWLOC_MEMBIND_MIXED;
  } else {
    err = hwloc_linux_membind_policy_to_hwloc(linuxpolicy, policy);
    if (err < 0)
      goto out_with_linuxmasks;
  }

  if (full) {
    hwloc_bitmap_copy(nodeset, hwloc_topology_get_topology_nodeset(topology));
  } else {
    hwloc_linux_membind_mask_to_nodeset(topology, nodeset, max_os_index, globallinuxmask);
  }

  free(linuxmask);
  free(globallinuxmask);
  return 0;

 out_with_linuxmasks:
  free(linuxmask);
  free(globallinuxmask);
  return -1;
}

static int
hwloc_linux_get_area_memlocation(hwloc_topology_t topology __hwloc_attribute_unused, const void *addr, size_t len, hwloc_nodeset_t nodeset, int flags __hwloc_attribute_unused)
{
  unsigned offset;
  unsigned long count;
  void **pages;
  int *status;
  int pagesize = hwloc_getpagesize();
  int ret;
  unsigned i;

  offset = ((unsigned long) addr) & (pagesize-1);
  addr = ((char*) addr) - offset;
  len += offset;
  count = (len + pagesize-1)/pagesize;
  pages = malloc(count*sizeof(*pages));
  status = malloc(count*sizeof(*status));
  if (!pages || !status) {
    ret = -1;
    goto out_with_pages;
  }

  for(i=0; i<count; i++)
    pages[i] = ((char*)addr) + i*pagesize;

  ret = hwloc_move_pages(0, count, pages, NULL, status, 0);
  if (ret  < 0)
    goto out_with_pages;

  hwloc_bitmap_zero(nodeset);
  for(i=0; i<count; i++)
    if (status[i] >= 0)
      hwloc_bitmap_set(nodeset, status[i]);
  ret = 0; /* not really useful since move_pages never returns > 0 */

 out_with_pages:
  free(pages);
  free(status);
  return ret;
}

static void hwloc_linux__get_allowed_resources(hwloc_topology_t topology, const char *root_path, int root_fd, char **cpuset_namep);

static int hwloc_linux_get_allowed_resources_hook(hwloc_topology_t topology)
{
  const char *fsroot_path;
  char *cpuset_name = NULL;
  int root_fd = -1;

  fsroot_path = getenv("HWLOC_FSROOT");
  if (!fsroot_path)
    fsroot_path = "/";

  if (strcmp(fsroot_path, "/")) {
#ifdef HAVE_OPENAT
    root_fd = open(fsroot_path, O_RDONLY | O_DIRECTORY);
    if (root_fd < 0)
      goto out;
#else
    errno = ENOSYS;
    goto out;
#endif
  }

  /* we could also error-out if the current topology doesn't actually match the system,
   * at least for PUs and NUMA nodes. But it would increase the overhead of loading XMLs.
   *
   * Just trust the user when he sets THISSYSTEM=1. It enables hacky
   * tests such as restricting random XML or synthetic to the current
   * machine (uses the default cgroup).
   */

  hwloc_linux__get_allowed_resources(topology, fsroot_path, root_fd, &cpuset_name);
  if (cpuset_name) {
    hwloc__add_info_nodup(&topology->levels[0][0]->infos, &topology->levels[0][0]->infos_count,
			  "LinuxCgroup", cpuset_name, 1 /* replace */);
    free(cpuset_name);
  }
  if (root_fd != -1)
    close(root_fd);

 out:
  return -1;
}

void
hwloc_set_linuxfs_hooks(struct hwloc_binding_hooks *hooks,
			struct hwloc_topology_support *support)
{
  hooks->set_thisthread_cpubind = hwloc_linux_set_thisthread_cpubind;
  hooks->get_thisthread_cpubind = hwloc_linux_get_thisthread_cpubind;
  hooks->set_thisproc_cpubind = hwloc_linux_set_thisproc_cpubind;
  hooks->get_thisproc_cpubind = hwloc_linux_get_thisproc_cpubind;
  hooks->set_proc_cpubind = hwloc_linux_set_proc_cpubind;
  hooks->get_proc_cpubind = hwloc_linux_get_proc_cpubind;
#if HAVE_DECL_PTHREAD_SETAFFINITY_NP
  hooks->set_thread_cpubind = hwloc_linux_set_thread_cpubind;
#endif /* HAVE_DECL_PTHREAD_SETAFFINITY_NP */
#if HAVE_DECL_PTHREAD_GETAFFINITY_NP
  hooks->get_thread_cpubind = hwloc_linux_get_thread_cpubind;
#endif /* HAVE_DECL_PTHREAD_GETAFFINITY_NP */
  hooks->get_thisthread_last_cpu_location = hwloc_linux_get_thisthread_last_cpu_location;
  hooks->get_thisproc_last_cpu_location = hwloc_linux_get_thisproc_last_cpu_location;
  hooks->get_proc_last_cpu_location = hwloc_linux_get_proc_last_cpu_location;
  hooks->set_thisthread_membind = hwloc_linux_set_thisthread_membind;
  hooks->get_thisthread_membind = hwloc_linux_get_thisthread_membind;
  hooks->get_area_membind = hwloc_linux_get_area_membind;
  hooks->set_area_membind = hwloc_linux_set_area_membind;
  hooks->get_area_memlocation = hwloc_linux_get_area_memlocation;
  hooks->alloc_membind = hwloc_linux_alloc_membind;
  hooks->alloc = hwloc_alloc_mmap;
  hooks->free_membind = hwloc_free_mmap;
  support->membind->firsttouch_membind = 1;
  support->membind->bind_membind = 1;
  support->membind->interleave_membind = 1;
  support->membind->migrate_membind = 1;
  hooks->get_allowed_resources = hwloc_linux_get_allowed_resources_hook;

  /* The get_allowed_resources() hook also works in the !thissystem case
   * (it just reads fsroot files) but hooks are only setup if thissystem.
   * Not an issue because this hook isn't used unless THISSYSTEM_ALLOWED_RESOURCES
   * which also requires THISSYSTEM which means this functions is called.
   */
}


/*******************************************
 *** Misc Helpers for Topology Discovery ***
 *******************************************/

/* cpuinfo array */
struct hwloc_linux_cpuinfo_proc {
  /* set during hwloc_linux_parse_cpuinfo */
  unsigned long Pproc;

  /* custom info, set during hwloc_linux_parse_cpuinfo */
  struct hwloc_info_s *infos;
  unsigned infos_count;
};

static void
hwloc_find_linux_cpuset_mntpnt(char **cgroup_mntpnt, char **cpuset_mntpnt, const char *root_path)
{
  char *mount_path;
  struct mntent mntent;
  char *buf;
  FILE *fd;
  int err;
  size_t bufsize;

  *cgroup_mntpnt = NULL;
  *cpuset_mntpnt = NULL;

  if (root_path) {
    /* setmntent() doesn't support openat(), so use the root_path directly */
    err = asprintf(&mount_path, "%s/proc/mounts", root_path);
    if (err < 0)
      return;
    fd = setmntent(mount_path, "r");
    free(mount_path);
  } else {
    fd = setmntent("/proc/mounts", "r");
  }
  if (!fd)
    return;

  /* getmntent_r() doesn't actually report an error when the buffer
   * is too small. It just silently truncates things. So we can't
   * dynamically resize things.
   *
   * Linux limits mount type, string, and options to one page each.
   * getmntent() limits the line size to 4kB.
   * so use 4*pagesize to be far above both.
   */
  bufsize = hwloc_getpagesize()*4;
  buf = malloc(bufsize);
  if (!buf)
    return;

  while (getmntent_r(fd, &mntent, buf, bufsize)) {
    if (!strcmp(mntent.mnt_type, "cpuset")) {
      hwloc_debug("Found cpuset mount point on %s\n", mntent.mnt_dir);
      *cpuset_mntpnt = strdup(mntent.mnt_dir);
      break;
    } else if (!strcmp(mntent.mnt_type, "cgroup")) {
      /* found a cgroup mntpnt */
      char *opt, *opts = mntent.mnt_opts;
      int cpuset_opt = 0;
      int noprefix_opt = 0;
      /* look at options */
      while ((opt = strsep(&opts, ",")) != NULL) {
	if (!strcmp(opt, "cpuset"))
	  cpuset_opt = 1;
	else if (!strcmp(opt, "noprefix"))
	  noprefix_opt = 1;
      }
      if (!cpuset_opt)
	continue;
      if (noprefix_opt) {
	hwloc_debug("Found cgroup emulating a cpuset mount point on %s\n", mntent.mnt_dir);
	*cpuset_mntpnt = strdup(mntent.mnt_dir);
      } else {
	hwloc_debug("Found cgroup/cpuset mount point on %s\n", mntent.mnt_dir);
	*cgroup_mntpnt = strdup(mntent.mnt_dir);
      }
      break;
    }
  }

  endmntent(fd);
  free(buf);
}

/*
 * Linux cpusets may be managed directly or through cgroup.
 * If cgroup is used, tasks get a /proc/pid/cgroup which may contain a
 * single line %d:cpuset:<name>. If cpuset are used they get /proc/pid/cpuset
 * containing <name>.
 */
static char *
hwloc_read_linux_cpuset_name(int fsroot_fd, hwloc_pid_t pid)
{
#define CPUSET_NAME_LEN 128
  char cpuset_name[CPUSET_NAME_LEN];
  FILE *file;
  int err;
  char *tmp;

  /* check whether a cgroup-cpuset is enabled */
  if (!pid)
    file = hwloc_fopen("/proc/self/cgroup", "r", fsroot_fd);
  else {
    char path[] = "/proc/XXXXXXXXXXX/cgroup";
    snprintf(path, sizeof(path), "/proc/%d/cgroup", pid);
    file = hwloc_fopen(path, "r", fsroot_fd);
  }
  if (file) {
    /* find a cpuset line */
#define CGROUP_LINE_LEN 256
    char line[CGROUP_LINE_LEN];
    while (fgets(line, sizeof(line), file)) {
      char *end, *colon = strchr(line, ':');
      if (!colon)
	continue;
      if (strncmp(colon, ":cpuset:", 8))
	continue;

      /* found a cgroup-cpuset line, return the name */
      fclose(file);
      end = strchr(colon, '\n');
      if (end)
	*end = '\0';
      hwloc_debug("Found cgroup-cpuset %s\n", colon+8);
      return strdup(colon+8);
    }
    fclose(file);
  }

  /* check whether a cpuset is enabled */
  if (!pid)
    err = hwloc_read_path_by_length("/proc/self/cpuset", cpuset_name, sizeof(cpuset_name), fsroot_fd);
  else {
    char path[] = "/proc/XXXXXXXXXXX/cpuset";
    snprintf(path, sizeof(path), "/proc/%d/cpuset", pid);
    err = hwloc_read_path_by_length(path, cpuset_name, sizeof(cpuset_name), fsroot_fd);
  }
  if (err < 0) {
    /* found nothing */
    hwloc_debug("%s", "No cgroup or cpuset found\n");
    return NULL;
  }

  /* found a cpuset, return the name */
  tmp = strchr(cpuset_name, '\n');
  if (tmp)
    *tmp = '\0';
  hwloc_debug("Found cpuset %s\n", cpuset_name);
  return strdup(cpuset_name);
}

/*
 * Then, the cpuset description is available from either the cgroup or
 * the cpuset filesystem (usually mounted in / or /dev) where there
 * are cgroup<name>/cpuset.{cpus,mems} or cpuset<name>/{cpus,mems} files.
 */
static void
hwloc_admin_disable_set_from_cpuset(int root_fd,
				    const char *cgroup_mntpnt, const char *cpuset_mntpnt, const char *cpuset_name,
				    const char *attr_name,
				    hwloc_bitmap_t admin_enabled_set)
{
#define CPUSET_FILENAME_LEN 256
  char cpuset_filename[CPUSET_FILENAME_LEN];
  int err;

  if (cgroup_mntpnt) {
    /* try to read the cpuset from cgroup */
    snprintf(cpuset_filename, CPUSET_FILENAME_LEN, "%s%s/cpuset.%s", cgroup_mntpnt, cpuset_name, attr_name);
    hwloc_debug("Trying to read cgroup file <%s>\n", cpuset_filename);
  } else if (cpuset_mntpnt) {
    /* try to read the cpuset directly */
    snprintf(cpuset_filename, CPUSET_FILENAME_LEN, "%s%s/%s", cpuset_mntpnt, cpuset_name, attr_name);
    hwloc_debug("Trying to read cpuset file <%s>\n", cpuset_filename);
  }

  err = hwloc__read_path_as_cpulist(cpuset_filename, admin_enabled_set, root_fd);

  if (err < 0) {
    hwloc_debug("failed to read cpuset '%s' attribute '%s'\n", cpuset_name, attr_name);
    hwloc_bitmap_fill(admin_enabled_set);
  } else {
    hwloc_debug_bitmap("cpuset includes %s\n", admin_enabled_set);
  }
}

static void
hwloc_parse_meminfo_info(struct hwloc_linux_backend_data_s *data,
			 const char *path,
			 uint64_t *local_memory)
{
  char *tmp;
  char buffer[4096];
  unsigned long long number;

  if (hwloc_read_path_by_length(path, buffer, sizeof(buffer), data->root_fd) < 0)
    return;

  tmp = strstr(buffer, "MemTotal: "); /* MemTotal: %llu kB */
  if (tmp) {
    number = strtoull(tmp+10, NULL, 10);
    *local_memory = number << 10;
  }
}

#define SYSFS_NUMA_NODE_PATH_LEN 128

static void
hwloc_parse_hugepages_info(struct hwloc_linux_backend_data_s *data,
			   const char *dirpath,
			   struct hwloc_numanode_attr_s *memory,
			   uint64_t *remaining_local_memory)
{
  DIR *dir;
  struct dirent *dirent;
  unsigned long index_ = 1; /* slot 0 is for normal pages */
  char line[64];
  char path[SYSFS_NUMA_NODE_PATH_LEN];

  dir = hwloc_opendir(dirpath, data->root_fd);
  if (dir) {
    while ((dirent = readdir(dir)) != NULL) {
      int err;
      if (strncmp(dirent->d_name, "hugepages-", 10))
        continue;
      memory->page_types[index_].size = strtoul(dirent->d_name+10, NULL, 0) * 1024ULL;
      err = snprintf(path, sizeof(path), "%s/%s/nr_hugepages", dirpath, dirent->d_name);
      if ((size_t) err < sizeof(path)
	  && !hwloc_read_path_by_length(path, line, sizeof(line), data->root_fd)) {
	/* these are the actual total amount of huge pages */
	memory->page_types[index_].count = strtoull(line, NULL, 0);
	*remaining_local_memory -= memory->page_types[index_].count * memory->page_types[index_].size;
	index_++;
      }
    }
    closedir(dir);
    memory->page_types_len = index_;
  }
}

static void
hwloc_get_machine_meminfo(struct hwloc_linux_backend_data_s *data,
			  struct hwloc_numanode_attr_s *memory)
{
  struct stat st;
  int has_sysfs_hugepages = 0;
  int types = 1; /* only normal pages by default */
  uint64_t remaining_local_memory;
  int err;

  err = hwloc_stat("/sys/kernel/mm/hugepages", &st, data->root_fd);
  if (!err) {
    types = 1 + st.st_nlink-2;
    has_sysfs_hugepages = 1;
  }

  memory->page_types = calloc(types, sizeof(*memory->page_types));
  if (!memory->page_types) {
    memory->page_types_len = 0;
    return;
  }
  memory->page_types_len = 1; /* we'll increase it when successfully getting hugepage info */

  /* get the total memory */
  hwloc_parse_meminfo_info(data, "/proc/meminfo",
			   &memory->local_memory);
  remaining_local_memory = memory->local_memory;

  if (has_sysfs_hugepages) {
    /* read from node%d/hugepages/hugepages-%skB/nr_hugepages */
    hwloc_parse_hugepages_info(data, "/sys/kernel/mm/hugepages", memory, &remaining_local_memory);
  }

  /* use remaining memory as normal pages */
  memory->page_types[0].size = data->pagesize;
  memory->page_types[0].count = remaining_local_memory / memory->page_types[0].size;
}

static void
hwloc_get_sysfs_node_meminfo(struct hwloc_linux_backend_data_s *data,
			     const char *syspath, int node,
			     struct hwloc_numanode_attr_s *memory)
{
  char path[SYSFS_NUMA_NODE_PATH_LEN];
  char meminfopath[SYSFS_NUMA_NODE_PATH_LEN];
  struct stat st;
  int has_sysfs_hugepages = 0;
  int types = 1; /* only normal pages by default */
  uint64_t remaining_local_memory;
  int err;

  sprintf(path, "%s/node%d/hugepages", syspath, node);
  err = hwloc_stat(path, &st, data->root_fd);
  if (!err) {
    types = 1 + st.st_nlink-2;
    has_sysfs_hugepages = 1;
  }

  memory->page_types = calloc(types, sizeof(*memory->page_types));
  if (!memory->page_types) {
    memory->page_types_len = 0;
    return;
  }
  memory->page_types_len = 1; /* we'll increase it when successfully getting hugepage info */

  /* get the total memory */
  sprintf(meminfopath, "%s/node%d/meminfo", syspath, node);
  hwloc_parse_meminfo_info(data, meminfopath,
			   &memory->local_memory);
  remaining_local_memory = memory->local_memory;

  if (has_sysfs_hugepages) {
    /* read from node%d/hugepages/hugepages-%skB/nr_hugepages */
    hwloc_parse_hugepages_info(data, path, memory, &remaining_local_memory);
  }

  /* use remaining memory as normal pages */
  memory->page_types[0].size = data->pagesize;
  memory->page_types[0].count = remaining_local_memory / memory->page_types[0].size;
}

static int
hwloc_parse_nodes_distances(const char *path, unsigned nbnodes, unsigned *indexes, uint64_t *distances, int fsroot_fd)
{
  size_t len = (10+1)*nbnodes;
  uint64_t *curdist = distances;
  char *string;
  unsigned i;

  string = malloc(len); /* space-separated %d */
  if (!string)
    goto out;

  for(i=0; i<nbnodes; i++) {
    unsigned osnode = indexes[i];
    char distancepath[SYSFS_NUMA_NODE_PATH_LEN];
    char *tmp, *next;
    unsigned found;

    /* Linux nodeX/distance file contains distance from X to other localities (from ACPI SLIT table or so),
     * store them in slots X*N...X*N+N-1 */
    sprintf(distancepath, "%s/node%u/distance", path, osnode);
    if (hwloc_read_path_by_length(distancepath, string, len, fsroot_fd) < 0)
      goto out_with_string;

    tmp = string;
    found = 0;
    while (tmp) {
      unsigned distance = strtoul(tmp, &next, 0); /* stored as a %d */
      if (next == tmp)
	break;
      *curdist = (uint64_t) distance;
      curdist++;
      found++;
      if (found == nbnodes)
	break;
      tmp = next+1;
    }
    if (found != nbnodes)
      goto out_with_string;
  }

  free(string);
  return 0;

 out_with_string:
  free(string);
 out:
  return -1;
}

static void
hwloc__get_dmi_id_one_info(struct hwloc_linux_backend_data_s *data,
			   hwloc_obj_t obj,
			   char *path, unsigned pathlen,
			   const char *dmi_name, const char *hwloc_name)
{
  char dmi_line[64];

  strcpy(path+pathlen, dmi_name);
  if (hwloc_read_path_by_length(path, dmi_line, sizeof(dmi_line), data->root_fd) < 0)
    return;

  if (dmi_line[0] != '\0') {
    char *tmp = strchr(dmi_line, '\n');
    if (tmp)
      *tmp = '\0';
    hwloc_debug("found %s '%s'\n", hwloc_name, dmi_line);
    hwloc_obj_add_info(obj, hwloc_name, dmi_line);
  }
}

static void
hwloc__get_dmi_id_info(struct hwloc_linux_backend_data_s *data, hwloc_obj_t obj)
{
  char path[128];
  unsigned pathlen;
  DIR *dir;

  strcpy(path, "/sys/devices/virtual/dmi/id");
  dir = hwloc_opendir(path, data->root_fd);
  if (dir) {
    pathlen = 27;
  } else {
    strcpy(path, "/sys/class/dmi/id");
    dir = hwloc_opendir(path, data->root_fd);
    if (dir)
      pathlen = 17;
    else
      return;
  }
  closedir(dir);

  path[pathlen++] = '/';

  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_name", "DMIProductName");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_version", "DMIProductVersion");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_serial", "DMIProductSerial");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "product_uuid", "DMIProductUUID");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_vendor", "DMIBoardVendor");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_name", "DMIBoardName");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_version", "DMIBoardVersion");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_serial", "DMIBoardSerial");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "board_asset_tag", "DMIBoardAssetTag");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_vendor", "DMIChassisVendor");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_type", "DMIChassisType");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_version", "DMIChassisVersion");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_serial", "DMIChassisSerial");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "chassis_asset_tag", "DMIChassisAssetTag");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_vendor", "DMIBIOSVendor");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_version", "DMIBIOSVersion");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "bios_date", "DMIBIOSDate");
  hwloc__get_dmi_id_one_info(data, obj, path, pathlen, "sys_vendor", "DMISysVendor");
}


/***********************************
 ****** Device tree Discovery ******
 ***********************************/

/* Reads the entire file and returns bytes read if bytes_read != NULL
 * Returned pointer can be freed by using free().  */
static void *
hwloc_read_raw(const char *p, const char *p1, size_t *bytes_read, int root_fd)
{
  char fname[256];
  char *ret = NULL;
  struct stat fs;
  int file = -1;

  snprintf(fname, sizeof(fname), "%s/%s", p, p1);

  file = hwloc_open(fname, root_fd);
  if (-1 == file) {
      goto out_no_close;
  }
  if (fstat(file, &fs)) {
    goto out;
  }

  ret = (char *) malloc(fs.st_size);
  if (NULL != ret) {
    ssize_t cb = read(file, ret, fs.st_size);
    if (cb == -1) {
      free(ret);
      ret = NULL;
    } else {
      if (NULL != bytes_read)
        *bytes_read = cb;
    }
  }

 out:
  close(file);
 out_no_close:
  return ret;
}

/* Reads the entire file and returns it as a 0-terminated string
 * Returned pointer can be freed by using free().  */
static char *
hwloc_read_str(const char *p, const char *p1, int root_fd)
{
  size_t cb = 0;
  char *ret = hwloc_read_raw(p, p1, &cb, root_fd);
  if ((NULL != ret) && (0 < cb) && (0 != ret[cb-1])) {
    char *tmp = realloc(ret, cb + 1);
    if (!tmp) {
      free(ret);
      return NULL;
    }
    ret = tmp;
    ret[cb] = 0;
  }
  return ret;
}

/* Reads first 32bit bigendian value */
static ssize_t
hwloc_read_unit32be(const char *p, const char *p1, uint32_t *buf, int root_fd)
{
  size_t cb = 0;
  uint32_t *tmp = hwloc_read_raw(p, p1, &cb, root_fd);
  if (sizeof(*buf) != cb) {
    errno = EINVAL;
    free(tmp); /* tmp is either NULL or contains useless things */
    return -1;
  }
  *buf = htonl(*tmp);
  free(tmp);
  return sizeof(*buf);
}

typedef struct {
  unsigned int n, allocated;
  struct {
    hwloc_bitmap_t cpuset;
    uint32_t phandle;
    uint32_t l2_cache;
    char *name;
  } *p;
} device_tree_cpus_t;

static void
add_device_tree_cpus_node(device_tree_cpus_t *cpus, hwloc_bitmap_t cpuset,
    uint32_t l2_cache, uint32_t phandle, const char *name)
{
  if (cpus->n == cpus->allocated) {
    void *tmp;
    unsigned allocated;
    if (!cpus->allocated)
      allocated = 64;
    else
      allocated = 2 * cpus->allocated;
    tmp = realloc(cpus->p, allocated * sizeof(cpus->p[0]));
    if (!tmp)
      return; /* failed to realloc, ignore this entry */
    cpus->p = tmp;
    cpus->allocated = allocated;
  }
  cpus->p[cpus->n].phandle = phandle;
  cpus->p[cpus->n].cpuset = (NULL == cpuset)?NULL:hwloc_bitmap_dup(cpuset);
  cpus->p[cpus->n].l2_cache = l2_cache;
  cpus->p[cpus->n].name = strdup(name);
  ++cpus->n;
}

/* Walks over the cache list in order to detect nested caches and CPU mask for each */
static int
look_powerpc_device_tree_discover_cache(device_tree_cpus_t *cpus,
    uint32_t phandle, unsigned int *level, hwloc_bitmap_t cpuset)
{
  unsigned int i;
  int ret = -1;
  if ((NULL == level) || (NULL == cpuset) || phandle == (uint32_t) -1)
    return ret;
  for (i = 0; i < cpus->n; ++i) {
    if (phandle != cpus->p[i].l2_cache)
      continue;
    if (NULL != cpus->p[i].cpuset) {
      hwloc_bitmap_or(cpuset, cpuset, cpus->p[i].cpuset);
      ret = 0;
    } else {
      ++(*level);
      if (0 == look_powerpc_device_tree_discover_cache(cpus,
            cpus->p[i].phandle, level, cpuset))
        ret = 0;
    }
  }
  return ret;
}

static void
try__add_cache_from_device_tree_cpu(struct hwloc_topology *topology,
				    unsigned int level, hwloc_obj_cache_type_t ctype,
				    uint32_t cache_line_size, uint32_t cache_size, uint32_t cache_sets,
				    hwloc_bitmap_t cpuset)
{
  struct hwloc_obj *c = NULL;
  hwloc_obj_type_t otype;

  if (0 == cache_size)
    return;

  otype = hwloc_cache_type_by_depth_type(level, ctype);
  if (otype == HWLOC_OBJ_TYPE_NONE)
    return;
  if (!hwloc_filter_check_keep_object_type(topology, otype))
    return;

  c = hwloc_alloc_setup_object(topology, otype, HWLOC_UNKNOWN_INDEX);
  c->attr->cache.depth = level;
  c->attr->cache.linesize = cache_line_size;
  c->attr->cache.size = cache_size;
  c->attr->cache.type = ctype;
  if (cache_sets == 1)
    /* likely wrong, make it unknown */
    cache_sets = 0;
  if (cache_sets && cache_line_size)
    c->attr->cache.associativity = cache_size / (cache_sets * cache_line_size);
  else
    c->attr->cache.associativity = 0;
  c->cpuset = hwloc_bitmap_dup(cpuset);
  hwloc_debug_2args_bitmap("cache (%s) depth %u has cpuset %s\n",
			   ctype == HWLOC_OBJ_CACHE_UNIFIED ? "unified" : (ctype == HWLOC_OBJ_CACHE_DATA ? "data" : "instruction"),
			   level, c->cpuset);
  hwloc_insert_object_by_cpuset(topology, c);
}

static void
try_add_cache_from_device_tree_cpu(struct hwloc_topology *topology,
				   struct hwloc_linux_backend_data_s *data,
				   const char *cpu, unsigned int level, hwloc_bitmap_t cpuset)
{
  /* d-cache-block-size - ignore */
  /* d-cache-line-size - to read, in bytes */
  /* d-cache-sets - ignore */
  /* d-cache-size - to read, in bytes */
  /* i-cache, same for instruction */
  /* cache-unified only exist if data and instruction caches are unified */
  /* d-tlb-sets - ignore */
  /* d-tlb-size - ignore, always 0 on power6 */
  /* i-tlb-*, same */
  uint32_t d_cache_line_size = 0, d_cache_size = 0, d_cache_sets = 0;
  uint32_t i_cache_line_size = 0, i_cache_size = 0, i_cache_sets = 0;
  char unified_path[1024];
  struct stat statbuf;
  int unified;

  snprintf(unified_path, sizeof(unified_path), "%s/cache-unified", cpu);
  unified = (hwloc_stat(unified_path, &statbuf, data->root_fd) == 0);

  hwloc_read_unit32be(cpu, "d-cache-line-size", &d_cache_line_size,
      data->root_fd);
  hwloc_read_unit32be(cpu, "d-cache-size", &d_cache_size,
      data->root_fd);
  hwloc_read_unit32be(cpu, "d-cache-sets", &d_cache_sets,
      data->root_fd);
  hwloc_read_unit32be(cpu, "i-cache-line-size", &i_cache_line_size,
      data->root_fd);
  hwloc_read_unit32be(cpu, "i-cache-size", &i_cache_size,
      data->root_fd);
  hwloc_read_unit32be(cpu, "i-cache-sets", &i_cache_sets,
      data->root_fd);

  if (!unified)
    try__add_cache_from_device_tree_cpu(topology, level, HWLOC_OBJ_CACHE_INSTRUCTION,
					i_cache_line_size, i_cache_size, i_cache_sets, cpuset);
  try__add_cache_from_device_tree_cpu(topology, level, unified ? HWLOC_OBJ_CACHE_UNIFIED : HWLOC_OBJ_CACHE_DATA,
				      d_cache_line_size, d_cache_size, d_cache_sets, cpuset);
}

/*
 * Discovers L1/L2/L3 cache information on IBM PowerPC systems for old kernels (RHEL5.*)
 * which provide NUMA nodes information without any details
 */
static void
look_powerpc_device_tree(struct hwloc_topology *topology,
			 struct hwloc_linux_backend_data_s *data)
{
  device_tree_cpus_t cpus;
  const char ofroot[] = "/proc/device-tree/cpus";
  unsigned int i;
  int root_fd = data->root_fd;
  DIR *dt = hwloc_opendir(ofroot, root_fd);
  struct dirent *dirent;

  if (NULL == dt)
    return;

  /* only works for Power so far, and not useful on ARM */
  if (data->arch != HWLOC_LINUX_ARCH_POWER) {
    closedir(dt);
    return;
  }

  cpus.n = 0;
  cpus.p = NULL;
  cpus.allocated = 0;

  while (NULL != (dirent = readdir(dt))) {
    char cpu[256];
    char *device_type;
    uint32_t reg = -1, l2_cache = -1, phandle = -1;
    int err;

    if ('.' == dirent->d_name[0])
      continue;

    err = snprintf(cpu, sizeof(cpu), "%s/%s", ofroot, dirent->d_name);
    if ((size_t) err >= sizeof(cpu))
      continue;

    device_type = hwloc_read_str(cpu, "device_type", root_fd);
    if (NULL == device_type)
      continue;

    hwloc_read_unit32be(cpu, "reg", &reg, root_fd);
    if (hwloc_read_unit32be(cpu, "next-level-cache", &l2_cache, root_fd) == -1)
      hwloc_read_unit32be(cpu, "l2-cache", &l2_cache, root_fd);
    if (hwloc_read_unit32be(cpu, "phandle", &phandle, root_fd) == -1)
      if (hwloc_read_unit32be(cpu, "ibm,phandle", &phandle, root_fd) == -1)
        hwloc_read_unit32be(cpu, "linux,phandle", &phandle, root_fd);

    if (0 == strcmp(device_type, "cache")) {
      add_device_tree_cpus_node(&cpus, NULL, l2_cache, phandle, dirent->d_name);
    }
    else if (0 == strcmp(device_type, "cpu")) {
      /* Found CPU */
      hwloc_bitmap_t cpuset = NULL;
      size_t cb = 0;
      uint32_t *threads = hwloc_read_raw(cpu, "ibm,ppc-interrupt-server#s", &cb, root_fd);
      uint32_t nthreads = cb / sizeof(threads[0]);

      if (NULL != threads) {
        cpuset = hwloc_bitmap_alloc();
        for (i = 0; i < nthreads; ++i) {
          if (hwloc_bitmap_isset(topology->levels[0][0]->complete_cpuset, ntohl(threads[i])))
            hwloc_bitmap_set(cpuset, ntohl(threads[i]));
        }
        free(threads);
      } else if ((unsigned int)-1 != reg) {
        /* Doesn't work on ARM because cpu "reg" do not start at 0.
	 * We know the first cpu "reg" is the lowest. The others are likely
	 * in order assuming the device-tree shows objects in order.
	 */
        cpuset = hwloc_bitmap_alloc();
        hwloc_bitmap_set(cpuset, reg);
      }

      if (NULL == cpuset) {
        hwloc_debug("%s has no \"reg\" property, skipping\n", cpu);
      } else {
        struct hwloc_obj *core = NULL;
        add_device_tree_cpus_node(&cpus, cpuset, l2_cache, phandle, dirent->d_name);

	if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_CORE)) {
	  /* Add core */
	  core = hwloc_alloc_setup_object(topology, HWLOC_OBJ_CORE, (unsigned) reg);
	  core->cpuset = hwloc_bitmap_dup(cpuset);
	  hwloc_insert_object_by_cpuset(topology, core);
	}

        /* Add L1 cache */
        try_add_cache_from_device_tree_cpu(topology, data, cpu, 1, cpuset);

        hwloc_bitmap_free(cpuset);
      }
    }
    free(device_type);
  }
  closedir(dt);

  /* No cores and L2 cache were found, exiting */
  if (0 == cpus.n) {
    hwloc_debug("No cores and L2 cache were found in %s, exiting\n", ofroot);
    return;
  }

#ifdef HWLOC_DEBUG
  for (i = 0; i < cpus.n; ++i) {
    hwloc_debug("%u: %s  ibm,phandle=%08X l2_cache=%08X ",
      i, cpus.p[i].name, cpus.p[i].phandle, cpus.p[i].l2_cache);
    if (NULL == cpus.p[i].cpuset) {
      hwloc_debug("%s\n", "no cpuset");
    } else {
      hwloc_debug_bitmap("cpuset %s\n", cpus.p[i].cpuset);
    }
  }
#endif

  /* Scan L2/L3/... caches */
  for (i = 0; i < cpus.n; ++i) {
    unsigned int level = 2;
    hwloc_bitmap_t cpuset;
    /* Skip real CPUs */
    if (NULL != cpus.p[i].cpuset)
      continue;

    /* Calculate cache level and CPU mask */
    cpuset = hwloc_bitmap_alloc();
    if (0 == look_powerpc_device_tree_discover_cache(&cpus,
          cpus.p[i].phandle, &level, cpuset)) {
      char cpu[256];
      snprintf(cpu, sizeof(cpu), "%s/%s", ofroot, cpus.p[i].name);
      try_add_cache_from_device_tree_cpu(topology, data, cpu, level, cpuset);
    }
    hwloc_bitmap_free(cpuset);
  }

  /* Do cleanup */
  for (i = 0; i < cpus.n; ++i) {
    hwloc_bitmap_free(cpus.p[i].cpuset);
    free(cpus.p[i].name);
  }
  free(cpus.p);
}

/***************************************
 * KNL NUMA quirks
 */

struct knl_hwdata {
  char memory_mode[32];
  char cluster_mode[32];
  long long int mcdram_cache_size; /* mcdram_cache_* is valid only if size > 0 */
  int mcdram_cache_associativity;
  int mcdram_cache_inclusiveness;
  int mcdram_cache_line_size;
};

struct knl_distances_summary {
  unsigned nb_values; /* number of different values found in the matrix */
  struct knl_distances_value {
    unsigned occurences;
    uint64_t value;
  } values[4]; /* sorted by occurences */
};

static int hwloc_knl_distances_value_compar(const void *_v1, const void *_v2)
{
  const struct knl_distances_value *v1 = _v1, *v2 = _v2;
  return v1->occurences - v2->occurences;
}

static int
hwloc_linux_knl_parse_numa_distances(unsigned nbnodes,
				     uint64_t *distances,
				     struct knl_distances_summary *summary)
{
  unsigned i, j, k;

  summary->nb_values = 1;
  summary->values[0].value = 10;
  summary->values[0].occurences = nbnodes;

  if (nbnodes == 1)
    /* nothing else needed */
    return 0;

  if (nbnodes != 2 && nbnodes != 4 && nbnodes != 8) {
    fprintf(stderr, "Ignoring KNL NUMA quirk, nbnodes (%u) isn't 2, 4 or 8.\n", nbnodes);
    return -1;
  }

  if (!distances) {
    fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix missing.\n");
    return -1;
  }

  for(i=0; i<nbnodes; i++) {
    /* check we have 10 on the diagonal */
    if (distances[i*nbnodes+i] != 10) {
      fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix does not contain 10 on the diagonal.\n");
      return -1;
    }
    for(j=i+1; j<nbnodes; j++) {
      uint64_t distance = distances[i*nbnodes+j];
      /* check things are symmetric */
      if (distance != distances[i+j*nbnodes]) {
	fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix isn't symmetric.\n");
	return -1;
      }
      /* check everything outside the diagonal is > 10 */
      if (distance <= 10) {
	fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix contains values <= 10.\n");
	return -1;
      }
      /* did we already see this value? */
      for(k=0; k<summary->nb_values; k++)
	if (distance == summary->values[k].value) {
	  summary->values[k].occurences++;
	  break;
	}
      if (k == summary->nb_values) {
	/* add a new value */
	if (k == 4) {
	  fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix contains more than 4 different values.\n");
	  return -1;
	}
	summary->values[k].value = distance;
	summary->values[k].occurences = 1;
	summary->nb_values++;
      }
    }
  }

  qsort(summary->values, summary->nb_values, sizeof(struct knl_distances_value), hwloc_knl_distances_value_compar);

  if (nbnodes == 2) {
    if (summary->nb_values != 2) {
      fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 2 nodes cannot contain %u different values instead of 2.\n",
	      summary->nb_values);
      return -1;
    }

  } else if (nbnodes == 4) {
    if (summary->nb_values != 2 && summary->nb_values != 4) {
      fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 8 nodes cannot contain %u different values instead of 2 or 4.\n",
	      summary->nb_values);
      return -1;
    }

  } else if (nbnodes == 8) {
    if (summary->nb_values != 4) {
      fprintf(stderr, "Ignoring KNL NUMA quirk, distance matrix for 8 nodes cannot contain %u different values instead of 4.\n",
	      summary->nb_values);
      return -1;
    }

  } else {
    abort(); /* checked above */
  }

  hwloc_debug("Summary of KNL distance matrix:\n");
  for(k=0; k<summary->nb_values; k++)
    hwloc_debug("  Found %u times distance %llu\n", summary->values[k].occurences, (unsigned long long) summary->values[k].value);
  return 0;
}

static int
hwloc_linux_knl_identify_4nodes(uint64_t *distances,
				struct knl_distances_summary *distsum,
				unsigned *ddr, unsigned *mcdram) /* ddr and mcdram arrays must be 2-long */
{
  uint64_t value;
  unsigned i;

  hwloc_debug("Trying to identify 4 KNL NUMA nodes in SNC-2 cluster mode...\n");

  /* The SNC2-Flat/Hybrid matrix should be something like
   * 10 21 31 41
   * 21 10 41 31
   * 31 41 10 41
   * 41 31 41 10
   * which means there are (above 4*10 on the diagonal):
   * 1 unique value for DDR to other DDR,
   * 2 identical values for DDR to local MCDRAM
   * 3 identical values for everything else.
   */
  if (distsum->nb_values != 4
      || distsum->values[0].occurences != 1 /* DDR to DDR */
      || distsum->values[1].occurences != 2 /* DDR to local MCDRAM */
      || distsum->values[2].occurences != 3 /* others */
      || distsum->values[3].occurences != 4 /* local */ )
    return -1;

  /* DDR:0 is always first */
  ddr[0] = 0;

  /* DDR:1 is at distance distsum->values[0].value from ddr[0] */
  value = distsum->values[0].value;
  ddr[1] = 0;
  hwloc_debug("  DDR#0 is NUMAnode#0\n");
  for(i=0; i<4; i++)
    if (distances[i] == value) {
      ddr[1] = i;
      hwloc_debug("  DDR#1 is NUMAnode#%u\n", i);
      break;
    }
  if (!ddr[1])
    return -1;

  /* MCDRAMs are at distance distsum->values[1].value from their local DDR */
  value = distsum->values[1].value;
  mcdram[0] = mcdram[1] = 0;
  for(i=1; i<4; i++) {
    if (distances[i] == value) {
      hwloc_debug("  MCDRAM#0 is NUMAnode#%u\n", i);
      mcdram[0] = i;
    } else if (distances[ddr[1]*4+i] == value) {
      hwloc_debug("  MCDRAM#1 is NUMAnode#%u\n", i);
      mcdram[1] = i;
    }
  }
  if (!mcdram[0] || !mcdram[1])
    return -1;

  return 0;
}

static int
hwloc_linux_knl_identify_8nodes(uint64_t *distances,
				struct knl_distances_summary *distsum,
				unsigned *ddr, unsigned *mcdram) /* ddr and mcdram arrays must be 4-long */
{
  uint64_t value;
  unsigned i, nb;

  hwloc_debug("Trying to identify 8 KNL NUMA nodes in SNC-4 cluster mode...\n");

  /* The SNC4-Flat/Hybrid matrix should be something like
   * 10 21 21 21 31 41 41 41
   * 21 10 21 21 41 31 41 41
   * 21 21 10 21 41 41 31 41
   * 21 21 21 10 41 41 41 31
   * 31 41 41 41 10 41 41 41
   * 41 31 41 41 41 10 41 41
   * 41 41 31 41 41 41 31 41
   * 41 41 41 31 41 41 41 41
   * which means there are (above 8*10 on the diagonal):
   * 4 identical values for DDR to local MCDRAM
   * 6 identical values for DDR to other DDR,
   * 18 identical values for everything else.
   */
  if (distsum->nb_values != 4
      || distsum->values[0].occurences != 4 /* DDR to local MCDRAM */
      || distsum->values[1].occurences != 6 /* DDR to DDR */
      || distsum->values[2].occurences != 8 /* local */
      || distsum->values[3].occurences != 18 /* others */ )
    return -1;

  /* DDR:0 is always first */
  ddr[0] = 0;
  hwloc_debug("  DDR#0 is NUMAnode#0\n");

  /* DDR:[1-3] are at distance distsum->values[1].value from ddr[0] */
  value = distsum->values[1].value;
  ddr[1] = ddr[2] = ddr[3] = 0;
  nb = 1;
  for(i=0; i<8; i++)
    if (distances[i] == value) {
      hwloc_debug("  DDR#%u is NUMAnode#%u\n", nb, i);
      ddr[nb++] = i;
      if (nb == 4)
	break;
    }
  if (nb != 4 || !ddr[1] || !ddr[2] || !ddr[3])
    return -1;

  /* MCDRAMs are at distance distsum->values[0].value from their local DDR */
  value = distsum->values[0].value;
  mcdram[0] = mcdram[1] = mcdram[2] = mcdram[3] = 0;
  for(i=1; i<8; i++) {
    if (distances[i] == value) {
      hwloc_debug("  MCDRAM#0 is NUMAnode#%u\n", i);
      mcdram[0] = i;
    } else if (distances[ddr[1]*8+i] == value) {
      hwloc_debug("  MCDRAM#1 is NUMAnode#%u\n", i);
      mcdram[1] = i;
    } else if (distances[ddr[2]*8+i] == value) {
      hwloc_debug("  MCDRAM#2 is NUMAnode#%u\n", i);
      mcdram[2] = i;
    } else if (distances[ddr[3]*8+i] == value) {
      hwloc_debug("  MCDRAM#3 is NUMAnode#%u\n", i);
      mcdram[3] = i;
    }
  }
  if (!mcdram[0] || !mcdram[1] || !mcdram[2] || !mcdram[3])
    return -1;

  return 0;
}

/* Try to handle knl hwdata properties
 * Returns 0 on success and -1 otherwise */
static int
hwloc_linux_knl_read_hwdata_properties(struct hwloc_linux_backend_data_s *data,
				       struct knl_hwdata *hwdata)
{
  char *knl_cache_file;
  int version = 0;
  char buffer[512] = {0};
  char *data_beg = NULL;

  if (asprintf(&knl_cache_file, "%s/knl_memoryside_cache", data->dumped_hwdata_dirname) < 0)
    return -1;

  hwloc_debug("Reading knl cache data from: %s\n", knl_cache_file);
  if (hwloc_read_path_by_length(knl_cache_file, buffer, sizeof(buffer), data->root_fd) < 0) {
    hwloc_debug("Unable to open KNL data file `%s' (%s)\n", knl_cache_file, strerror(errno));
    free(knl_cache_file);
    return -1;
  }
  free(knl_cache_file);

  data_beg = &buffer[0];

  /* file must start with version information */
  if (sscanf(data_beg, "version: %d", &version) != 1) {
    fprintf(stderr, "Invalid knl_memoryside_cache header, expected \"version: <int>\".\n");
    return -1;
  }

  while (1) {
    char *line_end = strstr(data_beg, "\n");
    if (!line_end)
        break;
    if (version >= 1) {
      if (!strncmp("cache_size:", data_beg, strlen("cache_size"))) {
          sscanf(data_beg, "cache_size: %lld", &hwdata->mcdram_cache_size);
          hwloc_debug("read cache_size=%lld\n", hwdata->mcdram_cache_size);
      } else if (!strncmp("line_size:", data_beg, strlen("line_size:"))) {
          sscanf(data_beg, "line_size: %d", &hwdata->mcdram_cache_line_size);
          hwloc_debug("read line_size=%d\n", hwdata->mcdram_cache_line_size);
      } else if (!strncmp("inclusiveness:", data_beg, strlen("inclusiveness:"))) {
          sscanf(data_beg, "inclusiveness: %d", &hwdata->mcdram_cache_inclusiveness);
          hwloc_debug("read inclusiveness=%d\n", hwdata->mcdram_cache_inclusiveness);
      } else if (!strncmp("associativity:", data_beg, strlen("associativity:"))) {
          sscanf(data_beg, "associativity: %d\n", &hwdata->mcdram_cache_associativity);
          hwloc_debug("read associativity=%d\n", hwdata->mcdram_cache_associativity);
      }
    }
    if (version >= 2) {
      if (!strncmp("cluster_mode: ", data_beg, strlen("cluster_mode: "))) {
	size_t length;
	data_beg += strlen("cluster_mode: ");
	length = line_end-data_beg;
	if (length > sizeof(hwdata->cluster_mode)-1)
	  length = sizeof(hwdata->cluster_mode)-1;
	memcpy(hwdata->cluster_mode, data_beg, length);
	hwdata->cluster_mode[length] = '\0';
        hwloc_debug("read cluster_mode=%s\n", hwdata->cluster_mode);
      } else if (!strncmp("memory_mode: ", data_beg, strlen("memory_mode: "))) {
	size_t length;
	data_beg += strlen("memory_mode: ");
	length = line_end-data_beg;
	if (length > sizeof(hwdata->memory_mode)-1)
	  length = sizeof(hwdata->memory_mode)-1;
	memcpy(hwdata->memory_mode, data_beg, length);
	hwdata->memory_mode[length] = '\0';
        hwloc_debug("read memory_mode=%s\n", hwdata->memory_mode);
      }
    }

    data_beg = line_end + 1;
  }

  if (hwdata->mcdram_cache_size == -1
      || hwdata->mcdram_cache_line_size == -1
      || hwdata->mcdram_cache_associativity == -1
      || hwdata->mcdram_cache_inclusiveness == -1) {
    hwloc_debug("Incorrect file format cache_size=%lld line_size=%d associativity=%d inclusiveness=%d\n",
		hwdata->mcdram_cache_size,
		hwdata->mcdram_cache_line_size,
		hwdata->mcdram_cache_associativity,
		hwdata->mcdram_cache_inclusiveness);
    hwdata->mcdram_cache_size = -1; /* mark cache as invalid */
  }

  return 0;
}

static void
hwloc_linux_knl_guess_hwdata_properties(struct knl_hwdata *hwdata,
					hwloc_obj_t *nodes, unsigned nbnodes,
					struct knl_distances_summary *distsum)
{
  /* Try to guess KNL configuration (Cluster mode, Memory mode, and MCDRAM cache info)
   * from the NUMA configuration (number of nodes, CPUless or not, distances).
   * Keep in mind that some CPUs might be offline (hence DDR could be CPUless too.
   * Keep in mind that part of the memory might be offline (hence MCDRAM could contain less than 16GB total).
   */

  hwloc_debug("Trying to guess missing KNL configuration information...\n");

  /* These MCDRAM cache attributes are always valid.
   * We'll only use them if mcdram_cache_size > 0
   */
  hwdata->mcdram_cache_associativity = 1;
  hwdata->mcdram_cache_inclusiveness = 1;
  hwdata->mcdram_cache_line_size = 64;
  /* All commercial KNL/KNM have 16GB of MCDRAM, we'll divide that in the number of SNC */

  if (hwdata->mcdram_cache_size > 0
      && hwdata->cluster_mode[0]
      && hwdata->memory_mode[0])
    /* Nothing to guess */
    return;

  /* Quadrant/All2All/Hemisphere are basically identical from the application point-of-view,
   * and Quadrant is recommended (except if underpopulating DIMMs).
   * Hence we'll assume Quadrant when unknown.
   */

  /* Flat/Hybrid25/Hybrid50 cannot be distinguished unless we know the Cache size
   * (if running a old hwloc-dump-hwdata that reports Cache size without modes)
   * or we're sure MCDRAM NUMAnode size was not decreased by offlining some memory.
   * Hence we'll assume Flat when unknown.
   */

  if (nbnodes == 1) {
    /* Quadrant-Cache */
    if (!hwdata->cluster_mode[0])
      strcpy(hwdata->cluster_mode, "Quadrant");
    if (!hwdata->memory_mode[0])
      strcpy(hwdata->memory_mode, "Cache");
    if (hwdata->mcdram_cache_size <= 0)
      hwdata->mcdram_cache_size = 16UL*1024*1024*1024;

  } else if (nbnodes == 2) {
    /* most likely Quadrant-Flat/Hybrid,
     * or SNC2/Cache (unlikely)
     */

    if (!strcmp(hwdata->memory_mode, "Cache")
	|| !strcmp(hwdata->cluster_mode, "SNC2")
	|| !hwloc_bitmap_iszero(nodes[1]->cpuset)) { /* MCDRAM cannot be nodes[0], and its cpuset is always empty */
      /* SNC2-Cache */
      if (!hwdata->cluster_mode[0])
	strcpy(hwdata->cluster_mode, "SNC2");
      if (!hwdata->memory_mode[0])
	strcpy(hwdata->memory_mode, "Cache");
      if (hwdata->mcdram_cache_size <= 0)
	hwdata->mcdram_cache_size = 8UL*1024*1024*1024;

    } else {
      /* Assume Quadrant-Flat/Hybrid.
       * Could also be SNC2-Cache with offline CPUs in nodes[1] (unlikely).
       */
      if (!hwdata->cluster_mode[0])
	strcpy(hwdata->cluster_mode, "Quadrant");
      if (!hwdata->memory_mode[0]) {
	if (hwdata->mcdram_cache_size == 4UL*1024*1024*1024)
	  strcpy(hwdata->memory_mode, "Hybrid25");
	else if (hwdata->mcdram_cache_size == 8UL*1024*1024*1024)
	  strcpy(hwdata->memory_mode, "Hybrid50");
	else
	  strcpy(hwdata->memory_mode, "Flat");
      } else {
	if (hwdata->mcdram_cache_size <= 0) {
	  if (!strcmp(hwdata->memory_mode, "Hybrid25"))
	    hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
	  else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
	    hwdata->mcdram_cache_size = 8UL*1024*1024*1024;
	}
      }
    }

  } else if (nbnodes == 4) {
    /* most likely SNC4-Cache
     * or SNC2-Flat/Hybrid (unlikely)
     *
     * SNC2-Flat/Hybrid has 4 different values in distsum,
     * while SNC4-Cache only has 2.
     */

    if (!strcmp(hwdata->cluster_mode, "SNC2") || distsum->nb_values == 4) {
      /* SNC2-Flat/Hybrid */
      if (!hwdata->cluster_mode[0])
	strcpy(hwdata->cluster_mode, "SNC2");
      if (!hwdata->memory_mode[0]) {
	if (hwdata->mcdram_cache_size == 2UL*1024*1024*1024)
	  strcpy(hwdata->memory_mode, "Hybrid25");
	else if (hwdata->mcdram_cache_size == 4UL*1024*1024*1024)
	  strcpy(hwdata->memory_mode, "Hybrid50");
	else
	  strcpy(hwdata->memory_mode, "Flat");
      } else {
	if (hwdata->mcdram_cache_size <= 0) {
	  if (!strcmp(hwdata->memory_mode, "Hybrid25"))
	    hwdata->mcdram_cache_size = 2UL*1024*1024*1024;
	  else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
	    hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
	}
      }

    } else {
      /* Assume SNC4-Cache.
       * SNC2 is unlikely.
       */
      if (!hwdata->cluster_mode[0])
	strcpy(hwdata->cluster_mode, "SNC4");
      if (!hwdata->memory_mode[0])
	strcpy(hwdata->memory_mode, "Cache");
      if (hwdata->mcdram_cache_size <= 0)
	hwdata->mcdram_cache_size = 4UL*1024*1024*1024;
    }

  } else if (nbnodes == 8) {
    /* SNC4-Flat/Hybrid */

    if (!hwdata->cluster_mode[0])
      strcpy(hwdata->cluster_mode, "SNC4");
    if (!hwdata->memory_mode[0]) {
      if (hwdata->mcdram_cache_size == 1UL*1024*1024*1024)
	strcpy(hwdata->memory_mode, "Hybrid25");
      else if (hwdata->mcdram_cache_size == 2UL*1024*1024*1024)
	strcpy(hwdata->memory_mode, "Hybrid50");
      else
	strcpy(hwdata->memory_mode, "Flat");
    } else {
      if (hwdata->mcdram_cache_size <= 0) {
	if (!strcmp(hwdata->memory_mode, "Hybrid25"))
	  hwdata->mcdram_cache_size = 1UL*1024*1024*1024;
	else if (!strcmp(hwdata->memory_mode, "Hybrid50"))
	  hwdata->mcdram_cache_size = 2UL*1024*1024*1024;
      }
    }
  }

  hwloc_debug("  Found cluster=%s memory=%s cache=%lld\n",
	      hwdata->cluster_mode, hwdata->memory_mode,
	      hwdata->mcdram_cache_size);
}

static void
hwloc_linux_knl_add_cluster(struct hwloc_topology *topology,
			    hwloc_obj_t ddr, hwloc_obj_t mcdram,
			    struct knl_hwdata *knl_hwdata,
			    int mscache_as_l3,
			    unsigned *failednodes)
{
  hwloc_obj_t cluster = NULL;

  if (mcdram) {
    mcdram->subtype = strdup("MCDRAM");
    /* Change MCDRAM cpuset to DDR cpuset for clarity.
     * Not actually useful if we insert with hwloc__attach_memory_object() below.
     * The cpuset will be updated by the core later anyway.
     */
    hwloc_bitmap_copy(mcdram->cpuset, ddr->cpuset);

    /* Add a Group for Cluster containing this MCDRAM + DDR */
    cluster = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, HWLOC_UNKNOWN_INDEX);
    hwloc_obj_add_other_obj_sets(cluster, ddr);
    hwloc_obj_add_other_obj_sets(cluster, mcdram);
    cluster->subtype = strdup("Cluster");
    cluster->attr->group.kind = HWLOC_GROUP_KIND_INTEL_KNL_SUBNUMA_CLUSTER;
    cluster = hwloc__insert_object_by_cpuset(topology, NULL, cluster, hwloc_report_os_error);
  }

  if (cluster) {
    /* Now insert NUMA nodes below this cluster */
    hwloc_obj_t res;
    res = hwloc__attach_memory_object(topology, cluster, ddr, hwloc_report_os_error);
    if (res != ddr) {
      (*failednodes)++;
      ddr = NULL;
    }
    res = hwloc__attach_memory_object(topology, cluster, mcdram, hwloc_report_os_error);
    if (res != mcdram)
      (*failednodes)++;

  } else {
    /* we don't know where to attach, let the core find or insert if needed */
    hwloc_obj_t res;
    res = hwloc__insert_object_by_cpuset(topology, NULL, ddr, hwloc_report_os_error);
    if (res != ddr) {
      (*failednodes)++;
      ddr = NULL;
    }
    if (mcdram) {
      res = hwloc__insert_object_by_cpuset(topology, NULL, mcdram, hwloc_report_os_error);
      if (res != mcdram)
	(*failednodes)++;
    }
  }

  if (ddr && knl_hwdata->mcdram_cache_size > 0) {
    /* Now insert the mscache if any */
    hwloc_obj_t cache = hwloc_alloc_setup_object(topology, HWLOC_OBJ_L3CACHE, HWLOC_UNKNOWN_INDEX);
    if (!cache)
      /* failure is harmless */
      return;
    cache->attr->cache.depth = 3;
    cache->attr->cache.type = HWLOC_OBJ_CACHE_UNIFIED;
    cache->attr->cache.size = knl_hwdata->mcdram_cache_size;
    cache->attr->cache.linesize = knl_hwdata->mcdram_cache_line_size;
    cache->attr->cache.associativity = knl_hwdata->mcdram_cache_associativity;
    hwloc_obj_add_info(cache, "Inclusive", knl_hwdata->mcdram_cache_inclusiveness ? "1" : "0");
    cache->cpuset = hwloc_bitmap_dup(ddr->cpuset);
    cache->nodeset = hwloc_bitmap_dup(ddr->nodeset); /* only applies to DDR */
    if (mscache_as_l3) {
      /* make it a L3 */
      cache->subtype = strdup("MemorySideCache");
      hwloc_insert_object_by_cpuset(topology, cache);
    } else {
      /* make it a real mscache */
      cache->type = HWLOC_OBJ_MEMCACHE;
      if (cluster)
	hwloc__attach_memory_object(topology, cluster, cache, hwloc_report_os_error);
      else
	hwloc__insert_object_by_cpuset(topology, NULL, cache, hwloc_report_os_error);
    }
  }
}

static void
hwloc_linux_knl_numa_quirk(struct hwloc_topology *topology,
			   struct hwloc_linux_backend_data_s *data,
			   hwloc_obj_t *nodes, unsigned nbnodes,
			   uint64_t * distances,
			   unsigned *failednodes)
{
  struct knl_hwdata hwdata;
  struct knl_distances_summary dist;
  unsigned i;
  char * fallback_env = getenv("HWLOC_KNL_HDH_FALLBACK");
  int fallback = fallback_env ? atoi(fallback_env) : -1; /* by default, only fallback if needed */
  char * mscache_as_l3_env = getenv("HWLOC_KNL_MSCACHE_L3");
  int mscache_as_l3 = mscache_as_l3_env ? atoi(mscache_as_l3_env) : 1; /* L3 by default, for backward compat */

  if (*failednodes)
    goto error;

  if (hwloc_linux_knl_parse_numa_distances(nbnodes, distances, &dist) < 0)
    goto error;

  hwdata.memory_mode[0] = '\0';
  hwdata.cluster_mode[0] = '\0';
  hwdata.mcdram_cache_size = -1;
  hwdata.mcdram_cache_associativity = -1;
  hwdata.mcdram_cache_inclusiveness = -1;
  hwdata.mcdram_cache_line_size = -1;
  if (fallback == 1)
    hwloc_debug("KNL dumped hwdata ignored, forcing fallback to heuristics\n");
  else
    hwloc_linux_knl_read_hwdata_properties(data, &hwdata);
  if (fallback != 0)
    hwloc_linux_knl_guess_hwdata_properties(&hwdata, nodes, nbnodes, &dist);

  if (strcmp(hwdata.cluster_mode, "All2All")
      && strcmp(hwdata.cluster_mode, "Hemisphere")
      && strcmp(hwdata.cluster_mode, "Quadrant")
      && strcmp(hwdata.cluster_mode, "SNC2")
      && strcmp(hwdata.cluster_mode, "SNC4")) {
    fprintf(stderr, "Failed to find a usable KNL cluster mode (%s)\n", hwdata.cluster_mode);
    goto error;
  }
  if (strcmp(hwdata.memory_mode, "Cache")
      && strcmp(hwdata.memory_mode, "Flat")
      && strcmp(hwdata.memory_mode, "Hybrid25")
      && strcmp(hwdata.memory_mode, "Hybrid50")) {
    fprintf(stderr, "Failed to find a usable KNL memory mode (%s)\n", hwdata.memory_mode);
    goto error;
  }

  if (mscache_as_l3) {
    if (!hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_L3CACHE))
      hwdata.mcdram_cache_size = 0;
  } else {
    if (!hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_MEMCACHE))
      hwdata.mcdram_cache_size = 0;
  }

  hwloc_obj_add_info(topology->levels[0][0], "ClusterMode", hwdata.cluster_mode);
  hwloc_obj_add_info(topology->levels[0][0], "MemoryMode", hwdata.memory_mode);

  if (!strcmp(hwdata.cluster_mode, "All2All")
      || !strcmp(hwdata.cluster_mode, "Hemisphere")
      || !strcmp(hwdata.cluster_mode, "Quadrant")) {
    if (!strcmp(hwdata.memory_mode, "Cache")) {
      /* Quadrant-Cache */
      if (nbnodes != 1) {
	fprintf(stderr, "Found %u NUMA nodes instead of 1 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);

    } else {
      /* Quadrant-Flat/Hybrid */
      if (nbnodes != 2) {
	fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      if (!strcmp(hwdata.memory_mode, "Flat"))
	hwdata.mcdram_cache_size = 0;
      hwloc_linux_knl_add_cluster(topology, nodes[0], nodes[1], &hwdata, mscache_as_l3, failednodes);
    }

  } else if (!strcmp(hwdata.cluster_mode, "SNC2")) {
    if (!strcmp(hwdata.memory_mode, "Cache")) {
      /* SNC2-Cache */
      if (nbnodes != 2) {
	fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[1], NULL, &hwdata, mscache_as_l3, failednodes);

    } else {
      /* SNC2-Flat/Hybrid */
      unsigned ddr[2], mcdram[2];
      if (nbnodes != 4) {
	fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      if (hwloc_linux_knl_identify_4nodes(distances, &dist, ddr, mcdram) < 0) {
	fprintf(stderr, "Unexpected distance layout for mode %s-%s\n", hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      if (!strcmp(hwdata.memory_mode, "Flat"))
	hwdata.mcdram_cache_size = 0;
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[0]], nodes[mcdram[0]], &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[1]], nodes[mcdram[1]], &hwdata, mscache_as_l3, failednodes);
    }

  } else if (!strcmp(hwdata.cluster_mode, "SNC4")) {
    if (!strcmp(hwdata.memory_mode, "Cache")) {
      /* SNC4-Cache */
      if (nbnodes != 4) {
	fprintf(stderr, "Found %u NUMA nodes instead of 4 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      hwloc_linux_knl_add_cluster(topology, nodes[0], NULL, &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[1], NULL, &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[2], NULL, &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[3], NULL, &hwdata, mscache_as_l3, failednodes);

    } else {
      /* SNC4-Flat/Hybrid */
      unsigned ddr[4], mcdram[4];
      if (nbnodes != 8) {
	fprintf(stderr, "Found %u NUMA nodes instead of 2 in mode %s-%s\n", nbnodes, hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      if (hwloc_linux_knl_identify_8nodes(distances, &dist, ddr, mcdram) < 0) {
	fprintf(stderr, "Unexpected distance layout for mode %s-%s\n", hwdata.cluster_mode, hwdata.memory_mode);
	goto error;
      }
      if (!strcmp(hwdata.memory_mode, "Flat"))
	hwdata.mcdram_cache_size = 0;
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[0]], nodes[mcdram[0]], &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[1]], nodes[mcdram[1]], &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[2]], nodes[mcdram[2]], &hwdata, mscache_as_l3, failednodes);
      hwloc_linux_knl_add_cluster(topology, nodes[ddr[3]], nodes[mcdram[3]], &hwdata, mscache_as_l3, failednodes);
    }
  }

  return;

 error:
  /* just insert nodes basically */
  for (i = 0; i < nbnodes; i++) {
    hwloc_obj_t node = nodes[i];
    if (node) {
      hwloc_obj_t res_obj = hwloc__insert_object_by_cpuset(topology, NULL, node, hwloc_report_os_error);
      if (res_obj != node)
	/* This NUMA node got merged somehow, could be a buggy BIOS reporting wrong NUMA node cpuset.
	 * This object disappeared, we'll ignore distances */
	(*failednodes)++;
    }
  }
}


/**************************************
 ****** Sysfs Topology Discovery ******
 **************************************/

/* try to find locality of CPU-less NUMA nodes by looking at their distances */
static int
fixup_cpuless_node_locality_from_distances(unsigned i,
					   unsigned nbnodes, hwloc_obj_t *nodes, uint64_t *distances)
{
  unsigned min = UINT_MAX;
  unsigned nb = 0, j;

  for(j=0; j<nbnodes; j++) {
    if (j==i || !nodes[j])
      continue;
    if (distances[i*nbnodes+j] < min) {
      min = distances[i*nbnodes+j];
      nb = 1;
    } else if (distances[i*nbnodes+j] == min) {
      nb++;
    }
  }

  if (min <= distances[i*nbnodes+i] || min == UINT_MAX || nb == nbnodes-1)
    return -1;

  /* not local, but closer to *some* other nodes */
  for(j=0; j<nbnodes; j++)
    if (j!=i && nodes[j] && distances[i*nbnodes+j] == min)
      hwloc_bitmap_or(nodes[i]->cpuset, nodes[i]->cpuset, nodes[j]->cpuset);
  return 0;
}

/* try to find locality of CPU-less NUMA nodes by looking at HMAT initiators.
 *
 * In theory, we may have HMAT info only for some nodes.
 * In practice, if this never occurs, we may want to assume HMAT for either all or no nodes.
 */
static int
read_node_initiators(struct hwloc_linux_backend_data_s *data,
		     hwloc_obj_t node, unsigned nbnodes, hwloc_obj_t *nodes,
		     const char *path)
{
  char accesspath[SYSFS_NUMA_NODE_PATH_LEN];
  DIR *dir;
  struct dirent *dirent;

  sprintf(accesspath, "%s/node%u/access0/initiators", path, node->os_index);
  dir = hwloc_opendir(accesspath, data->root_fd);
  if (!dir)
    return -1;

  while ((dirent = readdir(dir)) != NULL) {
    unsigned initiator_os_index;
    if (sscanf(dirent->d_name, "node%u", &initiator_os_index) == 1
	&& initiator_os_index != node->os_index) {
      /* we found an initiator that's not ourself,
       * find the corresponding node and add its cpuset
       */
      unsigned j;
      for(j=0; j<nbnodes; j++)
	if (nodes[j] && nodes[j]->os_index == initiator_os_index) {
	  hwloc_bitmap_or(node->cpuset, node->cpuset, nodes[j]->cpuset);
	  break;
	}
    }
  }
  closedir(dir);
  return 0;
}

/* return -1 if the kernel doesn't support mscache,
 * or update tree (containing only the node on input) with caches (if any)
 */
static int
read_node_mscaches(struct hwloc_topology *topology,
		   struct hwloc_linux_backend_data_s *data,
		   const char *path,
		   hwloc_obj_t *treep)
{
  hwloc_obj_t tree = *treep, node = tree;
  unsigned osnode = node->os_index;
  char mscpath[SYSFS_NUMA_NODE_PATH_LEN];
  DIR *mscdir;
  struct dirent *dirent;

  sprintf(mscpath, "%s/node%u/memory_side_cache", path, osnode);
  mscdir = hwloc_opendir(mscpath, data->root_fd);
  if (!mscdir)
    return -1;

  while ((dirent = readdir(mscdir)) != NULL) {
    unsigned depth;
    uint64_t size;
    unsigned line_size;
    unsigned associativity;
    hwloc_obj_t cache;

    if (strncmp(dirent->d_name, "index", 5))
      continue;

    depth = atoi(dirent->d_name+5);

    sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/size", path, osnode, depth);
    if (hwloc_read_path_as_uint64(mscpath, &size, data->root_fd) < 0)
      continue;

    sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/line_size", path, osnode, depth);
    if (hwloc_read_path_as_uint(mscpath, &line_size, data->root_fd) < 0)
      continue;

    sprintf(mscpath, "%s/node%u/memory_side_cache/index%u/indexing", path, osnode, depth);
    if (hwloc_read_path_as_uint(mscpath, &associativity, data->root_fd) < 0)
      continue;
    /* 0 for direct-mapped, 1 for indexed (don't know how many ways), 2 for custom/other */

    cache = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MEMCACHE, HWLOC_UNKNOWN_INDEX);
    if (cache) {
      cache->nodeset = hwloc_bitmap_dup(node->nodeset);
      cache->cpuset = hwloc_bitmap_dup(node->cpuset);
      cache->attr->cache.size = size;
      cache->attr->cache.depth = depth;
      cache->attr->cache.linesize = line_size;
      cache->attr->cache.type = HWLOC_OBJ_CACHE_UNIFIED;
      cache->attr->cache.associativity = !associativity ? 1 /* direct-mapped */ : 0 /* unknown */;
      hwloc_debug_1arg_bitmap("mscache %s has nodeset %s\n",
			      dirent->d_name, cache->nodeset);

      cache->memory_first_child = tree;
      tree = cache;
    }
  }
  closedir(mscdir);
  *treep = tree;
  return 0;
}

static unsigned *
list_sysfsnode(struct hwloc_topology *topology,
	       struct hwloc_linux_backend_data_s *data,
	       const char *path,
	       unsigned *nbnodesp)
{
  DIR *dir;
  unsigned osnode, nbnodes = 0;
  unsigned *indexes, index_;
  hwloc_bitmap_t nodeset;
  struct dirent *dirent;

  /* try to get the list of NUMA nodes at once.
   * otherwise we'll list the entire directory.
   *
   * offline nodes don't exist at all under /sys (they are in "possible", we may ignore them).
   *
   * don't use <path>/online, /sys/bus/node/devices only contains node%d
   */
  nodeset = hwloc__alloc_read_path_as_cpulist("/sys/devices/system/node/online", data->root_fd);
  if (nodeset) {
    int _nbnodes = hwloc_bitmap_weight(nodeset);
    assert(_nbnodes >= 1);
    nbnodes = (unsigned)_nbnodes;
    hwloc_debug_bitmap("possible NUMA nodes %s\n", nodeset);
    goto found;
  }

  /* Get the list of nodes first */
  dir = hwloc_opendir(path, data->root_fd);
  if (!dir)
    return NULL;

  nodeset = hwloc_bitmap_alloc();
  if (!nodeset) {
    closedir(dir);
    return NULL;
  }

  while ((dirent = readdir(dir)) != NULL) {
    if (strncmp(dirent->d_name, "node", 4))
      continue;
    osnode = strtoul(dirent->d_name+4, NULL, 0);
    hwloc_bitmap_set(nodeset, osnode);
    nbnodes++;
  }
  closedir(dir);

  assert(nbnodes >= 1); /* linux cannot have a "node/" subdirectory without at least one "node%d" */

  /* we don't know if sysfs returns nodes in order, we can't merge above and below loops */

 found:
  /* if there are already some nodes, we'll annotate them. make sure the indexes match */
  if (!hwloc_bitmap_iszero(topology->levels[0][0]->nodeset)
      && !hwloc_bitmap_isequal(nodeset, topology->levels[0][0]->nodeset)) {
    char *sn, *tn;
    hwloc_bitmap_asprintf(&sn, nodeset);
    hwloc_bitmap_asprintf(&tn, topology->levels[0][0]->nodeset);
    fprintf(stderr, "linux/sysfs: ignoring nodes because nodeset %s doesn't match existing nodeset %s.\n", tn, sn);
    free(sn);
    free(tn);
    hwloc_bitmap_free(nodeset);
    return NULL;
  }

  indexes = calloc(nbnodes, sizeof(*indexes));
  if (!indexes) {
    hwloc_bitmap_free(nodeset);
    return NULL;
  }

  /* Unsparsify node indexes.
   * We'll need them later because Linux groups sparse distances
   * and keeps them in order in the sysfs distance files.
   * It'll simplify things in the meantime.
   */
  index_ = 0;
  hwloc_bitmap_foreach_begin (osnode, nodeset) {
    indexes[index_] = osnode;
    index_++;
  } hwloc_bitmap_foreach_end();

  hwloc_bitmap_free(nodeset);

#ifdef HWLOC_DEBUG
  hwloc_debug("%s", "NUMA indexes: ");
  for (index_ = 0; index_ < nbnodes; index_++)
    hwloc_debug(" %u", indexes[index_]);
  hwloc_debug("%s", "\n");
#endif

  *nbnodesp = nbnodes;
  return indexes;
}

static int
annotate_sysfsnode(struct hwloc_topology *topology,
		   struct hwloc_linux_backend_data_s *data,
		   const char *path, unsigned *found)
{
  unsigned nbnodes;
  hwloc_obj_t * nodes; /* the array of NUMA node objects, to be used for inserting distances */
  hwloc_obj_t node;
  unsigned * indexes;
  uint64_t * distances;
  unsigned i;

  /* NUMA nodes cannot be filtered out */
  indexes = list_sysfsnode(topology, data, path, &nbnodes);
  if (!indexes)
    return 0;

  nodes = calloc(nbnodes, sizeof(hwloc_obj_t));
  distances = malloc(nbnodes*nbnodes*sizeof(*distances));
  if (NULL == nodes || NULL == distances) {
    free(nodes);
    free(indexes);
    free(distances);
    return 0;
  }

  for(node=hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, NULL);
      node != NULL;
      node = hwloc_get_next_obj_by_type(topology, HWLOC_OBJ_NUMANODE, node)) {
    assert(node); /* list_sysfsnode() ensured that sysfs nodes and existing nodes match */

    /* hwloc_parse_nodes_distances() requires nodes in physical index order,
     * and inserting distances requires nodes[] and indexes[] in same order.
     */
    for(i=0; i<nbnodes; i++)
      if (indexes[i] == node->os_index) {
	nodes[i] = node;
	break;
      }

    hwloc_get_sysfs_node_meminfo(data, path, node->os_index, &node->attr->numanode);
  }

  topology->support.discovery->numa = 1;
  topology->support.discovery->numa_memory = 1;
  topology->support.discovery->disallowed_numa = 1;

  if (nbnodes >= 2
      && data->use_numa_distances
      && !hwloc_parse_nodes_distances(path, nbnodes, indexes, distances, data->root_fd)) {
    hwloc_internal_distances_add(topology, "NUMALatency", nbnodes, nodes, distances,
				 HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_MEANS_LATENCY,
				 HWLOC_DISTANCES_ADD_FLAG_GROUP);
  } else {
    free(nodes);
    free(distances);
  }

  free(indexes);
  *found = nbnodes;
  return 0;
}

static int
look_sysfsnode(struct hwloc_topology *topology,
	       struct hwloc_linux_backend_data_s *data,
	       const char *path, unsigned *found)
{
  unsigned osnode;
  unsigned nbnodes;
  hwloc_obj_t * nodes; /* the array of NUMA node objects, to be used for inserting distances */
  unsigned nr_trees;
  hwloc_obj_t * trees; /* the array of memory hierarchies to insert */
  unsigned *indexes;
  uint64_t * distances;
  hwloc_bitmap_t nodes_cpuset;
  unsigned failednodes = 0;
  unsigned i;
  DIR *dir;
  int allow_overlapping_node_cpusets = (getenv("HWLOC_DEBUG_ALLOW_OVERLAPPING_NODE_CPUSETS") != NULL);
  int need_memcaches = hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_MEMCACHE);

  /* NUMA nodes cannot be filtered out */
  indexes = list_sysfsnode(topology, data, path, &nbnodes);
  if (!indexes)
    return 0;

  nodes = calloc(nbnodes, sizeof(hwloc_obj_t));
  trees = calloc(nbnodes, sizeof(hwloc_obj_t));
  distances = malloc(nbnodes*nbnodes*sizeof(*distances));
  nodes_cpuset  = hwloc_bitmap_alloc();
  if (NULL == nodes || NULL == trees || NULL == distances || NULL == nodes_cpuset) {
    free(nodes);
    free(trees);
    free(indexes);
    free(distances);
    hwloc_bitmap_free(nodes_cpuset);
    nbnodes = 0;
    goto out;
  }

  topology->support.discovery->numa = 1;
  topology->support.discovery->numa_memory = 1;
  topology->support.discovery->disallowed_numa = 1;

  /* Create NUMA objects */
  for (i = 0; i < nbnodes; i++) {
    hwloc_obj_t node;
    char nodepath[SYSFS_NUMA_NODE_PATH_LEN];
    hwloc_bitmap_t cpuset;

    osnode = indexes[i];
    sprintf(nodepath, "%s/node%u/cpumap", path, osnode);
    cpuset = hwloc__alloc_read_path_as_cpumask(nodepath, data->root_fd);
    if (!cpuset) {
      /* This NUMA object won't be inserted, we'll ignore distances */
      failednodes++;
      continue;
    }
    if (hwloc_bitmap_intersects(nodes_cpuset, cpuset)) {
      /* Buggy BIOS with overlapping NUMA node cpusets, impossible on Linux so far, we should ignore them.
       * But it may be useful for debugging the core.
       */
      if (!allow_overlapping_node_cpusets) {
	hwloc_debug_1arg_bitmap("node P#%u cpuset %s intersects with previous nodes, ignoring that node.\n", osnode, cpuset);
	hwloc_bitmap_free(cpuset);
	failednodes++;
	continue;
      }
      fprintf(stderr, "node P#%u cpuset intersects with previous nodes, forcing its acceptance\n", osnode);
    }
    hwloc_bitmap_or(nodes_cpuset, nodes_cpuset, cpuset);

    node = hwloc_alloc_setup_object(topology, HWLOC_OBJ_NUMANODE, osnode);
    node->cpuset = cpuset;
    node->nodeset = hwloc_bitmap_alloc();
    hwloc_bitmap_set(node->nodeset, osnode);
    hwloc_get_sysfs_node_meminfo(data, path, osnode, &node->attr->numanode);

    nodes[i] = node;
    hwloc_debug_1arg_bitmap("os node %u has cpuset %s\n",
			  osnode, node->cpuset);
  }

      /* try to find NUMA nodes that correspond to NVIDIA GPU memory */
      dir = hwloc_opendir("/proc/driver/nvidia/gpus", data->root_fd);
      if (dir) {
	struct dirent *dirent;
	char *env = getenv("HWLOC_KEEP_NVIDIA_GPU_NUMA_NODES");
	int keep = env && atoi(env);
	while ((dirent = readdir(dir)) != NULL) {
	  char nvgpunumapath[300], line[256];
	  int fd;
	  snprintf(nvgpunumapath, sizeof(nvgpunumapath), "/proc/driver/nvidia/gpus/%s/numa_status", dirent->d_name);
	  fd = hwloc_open(nvgpunumapath, data->root_fd);
	  if (fd >= 0) {
	    int ret;
	    ret = read(fd, line, sizeof(line)-1);
	    line[sizeof(line)-1] = '\0';
	    if (ret >= 0) {
	      const char *nvgpu_node_line = strstr(line, "Node:");
	      if (nvgpu_node_line) {
		unsigned nvgpu_node;
		const char *value = nvgpu_node_line+5;
		while (*value == ' ' || *value == '\t')
		  value++;
		nvgpu_node = atoi(value);
		hwloc_debug("os node %u is NVIDIA GPU %s integrated memory\n", nvgpu_node, dirent->d_name);
		for(i=0; i<nbnodes; i++) {
		  hwloc_obj_t node = nodes[i];
		  if (node && node->os_index == nvgpu_node) {
		    if (keep) {
		      /* keep this NUMA node but fixed its locality and add an info about the GPU */
		      char nvgpulocalcpuspath[300];
		      int err;
		      node->subtype = strdup("GPUMemory");
		      hwloc_obj_add_info(node, "PCIBusID", dirent->d_name);
		      snprintf(nvgpulocalcpuspath, sizeof(nvgpulocalcpuspath), "/sys/bus/pci/devices/%s/local_cpus", dirent->d_name);
		      err = hwloc__read_path_as_cpumask(nvgpulocalcpuspath, node->cpuset, data->root_fd);
		      if (err)
			/* the core will attach to the root */
			hwloc_bitmap_zero(node->cpuset);
		    } else {
		      /* drop this NUMA node */
		      hwloc_free_unlinked_object(node);
		      nodes[i] = NULL;
		    }
		    break;
		  }
		}
	      }
	    }
	    close(fd);
	  }
	}
	closedir(dir);
      }

      /* try to find DAX devices of KMEM NUMA nodes */
      dir = hwloc_opendir("/sys/bus/dax/devices/", data->root_fd);
      if (dir) {
	struct dirent *dirent;
	while ((dirent = readdir(dir)) != NULL) {
	  char daxpath[300];
	  int tmp;
	  osnode = (unsigned) -1;
	  snprintf(daxpath, sizeof(daxpath), "/sys/bus/dax/devices/%s/target_node", dirent->d_name);
	  if (!hwloc_read_path_as_int(daxpath, &tmp, data->root_fd)) { /* contains %d when added in 5.1 */
	    osnode = (unsigned) tmp;
	    for(i=0; i<nbnodes; i++) {
	      hwloc_obj_t node = nodes[i];
	      if (node && node->os_index == osnode)
		hwloc_obj_add_info(node, "DAXDevice", dirent->d_name);
	    }
	  }
	}
	closedir(dir);
      }

      topology->support.discovery->numa = 1;
      topology->support.discovery->numa_memory = 1;
      topology->support.discovery->disallowed_numa = 1;

      hwloc_bitmap_free(nodes_cpuset);

      if (nbnodes <= 1) {
	/* failed to read/create some nodes, don't bother reading/fixing
	 * a distance matrix that would likely be wrong anyway.
	 */
	data->use_numa_distances = 0;
      }

      if (!data->use_numa_distances) {
	free(distances);
	distances = NULL;
      }

      if (distances && hwloc_parse_nodes_distances(path, nbnodes, indexes, distances, data->root_fd) < 0) {
	free(distances);
	distances = NULL;
      }

      free(indexes);

      if (data->is_knl) {
	/* apply KNL quirks */
	char *env = getenv("HWLOC_KNL_NUMA_QUIRK");
	int noquirk = (env && !atoi(env));
	if (!noquirk) {
	  hwloc_linux_knl_numa_quirk(topology, data, nodes, nbnodes, distances, &failednodes);
	  free(distances);
	  free(nodes);
	  free(trees);
	  goto out;
	}
      }

      /* Fill the array of trees */
      nr_trees = 0;
      /* First list nodes that have a non-empty cpumap.
       * They are likely the default nodes where we want to allocate from (DDR),
       * make sure they are listed first in their parent memory subtree.
       */
      for (i = 0; i < nbnodes; i++) {
	hwloc_obj_t node = nodes[i];
	if (node && !hwloc_bitmap_iszero(node->cpuset)) {
	  hwloc_obj_t tree;
	  /* update from HMAT initiators if any */
	  if (data->use_numa_initiators)
	    read_node_initiators(data, node, nbnodes, nodes, path);

	  tree = node;
	  if (need_memcaches)
	    read_node_mscaches(topology, data, path, &tree);
	  trees[nr_trees++] = tree;
	}
      }
      /* Now look for empty-cpumap nodes.
       * Those may be the non-default nodes for allocation.
       * Hence we don't want them to be listed first,
       * especially if we end up fixing their actual cpumap.
       */
      for (i = 0; i < nbnodes; i++) {
	hwloc_obj_t node = nodes[i];
	if (node && hwloc_bitmap_iszero(node->cpuset)) {
	  hwloc_obj_t tree;
	  /* update from HMAT initiators if any */
	  if (data->use_numa_initiators)
	    if (!read_node_initiators(data, node, nbnodes, nodes, path))
	      if (!hwloc_bitmap_iszero(node->cpuset))
		goto fixed;

	  /* if HMAT didn't help, try to find locality of CPU-less NUMA nodes by looking at their distances */
	  if (distances && data->use_numa_distances_for_cpuless)
	    fixup_cpuless_node_locality_from_distances(i, nbnodes, nodes, distances);

	fixed:
	  tree = node;
	  if (need_memcaches)
	    read_node_mscaches(topology, data, path, &tree);
	  trees[nr_trees++] = tree;
	}
      }

      /* insert memory trees for real */
      for (i = 0; i < nr_trees; i++) {
	hwloc_obj_t tree = trees[i];
	while (tree) {
	  hwloc_obj_t cur_obj;
	  hwloc_obj_t res_obj;
	  hwloc_obj_type_t cur_type;
	  cur_obj = tree;
	  cur_type = cur_obj->type;
	  tree = cur_obj->memory_first_child;
	  assert(!cur_obj->next_sibling);
	  res_obj = hwloc__insert_object_by_cpuset(topology, NULL, cur_obj, hwloc_report_os_error);
	  if (res_obj != cur_obj && cur_type == HWLOC_OBJ_NUMANODE) {
	    /* This NUMA node got merged somehow, could be a buggy BIOS reporting wrong NUMA node cpuset.
	     * Update it in the array for the distance matrix. */
	    unsigned j;
	    for(j=0; j<nbnodes; j++)
	      if (nodes[j] == cur_obj)
		nodes[j] = res_obj;
	    failednodes++;
	  }
	}
      }
      free(trees);

      /* Inserted distances now that nodes are properly inserted */
      if (distances)
	hwloc_internal_distances_add(topology, "NUMALatency", nbnodes, nodes, distances,
				     HWLOC_DISTANCES_KIND_FROM_OS|HWLOC_DISTANCES_KIND_MEANS_LATENCY,
				     HWLOC_DISTANCES_ADD_FLAG_GROUP);
      else
	free(nodes);

 out:
  *found = nbnodes - failednodes;
  return 0;
}

/* Look at Linux' /sys/devices/system/cpu/cpu%d/topology/ */
static int
look_sysfscpu(struct hwloc_topology *topology,
	      struct hwloc_linux_backend_data_s *data,
	      const char *path, int old_filenames,
	      struct hwloc_linux_cpuinfo_proc * cpuinfo_Lprocs, unsigned cpuinfo_numprocs)
{
  hwloc_bitmap_t cpuset; /* Set of cpus for which we have topology information */
  hwloc_bitmap_t online_set; /* Set of online CPUs if easily available, or NULL */
#define CPU_TOPOLOGY_STR_LEN 128
  char str[CPU_TOPOLOGY_STR_LEN];
  DIR *dir;
  int i,j;
  unsigned caches_added;
  int threadwithcoreid = data->is_amd_with_CU ? -1 : 0; /* -1 means we don't know yet if threads have their own coreids within thread_siblings */

  /* try to get the list of online CPUs at once.
   * otherwise we'll use individual per-CPU "online" files.
   *
   * don't use <path>/online, /sys/bus/cpu/devices only contains cpu%d
   */
  online_set = hwloc__alloc_read_path_as_cpulist("/sys/devices/system/cpu/online", data->root_fd);
  if (online_set)
    hwloc_debug_bitmap("online CPUs %s\n", online_set);

  /* fill the cpuset of interesting cpus */
  dir = hwloc_opendir(path, data->root_fd);
  if (!dir) {
    hwloc_bitmap_free(online_set);
    return -1;
  } else {
    struct dirent *dirent;
    cpuset = hwloc_bitmap_alloc();

    while ((dirent = readdir(dir)) != NULL) {
      unsigned long cpu;
      char online[2];

      if (strncmp(dirent->d_name, "cpu", 3))
	continue;
      cpu = strtoul(dirent->d_name+3, NULL, 0);

      /* Maybe we don't have topology information but at least it exists */
      hwloc_bitmap_set(topology->levels[0][0]->complete_cpuset, cpu);

      /* check whether this processor is online */
      if (online_set) {
	if (!hwloc_bitmap_isset(online_set, cpu)) {
	  hwloc_debug("os proc %lu is offline\n", cpu);
	  continue;
	}
      } else {
	/* /sys/devices/system/cpu/online unavailable, check the cpu online file */
	sprintf(str, "%s/cpu%lu/online", path, cpu);
	if (hwloc_read_path_by_length(str, online, sizeof(online), data->root_fd) == 0) {
	  if (!atoi(online)) {
	    hwloc_debug("os proc %lu is offline\n", cpu);
	    continue;
	  }
	}
      }

      /* check whether the kernel exports topology information for this cpu */
      sprintf(str, "%s/cpu%lu/topology", path, cpu);
      if (hwloc_access(str, X_OK, data->root_fd) < 0 && errno == ENOENT) {
	hwloc_debug("os proc %lu has no accessible %s/cpu%lu/topology\n",
		   cpu, path, cpu);
	continue;
      }

      hwloc_bitmap_set(cpuset, cpu);
    }
    closedir(dir);
  }

  topology->support.discovery->pu = 1;
  topology->support.discovery->disallowed_pu = 1;
  hwloc_debug_1arg_bitmap("found %d cpu topologies, cpuset %s\n",
	     hwloc_bitmap_weight(cpuset), cpuset);

  caches_added = 0;
  hwloc_bitmap_foreach_begin(i, cpuset) {
    int tmpint;
    int notfirstofcore = 0; /* set if we have core info and if we're not the first PU of our core */
    int notfirstofdie = 0; /* set if we have die info and if we're not the first PU of our die */
    hwloc_bitmap_t dieset = NULL;

    if (hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_CORE)) {
      /* look at the core */
      hwloc_bitmap_t coreset;
      if (old_filenames)
	sprintf(str, "%s/cpu%d/topology/thread_siblings", path, i);
      else
	sprintf(str, "%s/cpu%d/topology/core_cpus", path, i);
      coreset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
      if (coreset) {
        unsigned mycoreid = (unsigned) -1;
	int gotcoreid = 0; /* to avoid reading the coreid twice */
	hwloc_bitmap_and(coreset, coreset, cpuset);
	if (hwloc_bitmap_weight(coreset) > 1 && threadwithcoreid == -1) {
	  /* check if this is hyper-threading or different coreids */
	  unsigned siblingid, siblingcoreid;

	  mycoreid = (unsigned) -1;
	  sprintf(str, "%s/cpu%d/topology/core_id", path, i); /* contains %d at least up to 4.19 */
	  if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
	    mycoreid = (unsigned) tmpint;
	  gotcoreid = 1;

	  siblingid = hwloc_bitmap_first(coreset);
	  if (siblingid == (unsigned) i)
	    siblingid = hwloc_bitmap_next(coreset, i);
	  siblingcoreid = (unsigned) -1;
	  sprintf(str, "%s/cpu%u/topology/core_id", path, siblingid); /* contains %d at least up to 4.19 */
	  if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
	    siblingcoreid = (unsigned) tmpint;
	  threadwithcoreid = (siblingcoreid != mycoreid);
	}
	if (hwloc_bitmap_first(coreset) != i)
	  notfirstofcore = 1;
	if (!notfirstofcore || threadwithcoreid) {
	  /* regular core */
	  struct hwloc_obj *core;

	  if (!gotcoreid) {
	    mycoreid = (unsigned) -1;
	    sprintf(str, "%s/cpu%d/topology/core_id", path, i); /* contains %d at least up to 4.19 */
	    if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
	      mycoreid = (unsigned) tmpint;
	  }

	  core = hwloc_alloc_setup_object(topology, HWLOC_OBJ_CORE, mycoreid);
	  if (threadwithcoreid)
	    /* amd multicore compute-unit, create one core per thread */
	    hwloc_bitmap_only(coreset, i);
	  core->cpuset = coreset;
	  hwloc_debug_1arg_bitmap("os core %u has cpuset %s\n",
				  mycoreid, core->cpuset);
	  hwloc_insert_object_by_cpuset(topology, core);
	  coreset = NULL; /* don't free it */
	} else

	hwloc_bitmap_free(coreset);
      }
    }

    if (!notfirstofcore /* don't look at the package unless we are the first of the core */
	&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_DIE)) {
      /* look at the die */
      sprintf(str, "%s/cpu%d/topology/die_cpus", path, i);
      dieset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
      if (dieset) {
	hwloc_bitmap_and(dieset, dieset, cpuset);
	if (hwloc_bitmap_first(dieset) != i) {
	  /* not first cpu in this die, ignore the die */
	  hwloc_bitmap_free(dieset);
	  dieset = NULL;
	  notfirstofdie = 1;
	}
	/* look at packages before deciding whether we keep that die or not */
      }
    }

    if (!notfirstofdie /* don't look at the package unless we are the first of the die */
	&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_PACKAGE)) {
      /* look at the package */
      hwloc_bitmap_t packageset;
      if (old_filenames)
	sprintf(str, "%s/cpu%d/topology/core_siblings", path, i);
      else
	sprintf(str, "%s/cpu%d/topology/package_cpus", path, i);
      packageset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
      if (packageset) {
	hwloc_bitmap_and(packageset, packageset, cpuset);
	if (dieset && hwloc_bitmap_isequal(packageset, dieset)) {
	  /* die is identical to package, ignore it */
	  hwloc_bitmap_free(dieset);
	  dieset = NULL;
	}
	if (hwloc_bitmap_first(packageset) == i) {
	  /* first cpu in this package, add the package */
	  struct hwloc_obj *package;
	  unsigned mypackageid;
	  mypackageid = (unsigned) -1;
	  sprintf(str, "%s/cpu%d/topology/physical_package_id", path, i); /* contains %d at least up to 4.19 */
	  if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
	    mypackageid = (unsigned) tmpint;

	  package = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PACKAGE, mypackageid);
	  package->cpuset = packageset;
	  hwloc_debug_1arg_bitmap("os package %u has cpuset %s\n",
				  mypackageid, packageset);
	  /* add cpuinfo */
	  if (cpuinfo_Lprocs) {
	    for(j=0; j<(int) cpuinfo_numprocs; j++)
	      if ((int) cpuinfo_Lprocs[j].Pproc == i) {
		hwloc__move_infos(&package->infos, &package->infos_count,
				  &cpuinfo_Lprocs[j].infos, &cpuinfo_Lprocs[j].infos_count);
	      }
	  }
	  hwloc_insert_object_by_cpuset(topology, package);
	  packageset = NULL; /* don't free it */
	}
	hwloc_bitmap_free(packageset);
      }
    }

    if (dieset) {
      struct hwloc_obj *die;
      unsigned mydieid;
      mydieid = (unsigned) -1;
      sprintf(str, "%s/cpu%d/topology/die_id", path, i); /* contains %d when added in 5.2 */
      if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0)
	mydieid = (unsigned) tmpint;

      die = hwloc_alloc_setup_object(topology, HWLOC_OBJ_DIE, mydieid);
      die->cpuset = dieset;
      hwloc_debug_1arg_bitmap("os die %u has cpuset %s\n",
			      mydieid, dieset);
      hwloc_insert_object_by_cpuset(topology, die);
    }

    if (data->arch == HWLOC_LINUX_ARCH_S390
	&& hwloc_filter_check_keep_object_type(topology, HWLOC_OBJ_GROUP)) {
      /* look at the books */
      hwloc_bitmap_t bookset, drawerset;
      sprintf(str, "%s/cpu%d/topology/book_siblings", path, i);
      bookset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
      if (bookset) {
	hwloc_bitmap_and(bookset, bookset, cpuset);
	if (hwloc_bitmap_first(bookset) == i) {
	  struct hwloc_obj *book;
	  unsigned mybookid;
	  mybookid = (unsigned) -1;
	  sprintf(str, "%s/cpu%d/topology/book_id", path, i); /* contains %d at least up to 4.19 */
	  if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0) {
	    mybookid = (unsigned) tmpint;

	    book = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, mybookid);
	    book->cpuset = bookset;
	    hwloc_debug_1arg_bitmap("os book %u has cpuset %s\n",
				    mybookid, bookset);
	    book->subtype = strdup("Book");
	    book->attr->group.kind = HWLOC_GROUP_KIND_S390_BOOK;
	    book->attr->group.subkind = 0;
	    hwloc_insert_object_by_cpuset(topology, book);
	    bookset = NULL; /* don't free it */
	  }
        }
	hwloc_bitmap_free(bookset);

	sprintf(str, "%s/cpu%d/topology/drawer_siblings", path, i);
	drawerset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
	if (drawerset) {
	  hwloc_bitmap_and(drawerset, drawerset, cpuset);
	  if (hwloc_bitmap_first(drawerset) == i) {
	    struct hwloc_obj *drawer;
	    unsigned mydrawerid;
	    mydrawerid = (unsigned) -1;
	    sprintf(str, "%s/cpu%d/topology/drawer_id", path, i); /* contains %d at least up to 4.19 */
	    if (hwloc_read_path_as_int(str, &tmpint, data->root_fd) == 0) {
	      mydrawerid = (unsigned) tmpint;

	      drawer = hwloc_alloc_setup_object(topology, HWLOC_OBJ_GROUP, mydrawerid);
	      drawer->cpuset = drawerset;
	      hwloc_debug_1arg_bitmap("os drawer %u has cpuset %s\n",
				      mydrawerid, drawerset);
	      drawer->subtype = strdup("Drawer");
	      drawer->attr->group.kind = HWLOC_GROUP_KIND_S390_BOOK;
	      drawer->attr->group.subkind = 1;
	      hwloc_insert_object_by_cpuset(topology, drawer);
	      drawerset = NULL; /* don't free it */
	    }
	  }
	  hwloc_bitmap_free(drawerset);
	}
      }
    }

    /* PU cannot be filtered-out */
    {
      /* look at the thread */
      hwloc_bitmap_t threadset;
      struct hwloc_obj *thread = hwloc_alloc_setup_object(topology, HWLOC_OBJ_PU, (unsigned) i);
      threadset = hwloc_bitmap_alloc();
      hwloc_bitmap_only(threadset, i);
      thread->cpuset = threadset;
      hwloc_debug_1arg_bitmap("thread %d has cpuset %s\n",
		 i, threadset);
      hwloc_insert_object_by_cpuset(topology, thread);
    }

    /* look at the caches */
    for(j=0; j<10; j++) {
      char str2[20]; /* enough for a level number (one digit) or a type (Data/Instruction/Unified) */
      hwloc_bitmap_t cacheset;

      sprintf(str, "%s/cpu%d/cache/index%d/shared_cpu_map", path, i, j);
      cacheset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
      if (cacheset) {
	if (hwloc_bitmap_iszero(cacheset)) {
	  /* ia64 returning empty L3 and L2i? use the core set instead */
	  hwloc_bitmap_t tmpset;
	  if (old_filenames)
	    sprintf(str, "%s/cpu%d/topology/thread_siblings", path, i);
	  else
	    sprintf(str, "%s/cpu%d/topology/core_cpus", path, i);
	  tmpset = hwloc__alloc_read_path_as_cpumask(str, data->root_fd);
	  /* only use it if we actually got something */
	  if (tmpset) {
	    hwloc_bitmap_free(cacheset);
	    cacheset = tmpset;
	  }
	}
	hwloc_bitmap_and(cacheset, cacheset, cpuset);

	if (hwloc_bitmap_first(cacheset) == i) {
	  unsigned kB;
	  unsigned linesize;
	  unsigned sets, lines_per_tag;
	  unsigned depth; /* 1 for L1, .... */
	  hwloc_obj_cache_type_t ctype = HWLOC_OBJ_CACHE_UNIFIED; /* default */
	  hwloc_obj_type_t otype;
	  struct hwloc_obj *cache;

	  /* get the cache level depth */
	  sprintf(str, "%s/cpu%d/cache/index%d/level", path, i, j); /* contains %u at least up to 4.19 */
	  if (hwloc_read_path_as_uint(str, &depth, data->root_fd) < 0) {
	    hwloc_bitmap_free(cacheset);
	    continue;
	  }

	  /* cache type */
	  sprintf(str, "%s/cpu%d/cache/index%d/type", path, i, j);
	  if (hwloc_read_path_by_length(str, str2, sizeof(str2), data->root_fd) == 0) {
	    if (!strncmp(str2, "Data", 4))
	      ctype = HWLOC_OBJ_CACHE_DATA;
	    else if (!strncmp(str2, "Unified", 7))
	      ctype = HWLOC_OBJ_CACHE_UNIFIED;
	    else if (!strncmp(str2, "Instruction", 11))
	      ctype = HWLOC_OBJ_CACHE_INSTRUCTION;
	  }

	  otype = hwloc_cache_type_by_depth_type(depth, ctype);
	  if (otype == HWLOC_OBJ_TYPE_NONE
	      || !hwloc_filter_check_keep_object_type(topology, otype)) {
	    hwloc_bitmap_free(cacheset);
	    continue;
	  }

	  /* FIXME: if Bulldozer/Piledriver, add compute unit Groups when L2/L1i filtered-out */
	  /* FIXME: if KNL, add tile Groups when L2/L1i filtered-out */

	  /* get the cache size */
	  kB = 0;
	  sprintf(str, "%s/cpu%d/cache/index%d/size", path, i, j); /* contains %uK at least up to 4.19 */
	  hwloc_read_path_as_uint(str, &kB, data->root_fd);
	  /* KNL reports L3 with size=0 and full cpuset in cpuid.
	   * Let hwloc_linux_try_add_knl_mcdram_cache() detect it better.
	   */
	  if (!kB && otype == HWLOC_OBJ_L3CACHE && data->is_knl) {
	    hwloc_bitmap_free(cacheset);
	    continue;
	  }

	  /* get the line size */
	  linesize = 0;
	  sprintf(str, "%s/cpu%d/cache/index%d/coherency_line_size", path, i, j); /* contains %u at least up to 4.19 */
	  hwloc_read_path_as_uint(str, &linesize, data->root_fd);

	  /* get the number of sets and lines per tag.
	   * don't take the associativity directly in "ways_of_associativity" because
	   * some archs (ia64, ppc) put 0 there when fully-associative, while others (x86) put something like -1 there.
	   */
	  sets = 0;
	  sprintf(str, "%s/cpu%d/cache/index%d/number_of_sets", path, i, j); /* contains %u at least up to 4.19 */
	  hwloc_read_path_as_uint(str, &sets, data->root_fd);

	  lines_per_tag = 1;
	  sprintf(str, "%s/cpu%d/cache/index%d/physical_line_partition", path, i, j); /* contains %u at least up to 4.19 */
	  hwloc_read_path_as_uint(str, &lines_per_tag, data->root_fd);

	  /* first cpu in this cache, add the cache */
	  cache = hwloc_alloc_setup_object(topology, otype, HWLOC_UNKNOWN_INDEX);
	  cache->attr->cache.size = ((uint64_t)kB) << 10;
	  cache->attr->cache.depth = depth;
	  cache->attr->cache.linesize = linesize;
	  cache->attr->cache.type = ctype;
	  if (!linesize || !lines_per_tag || !sets)
	    cache->attr->cache.associativity = 0; /* unknown */
	  else if (sets == 1)
	    cache->attr->cache.associativity = 0; /* likely wrong, make it unknown */
	  else
	    cache->attr->cache.associativity = (kB << 10) / linesize / lines_per_tag / sets;
	  cache->cpuset = cacheset;
	  hwloc_debug_1arg_bitmap("cache depth %u has cpuset %s\n",
				  depth, cacheset);
	  hwloc_insert_object_by_cpuset(topology, cache);
	  cacheset = NULL; /* don't free it */
	  ++caches_added;
	}
      }
      hwloc_bitmap_free(cacheset);
     }

  } hwloc_bitmap_foreach_end();

  if (0 == caches_added && data->use_dt)
    look_powerpc_device_tree(topology, data);

  hwloc_bitmap_free(cpuset);
  hwloc_bitmap_free(online_set);

  return 0;
}



/****************************************
 ****** cpuinfo Topology Discovery ******
 ****************************************/

static int
hwloc_linux_parse_cpuinfo_x86(const char *prefix, const char *value,
			      struct hwloc_info_s **infos, unsigned *infos_count,
			      int is_global __hwloc_attribute_unused)
{
  if (!strcmp("vendor_id", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUVendor", value);
  } else if (!strcmp("model name", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModel", value);
  } else if (!strcmp("model", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModelNumber", value);
  } else if (!strcmp("cpu family", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUFamilyNumber", value);
  } else if (!strcmp("stepping", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUStepping", value);
  }
  return 0;
}

static int
hwloc_linux_parse_cpuinfo_ia64(const char *prefix, const char *value,
			       struct hwloc_info_s **infos, unsigned *infos_count,
			       int is_global __hwloc_attribute_unused)
{
  if (!strcmp("vendor", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUVendor", value);
  } else if (!strcmp("model name", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModel", value);
  } else if (!strcmp("model", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModelNumber", value);
  } else if (!strcmp("family", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUFamilyNumber", value);
  }
  return 0;
}

static int
hwloc_linux_parse_cpuinfo_arm(const char *prefix, const char *value,
			      struct hwloc_info_s **infos, unsigned *infos_count,
			      int is_global __hwloc_attribute_unused)
{
  if (!strcmp("Processor", prefix) /* old kernels with one Processor header */
      || !strcmp("model name", prefix) /* new kernels with one model name per core */) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModel", value);
  } else if (!strcmp("CPU implementer", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUImplementer", value);
  } else if (!strcmp("CPU architecture", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUArchitecture", value);
  } else if (!strcmp("CPU variant", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUVariant", value);
  } else if (!strcmp("CPU part", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUPart", value);
  } else if (!strcmp("CPU revision", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPURevision", value);
  } else if (!strcmp("Hardware", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "HardwareName", value);
  } else if (!strcmp("Revision", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "HardwareRevision", value);
  } else if (!strcmp("Serial", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "HardwareSerial", value);
  }
  return 0;
}

static int
hwloc_linux_parse_cpuinfo_ppc(const char *prefix, const char *value,
			      struct hwloc_info_s **infos, unsigned *infos_count,
			      int is_global)
{
  /* common fields */
  if (!strcmp("cpu", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "CPUModel", value);
  } else if (!strcmp("platform", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "PlatformName", value);
  } else if (!strcmp("model", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "PlatformModel", value);
  }
  /* platform-specific fields */
  else if (!strcasecmp("vendor", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "PlatformVendor", value);
  } else if (!strcmp("Board ID", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "PlatformBoardID", value);
  } else if (!strcmp("Board", prefix)
	     || !strcasecmp("Machine", prefix)) {
    /* machine and board are similar (and often more precise) than model above */
    if (value[0])
      hwloc__add_info_nodup(infos, infos_count, "PlatformModel", value, 1);
  } else if (!strcasecmp("Revision", prefix)
	     || !strcmp("Hardware rev", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, is_global ? "PlatformRevision" : "CPURevision", value);
  } else if (!strcmp("SVR", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "SystemVersionRegister", value);
  } else if (!strcmp("PVR", prefix)) {
    if (value[0])
      hwloc__add_info(infos, infos_count, "ProcessorVersionRegister", value);
  }
  /* don't match 'board*' because there's also "board l2" on some platforms */
  return 0;
}

/*
 * avr32: "chip type\t:"			=> OK
 * blackfin: "model name\t:"			=> OK
 * h8300: "CPU:"				=> OK
 * m68k: "CPU:"					=> OK
 * mips: "cpu model\t\t:"			=> OK
 * openrisc: "CPU:"				=> OK
 * sparc: "cpu\t\t:"				=> OK
 * tile: "model name\t:"			=> OK
 * unicore32: "Processor\t:"			=> OK
 * alpha: "cpu\t\t\t: Alpha" + "cpu model\t\t:"	=> "cpu" overwritten by "cpu model", no processor indexes
 * cris: "cpu\t\t:" + "cpu model\t:"		=> only "cpu"
 * frv: "CPU-Core:" + "CPU:"			=> only "CPU"
 * mn10300: "cpu core   :" + "model name :"	=> only "model name"
 * parisc: "cpu family\t:" + "cpu\t\t:"		=> only "cpu"
 *
 * not supported because of conflicts with other arch minor lines:
 * m32r: "cpu family\t:"			=> KO (adding "cpu family" would break "blackfin")
 * microblaze: "CPU-Family:"			=> KO
 * sh: "cpu family\t:" + "cpu type\t:"		=> KO
 * xtensa: "model\t\t:"				=> KO
 */
static int
hwloc_linux_parse_cpuinfo_generic(const char *prefix, const char *value,
				  struct hwloc_info_s **infos, unsigned *infos_count,
				  int is_global __hwloc_attribute_unused)
{
  if (!strcmp("model name", prefix)
      || !strcmp("Processor", prefix)
      || !strcmp("chip type", prefix)
      || !strcmp("cpu model", prefix)
      || !strcasecmp("cpu", prefix)) {
    /* keep the last one, assume it's more precise than the first one.
     * we should have the Architecture keypair for basic information anyway.
     */
    if (value[0])
      hwloc__add_info_nodup(infos, infos_count, "CPUModel", value, 1);
  }
  return 0;
}

/* Lprocs_p set to NULL unless returns > 0 */
static int
hwloc_linux_parse_cpuinfo(struct hwloc_linux_backend_data_s *data,
			  const char *path,
			  struct hwloc_linux_cpuinfo_proc ** Lprocs_p,
			  struct hwloc_info_s **global_infos, unsigned *global_infos_count)
{
  /* FIXME: only parse once per package and once for globals? */
  FILE *fd;
  char str[128]; /* vendor/model can be very long */
  char *endptr;
  unsigned allocated_Lprocs = 0;
  struct hwloc_linux_cpuinfo_proc * Lprocs = NULL;
  unsigned numprocs = 0;
  int curproc = -1;
  int (*parse_cpuinfo_func)(const char *, const char *, struct hwloc_info_s **, unsigned *, int) = NULL;

  if (!(fd=hwloc_fopen(path,"r", data->root_fd)))
    {
      hwloc_debug("could not open %s\n", path);
      return -1;
    }

#      define PROCESSOR	"processor"
  hwloc_debug("\n\n * Topology extraction from %s *\n\n", path);
  while (fgets(str, sizeof(str), fd)!=NULL) {
    unsigned long Pproc;
    char *end, *dot, *prefix, *value;
    int noend = 0;

    /* remove the ending \n */
    end = strchr(str, '\n');
    if (end)
      *end = 0;
    else
      noend = 1;
    /* if empty line, skip and reset curproc */
    if (!*str) {
      curproc = -1;
      continue;
    }
    /* skip lines with no dot */
    dot = strchr(str, ':');
    if (!dot)
      continue;
    /* skip lines not starting with a letter */
    if ((*str > 'z' || *str < 'a')
	&& (*str > 'Z' || *str < 'A'))
      continue;

    /* mark the end of the prefix */
    prefix = str;
    end = dot;
    while (end[-1] == ' ' || end[-1] == '\t') end--; /* need a strrspn() */
    *end = 0;
    /* find beginning of value, its end is already marked */
    value = dot+1 + strspn(dot+1, " \t");

    /* defines for parsing numbers */
#   define getprocnb_begin(field, var)					\
    if (!strcmp(field,prefix)) {					\
      var = strtoul(value,&endptr,0);					\
      if (endptr==value) {						\
	hwloc_debug("no number in "field" field of %s\n", path);	\
	goto err;							\
      } else if (var==ULONG_MAX) {					\
	hwloc_debug("too big "field" number in %s\n", path); 		\
	goto err;							\
      }									\
      hwloc_debug(field " %lu\n", var)
#   define getprocnb_end()						\
    }
    /* actually parse numbers */
    getprocnb_begin(PROCESSOR, Pproc);
    curproc = numprocs++;
    if (numprocs > allocated_Lprocs) {
      struct hwloc_linux_cpuinfo_proc * tmp;
      if (!allocated_Lprocs)
	allocated_Lprocs = 8;
      else
        allocated_Lprocs *= 2;
      tmp = realloc(Lprocs, allocated_Lprocs * sizeof(*Lprocs));
      if (!tmp)
	goto err;
      Lprocs = tmp;
    }
    Lprocs[curproc].Pproc = Pproc;
    Lprocs[curproc].infos = NULL;
    Lprocs[curproc].infos_count = 0;
    getprocnb_end() else {

      /* architecture specific or default routine for parsing cpumodel */
      switch (data->arch) {
      case HWLOC_LINUX_ARCH_X86:
	parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_x86;
	break;
      case HWLOC_LINUX_ARCH_ARM:
	parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_arm;
	break;
      case HWLOC_LINUX_ARCH_POWER:
	parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_ppc;
	break;
      case HWLOC_LINUX_ARCH_IA64:
	parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_ia64;
	break;
      default:
	parse_cpuinfo_func = hwloc_linux_parse_cpuinfo_generic;
      }

      /* we can't assume that we already got a processor index line:
       * alpha/frv/h8300/m68k/microblaze/sparc have no processor lines at all, only a global entry.
       * tile has a global section with model name before the list of processor lines.
       */
      parse_cpuinfo_func(prefix, value,
			 curproc >= 0 ? &Lprocs[curproc].infos : global_infos,
			 curproc >= 0 ? &Lprocs[curproc].infos_count : global_infos_count,
			 curproc < 0);
    }

    if (noend) {
      /* ignore end of line */
      if (fscanf(fd,"%*[^\n]") == EOF)
	break;
      getc(fd);
    }
  }
  fclose(fd);

  *Lprocs_p = Lprocs;
  return numprocs;

 err:
  fclose(fd);
  free(Lprocs);
  *Lprocs_p = NULL;
  return -1;
}

static void
hwloc_linux_free_cpuinfo(struct hwloc_linux_cpuinfo_proc * Lprocs, unsigned numprocs,
			 struct hwloc_info_s *global_infos, unsigned global_infos_count)
{
  if (Lprocs) {
    unsigned i;
    for(i=0; i<numprocs; i++) {
      hwloc__free_infos(Lprocs[i].infos, Lprocs[i].infos_count);
    }
    free(Lprocs);
  }
  hwloc__free_infos(global_infos, global_infos_count);
}



/*************************************
 ****** Main Topology Discovery ******
 *************************************/

static void
hwloc__linux_get_mic_sn(struct hwloc_topology *topology, struct hwloc_linux_backend_data_s *data)
{
  char line[64], *tmp, *end;
  if (hwloc_read_path_by_length("/proc/elog", line, sizeof(line), data->root_fd) < 0)
    return;
  if (strncmp(line, "Card ", 5))
    return;
  tmp = line + 5;
  end = strchr(tmp, ':');
  if (!end)
    return;
  *end = '\0';

  if (tmp[0])
    hwloc_obj_add_info(hwloc_get_root_obj(topology), "MICSerialNumber", tmp);
}

static void
hwloc_gather_system_info(struct hwloc_topology *topology,
			 struct hwloc_linux_backend_data_s *data)
{
  FILE *file;
  char line[128]; /* enough for utsname fields */
  const char *env;

  /* initialize to something sane, in case !is_thissystem and we can't find things in /proc/hwloc-nofile-info */
  memset(&data->utsname, 0, sizeof(data->utsname));
  data->fallback_nbprocessors = -1; /* unknown yet */
  data->pagesize = 4096;

  /* read thissystem info */
  if (topology->is_thissystem) {
    uname(&data->utsname);
    data->fallback_nbprocessors = hwloc_fallback_nbprocessors(0); /* errors managed in hwloc_linux_fallback_pu_level() */
    data->pagesize = hwloc_getpagesize();
  }

  if (!data->is_real_fsroot) {
   /* overwrite with optional /proc/hwloc-nofile-info */
   file = hwloc_fopen("/proc/hwloc-nofile-info", "r", data->root_fd);
   if (file) {
    while (fgets(line, sizeof(line), file)) {
      char *tmp = strchr(line, '\n');
      if (!strncmp("OSName: ", line, 8)) {
	if (tmp)
	  *tmp = '\0';
	strncpy(data->utsname.sysname, line+8, sizeof(data->utsname.sysname));
	data->utsname.sysname[sizeof(data->utsname.sysname)-1] = '\0';
      } else if (!strncmp("OSRelease: ", line, 11)) {
	if (tmp)
	  *tmp = '\0';
	strncpy(data->utsname.release, line+11, sizeof(data->utsname.release));
	data->utsname.release[sizeof(data->utsname.release)-1] = '\0';
      } else if (!strncmp("OSVersion: ", line, 11)) {
	if (tmp)
	  *tmp = '\0';
	strncpy(data->utsname.version, line+11, sizeof(data->utsname.version));
	data->utsname.version[sizeof(data->utsname.version)-1] = '\0';
      } else if (!strncmp("HostName: ", line, 10)) {
	if (tmp)
	  *tmp = '\0';
	strncpy(data->utsname.nodename, line+10, sizeof(data->utsname.nodename));
	data->utsname.nodename[sizeof(data->utsname.nodename)-1] = '\0';
      } else if (!strncmp("Architecture: ", line, 14)) {
	if (tmp)
	  *tmp = '\0';
	strncpy(data->utsname.machine, line+14, sizeof(data->utsname.machine));
	data->utsname.machine[sizeof(data->utsname.machine)-1] = '\0';
      } else if (!strncmp("FallbackNbProcessors: ", line, 22)) {
	if (tmp)
	  *tmp = '\0';
	data->fallback_nbprocessors = atoi(line+22);
      } else if (!strncmp("PageSize: ", line, 10)) {
	if (tmp)
	 *tmp = '\0';
	data->pagesize = strtoull(line+10, NULL, 10);
      } else {
	hwloc_debug("ignored /proc/hwloc-nofile-info line %s\n", line);
	/* ignored */
      }
    }
    fclose(file);
   }
  }

  env = getenv("HWLOC_DUMP_NOFILE_INFO");
  if (env && *env) {
    file = fopen(env, "w");
    if (file) {
      if (*data->utsname.sysname)
	fprintf(file, "OSName: %s\n", data->utsname.sysname);
      if (*data->utsname.release)
	fprintf(file, "OSRelease: %s\n", data->utsname.release);
      if (*data->utsname.version)
	fprintf(file, "OSVersion: %s\n", data->utsname.version);
      if (*data->utsname.nodename)
	fprintf(file, "HostName: %s\n", data->utsname.nodename);
      if (*data->utsname.machine)
	fprintf(file, "Architecture: %s\n", data->utsname.machine);
      fprintf(file, "FallbackNbProcessors: %d\n", data->fallback_nbprocessors);
      fprintf(file, "PageSize: %llu\n", (unsigned long long) data->pagesize);
      fclose(file);
    }
  }

  /* detect arch for quirks, using configure #defines if possible, or uname */
#if (defined HWLOC_X86_32_ARCH) || (defined HWLOC_X86_64_ARCH) /* does not cover KNC */
  if (topology->is_thissystem)
    data->arch = HWLOC_LINUX_ARCH_X86;
#endif
  if (data->arch == HWLOC_LINUX_ARCH_UNKNOWN && *data->utsname.machine) {
    if (!strcmp(data->utsname.machine, "x86_64")
	|| (data->utsname.machine[0] == 'i' && !strcmp(data->utsname.machine+2, "86"))
	|| !strcmp(data->utsname.machine, "k1om"))
      data->arch = HWLOC_LINUX_ARCH_X86;
    else if (!strncmp(data->utsname.machine, "arm", 3))
      data->arch = HWLOC_LINUX_ARCH_ARM;
    else if (!strncmp(data->utsname.machine, "ppc", 3)
	     || !strncmp(data->utsname.machine, "power", 5))
      data->arch = HWLOC_LINUX_ARCH_POWER;
    else if (!strncmp(data->utsname.machine, "s390", 4))
      data->arch = HWLOC_LINUX_ARCH_S390;
    else if (!strcmp(data->utsname.machine, "ia64"))
      data->arch = HWLOC_LINUX_ARCH_IA64;
  }
}

/* returns 0 on success, -1 on non-match or error during hardwired load */
static int
hwloc_linux_try_hardwired_cpuinfo(struct hwloc_backend *backend)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_linux_backend_data_s *data = backend->private_data;

  if (getenv("HWLOC_NO_HARDWIRED_TOPOLOGY"))
    return -1;

  if (!strcmp(data->utsname.machine, "s64fx")) {
    char line[128];
    /* Fujistu K-computer, FX10, and FX100 use specific processors
     * whose Linux topology support is broken until 4.1 (acc455cffa75070d55e74fc7802b49edbc080e92and)
     * and existing machines will likely never be fixed by kernel upgrade.
     */

    /* /proc/cpuinfo starts with one of these lines:
     * "cpu             : Fujitsu SPARC64 VIIIfx"
     * "cpu             : Fujitsu SPARC64 XIfx"
     * "cpu             : Fujitsu SPARC64 IXfx"
     */
    if (hwloc_read_path_by_length("/proc/cpuinfo", line, sizeof(line), data->root_fd) < 0)
      return -1;

    if (strncmp(line, "cpu\t", 4))
      return -1;

    if (strstr(line, "Fujitsu SPARC64 VIIIfx"))
      return hwloc_look_hardwired_fujitsu_k(topology);
    else if (strstr(line, "Fujitsu SPARC64 IXfx"))
      return hwloc_look_hardwired_fujitsu_fx10(topology);
    else if (strstr(line, "FUJITSU SPARC64 XIfx"))
      return hwloc_look_hardwired_fujitsu_fx100(topology);
  }
  return -1;
}

static void hwloc_linux__get_allowed_resources(hwloc_topology_t topology, const char *root_path, int root_fd, char **cpuset_namep)
{
  char *cpuset_mntpnt, *cgroup_mntpnt, *cpuset_name = NULL;

  hwloc_find_linux_cpuset_mntpnt(&cgroup_mntpnt, &cpuset_mntpnt, root_path);
  if (cgroup_mntpnt || cpuset_mntpnt) {
    cpuset_name = hwloc_read_linux_cpuset_name(root_fd, topology->pid);
    if (cpuset_name) {
      hwloc_admin_disable_set_from_cpuset(root_fd, cgroup_mntpnt, cpuset_mntpnt, cpuset_name, "cpus", topology->allowed_cpuset);
      hwloc_admin_disable_set_from_cpuset(root_fd, cgroup_mntpnt, cpuset_mntpnt, cpuset_name, "mems", topology->allowed_nodeset);
    }
    free(cgroup_mntpnt);
    free(cpuset_mntpnt);
  }
  *cpuset_namep = cpuset_name;
}

static void
hwloc_linux_fallback_pu_level(struct hwloc_backend *backend)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_linux_backend_data_s *data = backend->private_data;

  if (data->fallback_nbprocessors >= 1)
    topology->support.discovery->pu = 1;
  else
    data->fallback_nbprocessors = 1;
  hwloc_setup_pu_level(topology, data->fallback_nbprocessors);
}

static const char *find_sysfs_cpu_path(int root_fd, int *old_filenames)
{
  if (!hwloc_access("/sys/bus/cpu/devices", R_OK|X_OK, root_fd)) {
    if (!hwloc_access("/sys/bus/cpu/devices/cpu0/topology/package_cpus", R_OK, root_fd)
	|| !hwloc_access("/sys/bus/cpu/devices/cpu0/topology/core_cpus", R_OK, root_fd)) {
      return "/sys/bus/cpu/devices";
    }

    if (!hwloc_access("/sys/bus/cpu/devices/cpu0/topology/core_siblings", R_OK, root_fd)
	|| !hwloc_access("/sys/bus/cpu/devices/cpu0/topology/thread_siblings", R_OK, root_fd)) {
      *old_filenames = 1;
      return "/sys/bus/cpu/devices";
    }
  }

  if (!hwloc_access("/sys/devices/system/cpu", R_OK|X_OK, root_fd)) {
    if (!hwloc_access("/sys/devices/system/cpu/cpu0/topology/package_cpus", R_OK, root_fd)
	|| !hwloc_access("/sys/devices/system/cpu/cpu0/topology/core_cpus", R_OK, root_fd)) {
      return "/sys/devices/system/cpu";
    }

    if (!hwloc_access("/sys/devices/system/cpu/cpu0/topology/core_siblings", R_OK, root_fd)
	|| !hwloc_access("/sys/devices/system/cpu/cpu0/topology/thread_siblings", R_OK, root_fd)) {
      *old_filenames = 1;
      return "/sys/devices/system/cpu";
    }
  }

  return NULL;
}

static const char *find_sysfs_node_path(int root_fd)
{
  if (!hwloc_access("/sys/bus/node/devices", R_OK|X_OK, root_fd)
      && !hwloc_access("/sys/bus/node/devices/node0/cpumap", R_OK, root_fd))
    return "/sys/bus/node/devices";

  if (!hwloc_access("/sys/devices/system/node", R_OK|X_OK, root_fd)
      && !hwloc_access("/sys/devices/system/node/node0/cpumap", R_OK, root_fd))
    return "/sys/devices/system/node";

  return NULL;
}

static int
hwloc_linuxfs_look_cpu(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  /*
   * This backend may be used with topology->is_thissystem set (default)
   * or not (modified fsroot path).
   */

  struct hwloc_topology *topology = backend->topology;
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  unsigned nbnodes;
  char *cpuset_name = NULL;
  struct hwloc_linux_cpuinfo_proc * Lprocs = NULL;
  struct hwloc_info_s *global_infos = NULL;
  unsigned global_infos_count = 0;
  int numprocs;
  int already_pus;
  int already_numanodes;
  const char *sysfs_cpu_path;
  const char *sysfs_node_path;
  int old_siblings_filenames = 0;
  int err;

  /* look for sysfs cpu path containing at least one of core_siblings and thread_siblings */
  sysfs_cpu_path = find_sysfs_cpu_path(data->root_fd, &old_siblings_filenames);
  hwloc_debug("Found sysfs cpu files under %s with %s topology filenames\n",
	      sysfs_cpu_path, old_siblings_filenames ? "old" : "new");

  /* look for sysfs node path */
  sysfs_node_path = find_sysfs_node_path(data->root_fd);
  hwloc_debug("Found sysfs node files under %s\n",
	      sysfs_node_path);

  already_pus = (topology->levels[0][0]->complete_cpuset != NULL
		 && !hwloc_bitmap_iszero(topology->levels[0][0]->complete_cpuset));
  /* if there are PUs, still look at memory information
   * since x86 misses NUMA node information (unless we forced AMD topoext NUMA nodes)
   * memory size.
   */
  already_numanodes = (topology->levels[0][0]->complete_nodeset != NULL
		       && !hwloc_bitmap_iszero(topology->levels[0][0]->complete_nodeset));
  /* if there are already NUMA nodes, we'll just annotate them with memory information,
   * which requires the NUMA level to be connected.
   */
  if (already_numanodes)
    hwloc_topology_reconnect(topology, 0);

  hwloc_alloc_root_sets(topology->levels[0][0]);

  /*********************************
   * Platform information for later
   */
  hwloc_gather_system_info(topology, data);

  /**********************
   * /proc/cpuinfo
   */
  numprocs = hwloc_linux_parse_cpuinfo(data, "/proc/cpuinfo", &Lprocs, &global_infos, &global_infos_count);
  if (numprocs < 0)
    numprocs = 0;

  /**************************
   * detect model for quirks
   */
  if (data->arch == HWLOC_LINUX_ARCH_X86 && numprocs > 0) {
      unsigned i;
      const char *cpuvendor = NULL, *cpufamilynumber = NULL, *cpumodelnumber = NULL;
      for(i=0; i<Lprocs[0].infos_count; i++) {
	if (!strcmp(Lprocs[0].infos[i].name, "CPUVendor")) {
	  cpuvendor = Lprocs[0].infos[i].value;
	} else if (!strcmp(Lprocs[0].infos[i].name, "CPUFamilyNumber")) {
	  cpufamilynumber = Lprocs[0].infos[i].value;
	} else if (!strcmp(Lprocs[0].infos[i].name, "CPUModelNumber")) {
	  cpumodelnumber = Lprocs[0].infos[i].value;
	}
      }
      if (cpuvendor && !strcmp(cpuvendor, "GenuineIntel")
	  && cpufamilynumber && !strcmp(cpufamilynumber, "6")
	  && cpumodelnumber && (!strcmp(cpumodelnumber, "87")
	  || !strcmp(cpumodelnumber, "133")))
	data->is_knl = 1;
      if (cpuvendor && !strcmp(cpuvendor, "AuthenticAMD")
	  && cpufamilynumber
	  && (!strcmp(cpufamilynumber, "21")
	      || !strcmp(cpufamilynumber, "22")))
	data->is_amd_with_CU = 1;
  }

  /**********************
   * Gather the list of admin-disabled cpus and mems
   */
  if (!(dstatus->flags & HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES)) {
    hwloc_linux__get_allowed_resources(topology, data->root_path, data->root_fd, &cpuset_name);
    dstatus->flags |= HWLOC_DISC_STATUS_FLAG_GOT_ALLOWED_RESOURCES;
  }

  /**********************
   * CPU information
   */

  /* Don't rediscover CPU resources if already done */
  if (already_pus)
    goto cpudone;

  /* Gather the list of cpus now */
  err = hwloc_linux_try_hardwired_cpuinfo(backend);
  if (!err)
    goto cpudone;

  /* setup root info */
  hwloc__move_infos(&hwloc_get_root_obj(topology)->infos, &hwloc_get_root_obj(topology)->infos_count,
		    &global_infos, &global_infos_count);

  if (!sysfs_cpu_path) {
    /* /sys/.../topology unavailable (before 2.6.16)
     * or not containing anything interesting */
    hwloc_linux_fallback_pu_level(backend);
    if (data->use_dt)
      look_powerpc_device_tree(topology, data);

  } else {
    /* sysfs */
    if (look_sysfscpu(topology, data, sysfs_cpu_path, old_siblings_filenames, Lprocs, numprocs) < 0)
      /* sysfs but we failed to read cpu topology, fallback */
      hwloc_linux_fallback_pu_level(backend);
  }

 cpudone:

  /*********************
   * Memory information
   */

  /* Get the machine memory attributes */
  hwloc_get_machine_meminfo(data, &topology->machine_memory);

  /* Gather NUMA information. */
  if (sysfs_node_path) {
    if (hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_NUMANODE) > 0)
      annotate_sysfsnode(topology, data, sysfs_node_path, &nbnodes);
    else
      look_sysfsnode(topology, data, sysfs_node_path, &nbnodes);
  } else
    nbnodes = 0;

  /**********************
   * Misc
   */

  /* Gather DMI info */
  hwloc__get_dmi_id_info(data, topology->levels[0][0]);

  hwloc_obj_add_info(topology->levels[0][0], "Backend", "Linux");
  if (cpuset_name) {
    hwloc_obj_add_info(topology->levels[0][0], "LinuxCgroup", cpuset_name);
    free(cpuset_name);
  }

  hwloc__linux_get_mic_sn(topology, data);

  /* data->utsname was filled with real uname or \0, we can safely pass it */
  hwloc_add_uname_info(topology, &data->utsname);

  hwloc_linux_free_cpuinfo(Lprocs, numprocs, global_infos, global_infos_count);
  return 0;
}



/****************************************
 ***** Linux PCI backend callbacks ******
 ****************************************/

/*
 * backend callback for retrieving the location of a pci device
 */
static int
hwloc_linux_backend_get_pci_busid_cpuset(struct hwloc_backend *backend,
					 struct hwloc_pcidev_attr_s *busid, hwloc_bitmap_t cpuset)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  char path[256];
  int err;

  snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%01x/local_cpus",
	   busid->domain, busid->bus,
	   busid->dev, busid->func);
  err = hwloc__read_path_as_cpumask(path, cpuset, data->root_fd);
  if (!err && !hwloc_bitmap_iszero(cpuset))
    return 0;
  return -1;
}



#ifdef HWLOC_HAVE_LINUXIO

/***********************************
 ******* Linux I/O discovery *******
 ***********************************/

#define HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL (1U<<0)
#define HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB (1U<<1)
#define HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS (1U<<2)
#define HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS (1U<<31)

static hwloc_obj_t
hwloc_linuxfs_find_osdev_parent(struct hwloc_backend *backend, int root_fd,
				const char *osdevpath, unsigned osdev_flags)
{
  struct hwloc_topology *topology = backend->topology;
  char path[256], buf[10];
  int fd;
  int foundpci;
  unsigned pcidomain = 0, pcibus = 0, pcidev = 0, pcifunc = 0;
  unsigned _pcidomain, _pcibus, _pcidev, _pcifunc;
  hwloc_bitmap_t cpuset;
  const char *tmp;
  hwloc_obj_t parent;
  char *devicesubdir;
  int err;

  if (osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS)
    devicesubdir = "..";
  else
    devicesubdir = "device";

  err = hwloc_readlink(osdevpath, path, sizeof(path), root_fd);
  if (err < 0) {
    /* /sys/class/<class>/<name> is a directory instead of a symlink on old kernels (at least around 2.6.18 and 2.6.25).
     * The link to parse can be found in /sys/class/<class>/<name>/device instead, at least for "/pci..."
     */
    char olddevpath[256];
    snprintf(olddevpath, sizeof(olddevpath), "%s/device", osdevpath);
    err = hwloc_readlink(olddevpath, path, sizeof(path), root_fd);
    if (err < 0)
      return NULL;
  }
  path[err] = '\0';

  if (!(osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL)) {
    if (strstr(path, "/virtual/"))
      return NULL;
  }

  if (!(osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB)) {
    if (strstr(path, "/usb"))
      return NULL;
  }

  tmp = strstr(path, "/pci");
  if (!tmp)
    goto nopci;
  tmp = strchr(tmp+4, '/');
  if (!tmp)
    goto nopci;
  tmp++;

  /* iterate through busid to find the last one (previous ones are bridges) */
  foundpci = 0;
 nextpci:
  if (sscanf(tmp+1, "%x:%x:%x.%x", &_pcidomain, &_pcibus, &_pcidev, &_pcifunc) == 4) {
    foundpci = 1;
    pcidomain = _pcidomain;
    pcibus = _pcibus;
    pcidev = _pcidev;
    pcifunc = _pcifunc;
    tmp += 13;
    goto nextpci;
  }
  if (sscanf(tmp+1, "%x:%x.%x", &_pcibus, &_pcidev, &_pcifunc) == 3) {
    foundpci = 1;
    pcidomain = 0;
    pcibus = _pcibus;
    pcidev = _pcidev;
    pcifunc = _pcifunc;
    tmp += 8;
    goto nextpci;
  }

  if (foundpci) {
    /* attach to a PCI parent or to a normal (non-I/O) parent found by PCI affinity */
    parent = hwloc_pci_find_parent_by_busid(topology, pcidomain, pcibus, pcidev, pcifunc);
    if (parent)
      return parent;
  }

 nopci:
  /* attach directly near the right NUMA node */
  snprintf(path, sizeof(path), "%s/%s/numa_node", osdevpath, devicesubdir);
  fd = hwloc_open(path, root_fd);
  if (fd >= 0) {
    err = read(fd, buf, sizeof(buf));
    close(fd);
    if (err > 0) {
      int node = atoi(buf);
      if (node >= 0) {
	parent = hwloc_get_numanode_obj_by_os_index(topology, (unsigned) node);
	if (parent) {
	  /* don't attach I/O under numa node, attach to the same normal parent */
	  while (hwloc__obj_type_is_memory(parent->type))
	    parent = parent->parent;
	  return parent;
	}
      }
    }
  }

  /* attach directly to the right cpuset */
  snprintf(path, sizeof(path), "%s/%s/local_cpus", osdevpath, devicesubdir);
  cpuset = hwloc__alloc_read_path_as_cpumask(path, root_fd);
  if (cpuset) {
    parent = hwloc_find_insert_io_parent_by_complete_cpuset(topology, cpuset);
    hwloc_bitmap_free(cpuset);
    if (parent)
      return parent;
  }

  /* FIXME: {numa_node,local_cpus} may be missing when the device link points to a subdirectory.
   * For instance, device of scsi blocks may point to foo/ata1/host0/target0:0:0/0:0:0:0/ instead of foo/
   * In such case, we should look for device/../../../../{numa_node,local_cpus} instead of device/{numa_node,local_cpus}
   * Not needed yet since scsi blocks use the PCI locality above.
   */

  /* fallback to the root object */
  return hwloc_get_root_obj(topology);
}

static hwloc_obj_t
hwloc_linux_add_os_device(struct hwloc_backend *backend, struct hwloc_obj *pcidev, hwloc_obj_osdev_type_t type, const char *name)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_obj *obj = hwloc_alloc_setup_object(topology, HWLOC_OBJ_OS_DEVICE, HWLOC_UNKNOWN_INDEX);
  obj->name = strdup(name);
  obj->attr->osdev.type = type;

  hwloc_insert_object_by_parent(topology, pcidev, obj);
  /* insert_object_by_parent() doesn't merge during insert, so obj is still valid */

  return obj;
}

static void
hwloc_linuxfs_block_class_fillinfos(struct hwloc_backend *backend __hwloc_attribute_unused, int root_fd,
				    struct hwloc_obj *obj, const char *osdevpath, unsigned osdev_flags)
{
#ifdef HWLOC_HAVE_LIBUDEV
  struct hwloc_linux_backend_data_s *data = backend->private_data;
#endif
  FILE *file;
  char path[296]; /* osdevpath <= 256 */
  char line[128];
  char vendor[64] = "";
  char model[64] = "";
  char serial[64] = "";
  char revision[64] = "";
  char blocktype[64] = "";
  unsigned sectorsize = 0;
  unsigned major_id, minor_id;
  char *devicesubdir;
  char *tmp;

  if (osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS)
    devicesubdir = "..";
  else
    devicesubdir = "device";

  snprintf(path, sizeof(path), "%s/size", osdevpath);
  if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
    unsigned long long value = strtoull(line, NULL, 10);
    /* linux always reports size in 512-byte units for blocks, and bytes for dax, we want kB */
    snprintf(line, sizeof(line), "%llu",
	     (osdev_flags & HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS) ? value / 2 : value >> 10);
    hwloc_obj_add_info(obj, "Size", line);
  }

  snprintf(path, sizeof(path), "%s/queue/hw_sector_size", osdevpath);
  if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
    sectorsize = strtoul(line, NULL, 10);
  }

  snprintf(path, sizeof(path), "%s/%s/devtype", osdevpath, devicesubdir);
  if (!hwloc_read_path_by_length(path, line, sizeof(line), root_fd)) {
    /* non-volatile devices use the following subtypes:
     * nd_namespace_pmem for pmem/raw (/dev/pmemX)
     * nd_btt for pmem/sector (/dev/pmemXs)
     * nd_pfn for pmem/fsdax (/dev/pmemX)
     * nd_dax for pmem/devdax (/dev/daxX) but it's not a block device anyway
     * nd_namespace_blk for blk/raw and blk/sector (/dev/ndblkX) ?
     *
     * Note that device/sector_size in btt devices includes integrity metadata
     * (512/4096 block + 0/N) while queue/hw_sector_size above is the user sectorsize
     * without metadata.
     */
    if (!strncmp(line, "nd_", 3))
      strcpy(blocktype, "NVDIMM"); /* Save the blocktype now since udev reports "" so far */
  }
  if (sectorsize) {
    snprintf(line, sizeof(line), "%u", sectorsize);
    hwloc_obj_add_info(obj, "SectorSize", line);
  }

  snprintf(path, sizeof(path), "%s/dev", osdevpath);
  if (hwloc_read_path_by_length(path, line, sizeof(line), root_fd) < 0)
    goto done;
  if (sscanf(line, "%u:%u", &major_id, &minor_id) != 2)
    goto done;
  tmp = strchr(line, '\n');
  if (tmp)
    *tmp = '\0';
  hwloc_obj_add_info(obj, "LinuxDeviceID", line);

#ifdef HWLOC_HAVE_LIBUDEV
  if (data->udev) {
    struct udev_device *dev;
    const char *prop;
    dev = udev_device_new_from_subsystem_sysname(data->udev, "block", obj->name);
    if (!dev)
      goto done;
    prop = udev_device_get_property_value(dev, "ID_VENDOR");
    if (prop) {
      strncpy(vendor, prop, sizeof(vendor));
      vendor[sizeof(vendor)-1] = '\0';
    }
    prop = udev_device_get_property_value(dev, "ID_MODEL");
    if (prop) {
      strncpy(model, prop, sizeof(model));
      model[sizeof(model)-1] = '\0';
    }
    prop = udev_device_get_property_value(dev, "ID_REVISION");
    if (prop) {
      strncpy(revision, prop, sizeof(revision));
      revision[sizeof(revision)-1] = '\0';
    }
    prop = udev_device_get_property_value(dev, "ID_SERIAL_SHORT");
    if (prop) {
      strncpy(serial, prop, sizeof(serial));
      serial[sizeof(serial)-1] = '\0';
    }
    prop = udev_device_get_property_value(dev, "ID_TYPE");
    if (prop) {
      strncpy(blocktype, prop, sizeof(blocktype));
      blocktype[sizeof(blocktype)-1] = '\0';
    }

    udev_device_unref(dev);
  } else
    /* fallback to reading files, works with any fsroot */
#endif
 {
  snprintf(path, sizeof(path), "/run/udev/data/b%u:%u", major_id, minor_id);
  file = hwloc_fopen(path, "r", root_fd);
  if (!file)
    goto done;

  while (NULL != fgets(line, sizeof(line), file)) {
    tmp = strchr(line, '\n');
    if (tmp)
      *tmp = '\0';
    if (!strncmp(line, "E:ID_VENDOR=", strlen("E:ID_VENDOR="))) {
      strncpy(vendor, line+strlen("E:ID_VENDOR="), sizeof(vendor));
      vendor[sizeof(vendor)-1] = '\0';
    } else if (!strncmp(line, "E:ID_MODEL=", strlen("E:ID_MODEL="))) {
      strncpy(model, line+strlen("E:ID_MODEL="), sizeof(model));
      model[sizeof(model)-1] = '\0';
    } else if (!strncmp(line, "E:ID_REVISION=", strlen("E:ID_REVISION="))) {
      strncpy(revision, line+strlen("E:ID_REVISION="), sizeof(revision));
      revision[sizeof(revision)-1] = '\0';
    } else if (!strncmp(line, "E:ID_SERIAL_SHORT=", strlen("E:ID_SERIAL_SHORT="))) {
      strncpy(serial, line+strlen("E:ID_SERIAL_SHORT="), sizeof(serial));
      serial[sizeof(serial)-1] = '\0';
    } else if (!strncmp(line, "E:ID_TYPE=", strlen("E:ID_TYPE="))) {
      strncpy(blocktype, line+strlen("E:ID_TYPE="), sizeof(blocktype));
      blocktype[sizeof(blocktype)-1] = '\0';
    }
  }
  fclose(file);
 }

 done:
  /* clear fake "ATA" vendor name */
  if (!strcasecmp(vendor, "ATA"))
    *vendor = '\0';
  /* overwrite vendor name from model when possible */
  if (!*vendor) {
    if (!strncasecmp(model, "wd", 2))
      strcpy(vendor, "Western Digital");
    else if (!strncasecmp(model, "st", 2))
      strcpy(vendor, "Seagate");
    else if (!strncasecmp(model, "samsung", 7))
      strcpy(vendor, "Samsung");
    else if (!strncasecmp(model, "sandisk", 7))
      strcpy(vendor, "SanDisk");
    else if (!strncasecmp(model, "toshiba", 7))
      strcpy(vendor, "Toshiba");
  }

  if (*vendor)
    hwloc_obj_add_info(obj, "Vendor", vendor);
  if (*model)
    hwloc_obj_add_info(obj, "Model", model);
  if (*revision)
    hwloc_obj_add_info(obj, "Revision", revision);
  if (*serial)
    hwloc_obj_add_info(obj, "SerialNumber", serial);

  if (!strcmp(blocktype, "disk") || !strncmp(obj->name, "nvme", 4))
    obj->subtype = strdup("Disk");
  else if (!strcmp(blocktype, "NVDIMM")) /* FIXME: set by us above, to workaround udev returning "" so far */
    obj->subtype = strdup("NVDIMM");
  else if (!strcmp(blocktype, "tape"))
    obj->subtype = strdup("Tape");
  else if (!strcmp(blocktype, "cd") || !strcmp(blocktype, "floppy") || !strcmp(blocktype, "optical"))
    obj->subtype = strdup("Removable Media Device");
  else {
    /* generic, usb mass storage/rbc, usb mass storage/scsi */
  }
}

static int
hwloc_linuxfs_lookup_block_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/block", root_fd);
  if (!dir)
    return 0;

  osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_BLOCK_WITH_SECTORS; /* uses 512B sectors */

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    struct stat stbuf;
    hwloc_obj_t obj, parent;
    int err;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;

    /* ignore partitions */
    err = snprintf(path, sizeof(path), "/sys/class/block/%s/partition", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& hwloc_stat(path, &stbuf, root_fd) >= 0)
      continue;

    err = snprintf(path, sizeof(path), "/sys/class/block/%s", dirent->d_name);
    if ((size_t) err >= sizeof(path))
      continue;
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    /* USB device are created here but removed later when USB PCI devices get filtered out
     * (unless WHOLE_IO is enabled).
     */

    obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);

    hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags);
  }

  closedir(dir);

  return 0;
}

static int
hwloc_linuxfs_lookup_dax_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  /* depending on the kernel config, dax devices may appear either in /sys/bus/dax or /sys/class/dax */

  dir = hwloc_opendir("/sys/bus/dax/devices", root_fd);
  if (dir) {
    int found = 0;
    while ((dirent = readdir(dir)) != NULL) {
      char path[300];
      char driver[256];
      hwloc_obj_t obj, parent;
      int err;

      if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
	continue;
      found++;

      /* ignore kmem-device, those appear as additional NUMA nodes */
      err = snprintf(path, sizeof(path), "/sys/bus/dax/devices/%s/driver", dirent->d_name);
      if ((size_t) err >= sizeof(path))
	continue;
      err = hwloc_readlink(path, driver, sizeof(driver), root_fd);
      if (err >= 0) {
	driver[err] = '\0';
	if (!strcmp(driver+err-5, "/kmem"))
	  continue;
      }

      snprintf(path, sizeof(path), "/sys/bus/dax/devices/%s", dirent->d_name);
      parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags | HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS);
      if (!parent)
	continue;

      obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);

      hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags | HWLOC_LINUXFS_OSDEV_FLAG_UNDER_BUS);
    }
    closedir(dir);

    /* don't look in /sys/class/dax if we found something in /sys/bus/dax */
    if (found)
      return 0;
  }

  dir = hwloc_opendir("/sys/class/dax", root_fd);
  if (dir) {
    while ((dirent = readdir(dir)) != NULL) {
      char path[256];
      hwloc_obj_t obj, parent;
      int err;

      if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
	continue;

      /* kmem not supported in class mode, driver may only be changed under bus */

      err = snprintf(path, sizeof(path), "/sys/class/dax/%s", dirent->d_name);
      if ((size_t) err >= sizeof(path))
	continue;
      parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
      if (!parent)
	continue;

      obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_BLOCK, dirent->d_name);

      hwloc_linuxfs_block_class_fillinfos(backend, root_fd, obj, path, osdev_flags);
    }
    closedir(dir);
  }

  return 0;
}

static void
hwloc_linuxfs_net_class_fillinfos(int root_fd,
				  struct hwloc_obj *obj, const char *osdevpath)
{
  struct stat st;
  char path[296]; /* osdevpath <= 256 */
  char address[128];
  int err;
  snprintf(path, sizeof(path), "%s/address", osdevpath);
  if (!hwloc_read_path_by_length(path, address, sizeof(address), root_fd)) {
    char *eol = strchr(address, '\n');
    if (eol)
      *eol = 0;
    hwloc_obj_add_info(obj, "Address", address);
  }
  snprintf(path, sizeof(path), "%s/device/infiniband", osdevpath);
  if (!hwloc_stat(path, &st, root_fd)) {
    char hexid[16];
    snprintf(path, sizeof(path), "%s/dev_port", osdevpath);
    err = hwloc_read_path_by_length(path, hexid, sizeof(hexid), root_fd);
    if (err < 0) {
      /* fallback t dev_id for old kernels/drivers */
      snprintf(path, sizeof(path), "%s/dev_id", osdevpath);
      err = hwloc_read_path_by_length(path, hexid, sizeof(hexid), root_fd);
    }
    if (!err) {
      char *eoid;
      unsigned long port;
      port = strtoul(hexid, &eoid, 0);
      if (eoid != hexid) {
	char portstr[21];
	snprintf(portstr, sizeof(portstr), "%lu", port+1);
	hwloc_obj_add_info(obj, "Port", portstr);
      }
    }
  }
}

static int
hwloc_linuxfs_lookup_net_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/net", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    hwloc_obj_t obj, parent;
    int err;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;

    err = snprintf(path, sizeof(path), "/sys/class/net/%s", dirent->d_name);
    if ((size_t) err >= sizeof(path))
      continue;
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_NETWORK, dirent->d_name);

    hwloc_linuxfs_net_class_fillinfos(root_fd, obj, path);
  }

  closedir(dir);

  return 0;
}

static void
hwloc_linuxfs_infiniband_class_fillinfos(int root_fd,
					 struct hwloc_obj *obj, const char *osdevpath)
{
  char path[296]; /* osdevpath <= 256 */
  char guidvalue[20];
  unsigned i,j;

  snprintf(path, sizeof(path), "%s/node_guid", osdevpath);
  if (!hwloc_read_path_by_length(path, guidvalue, sizeof(guidvalue), root_fd)) {
    size_t len;
    len = strspn(guidvalue, "0123456789abcdefx:");
    guidvalue[len] = '\0';
    hwloc_obj_add_info(obj, "NodeGUID", guidvalue);
  }

  snprintf(path, sizeof(path), "%s/sys_image_guid", osdevpath);
  if (!hwloc_read_path_by_length(path, guidvalue, sizeof(guidvalue), root_fd)) {
    size_t len;
    len = strspn(guidvalue, "0123456789abcdefx:");
    guidvalue[len] = '\0';
    hwloc_obj_add_info(obj, "SysImageGUID", guidvalue);
  }

  for(i=1; ; i++) {
    char statevalue[2];
    char lidvalue[11];
    char gidvalue[40];

    snprintf(path, sizeof(path), "%s/ports/%u/state", osdevpath, i);
    if (!hwloc_read_path_by_length(path, statevalue, sizeof(statevalue), root_fd)) {
      char statename[32];
      statevalue[1] = '\0'; /* only keep the first byte/digit */
      snprintf(statename, sizeof(statename), "Port%uState", i);
      hwloc_obj_add_info(obj, statename, statevalue);
    } else {
      /* no such port */
      break;
    }

    snprintf(path, sizeof(path), "%s/ports/%u/lid", osdevpath, i);
    if (!hwloc_read_path_by_length(path, lidvalue, sizeof(lidvalue), root_fd)) {
      char lidname[32];
      size_t len;
      len = strspn(lidvalue, "0123456789abcdefx");
      lidvalue[len] = '\0';
      snprintf(lidname, sizeof(lidname), "Port%uLID", i);
      hwloc_obj_add_info(obj, lidname, lidvalue);
    }

    snprintf(path, sizeof(path), "%s/ports/%u/lid_mask_count", osdevpath, i);
    if (!hwloc_read_path_by_length(path, lidvalue, sizeof(lidvalue), root_fd)) {
      char lidname[32];
      size_t len;
      len = strspn(lidvalue, "0123456789");
      lidvalue[len] = '\0';
      snprintf(lidname, sizeof(lidname), "Port%uLMC", i);
      hwloc_obj_add_info(obj, lidname, lidvalue);
    }

    for(j=0; ; j++) {
      snprintf(path, sizeof(path), "%s/ports/%u/gids/%u", osdevpath, i, j);
      if (!hwloc_read_path_by_length(path, gidvalue, sizeof(gidvalue), root_fd)) {
	char gidname[32];
	size_t len;
	len = strspn(gidvalue, "0123456789abcdefx:");
	gidvalue[len] = '\0';
	if (strncmp(gidvalue+20, "0000:0000:0000:0000", 19)) {
	  /* only keep initialized GIDs */
	  snprintf(gidname, sizeof(gidname), "Port%uGID%u", i, j);
	  hwloc_obj_add_info(obj, gidname, gidvalue);
	}
      } else {
	/* no such port */
	break;
      }
    }
  }
}

static int
hwloc_linuxfs_lookup_infiniband_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/infiniband", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    hwloc_obj_t obj, parent;
    int err;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;

    /* blocklist scif* fake devices */
    if (!strncmp(dirent->d_name, "scif", 4))
      continue;

    err = snprintf(path, sizeof(path), "/sys/class/infiniband/%s", dirent->d_name);
    if ((size_t) err > sizeof(path))
      continue;
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_OPENFABRICS, dirent->d_name);

    hwloc_linuxfs_infiniband_class_fillinfos(root_fd, obj, path);
  }

  closedir(dir);

  return 0;
}

static void
hwloc_linuxfs_mic_class_fillinfos(int root_fd,
				  struct hwloc_obj *obj, const char *osdevpath)
{
  char path[296]; /* osdevpath <= 256 */
  char family[64];
  char sku[64];
  char sn[64];
  char string[21];

  obj->subtype = strdup("MIC");

  snprintf(path, sizeof(path), "%s/family", osdevpath);
  if (!hwloc_read_path_by_length(path, family, sizeof(family), root_fd)) {
    char *eol = strchr(family, '\n');
    if (eol)
      *eol = 0;
    hwloc_obj_add_info(obj, "MICFamily", family);
  }

  snprintf(path, sizeof(path), "%s/sku", osdevpath);
  if (!hwloc_read_path_by_length(path, sku, sizeof(sku), root_fd)) {
    char *eol = strchr(sku, '\n');
    if (eol)
      *eol = 0;
    hwloc_obj_add_info(obj, "MICSKU", sku);
  }

  snprintf(path, sizeof(path), "%s/serialnumber", osdevpath);
  if (!hwloc_read_path_by_length(path, sn, sizeof(sn), root_fd)) {
    char *eol;
    eol = strchr(sn, '\n');
    if (eol)
      *eol = 0;
    hwloc_obj_add_info(obj, "MICSerialNumber", sn);
  }

  snprintf(path, sizeof(path), "%s/active_cores", osdevpath);
  if (!hwloc_read_path_by_length(path, string, sizeof(string), root_fd)) {
    unsigned long count = strtoul(string, NULL, 16);
    snprintf(string, sizeof(string), "%lu", count);
    hwloc_obj_add_info(obj, "MICActiveCores", string);
  }

  snprintf(path, sizeof(path), "%s/memsize", osdevpath);
  if (!hwloc_read_path_by_length(path, string, sizeof(string), root_fd)) {
    unsigned long count = strtoul(string, NULL, 16);
    snprintf(string, sizeof(string), "%lu", count);
    hwloc_obj_add_info(obj, "MICMemorySize", string);
  }
}

static int
hwloc_linuxfs_lookup_mic_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  unsigned idx;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/mic", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    hwloc_obj_t obj, parent;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;
    if (sscanf(dirent->d_name, "mic%u", &idx) != 1)
      continue;

    snprintf(path, sizeof(path), "/sys/class/mic/mic%u", idx);
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    obj = hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_COPROC, dirent->d_name);

    hwloc_linuxfs_mic_class_fillinfos(root_fd, obj, path);
  }

  closedir(dir);

  return 0;
}

static int
hwloc_linuxfs_lookup_drm_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/drm", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    hwloc_obj_t parent;
    struct stat stbuf;
    int err;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;

    /* only keep main devices, not subdevices for outputs */
    err = snprintf(path, sizeof(path), "/sys/class/drm/%s/dev", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& hwloc_stat(path, &stbuf, root_fd) < 0)
      continue;

    /* Most drivers expose a card%d device.
     * Some (free?) drivers also expose render%d.
     * Old kernels also have a controlD%d. On recent kernels, it's a symlink to card%d (deprecated?).
     * There can also exist some output-specific files such as card0-DP-1.
     *
     * All these aren't very useful compared to CUDA/OpenCL/...
     * Hence the DRM class is only enabled when KEEP_ALL.
     *
     * FIXME: We might want to filter everything out but card%d.
     * Maybe look at the driver (read the end of /sys/class/drm/<name>/device/driver symlink),
     * to decide whether card%d could be useful (likely not for NVIDIA).
     */

    err = snprintf(path, sizeof(path), "/sys/class/drm/%s", dirent->d_name);
    if ((size_t) err >= sizeof(path))
      continue;
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_GPU, dirent->d_name);
  }

  closedir(dir);

  return 0;
}

static int
hwloc_linuxfs_lookup_dma_class(struct hwloc_backend *backend, unsigned osdev_flags)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/class/dma", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
    char path[256];
    hwloc_obj_t parent;
    int err;

    if (!strcmp(dirent->d_name, ".") || !strcmp(dirent->d_name, ".."))
      continue;

    err = snprintf(path, sizeof(path), "/sys/class/dma/%s", dirent->d_name);
    if ((size_t) err >= sizeof(path))
      continue;
    parent = hwloc_linuxfs_find_osdev_parent(backend, root_fd, path, osdev_flags);
    if (!parent)
      continue;

    hwloc_linux_add_os_device(backend, parent, HWLOC_OBJ_OSDEV_DMA, dirent->d_name);
  }

  closedir(dir);

  return 0;
}

struct hwloc_firmware_dmi_mem_device_header {
  unsigned char type;
  unsigned char length;
  unsigned char handle[2];
  unsigned char phy_mem_handle[2];
  unsigned char mem_err_handle[2];
  unsigned char tot_width[2];
  unsigned char dat_width[2];
  unsigned char size[2];
  unsigned char ff;
  unsigned char dev_set;
  unsigned char dev_loc_str_num;
  unsigned char bank_loc_str_num;
  unsigned char mem_type;
  unsigned char type_detail[2];
  unsigned char speed[2];
  unsigned char manuf_str_num;
  unsigned char serial_str_num;
  unsigned char asset_tag_str_num;
  unsigned char part_num_str_num;
  /* don't include the following fields since we don't need them,
   * some old implementations may miss them.
   */
};

static int check_dmi_entry(const char *buffer)
{
  /* reject empty strings */
  if (!*buffer)
    return 0;
  /* reject strings of spaces (at least Dell use this for empty memory slots) */
  if (strspn(buffer, " ") == strlen(buffer))
    return 0;
  return 1;
}

static int
hwloc__get_firmware_dmi_memory_info_one(struct hwloc_topology *topology,
					unsigned idx, const char *path, FILE *fd,
					struct hwloc_firmware_dmi_mem_device_header *header)
{
  unsigned slen;
  char buffer[256]; /* enough for memory device strings, or at least for each of them */
  unsigned foff; /* offset in raw file */
  unsigned boff; /* offset in buffer read from raw file */
  unsigned i;
  struct hwloc_info_s *infos = NULL;
  unsigned infos_count = 0;
  hwloc_obj_t misc;
  int foundinfo = 0;

  /* start after the header */
  foff = header->length;
  i = 1;
  while (1) {
    /* read one buffer */
    if (fseek(fd, foff, SEEK_SET) < 0)
      break;
    if (!fgets(buffer, sizeof(buffer), fd))
      break;
    /* read string at the beginning of the buffer */
    boff = 0;
    while (1) {
      /* stop on empty string */
      if (!buffer[boff])
        goto done;
      /* stop if this string goes to the end of the buffer */
      slen = strlen(buffer+boff);
      if (boff + slen+1 == sizeof(buffer))
        break;
      /* string didn't get truncated, should be OK */
      if (i == header->manuf_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "Vendor", buffer+boff);
	  foundinfo = 1;
	}
      }	else if (i == header->serial_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "SerialNumber", buffer+boff);
	  foundinfo = 1;
	}
      } else if (i == header->asset_tag_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "AssetTag", buffer+boff);
	  foundinfo = 1;
	}
      } else if (i == header->part_num_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "PartNumber", buffer+boff);
	  foundinfo = 1;
	}
      } else if (i == header->dev_loc_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "DeviceLocation", buffer+boff);
	  /* only a location, not an actual info about the device */
	}
      } else if (i == header->bank_loc_str_num) {
	if (check_dmi_entry(buffer+boff)) {
	  hwloc__add_info(&infos, &infos_count, "BankLocation", buffer+boff);
	  /* only a location, not an actual info about the device */
	}
      } else {
	goto done;
      }
      /* next string in buffer */
      boff += slen+1;
      i++;
    }
    /* couldn't read a single full string from that buffer, we're screwed */
    if (!boff) {
      fprintf(stderr, "hwloc could read a DMI firmware entry #%u in %s\n",
	      i, path);
      break;
    }
    /* reread buffer after previous string */
    foff += boff;
  }

done:
  if (!foundinfo) {
    /* found no actual info about the device. if there's only location info, the slot may be empty */
    goto out_with_infos;
  }

  misc = hwloc_alloc_setup_object(topology, HWLOC_OBJ_MISC, idx);
  if (!misc)
    goto out_with_infos;

  misc->subtype = strdup("MemoryModule");

  hwloc__move_infos(&misc->infos, &misc->infos_count, &infos, &infos_count);
  /* FIXME: find a way to identify the corresponding NUMA node and attach these objects there.
   * but it means we need to parse DeviceLocation=DIMM_B4 but these vary significantly
   * with the vendor, and it's hard to be 100% sure 'B' is second socket.
   * Examples at http://sourceforge.net/p/edac-utils/code/HEAD/tree/trunk/src/etc/labels.db
   * or https://github.com/grondo/edac-utils/blob/master/src/etc/labels.db
   */
  hwloc_insert_object_by_parent(topology, hwloc_get_root_obj(topology), misc);
  return 1;

 out_with_infos:
  hwloc__free_infos(infos, infos_count);
  return 0;
}

static int
hwloc__get_firmware_dmi_memory_info(struct hwloc_topology *topology,
				    struct hwloc_linux_backend_data_s *data)
{
  char path[128];
  unsigned i;

  for(i=0; ; i++) {
    FILE *fd;
    struct hwloc_firmware_dmi_mem_device_header header;
    int err;

    snprintf(path, sizeof(path), "/sys/firmware/dmi/entries/17-%u/raw", i);
    fd = hwloc_fopen(path, "r", data->root_fd);
    if (!fd)
      break;

    err = fread(&header, sizeof(header), 1, fd);
    if (err != 1) {
      fclose(fd);
      break;
    }
    if (header.length < sizeof(header)) {
      /* invalid, or too old entry/spec that doesn't contain what we need */
      fclose(fd);
      break;
    }

    hwloc__get_firmware_dmi_memory_info_one(topology, i, path, fd, &header);

    fclose(fd);
  }

  return 0;
}

#ifdef HWLOC_HAVE_LINUXPCI

#define HWLOC_PCI_REVISION_ID 0x08
#define HWLOC_PCI_CAP_ID_EXP 0x10
#define HWLOC_PCI_CLASS_NOT_DEFINED 0x0000

static int
hwloc_linuxfs_pci_look_pcidevices(struct hwloc_backend *backend)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  struct hwloc_topology *topology = backend->topology;
  hwloc_obj_t tree = NULL;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  /* We could lookup /sys/devices/pci.../.../busid1/.../busid2 recursively
   * to build the hierarchy of bridges/devices directly.
   * But that would require readdirs in all bridge sysfs subdirectories.
   * Do a single readdir in the linear list in /sys/bus/pci/devices/...
   * and build the hierarchy manually instead.
   */
  dir = hwloc_opendir("/sys/bus/pci/devices/", root_fd);
  if (!dir)
    return 0;

  while ((dirent = readdir(dir)) != NULL) {
#define CONFIG_SPACE_CACHESIZE 256
    unsigned char config_space_cache[CONFIG_SPACE_CACHESIZE];
    unsigned domain, bus, dev, func;
    unsigned secondary_bus, subordinate_bus;
    unsigned short class_id;
    hwloc_obj_type_t type;
    hwloc_obj_t obj;
    struct hwloc_pcidev_attr_s *attr;
    unsigned offset;
    char path[64];
    char value[16];
    size_t ret;
    int fd, err;

    if (sscanf(dirent->d_name, "%04x:%02x:%02x.%01x", &domain, &bus, &dev, &func) != 4)
      continue;

    if (domain > 0xffff) {
      static int warned = 0;
      if (!warned)
	fprintf(stderr, "Ignoring PCI device with non-16bit domain\n");
      warned = 1;
      continue;
    }

    /* initialize the config space in case we fail to read it (missing permissions, etc). */
    memset(config_space_cache, 0xff, CONFIG_SPACE_CACHESIZE);
    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/config", dirent->d_name);
    if ((size_t) err < sizeof(path)) {
      /* don't use hwloc_read_path_by_length() because we don't want the ending \0 */
      fd = hwloc_open(path, root_fd);
      if (fd >= 0) {
	ret = read(fd, config_space_cache, CONFIG_SPACE_CACHESIZE);
	(void) ret; /* we initialized config_space_cache in case we don't read enough, ignore the read length */
	close(fd);
      }
    }

    class_id = HWLOC_PCI_CLASS_NOT_DEFINED;
    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/class", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
      class_id = strtoul(value, NULL, 16) >> 8;

    type = hwloc_pcidisc_check_bridge_type(class_id, config_space_cache);

    if (type == HWLOC_OBJ_BRIDGE) {
      /* since 4.13, there's secondary_bus_number and subordinate_bus_number in sysfs,
       * but reading them from the config-space is easy anyway.
       */
      if (hwloc_pcidisc_find_bridge_buses(domain, bus, dev, func,
					  &secondary_bus, &subordinate_bus,
					  config_space_cache) < 0)
	continue;
    }

    /* filtered? */
    if (type == HWLOC_OBJ_PCI_DEVICE) {
      enum hwloc_type_filter_e filter;
      hwloc_topology_get_type_filter(topology, HWLOC_OBJ_PCI_DEVICE, &filter);
      if (filter == HWLOC_TYPE_FILTER_KEEP_NONE)
	continue;
      if (filter == HWLOC_TYPE_FILTER_KEEP_IMPORTANT
	  && !hwloc_filter_check_pcidev_subtype_important(class_id))
	continue;
    } else if (type == HWLOC_OBJ_BRIDGE) {
      enum hwloc_type_filter_e filter;
      hwloc_topology_get_type_filter(topology, HWLOC_OBJ_BRIDGE, &filter);
      if (filter == HWLOC_TYPE_FILTER_KEEP_NONE)
	continue;
      /* HWLOC_TYPE_FILTER_KEEP_IMPORTANT filtered later in the core */
    }

    obj = hwloc_alloc_setup_object(topology, type, HWLOC_UNKNOWN_INDEX);
    if (!obj)
      break;
    attr = &obj->attr->pcidev;

    attr->domain = domain;
    attr->bus = bus;
    attr->dev = dev;
    attr->func = func;

    /* bridge specific attributes */
    if (type == HWLOC_OBJ_BRIDGE) {
      struct hwloc_bridge_attr_s *battr = &obj->attr->bridge;
      battr->upstream_type = HWLOC_OBJ_BRIDGE_PCI;
      battr->downstream_type = HWLOC_OBJ_BRIDGE_PCI;
      battr->downstream.pci.domain = domain;
      battr->downstream.pci.secondary_bus = secondary_bus;
      battr->downstream.pci.subordinate_bus = subordinate_bus;
    }

    /* default (unknown) values */
    attr->vendor_id = 0;
    attr->device_id = 0;
    attr->class_id = class_id;
    attr->revision = 0;
    attr->subvendor_id = 0;
    attr->subdevice_id = 0;
    attr->linkspeed = 0;

    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/vendor", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
      attr->vendor_id = strtoul(value, NULL, 16);

    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/device", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
      attr->device_id = strtoul(value, NULL, 16);

    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_vendor", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
      attr->subvendor_id = strtoul(value, NULL, 16);

    err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/subsystem_device", dirent->d_name);
    if ((size_t) err < sizeof(path)
	&& !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
      attr->subdevice_id = strtoul(value, NULL, 16);

    /* get the revision */
    attr->revision = config_space_cache[HWLOC_PCI_REVISION_ID];

    /* try to get the link speed */
    offset = hwloc_pcidisc_find_cap(config_space_cache, HWLOC_PCI_CAP_ID_EXP);
    if (offset > 0 && offset + 20 /* size of PCI express block up to link status */ <= CONFIG_SPACE_CACHESIZE) {
      hwloc_pcidisc_find_linkspeed(config_space_cache, offset, &attr->linkspeed);
    } else {
      /* if not available from config-space (extended part is root-only), look in sysfs files added in 4.13 */
      float speed = 0.f;
      unsigned width = 0;
      err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_speed", dirent->d_name);
      if ((size_t) err < sizeof(path)
	  && !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
	speed = hwloc_linux_pci_link_speed_from_string(value);
      err = snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/current_link_width", dirent->d_name);
      if ((size_t) err < sizeof(path)
	  && !hwloc_read_path_by_length(path, value, sizeof(value), root_fd))
	width = atoi(value);
      attr->linkspeed = speed*width/8;
    }

    hwloc_pcidisc_tree_insert_by_busid(&tree, obj);
  }

  closedir(dir);

  hwloc_pcidisc_tree_attach(backend->topology, tree);
  return 0;
}

static int
hwloc_linuxfs_pci_look_pcislots(struct hwloc_backend *backend)
{
  struct hwloc_topology *topology = backend->topology;
  struct hwloc_linux_backend_data_s *data = backend->private_data;
  int root_fd = data->root_fd;
  DIR *dir;
  struct dirent *dirent;

  dir = hwloc_opendir("/sys/bus/pci/slots/", root_fd);
  if (dir) {
    while ((dirent = readdir(dir)) != NULL) {
      char path[64];
      char buf[64];
      unsigned domain, bus, dev;
      int err;

      if (dirent->d_name[0] == '.')
	continue;
      err = snprintf(path, sizeof(path), "/sys/bus/pci/slots/%s/address", dirent->d_name);
      if ((size_t) err < sizeof(path)
	  && !hwloc_read_path_by_length(path, buf, sizeof(buf), root_fd)
	  && sscanf(buf, "%x:%x:%x", &domain, &bus, &dev) == 3) {
	/* may also be %x:%x without a device number but that's only for hotplug when nothing is plugged, ignore those */
	hwloc_obj_t obj = hwloc_pci_find_by_busid(topology, domain, bus, dev, 0);
	/* obj may be higher in the hierarchy that requested (if that exact bus didn't exist),
	 * we'll check below whether the bus ID is correct.
	 */
	while (obj) {
	  /* Apply the slot to that device and its siblings with same domain/bus/dev ID.
	   * Make sure that siblings are still PCI and on the same bus
	   * (optional bridge filtering can put different things together).
	   */
	  if (obj->type != HWLOC_OBJ_PCI_DEVICE &&
	      (obj->type != HWLOC_OBJ_BRIDGE || obj->attr->bridge.upstream_type != HWLOC_OBJ_BRIDGE_PCI))
	    break;
	  if (obj->attr->pcidev.domain != domain
	      || obj->attr->pcidev.bus != bus
	      || obj->attr->pcidev.dev != dev)
	    break;

	  hwloc_obj_add_info(obj, "PCISlot", dirent->d_name);
	  obj = obj->next_sibling;
	}
      }
    }
    closedir(dir);
  }

  return 0;
}
#endif /* HWLOC_HAVE_LINUXPCI */
#endif /* HWLOC_HAVE_LINUXIO */

static int
hwloc_look_linuxfs(struct hwloc_backend *backend, struct hwloc_disc_status *dstatus)
{
  /*
   * This backend may be used with topology->is_thissystem set (default)
   * or not (modified fsroot path).
   */

  struct hwloc_topology *topology = backend->topology;
#ifdef HWLOC_HAVE_LINUXIO
  enum hwloc_type_filter_e pfilter, bfilter, ofilter, mfilter;
#endif /* HWLOC_HAVE_LINUXIO */

  if (dstatus->phase == HWLOC_DISC_PHASE_CPU) {
    hwloc_linuxfs_look_cpu(backend, dstatus);
    return 0;
  }

#ifdef HWLOC_HAVE_LINUXIO
  hwloc_topology_get_type_filter(topology, HWLOC_OBJ_PCI_DEVICE, &pfilter);
  hwloc_topology_get_type_filter(topology, HWLOC_OBJ_BRIDGE, &bfilter);
  hwloc_topology_get_type_filter(topology, HWLOC_OBJ_OS_DEVICE, &ofilter);
  hwloc_topology_get_type_filter(topology, HWLOC_OBJ_MISC, &mfilter);

  if (dstatus->phase == HWLOC_DISC_PHASE_PCI
      && (bfilter != HWLOC_TYPE_FILTER_KEEP_NONE
	  || pfilter != HWLOC_TYPE_FILTER_KEEP_NONE)) {
#ifdef HWLOC_HAVE_LINUXPCI
    hwloc_linuxfs_pci_look_pcidevices(backend);
    /* no need to run another PCI phase */
    dstatus->excluded_phases |= HWLOC_DISC_PHASE_PCI;
#endif /* HWLOC_HAVE_LINUXPCI */
  }

  if (dstatus->phase == HWLOC_DISC_PHASE_ANNOTATE
      && (bfilter != HWLOC_TYPE_FILTER_KEEP_NONE
	  || pfilter != HWLOC_TYPE_FILTER_KEEP_NONE)) {
#ifdef HWLOC_HAVE_LINUXPCI
    hwloc_linuxfs_pci_look_pcislots(backend);
#endif /* HWLOC_HAVE_LINUXPCI */
  }

  if (dstatus->phase == HWLOC_DISC_PHASE_IO
      && ofilter != HWLOC_TYPE_FILTER_KEEP_NONE) {
    unsigned osdev_flags = 0;
    if (getenv("HWLOC_VIRTUAL_LINUX_OSDEV"))
      osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_FIND_VIRTUAL;
    if (ofilter == HWLOC_TYPE_FILTER_KEEP_ALL)
      osdev_flags |= HWLOC_LINUXFS_OSDEV_FLAG_FIND_USB;

    hwloc_linuxfs_lookup_block_class(backend, osdev_flags);
    hwloc_linuxfs_lookup_dax_class(backend, osdev_flags);
    hwloc_linuxfs_lookup_net_class(backend, osdev_flags);
    hwloc_linuxfs_lookup_infiniband_class(backend, osdev_flags);
    hwloc_linuxfs_lookup_mic_class(backend, osdev_flags);
    if (ofilter != HWLOC_TYPE_FILTER_KEEP_IMPORTANT) {
      hwloc_linuxfs_lookup_drm_class(backend, osdev_flags);
      hwloc_linuxfs_lookup_dma_class(backend, osdev_flags);
    }
  }

  if (dstatus->phase == HWLOC_DISC_PHASE_MISC
      && mfilter != HWLOC_TYPE_FILTER_KEEP_NONE) {
    hwloc__get_firmware_dmi_memory_info(topology, backend->private_data);
  }
#endif /* HWLOC_HAVE_LINUXIO */

  return 0;
}

/*******************************
 ******* Linux component *******
 *******************************/

static void
hwloc_linux_backend_disable(struct hwloc_backend *backend)
{
  struct hwloc_linux_backend_data_s *data = backend->private_data;
#ifdef HAVE_OPENAT
  if (data->root_fd >= 0) {
    free(data->root_path);
    close(data->root_fd);
  }
#endif
#ifdef HWLOC_HAVE_LIBUDEV
  if (data->udev)
    udev_unref(data->udev);
#endif
  free(data);
}

static struct hwloc_backend *
hwloc_linux_component_instantiate(struct hwloc_topology *topology,
				  struct hwloc_disc_component *component,
				  unsigned excluded_phases __hwloc_attribute_unused,
				  const void *_data1 __hwloc_attribute_unused,
				  const void *_data2 __hwloc_attribute_unused,
				  const void *_data3 __hwloc_attribute_unused)
{
  struct hwloc_backend *backend;
  struct hwloc_linux_backend_data_s *data;
  const char * fsroot_path;
  int root = -1;
  char *env;

  backend = hwloc_backend_alloc(topology, component);
  if (!backend)
    goto out;

  data = malloc(sizeof(*data));
  if (!data) {
    errno = ENOMEM;
    goto out_with_backend;
  }

  backend->private_data = data;
  backend->discover = hwloc_look_linuxfs;
  backend->get_pci_busid_cpuset = hwloc_linux_backend_get_pci_busid_cpuset;
  backend->disable = hwloc_linux_backend_disable;

  /* default values */
  data->arch = HWLOC_LINUX_ARCH_UNKNOWN;
  data->is_knl = 0;
  data->is_amd_with_CU = 0;
  data->use_dt = 0;
  data->is_real_fsroot = 1;
  data->root_path = NULL;
  fsroot_path = getenv("HWLOC_FSROOT");
  if (!fsroot_path)
    fsroot_path = "/";

  if (strcmp(fsroot_path, "/")) {
#ifdef HAVE_OPENAT
    int flags;

    root = open(fsroot_path, O_RDONLY | O_DIRECTORY);
    if (root < 0)
      goto out_with_data;

    backend->is_thissystem = 0;
    data->is_real_fsroot = 0;
    data->root_path = strdup(fsroot_path);

    /* Since this fd stays open after hwloc returns, mark it as
       close-on-exec so that children don't inherit it.  Stevens says
       that we should GETFD before we SETFD, so we do. */
    flags = fcntl(root, F_GETFD, 0);
    if (-1 == flags ||
	-1 == fcntl(root, F_SETFD, FD_CLOEXEC | flags)) {
      close(root);
      root = -1;
      goto out_with_data;
    }
#else
    fprintf(stderr, "Cannot change Linux fsroot without openat() support.\n");
    errno = ENOSYS;
    goto out_with_data;
#endif
  }
  data->root_fd = root;

#ifdef HWLOC_HAVE_LIBUDEV
  data->udev = NULL;
  if (data->is_real_fsroot) {
    data->udev = udev_new();
  }
#endif

  data->dumped_hwdata_dirname = getenv("HWLOC_DUMPED_HWDATA_DIR");
  if (!data->dumped_hwdata_dirname)
    data->dumped_hwdata_dirname = (char *) RUNSTATEDIR "/hwloc/";

  data->use_numa_distances = 1;
  data->use_numa_distances_for_cpuless = 1;
  data->use_numa_initiators = 1;
  env = getenv("HWLOC_USE_NUMA_DISTANCES");
  if (env) {
    unsigned val = atoi(env);
    data->use_numa_distances = !!(val & 3); /* 2 implies 1 */
    data->use_numa_distances_for_cpuless = !!(val & 2);
    data->use_numa_initiators = !!(val & 4);
  }

  env = getenv("HWLOC_USE_DT");
  if (env)
    data->use_dt = atoi(env);

  return backend;

 out_with_data:
#ifdef HAVE_OPENAT
  free(data->root_path);
#endif
  free(data);
 out_with_backend:
  free(backend);
 out:
  return NULL;
}

static struct hwloc_disc_component hwloc_linux_disc_component = {
  "linux",
  HWLOC_DISC_PHASE_CPU | HWLOC_DISC_PHASE_PCI | HWLOC_DISC_PHASE_IO | HWLOC_DISC_PHASE_MISC | HWLOC_DISC_PHASE_ANNOTATE,
  HWLOC_DISC_PHASE_GLOBAL,
  hwloc_linux_component_instantiate,
  50,
  1,
  NULL
};

const struct hwloc_component hwloc_linux_component = {
  HWLOC_COMPONENT_ABI,
  NULL, NULL,
  HWLOC_COMPONENT_TYPE_DISC,
  0,
  &hwloc_linux_disc_component
};
