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
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#include "gotcha_auxv.h"

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "gotcha_utils.h"
#include "libc_wrappers.h"

static ElfW(Ehdr) *vdso_ehdr = NULL;
static unsigned int auxv_pagesz = 0;
#define BUFFER_LEN 4096

int parse_auxv_contents() {
  char name[] = "/proc/self/auxv";
  int fd, done = 0;
  char buffer[BUFFER_LEN];
  const ssize_t buffer_size = BUFFER_LEN;
  ssize_t offset = 0, result;
  ElfW(auxv_t) * auxv, *a;
  static int parsed_auxv = 0;

  if (parsed_auxv) return parsed_auxv == -1 ? parsed_auxv : 0;
  parsed_auxv = 1;

  fd = gotcha_open(name, O_RDONLY);
  if (fd == -1) {
    parsed_auxv = -1;  // GCOVR_EXCL_LINE
    return -1;         // GCOVR_EXCL_LINE
  }

  do {
    for (;;) {
      result = gotcha_read(fd, buffer + offset, buffer_size - offset);
      if (result == -1) {
        if (errno == EINTR)  // GCOVR_EXCL_START
          continue;
        gotcha_close(fd);
        parsed_auxv = -1;
        return -1;
      }  // GCOVR_EXCL_STOP
      if (result == 0) {
        gotcha_close(fd);
        done = 1;
        break;
      }
      if (offset == buffer_size) {
        break;  // GCOVR_EXCL_LINE
      }
      offset += result;
    }

    auxv = (ElfW(auxv_t) *)buffer;
    for (a = auxv; a->a_type != AT_NULL; a++) {
      if (a->a_type == AT_SYSINFO_EHDR) {
        vdso_ehdr = (ElfW(Ehdr) *)a->a_un.a_val;
      } else if (a->a_type == AT_PAGESZ) {
        auxv_pagesz = (int)a->a_un.a_val;
      }
    }
  } while (!done);

  return 0;
}

struct link_map *get_vdso_from_auxv() {
  struct link_map *m;

  ElfW(Phdr) *vdso_phdrs = NULL;
  ElfW(Half) vdso_phdr_num, p;
  ElfW(Addr) vdso_dynamic = 0;

  parse_auxv_contents();
  if (vdso_ehdr) {
    vdso_phdrs =
        (ElfW(Phdr) *)(vdso_ehdr->e_phoff + ((unsigned char *)vdso_ehdr));
    vdso_phdr_num = vdso_ehdr->e_phnum;

    for (p = 0; p < vdso_phdr_num; p++) {
      if (vdso_phdrs[p].p_type == PT_DYNAMIC) {
        vdso_dynamic = (ElfW(Addr))vdso_phdrs[p].p_vaddr;
      }
    }

    for (m = _r_debug.r_map; m; m = m->l_next) {
      if (m->l_addr + vdso_dynamic == (ElfW(Addr))m->l_ld) {
        return m;
      }
    }
  }
  return NULL;  // GCOVR_EXCL_LINE
}

unsigned int get_auxv_pagesize() {
  int result;
  result = parse_auxv_contents();
  return result == -1 ? 0 : auxv_pagesz;
}

static char *vdso_aliases[] = {"linux-vdso.so", "linux-gate.so", NULL};

struct link_map *get_vdso_from_aliases() {
  struct link_map *m;
  char **aliases;

  for (m = _r_debug.r_map; m; m = m->l_next) {
    for (aliases = vdso_aliases; *aliases; aliases++) {
      if (m->l_name && gotcha_strcmp(m->l_name, *aliases) == 0) {
        return m;  // GCOVR_EXCL_LINE
      }
    }
  }
  return NULL;
}

static int read_line(char *line, int size, int fd) {
  int i;
  for (i = 0; i < size - 1; i++) {
    int result = gotcha_read(fd, line + i, 1);
    if (result == -1 && errno == EINTR) continue;  // GCOVR_EXCL_LINE
    if (result == -1 || result == 0) {
      line[i] = '\0';  // GCOVR_EXCL_LINE
      return -1;       // GCOVR_EXCL_LINE
    }
    if (line[i] == '\n') {
      line[i + 1] = '\0';
      return 0;
    }
  }
  line[size - 1] = '\0';  // GCOVR_EXCL_LINE
  return 0;               // GCOVR_EXCL_LINE
}

static int read_hex(char *str, unsigned long *val) {
  unsigned long local_val = 0, len = 0;
  for (;;) {
    if (*str >= '0' && *str <= '9') {
      local_val = (local_val * 16) + (*str - '0');
      len++;
    } else if (*str >= 'a' && *str <= 'f') {
      local_val = (local_val * 16) + (*str - 'a' + 10);
      len++;
    } else if (*str >= 'A' && *str <= 'F') {
      local_val = (local_val * 16) + (*str - 'A' + 10);
      len++;
    } else {
      *val = local_val;
      return len;
    }
    str++;
  }
}

static int read_word(char *str, char *word, int word_size) {
  int word_cur = 0;
  int len = 0;
  while (*str == ' ' || *str == '\t' || *str == '\n') {
    str++;
    len++;
  }
  if (*str == '\0') {
    *word = '\0';
    return len;
  }
  while (*str != ' ' && *str != '\t' && *str != '\n' && *str != '\0') {
    if (word && word_cur >= word_size) {
      if (word_size > 0 && word)  // GCOVR_EXCL_START
        word[word_size - 1] = '\0';
      return word_cur;
    }  // GCOVR_EXCL_STOP
    if (word) word[word_cur] = *str;
    word_cur++;
    str++;
    len++;
  }
  if (word_cur >= word_size) word_cur--;
  if (word) word[word_cur] = '\0';
  return len;
}

struct link_map *get_vdso_from_maps() {
  int maps, hit_eof;
  ElfW(Addr) addr_begin, addr_end, dynamic;
  char name[BUFFER_LEN], line[BUFFER_LEN], *line_pos;
  struct link_map *m;
  maps = gotcha_open("/proc/self/maps", O_RDONLY);
  for (;;) {
    hit_eof = read_line(line, BUFFER_LEN, maps);
    if (hit_eof) {
      gotcha_close(maps);  // GCOVR_EXCL_LINE
      return NULL;         // GCOVR_EXCL_LINE
    }
    line_pos = line;
    line_pos += read_hex(line_pos, &addr_begin);
    if (*line_pos != '-') continue;  // GCOVR_EXCL_LINE
    line_pos++;
    line_pos += read_hex(line_pos, &addr_end);
    line_pos += read_word(line_pos, NULL, 0);
    line_pos += read_word(line_pos, NULL, 0);
    line_pos += read_word(line_pos, NULL, 0);
    line_pos += read_word(line_pos, NULL, 0);
    line_pos += read_word(line_pos, name, sizeof(name));
    if (gotcha_strcmp(name, "[vdso]") == 0) {
      gotcha_close(maps);
      break;
    }
  }

  for (m = _r_debug.r_map; m; m = m->l_next) {
    dynamic = (ElfW(Addr))m->l_ld;
    if (dynamic >= addr_begin && dynamic < addr_end) return m;
  }

  return NULL;  // GCOVR_EXCL_LINE
}

int is_vdso(const struct link_map *map) {
  static int vdso_checked = 0;
  static struct link_map *vdso = NULL;
  struct link_map *result;

  if (!map) return 0;
  if (vdso_checked) return (map == vdso);

  vdso_checked = 1;

  result = get_vdso_from_aliases();
  if (result) {
    vdso = result;         // GCOVR_EXCL_LINE
    return (map == vdso);  // GCOVR_EXCL_LINE
  }

  result = get_vdso_from_auxv();
  if (result) {
    vdso = result;
    return (map == vdso);
  }

  result = get_vdso_from_maps();  // GCOVR_EXCL_START
  if (result) {
    vdso = result;
    return (map == vdso);
  }

  return 0;
}  // GCOVR_EXCL_STOP
