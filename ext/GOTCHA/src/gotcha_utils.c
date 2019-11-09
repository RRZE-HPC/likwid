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

#include "gotcha_utils.h"
#include "gotcha_dl.h"
#include "tool.h"
#include "libc_wrappers.h"
#include "elf_ops.h"
#include "gotcha/gotcha.h"
#include <stdlib.h>
#include "hash.h"

int debug_level;
static void debug_init()
{
   static int debug_initialized = 0;

   char *debug_str;
   if (debug_initialized) {
      return;
   }
   debug_initialized = 1;
   
   debug_str = gotcha_getenv(GOTCHA_DEBUG_ENV);
   if (!debug_str) {
      return;
   }

   debug_level = gotcha_atoi(debug_str);
   if (debug_level <= 0)
      debug_level = 1;

   debug_printf(0, "Gotcha debug initialized at level %d\n", debug_level);
}

hash_table_t function_hash_table;
hash_table_t notfound_binding_table;

static hash_table_t library_table;
static library_t *library_list = NULL;
unsigned int current_generation;

static hash_hashvalue_t link_map_hash(struct link_map *map)
{
   hash_hashvalue_t hashval = (hash_hashvalue_t) ((unsigned long) map);
   hashval ^= strhash(LIB_NAME(map));
   return hashval;
}

static int link_map_cmp(struct link_map *a, struct link_map *b)
{
   return ((unsigned long) a) < ((unsigned long) b);
}

static void setup_hash_tables() {
   create_hashtable(&library_table, 128, (hash_func_t) link_map_hash, (hash_cmp_t) link_map_cmp);
   create_hashtable(&function_hash_table, 4096, (hash_func_t) strhash, (hash_cmp_t) gotcha_strcmp);
   create_hashtable(&notfound_binding_table, 128, (hash_func_t) strhash, (hash_cmp_t) gotcha_strcmp);    
}

struct library_t *get_library(struct link_map *map)
{
   library_t *lib;
   int result;
   result = lookup_hashtable(&library_table, (hash_key_t) map, (hash_data_t *) &lib);
   if (result == -1)
      return NULL;
   return lib;
}

struct library_t *add_library(struct link_map *map)
{
   library_t *newlib = gotcha_malloc(sizeof(library_t));
   newlib->map = map;
   newlib->flags = 0;
   newlib->generation = 0;
   newlib->next = library_list;
   newlib->prev = NULL;
   if (library_list)
      library_list->prev = newlib;
   library_list = newlib;
   addto_hashtable(&library_table, (hash_key_t) map, (hash_data_t) newlib);
   return newlib;
}

void remove_library(struct link_map *map)
{
   library_t *lib = get_library(map);
   if (!lib)
      return;
   if (lib->prev)
      lib->prev->next = lib->next;
   if (lib->next)
      lib->next->prev = lib->prev;
   if (lib == library_list)
      library_list = library_list->next;
   removefrom_hashtable(&library_table, (hash_key_t) map);
   memset(lib, 0, sizeof(library_t));
   gotcha_free(lib);
}

void gotcha_init(){
   static int gotcha_initialized = 0;
   if(gotcha_initialized){
     return;
   }
   gotcha_initialized = 1;
   debug_init();
   setup_hash_tables();
   handle_libdl();
}

