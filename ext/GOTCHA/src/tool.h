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

#if !defined(TOOL_H_)
#define TOOL_H_

#include "gotcha/gotcha.h"
#include "gotcha/gotcha_types.h"
#include "hash.h"

struct tool_t;

#define UNSET_PRIORITY (-1)

enum gotcha_config_key_t {
  GOTCHA_PRIORITY
};

/**
 * A structure representing how a given tool's bindings are configured
 */
struct gotcha_configuration_t {
  int priority;
};

/**
 * A per-library structure
 **/
#define LIB_GOT_MARKED_WRITEABLE (1 << 0)
#define LIB_PRESENT              (1 << 1)
typedef struct library_t {
   struct link_map *map;
   struct library_t *next;
   struct library_t *prev;
   unsigned int generation;
   int flags;
} library_t;
struct library_t *get_library(struct link_map *map);
struct library_t *add_library(struct link_map *map);
void remove_library(struct link_map *map);
extern unsigned int current_generation;
   
/**
 * The internal structure that matches the external gotcha_binding_t.
 * In addition to the data specified in the gotcha_binding_t, we add:
 * - a linked-list pointer to the next binding table for this tool
 * - a linked-list pointer to the next binding table
 **/
typedef struct binding_t {
   struct tool_t *tool;
   struct internal_binding_t *internal_bindings;
   int internal_bindings_size;
   hash_table_t binding_hash;
   struct binding_t *next_tool_binding;
   struct binding_t *next_binding;
} binding_t;

/**
 * A structure for representing tools. Once we support stacking multiple
 * tools this will become more important.
 **/
typedef struct tool_t {
   const char *tool_name;
   binding_t *binding;
   struct tool_t *next_tool;
   struct gotcha_configuration_t config;
   hash_table_t child_tools;
   struct tool_t * parent_tool;
} tool_t;

struct internal_binding_t {
  struct binding_t* associated_binding_table;
  struct gotcha_binding_t* user_binding;
  struct internal_binding_t* next_binding;
  void* wrappee_pointer;
};

tool_t *create_tool(const char *tool_name);
tool_t *get_tool(const char *tool_name);
tool_t *get_tool_list();
void reorder_tool(tool_t* new_tool);
void remove_tool_from_list(struct tool_t* target);
void print_tools();

binding_t *add_binding_to_tool(tool_t *tool, struct gotcha_binding_t *user_binding, int user_binding_size);
binding_t *get_bindings();
binding_t *get_tool_bindings(tool_t *tool);

struct gotcha_configuration_t get_default_configuration();
enum gotcha_error_t get_configuration_value(const char* tool_name, enum gotcha_config_key_t key, void* location_to_store_result);
int get_priority(tool_t *tool);
int tool_equal(tool_t* tool_1, tool_t* tool_2);

#endif
