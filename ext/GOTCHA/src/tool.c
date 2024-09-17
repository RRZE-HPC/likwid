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

#include "tool.h"

#include "gotcha_utils.h"
#include "libc_wrappers.h"

static tool_t *tools = NULL;
static binding_t *all_bindings = NULL;

tool_t *get_tool_list() { return tools; }

int tool_equal(tool_t *t1, tool_t *t2) {
  return gotcha_strcmp(t1->tool_name, t2->tool_name);
}

void remove_tool_from_list(struct tool_t *target) {
  if (!tools) {
    return;  // GCOVR_EXCL_LINE
  }
  if (!tool_equal(tools, target)) {
    tools = tools->next_tool;
    return;
  }
  struct tool_t *cur = tools;
  while ((cur != NULL) && (cur->next_tool != NULL) &&
         (tool_equal(cur->next_tool, target))) {
    cur = cur->next_tool;
  }
  if (!tool_equal(cur->next_tool, target)) {
    cur->next_tool = target->next_tool;
  }
}

void reorder_tool(tool_t *new_tool) {
  int new_priority = new_tool->config.priority;
  if (tools == NULL || tools->config.priority >= new_priority) {
    new_tool->next_tool = tools;
    tools = new_tool;
  } else {
    struct tool_t *cur = tools;
    while ((cur->next_tool != NULL) &&
           cur->next_tool->config.priority < new_priority) {
      cur = cur->next_tool;
    }
    new_tool->next_tool = cur->next_tool;
    cur->next_tool = new_tool;
  }
}

tool_t *create_tool(const char *tool_name) {
  debug_printf(1, "Found no existing tool with name %s\n", tool_name);
  // TODO: ensure free
  tool_t *newtool = (tool_t *)gotcha_malloc(sizeof(tool_t));
  if (!newtool) {
    error_printf("Failed to malloc tool %s\n", tool_name);  // GCOVR_EXCL_LINE
    return NULL;                                            // GCOVR_EXCL_LINE
  }
  newtool->tool_name = tool_name;
  newtool->binding = NULL;
  // newtool->next_tool = tools;
  newtool->config = get_default_configuration();
  reorder_tool(newtool);
  newtool->parent_tool = NULL;
  create_hashtable(&newtool->child_tools, 24, (hash_func_t)strhash,
                   (hash_cmp_t)gotcha_strcmp);
  // tools = newtool;
  debug_printf(1, "Created new tool %s\n", tool_name);
  return newtool;
}

tool_t *get_tool(const char *tool_name) {
  tool_t *t;
  for (t = tools; t; t = t->next_tool) {
    if (gotcha_strcmp(tool_name, t->tool_name) == 0) {
      return t;
    }
  }
  return NULL;
}

binding_t *add_binding_to_tool(tool_t *tool,
                               struct gotcha_binding_t *user_binding,
                               int user_binding_size) {
  binding_t *newbinding;
  int result, i;
  newbinding = (binding_t *)gotcha_malloc(sizeof(binding_t));
  newbinding->tool = tool;
  struct internal_binding_t *internal_bindings =
      (struct internal_binding_t *)gotcha_malloc(
          sizeof(struct internal_binding_t) * user_binding_size);
  for (i = 0; i < user_binding_size; i++) {
    internal_bindings[i].next_binding = NULL;
    internal_bindings[i].user_binding = &user_binding[i];
    *(user_binding[i].function_handle) = &internal_bindings[i];
    internal_bindings[i].associated_binding_table = newbinding;
  }
  newbinding->internal_bindings = internal_bindings;
  newbinding->internal_bindings_size = user_binding_size;
  result = create_hashtable(&newbinding->binding_hash, user_binding_size * 2,
                            (hash_func_t)strhash, (hash_cmp_t)gotcha_strcmp);
  if (result != 0) {  // GCOVR_EXCL_START
    error_printf("Could not create hash table for %s\n", tool->tool_name);
    goto error;  // error is a label which frees allocated resources and returns
  }              // GCOVR_EXCL_STOP

  for (i = 0; i < user_binding_size; i++) {
    result =
        addto_hashtable(&newbinding->binding_hash, (void *)user_binding[i].name,
                        (void *)(internal_bindings + i));
    if (result != 0) {  // GCOVR_EXCL_START
      error_printf("Could not add hash entry for %s to table for tool %s\n",
                   user_binding[i].name, tool->tool_name);
      goto error;  // error is a label which frees allocated resources and
                   // returns NULL
    }              // GCOVR_EXCL_STOP
  }

  newbinding->next_tool_binding = tool->binding;
  tool->binding = newbinding;

  newbinding->next_binding = all_bindings;
  all_bindings = newbinding;

  debug_printf(2, "Created new binding table of size %d for tool %s\n",
               user_binding_size, tool->tool_name);
  return newbinding;

error:  // GCOVR_EXCL_START
  if (newbinding) gotcha_free(newbinding);
  return NULL;
}  // GCOVR_EXCL_STOP

binding_t *get_bindings() { return all_bindings; }

binding_t *get_tool_bindings(tool_t *tool) { return tool->binding; }

struct gotcha_configuration_t get_default_configuration() {
  struct gotcha_configuration_t result;
  result.priority = UNSET_PRIORITY;
  return result;
}

enum gotcha_error_t get_default_configuration_value(
    enum gotcha_config_key_t key, void *data) {
  struct gotcha_configuration_t config = get_default_configuration();
  if (key == GOTCHA_PRIORITY) {
    *((int *)(data)) = config.priority;
  }
  return GOTCHA_SUCCESS;
}

enum gotcha_error_t get_configuration_value(const char *tool_name,
                                            enum gotcha_config_key_t key,
                                            void *location_to_store_result) {
  struct tool_t *tool = get_tool(tool_name);
  if (tool == NULL) {
    error_printf("Property being examined for nonexistent tool %s\n",
                 tool_name);
    return GOTCHA_INVALID_TOOL;
  }
  get_default_configuration_value(key, location_to_store_result);
  int found_valid_value = 0;
  while ((tool != NULL) && !(found_valid_value)) {
    struct gotcha_configuration_t config = tool->config;
    if (key == GOTCHA_PRIORITY) {
      int current_priority = config.priority;
      if (current_priority != UNSET_PRIORITY) {
        *((int *)(location_to_store_result)) = config.priority;
        found_valid_value = 1;
        return GOTCHA_SUCCESS;
      }
    } else {
      error_printf("Invalid property being configured on tool %s\n", tool_name);
      return GOTCHA_INTERNAL;
    }
    tool = tool->parent_tool;
  }
  return GOTCHA_SUCCESS;
}

int get_priority(tool_t *tool) { return tool->config.priority; }
