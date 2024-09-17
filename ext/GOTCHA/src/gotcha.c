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

#define _GNU_SOURCE
#include "gotcha/gotcha.h"

#include <link.h>

#include "elf_ops.h"
#include "gotcha/gotcha_types.h"
#include "gotcha_auxv.h"
#include "gotcha_dl.h"
#include "gotcha_utils.h"
#include "libc_wrappers.h"
#include "tool.h"
#include "translations.h"

static void writeAddress(void *write, void *value) { *(void **)write = value; }

static void **getBindingAddressPointer(struct gotcha_binding_t *in) {
  return (void **)in->function_handle;
}

static void setBindingAddressPointer(struct gotcha_binding_t *in, void *value) {
  void **target = getBindingAddressPointer(in);
  debug_printf(3, "Updating binding address pointer at %p to %p\n", target,
               value);
  writeAddress(target, value);
}

void **getInternalBindingAddressPointer(struct internal_binding_t **in) {
  return (void **)&((*in)->wrappee_pointer);
}

static void setInternalBindingAddressPointer(void **in, void *value) {
  void **target =
      getInternalBindingAddressPointer((struct internal_binding_t **)in);
  debug_printf(3, "Updating binding address pointer at %p to %p\n", target,
               value);
  writeAddress(target, value);
}

long lookup_exported_symbol(const char *name, const struct link_map *lib,
                            void **symbol) {
  long result;
  if (is_vdso(lib)) {
    debug_printf(2, "Skipping VDSO library at 0x%lx with name %s\n",
                 lib->l_addr, LIB_NAME(lib));
    return -1;
  }
  debug_printf(2, "Searching for exported symbols in %s\n", LIB_NAME(lib));
  INIT_DYNAMIC(lib);

  if (!gnu_hash && !elf_hash) {  // GCOVR_EXCL_START
    debug_printf(3, "Library %s does not export or import symbols\n",
                 LIB_NAME(lib));
    return -1;
  }  // GCOVR_EXCL_STOP
  result = -1;
  if (gnu_hash) {
    debug_printf(3, "Checking GNU hash for %s in %s\n", name, LIB_NAME(lib));
    result = lookup_gnu_hash_symbol(name, symtab, versym, strtab,
                                    (struct gnu_hash_header *)gnu_hash);
  }
  if (elf_hash && result == -1) {
    debug_printf(3, "Checking ELF hash for %s in %s\n", name, LIB_NAME(lib));
    result = lookup_elf_hash_symbol(name, symtab, versym, strtab,
                                    (ElfW(Word) *)elf_hash);
  }
  if (result == -1) {
    debug_printf(3, "%s not found in %s\n", name, LIB_NAME(lib));
    return -1;
  }
  if (!GOTCHA_CHECK_VISIBILITY(symtab[result])) {  // GCOVR_EXCL_START
    debug_printf(3, "Symbol %s found but not exported in %s\n", name,
                 LIB_NAME(lib));
    return -1;
  }  // GCOVR_EXCL_STOP

  debug_printf(2, "Symbol %s found in %s at 0x%lx\n", name, LIB_NAME(lib),
               symtab[result].st_value + lib->l_addr);
  *symbol = (void *)(symtab[result].st_value + lib->l_addr);
  return result;
}

int prepare_symbol(struct internal_binding_t *binding) {
  int result;
  struct link_map *lib;
  struct gotcha_binding_t *user_binding = binding->user_binding;

  debug_printf(2, "Looking up exported symbols for %s\n", user_binding->name);
  for (lib = _r_debug.r_map; lib != 0; lib = lib->l_next) {
    struct library_t *int_library = get_library(lib);
    if (!int_library) {
      debug_printf(3, "Creating new library object for %s\n", LIB_NAME(lib));
      int_library = add_library(lib);
    }
    void *symbol;
    result = lookup_exported_symbol(user_binding->name, lib, &symbol);
    if (result != -1) {
      setInternalBindingAddressPointer(user_binding->function_handle, symbol);
      return 0;
    }
  }
  debug_printf(1, "Symbol %s was found in program\n", user_binding->name);
  return -1;
}

static void insert_at_head(struct internal_binding_t *binding,
                           struct internal_binding_t *head) {
  binding->next_binding = head;
  setInternalBindingAddressPointer(binding->user_binding->function_handle,
                                   head->user_binding->wrapper_pointer);
  removefrom_hashtable(&function_hash_table,
                       (void *)binding->user_binding->name);
  addto_hashtable(&function_hash_table, (void *)binding->user_binding->name,
                  (void *)binding);
}

static void insert_after_pos(struct internal_binding_t *binding,
                             struct internal_binding_t *pos) {
  setInternalBindingAddressPointer(binding->user_binding->function_handle,
                                   pos->wrappee_pointer);
  setInternalBindingAddressPointer(pos->user_binding->function_handle,
                                   binding->user_binding->wrapper_pointer);
  binding->next_binding = pos->next_binding;
  pos->next_binding = binding;
}

#define RWO_NOCHANGE 0
#define RWO_NEED_LOOKUP (1 << 0)
#define RWO_NEED_BINDING (1 << 1)
static int rewrite_wrapper_orders(struct internal_binding_t *binding) {
  const char *name = binding->user_binding->name;
  int insert_priority = get_priority(binding->associated_binding_table->tool);

  if (gotcha_strcmp(name, "main") == 0) {
    if (!main_wrapped) {
      debug_printf(2, "Wrapping main with Gotcha's internal wrappers");
      main_wrapped = 1;
      gotcha_wrap(libc_main_wrappers, 1, "gotcha");
      gotcha_wrap(main_wrappers, 1, "gotcha");
    }
  }

  debug_printf(2,
               "gotcha_rewrite_wrapper_orders for binding %s in tool %s of "
               "priority %d\n",
               name, binding->associated_binding_table->tool->tool_name,
               insert_priority);

  struct internal_binding_t *head;
  int hash_result;
  hash_result =
      lookup_hashtable(&function_hash_table, (void *)name, (void **)&head);
  if (hash_result != 0) {
    debug_printf(2, "Adding new entry for %s to hash table\n", name);
    addto_hashtable(&function_hash_table, (void *)name, (void *)binding);
    return (RWO_NEED_LOOKUP | RWO_NEED_BINDING);
  }

  int head_priority = get_priority(head->associated_binding_table->tool);
  if (head_priority < insert_priority) {
    debug_printf(2,
                 "New binding priority %d is greater than head priority %d, "
                 "adding to head\n",
                 insert_priority, head_priority);
    insert_at_head(binding, head);
    return RWO_NEED_BINDING;
  }

  struct internal_binding_t *cur;
  for (cur = head; cur->next_binding; cur = cur->next_binding) {
    int next_priority =
        get_priority(cur->next_binding->associated_binding_table->tool);
    debug_printf(
        3,
        "Comparing binding for new insertion %d to binding for tool %s at %d\n",
        insert_priority,
        cur->next_binding->associated_binding_table->tool->tool_name,
        next_priority);
    if (next_priority < insert_priority) {
      break;
    }
    if (cur->user_binding->wrapper_pointer ==
        binding->user_binding->wrapper_pointer) {
      debug_printf(3, "Tool is already inserted.  Skipping binding rewrite\n");
      return RWO_NOCHANGE;
    }
  }
  debug_printf(2, "Inserting binding after tool %s\n",
               cur->associated_binding_table->tool->tool_name);
  insert_after_pos(binding, cur);
  return RWO_NOCHANGE;
}

static int update_lib_bindings(ElfW(Sym) * symbol KNOWN_UNUSED, char *name,
                               ElfW(Addr) offset, struct link_map *lmap,
                               hash_table_t *lookuptable) {
  int result;
  struct internal_binding_t *internal_binding;
  void **got_address;

  result = lookup_hashtable(lookuptable, name, (void **)&internal_binding);
  if (result != 0) return -1;
  got_address = (void **)(lmap->l_addr + offset);
  writeAddress(got_address, internal_binding->user_binding->wrapper_pointer);
  debug_printf(3, "Remapped call to %s at 0x%lx in %s to wrapper at 0x%p\n",
               name, (lmap->l_addr + offset), LIB_NAME(lmap),
               internal_binding->user_binding->wrapper_pointer);
  return 0;
}

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif
/**
 * structure used to pass lookup addr and return library address.
 */
struct Boundary {
  const char *l_name;     // input
  ElfW(Addr) load_addr;   // input
  ElfW(Addr) start_addr;  // output
  ElfW(Addr) end_addr;    // output
  int found;
};
int find_relro_boundary(struct dl_phdr_info *info, size_t size, void *data) {
  struct Boundary *boundary = data;
  int found = 0;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    if (info->dlpi_phdr[i].p_type == PT_LOAD) {
      if (strcmp(boundary->l_name, info->dlpi_name) == 0 &&
          boundary->load_addr == info->dlpi_addr) {
        found = 1;
        break;
      }
    }
  }
  if (found) {
    for (int i = 0; i < info->dlpi_phnum; ++i) {
      if (info->dlpi_phdr[i].p_type == PT_GNU_RELRO) {
        boundary->start_addr = boundary->load_addr + info->dlpi_phdr[i].p_vaddr;
        boundary->end_addr = boundary->start_addr + info->dlpi_phdr[i].p_memsz;
        boundary->found = 1;
        return 1;
      }
    }
  }
  return 0;
}

static int mark_got_writable(struct link_map *lib) {
  static ElfW(Addr) page_size = 0;
  INIT_DYNAMIC(lib);
  if (!got) return 0;

  if (!page_size) page_size = gotcha_getpagesize();

  ElfW(Addr) plt_got_size = MAX(rel_size, page_size);
  if (plt_got_size % page_size) {  // GCOVR_EXCL_START
    plt_got_size += page_size - ((plt_got_size) % page_size);
  }  // GCOVR_EXCL_STOP
  ElfW(Addr) plt_got_addr = BOUNDARY_BEFORE(got, (ElfW(Addr))page_size);
  struct Boundary boundary;
  boundary.l_name = lib->l_name;
  boundary.load_addr = lib->l_addr;
  boundary.found = 0;
  dl_iterate_phdr(find_relro_boundary, &boundary);
  int plt_got_written = 0;
  if (boundary.found) {
    ElfW(Addr) got_end = MAX(boundary.end_addr, page_size);
    if (got_end % page_size) {
      got_end += page_size - ((got_end) % page_size);  // GCOVR_EXCL_LINE
    }
    ElfW(Addr) got_addr =
        BOUNDARY_BEFORE(boundary.start_addr, (ElfW(Addr))page_size);
    ElfW(Addr) got_size = got_end - got_addr;
    /**
     * The next two cases are to optimize mprotect calls to do both pages
     * together if they align. We do not have such usecase yet and hence
     * ignoring from coverage.
     */
    if (got_addr == plt_got_addr + plt_got_size) {
      // Haven't seen a library till now where got_addr > plt_got_addr
      // GCOVR_EXCL_START
      debug_printf(3,
                   "Setting library %s GOT and PLT table "
                   "from %p to +%lu to writeable\n",
                   LIB_NAME(lib), (void *)plt_got_addr,
                   plt_got_size + got_size);
      int res = gotcha_mprotect((void *)plt_got_addr, plt_got_size + got_size,
                                PROT_READ | PROT_WRITE | PROT_EXEC);
      // mprotect returns -1 on an error
      if (res == -1) {
        error_printf(
            "GOTCHA attempted to mark both GOT and PLT GOT tables as writable "
            "and was unable to do so, "
            "calls to wrapped functions may likely fail.\n");
      }
      plt_got_written = 1;
      // GCOVR_EXCL_STOP
    } else if (plt_got_addr == got_addr + got_size) {
      // This is a more common case.
      debug_printf(3,
                   "Setting library %s GOT and PLT table "
                   "from %p to +%lu to writeable\n",
                   LIB_NAME(lib), (void *)got_addr, plt_got_size + got_size);
      int res = gotcha_mprotect((void *)got_addr, plt_got_size + got_size,
                                PROT_READ | PROT_WRITE | PROT_EXEC);
      // mprotect returns -1 on an error
      if (res == -1) {  // GCOVR_EXCL_START
        error_printf(
            "GOTCHA attempted to mark both GOT and PLT GOT tables as writable "
            "and was unable to do so, "
            "calls to wrapped functions may likely fail.\n");
      }  // GCOVR_EXCL_STOP
      plt_got_written = 1;
    } else {
      debug_printf(
          3, "Setting library %s only GOT table from %p to +%lu to writeable\n",
          LIB_NAME(lib), (void *)got_addr, got_size);
      int res = gotcha_mprotect((void *)got_addr, got_size,
                                PROT_READ | PROT_WRITE | PROT_EXEC);
      if (res == -1) {  // GCOVR_EXCL_START
        error_printf(
            "GOTCHA attempted to mark the only GOT table as writable and was "
            "unable to do so, "
            "calls to wrapped functions may likely fail.\n");
      }  // GCOVR_EXCL_STOP
    }
  }
  if (!plt_got_written) {
    debug_printf(
        3,
        "Setting library %s only PLT GOT table from %p to +%lu to writeable\n",
        LIB_NAME(lib), (void *)plt_got_addr, plt_got_size);
    int res = gotcha_mprotect((void *)plt_got_addr, plt_got_size,
                              PROT_READ | PROT_WRITE | PROT_EXEC);
    // mprotect returns -1 on an error
    if (res == -1) {  // GCOVR_EXCL_START
      error_printf(
          "GOTCHA attempted to mark the only PLT GOT table as writable and was "
          "unable to do so, "
          "calls to wrapped functions may likely fail.\n");
    }  // GCOVR_EXCL_STOP
  }
  return 0;
}

static int update_library_got(struct link_map *map,
                              hash_table_t *bindingtable) {
  struct library_t *lib = get_library(map);
  if (!lib) {
    debug_printf(3, "Creating new library object for %s\n", LIB_NAME(map));
    lib = add_library(map);
  }

  if (!libraryFilterFunc(map)) {
    debug_printf(3, "Skipping library %s due to libraryFilterFunc\n",
                 LIB_NAME(map));
    return 0;
  }

  if (lib->generation == current_generation) {
    debug_printf(2,
                 "Library %s is already up-to-date.  Skipping GOT rewriting\n",
                 LIB_NAME(map));
    return 0;
  }

  if (!(lib->flags & LIB_GOT_MARKED_WRITEABLE)) {
    mark_got_writable(map);
    lib->flags |= LIB_GOT_MARKED_WRITEABLE;
  }

  FOR_EACH_PLTREL(map, update_lib_bindings, map, bindingtable);
  lib->generation = current_generation;
  return 0;
}

void update_all_library_gots(hash_table_t *bindings) {
  struct link_map *lib_iter;
  debug_printf(2, "Searching all callsites for %lu bindings\n",
               (unsigned long)bindings->entry_count);
  for (lib_iter = _r_debug.r_map; lib_iter != 0; lib_iter = lib_iter->l_next) {
    update_library_got(lib_iter, bindings);
  }
}

GOTCHA_EXPORT enum gotcha_error_t gotcha_wrap(
    struct gotcha_binding_t *user_bindings, int num_actions,
    const char *tool_name) {
  int i, not_found = 0, new_bindings_count = 0;
  tool_t *tool;
  hash_table_t new_bindings;

  gotcha_init();

  debug_printf(1, "User called gotcha_wrap for tool %s with %d bindings\n",
               tool_name, num_actions);
  if (debug_level >= 3) {
    for (i = 0; i < num_actions; i++) {
      debug_bare_printf(3, "\t%d: %s will map to %p\n", i,
                        user_bindings[i].name,
                        user_bindings[i].wrapper_pointer);
    }
  }
  debug_printf(3, "Initializing %d user binding entries to NULL\n",
               num_actions);
  for (i = 0; i < num_actions; i++) {
    setBindingAddressPointer(&user_bindings[i], NULL);
  }

  if (!tool_name) tool_name = "[UNSPECIFIED]";
  tool = get_tool(tool_name);
  if (!tool) tool = create_tool(tool_name);
  if (!tool) {
    error_printf("Failed to create tool %s\n", tool_name);  // GCOVR_EXCL_LINE
    return GOTCHA_INTERNAL;                                 // GCOVR_EXCL_LINE
  }

  current_generation++;
  debug_printf(2, "Moved current_generation to %u in gotcha_wrap\n",
               current_generation);

  debug_printf(
      2,
      "Creating internal binding data structures and adding binding to tool\n");
  binding_t *bindings = add_binding_to_tool(tool, user_bindings, num_actions);
  if (!bindings) {  // GCOVR_EXCL_START
    error_printf("Failed to create bindings for tool %s\n", tool_name);
    return GOTCHA_INTERNAL;
  }  // GCOVR_EXCL_STOP

  debug_printf(2, "Processing %d bindings\n", num_actions);
  for (i = 0; i < num_actions; i++) {
    struct internal_binding_t *binding = bindings->internal_bindings + i;
    int result = rewrite_wrapper_orders(binding);
    if (result & RWO_NEED_LOOKUP) {
      debug_printf(2, "Symbol %s needs lookup operation\n",
                   binding->user_binding->name);
      int presult = prepare_symbol(binding);
      if (presult == -1) {
        debug_printf(
            2,
            "Stashing %s in notfound_binding table to re-lookup on dlopens\n",
            binding->user_binding->name);
        addto_hashtable(&notfound_binding_table,
                        (hash_key_t)binding->user_binding->name,
                        (hash_data_t)binding);
        not_found++;
      }
    }
    if (result & RWO_NEED_BINDING) {
      debug_printf(2, "Symbol %s needs binding from application\n",
                   binding->user_binding->name);
      if (!new_bindings_count) {
        create_hashtable(&new_bindings, num_actions * 2, (hash_func_t)strhash,
                         (hash_cmp_t)gotcha_strcmp);
      }
      addto_hashtable(&new_bindings, (void *)binding->user_binding->name,
                      (void *)binding);
      new_bindings_count++;
    }
  }

  if (new_bindings_count) {
    update_all_library_gots(&new_bindings);
    destroy_hashtable(&new_bindings);
  }

  if (not_found) {
    debug_printf(1, "Could not find bindings for %d / %d functions\n",
                 not_found, num_actions);
    return GOTCHA_FUNCTION_NOT_FOUND;
  }
  debug_printf(1, "Gotcha wrap completed successfully\n");
  return GOTCHA_SUCCESS;
}

static enum gotcha_error_t gotcha_configure_int(
    const char *tool_name, enum gotcha_config_key_t configuration_key,
    int value) {
  tool_t *tool = get_tool(tool_name);
  if (tool == NULL) {
    tool = create_tool(tool_name);
  }
  if (configuration_key == GOTCHA_PRIORITY) {
    tool->config.priority = value;
  } else {
    error_printf("Invalid property being configured on tool %s\n", tool_name);
    return GOTCHA_INTERNAL;
  }
  return GOTCHA_SUCCESS;
}

GOTCHA_EXPORT enum gotcha_error_t gotcha_set_priority(const char *tool_name,
                                                      int value) {
  gotcha_init();
  debug_printf(1, "User called gotcha_set_priority(%s, %d)\n", tool_name,
               value);
  enum gotcha_error_t error_on_set =
      gotcha_configure_int(tool_name, GOTCHA_PRIORITY, value);
  if (error_on_set != GOTCHA_SUCCESS) {
    return error_on_set;  // GCOVR_EXCL_LINE
  }
  tool_t *tool_to_place = get_tool(tool_name);
  if (!tool_to_place) {  // will not happen as gotcha_configure_init creats tool
                         // if not exists
    tool_to_place = create_tool(tool_name);  // GCOVR_EXCL_LINE
  }
  remove_tool_from_list(tool_to_place);
  reorder_tool(tool_to_place);
  return GOTCHA_SUCCESS;
}

GOTCHA_EXPORT enum gotcha_error_t gotcha_get_priority(const char *tool_name,
                                                      int *priority) {
  gotcha_init();
  return get_configuration_value(tool_name, GOTCHA_PRIORITY, priority);
}

GOTCHA_EXPORT void *gotcha_get_wrappee(gotcha_wrappee_handle_t handle) {
  return ((struct internal_binding_t *)handle)->wrappee_pointer;
}
