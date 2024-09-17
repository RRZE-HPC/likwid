#define _GNU_SOURCE
#include "gotcha_dl.h"

#include <dlfcn.h>

#include "elf_ops.h"
#include "libc_wrappers.h"
#include "tool.h"

/**
 * structure used to pass lookup addr and return library address.
 */
struct Addrs {
  ElfW(Addr) lookup_addr;  // input
  struct link_map *lmap;   // output
  int found;
};
/**
 * This is a callback to get headers for each library.
 * We check if the caller's virtual address is between base address and the
 * virtual addr + memory size. Then we select the lmap based on base_addr and
 * name
 *
 * @param info, information about headers
 * @param size, size of headers
 * @param data, data from caller
 * @return
 */
int lib_header_callback(struct dl_phdr_info *info, size_t size, void *data) {
  struct Addrs *addrs = data;
  const char *name = NULL;
  ElfW(Addr) load_address;
  for (int i = 0; i < info->dlpi_phnum; ++i) {
    if (info->dlpi_phdr[i].p_type == PT_LOAD) {
      ElfW(Addr) base_addr = info->dlpi_addr;
      ElfW(Addr) start_addr = base_addr + info->dlpi_phdr[i].p_vaddr;
      ElfW(Addr) end_addr = start_addr + info->dlpi_phdr[i].p_memsz;
      if (addrs->lookup_addr >= start_addr && addrs->lookup_addr < end_addr) {
        name = info->dlpi_name;
        load_address = info->dlpi_addr;
        break;
      }
    }
  }
  if (name) {
    struct link_map *current = addrs->lmap;
    while (current) {
      if (strcmp(current->l_name, name) == 0 &&
          load_address == current->l_addr) {
        addrs->lmap = current;
        addrs->found = 1;
        return 1;
      }
      current = current->l_next;  // GCOVR_EXCL_LINE
    }
  }
  return 0;  // GCOVR_EXCL_LINE
}

/**
 * Implement the logic of _dl_sym for supporting RTLD_NEXT
 * 1. find the caller library using the program headers.
 * 2. find the second library which has the symbol for RTLD_NEXT
 * @param handle, handle for dl operation
 * @param name, name of the symbol
 * @param who, the virtual address of the caller
 * @return link_map pointer
 */

static struct link_map *gotchas_dlsym_rtld_next_lookup(const char *name,
                                                       void *who) {
  ElfW(Addr) caller = (ElfW(Addr))who;
  /* Iterative over the library headers and find the caller
   * the address of the caller is set in addrs->library_laddr
   **/
  struct Addrs addrs;
  addrs.lookup_addr = caller;
  addrs.lmap = _r_debug.r_map;
  addrs.found = 0;
  void *symbol;
  dl_iterate_phdr(lib_header_callback, &addrs);
  if (!addrs.found) {  // GCOVR_EXCL_START
    error_printf("RTLD_NEXT used in code not dynamically loaded");
    exit(127);
  }  // GCOVR_EXCL_STOP
  struct link_map *handle = addrs.lmap->l_next;
  while (handle) {
    /* lookup symbol on the next-to-next lib which has symbol
     * for RTLD_NEXT
     **/
    long result = lookup_exported_symbol(name, handle, &symbol);
    if (result != -1) {
      return handle;
    } else {
      debug_printf(3, "Symbol %s not found in the library %s\n", name,
                   LIB_NAME(handle));
    }
    handle = handle->l_next;
  }
  debug_printf(3, "Symbol %s not found in the libraries after caller\n", name);
  return NULL;
}

gotcha_wrappee_handle_t orig_dlopen_handle;
gotcha_wrappee_handle_t orig_dlsym_handle;

static int per_binding(hash_key_t key, hash_data_t data,
                       void *opaque KNOWN_UNUSED) {
  int result;
  struct internal_binding_t *binding = (struct internal_binding_t *)data;

  debug_printf(3, "Trying to re-bind %s from tool %s after dlopen\n",
               binding->user_binding->name,
               binding->associated_binding_table->tool->tool_name);

  if (!binding->user_binding->name) return 0;
  while (binding->next_binding) {
    binding = binding->next_binding;  // GCOVR_EXCL_START
    debug_printf(3,
                 "Selecting new innermost version of binding %s from tool "
                 "%s.\n",  // GCOVR_EXCL_LINE
                 binding->user_binding->name,
                 binding->associated_binding_table->tool->tool_name);
  }  // GCOVR_EXCL_STOP

  result = prepare_symbol(binding);
  if (result == -1) {  // GCOVR_EXCL_START
    debug_printf(3, "Still could not prepare binding %s after dlopen\n",
                 binding->user_binding->name);
    return 0;
  }  // GCOVR_EXCL_STOP

  removefrom_hashtable(&notfound_binding_table, key);
  return 0;
}

static void *dlopen_wrapper(const char *filename, int flags) {
  typeof(&dlopen_wrapper) orig_dlopen = gotcha_get_wrappee(orig_dlopen_handle);
  void *handle;
  debug_printf(1, "User called dlopen(%s, 0x%x)\n", filename,
               (unsigned int)flags);
  handle = orig_dlopen(filename, flags);

  debug_printf(
      2, "Searching new dlopened libraries for previously-not-found exports\n");
  foreach_hash_entry(&notfound_binding_table, NULL, per_binding);

  debug_printf(2, "Updating GOT entries for new dlopened libraries\n");
  update_all_library_gots(&function_hash_table);

  return handle;
}

static void *dlsym_wrapper(void *handle, const char *symbol_name) {
  typeof(&dlopen_wrapper) orig_dlopen = gotcha_get_wrappee(orig_dlopen_handle);
  typeof(&dlsym_wrapper) orig_dlsym = gotcha_get_wrappee(orig_dlsym_handle);
  struct internal_binding_t *binding;
  debug_printf(1, "User called dlsym(%p, %s)\n", handle, symbol_name);
  int result = lookup_hashtable(&function_hash_table, (hash_key_t)symbol_name,
                                (hash_data_t *)&binding);
  void *val = orig_dlsym(handle, symbol_name);
  if (result != -1) {
    void **wrappee_ptr = getInternalBindingAddressPointer(
        (struct internal_binding_t **)binding->user_binding->function_handle);
    if (val == NULL || *wrappee_ptr == val) {
      // if the wrapper is found and the wrappee is the function requested.
      // This is needed in cases where we wrap a function F1 from library A and
      // we dynamically load function F1 from library B. As name is same, we
      // need to make sure the wrappee are the same as well
      return binding->user_binding->wrapper_pointer;
    }
  }
  if (handle == RTLD_NEXT) {
    struct link_map *lib = gotchas_dlsym_rtld_next_lookup(
        symbol_name, __builtin_return_address(0));
    if (lib) {
      void *handle = orig_dlopen(lib->l_name, RTLD_NOW);
      void *symbol = orig_dlsym(handle, symbol_name);
      return symbol;
    }
    return NULL;
  } else {
    return val;
  }
}

struct gotcha_binding_t dl_binds[] = {
    {"dlopen", dlopen_wrapper, &orig_dlopen_handle},
    {"dlsym", dlsym_wrapper, &orig_dlsym_handle}};
void handle_libdl() { gotcha_wrap(dl_binds, 2, "gotcha"); }
