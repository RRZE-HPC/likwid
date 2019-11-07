#define _GNU_SOURCE
#include "gotcha_dl.h"
#include "tool.h"
#include "libc_wrappers.h"
#include "elf_ops.h"
#include <dlfcn.h>

void* _dl_sym(void* handle, const char* name, void* where);

gotcha_wrappee_handle_t orig_dlopen_handle;
gotcha_wrappee_handle_t orig_dlsym_handle;

static int per_binding(hash_key_t key, hash_data_t data, void *opaque KNOWN_UNUSED)
{
   int result;
   struct internal_binding_t *binding = (struct internal_binding_t *) data;

   debug_printf(3, "Trying to re-bind %s from tool %s after dlopen\n",
                binding->user_binding->name, binding->associated_binding_table->tool->tool_name);
   
   while (binding->next_binding) {
      binding = binding->next_binding;
      debug_printf(3, "Selecting new innermost version of binding %s from tool %s.\n",
                   binding->user_binding->name, binding->associated_binding_table->tool->tool_name);
   }
   
   result = prepare_symbol(binding);
   if (result == -1) {
      debug_printf(3, "Still could not prepare binding %s after dlopen\n", binding->user_binding->name);
      return 0;
   }

   removefrom_hashtable(&notfound_binding_table, key);
   return 0;
}

static void* dlopen_wrapper(const char* filename, int flags) {
   typeof(&dlopen_wrapper) orig_dlopen = gotcha_get_wrappee(orig_dlopen_handle);
   void *handle;
   debug_printf(1, "User called dlopen(%s, 0x%x)\n", filename, (unsigned int) flags);
   handle = orig_dlopen(filename,flags);

   debug_printf(2, "Searching new dlopened libraries for previously-not-found exports\n");
   foreach_hash_entry(&notfound_binding_table, NULL, per_binding);

   debug_printf(2, "Updating GOT entries for new dlopened libraries\n");
   update_all_library_gots(&function_hash_table);
  
   return handle;
}

static void* dlsym_wrapper(void* handle, const char* symbol_name){
  typeof(&dlsym_wrapper) orig_dlsym = gotcha_get_wrappee(orig_dlsym_handle);
  struct internal_binding_t *binding;
  int result;
  
  if(handle == RTLD_NEXT){
    return _dl_sym(RTLD_NEXT, symbol_name ,__builtin_return_address(0));
  }
  
  result = lookup_hashtable(&function_hash_table, (hash_key_t) symbol_name, (hash_data_t *) &binding);
  if (result == -1)
     return orig_dlsym(handle, symbol_name);
  else
     return binding->user_binding->wrapper_pointer;
}

struct gotcha_binding_t dl_binds[] = {
  {"dlopen", dlopen_wrapper, &orig_dlopen_handle},
  {"dlsym", dlsym_wrapper, &orig_dlsym_handle}
};     
void handle_libdl(){
  gotcha_wrap(dl_binds, 2, "gotcha");
}

