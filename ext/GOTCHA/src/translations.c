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
#include <gotcha/gotcha.h>
#include "translations.h"
#include "gotcha_utils.h"

int main_wrapped;
gotcha_wrappee_handle_t gotcha_internal_libc_main_wrappee_handle;
gotcha_wrappee_handle_t gotcha_internal_main_wrappee_handle;

int gotcha_internal_main(int argc, char** argv, char** envp){
  main_t underlying_main = gotcha_get_wrappee(gotcha_internal_main_wrappee_handle); 
  return underlying_main(argc, argv, envp);
}
int gotcha_internal_libc_start_main(int (*main_arg)(int, char**, char**) KNOWN_UNUSED, int argc, char** argv, void (*init)(), void (*fini)(), void (*rtld_fini)(), void* stack_end){
   libc_start_main_t underlying_libc_main = gotcha_get_wrappee(gotcha_internal_libc_main_wrappee_handle);
   main_t underlying_main = gotcha_get_wrappee(gotcha_internal_main_wrappee_handle);
   return underlying_libc_main(underlying_main, argc, argv, init, fini, rtld_fini, stack_end);
}

struct gotcha_binding_t libc_main_wrappers[] = {
  {"__libc_start_main", gotcha_internal_libc_start_main, &gotcha_internal_libc_main_wrappee_handle}
};
struct gotcha_binding_t main_wrappers[] = {
  {"main", gotcha_internal_main, &gotcha_internal_main_wrappee_handle}
};
