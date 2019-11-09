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
 * This file contains utilities for cases where users say something (wrap main) 
 * that doesn't work, but can be translated to something that does work
 */
#ifndef GOTCHA_SRC_TRANSLATIONS_H
#define GOTCHA_SRC_TRANSLATIONS_H
#include <gotcha/gotcha.h>

/** "int main" wrapping handling */
typedef int (*libc_start_main_t) (int (*)(int, char**, char**), int, char**, void (*)(), void (*)(), void (*)(), void*);
typedef int (*main_t)            (int argc, char** argv, char** envp);

extern int main_wrapped;
extern gotcha_wrappee_handle_t gotcha_internal_libc_main_wrappee_handle;
extern gotcha_wrappee_handle_t gotcha_internal_main_wrappee_handle;

int gotcha_internal_main(int argc, char** argv, char** envp);
int gotcha_internal_libc_start_main (int (*)(int, char**, char**), int, char**, void (*)(), void (*)(), void (*)(), void*);

extern struct gotcha_binding_t libc_main_wrappers[];
extern struct gotcha_binding_t main_wrappers[];

#endif
