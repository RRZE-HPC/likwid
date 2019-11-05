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
#include "library_filters.h"
#include "libc_wrappers.h"

static const char* filter;
int (*libraryFilterFunc)(struct link_map*) = alwaysTrue;

int alwaysTrue(struct link_map* candidate KNOWN_UNUSED){
  return 1;
}

int trueIfNameMatches(struct link_map* target){
  int match = (filter) && (target) && (gotcha_strstr(target->l_name, filter) != 0);
  return match;
}
int trueIfLast(struct link_map* target){
  int ret = (target->l_next) ? 0 : 1;
  return ret;
}
void onlyFilterLast(){
  setLibraryFilterFunc(trueIfLast);
}
void setLibraryFilterFunc(int(*new_func)(struct link_map*)){
  libraryFilterFunc = new_func;
}
void restoreLibraryFilterFunc(){
  setLibraryFilterFunc(alwaysTrue);
}

void filterLibrariesByName(const char* nameFilter){
  filter = nameFilter;
  setLibraryFilterFunc(trueIfNameMatches);
}

