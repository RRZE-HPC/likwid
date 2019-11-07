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

// TODO: Determine whether this interface should stay on in this form

#ifndef GOTCHA_LIBRARY_FILTERS_H
#define GOTCHA_LIBRARY_FILTERS_H
#include <link.h>
#include "gotcha_utils.h"

int alwaysTrue(struct link_map* candidate KNOWN_UNUSED);
extern int (*libraryFilterFunc)(struct link_map*);

int trueIfNameMatches(struct link_map* target);
int trueIfLast(struct link_map* target);
void filterLibrariesByName(const char* nameFilter);
void onlyFilterLast();
void setLibraryFilterFunc(int(*new_func)(struct link_map*));
void restoreLibraryFilterFunc();
#endif
