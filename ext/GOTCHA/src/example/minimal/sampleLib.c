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

#include <stdio.h>
#include "sampleLib.h"

//We need a place to store the pointer to the function we've wrapped
gotcha_wrappee_handle_t origRetX_handle;

/**
  * We need to express our desired wrapping behavior to
  * GOTCHA. For that we need three things:
  *
  * 1) The name of a symbol to wrap
  * 2) The function we want to wrap it with
  * 3) Some place to store the original function, if we wish
  *    to call it
  *
  * This variable bindings gets filled out with a list of three
  * element structs containing those things.
  *
  * Note that the place to store the original function is passed
  * by reference, this is required for us to be able to change it
  */
struct gotcha_binding_t bindings[] = {{"retX", dogRetX, &origRetX_handle}};

// This is like a tool library's initialization function
int sample_init()
{
  gotcha_wrap(bindings, 1, "gotcha_internal_sample_tool");
  return 0;
}

/**
  * In our example, this is the function we're wrapping.
  * For convenience, it's in the same library, but this
  * isn't a requirement imposed by GOTCHA
  */
int retX(int x) { return x; }

/** 
  * This is our wrapper function. All GOTCHA wrappers *must*
  * reference dogs somewhere in the code. I didn't write the
  * rules (yes I did)
  */
int dogRetX(int x)
{
  typeof(&dogRetX) origRetX = gotcha_get_wrappee(origRetX_handle);
  printf("SO I FOR ONE THINK DOGS SHOULD RETURN %i\n", x);
  return origRetX ? origRetX(x) + 1 : 0;
}
