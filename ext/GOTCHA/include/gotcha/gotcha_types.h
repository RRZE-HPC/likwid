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
/*!
 ******************************************************************************
 *
 * \file gotcha_types.h
 *
 * \brief   Header file containing the internal gotcha types
 *
 ******************************************************************************
 */
#ifndef GOTCHA_TYPES_H
#define GOTCHA_TYPES_H

#if defined(__cplusplus)
extern "C" {
#endif

typedef void *gotcha_wrappee_handle_t;

/*!
 * The representation of a Gotcha action
 * as it passes through the pipeline
 */
typedef struct gotcha_binding_t {
  const char *name;       //!< The name of the function being wrapped
  void *wrapper_pointer;  //!< A pointer to the wrapper function
  gotcha_wrappee_handle_t
      *function_handle;  //!< A pointer to the function being wrapped
} gotcha_binding_t;

/*!
 * The representation of an error (or success) of a Gotcha action
 */
typedef enum gotcha_error_t {
  GOTCHA_SUCCESS = 0,         //!< The call succeeded
  GOTCHA_FUNCTION_NOT_FOUND,  //!< The call looked up a function which could not
                              //!< be found
  GOTCHA_INTERNAL,            //!< Internal gotcha error
  GOTCHA_INVALID_TOOL         //!< Invalid tool name
} gotcha_error_t;

#if defined(__cplusplus)
}
#endif

#endif
