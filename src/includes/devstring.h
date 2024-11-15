/*
 * =======================================================================================
 *
 *      Filename:  devstring.h
 *
 *      Description:  Header File to resolve a LIKWID device string to a list of devices
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Michael Panzlaff, michael.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#ifndef DEVSTRING_H
#define DEVSTRING_H

#include <stddef.h>

#include <likwid.h>

int likwid_devstr_to_devlist(const char *str, LikwidDeviceList_t *dev_list);

#endif // DEVSTRING_H
