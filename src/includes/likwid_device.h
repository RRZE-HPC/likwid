/*
 * =======================================================================================
 *
 *      Filename:  likwid_device.h
 *
 *      Description:  Interface for LIKWID's device handling
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
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

#ifndef LIKWID_DEVICE_H
#define LIKWID_DEVICE_H

#include <likwid.h>

int likwid_device_create(LikwidDeviceType scope, int id, LikwidDevice_t* device) __attribute__ ((visibility ("default") ));
void likwid_device_destroy(LikwidDevice_t device) __attribute__ ((visibility ("default") ));

char* device_type_name(LikwidDeviceType type) __attribute__ ((visibility ("default") ));

#endif /* LIKWID_DEVICE_H */
