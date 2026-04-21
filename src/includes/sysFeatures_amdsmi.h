/*
 * =======================================================================================
 *
 *      Filename:  sysFeatures_amdsmi.h
 *
 *      Description:  Interface to control various AMD SMI based features
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber, thomas.gruber@fau.de
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

#ifndef SYSFEATURES_AMDSMI_H
#define SYSFEATURES_AMDSMI_H

#include <sysFeatures_types.h>

int likwid_sysft_init_amdsmi(_SysFeatureList *list);

#endif // SYSFEATURES_AMDSMI_H
