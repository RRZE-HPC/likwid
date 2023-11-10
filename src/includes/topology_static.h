/*
 * =======================================================================================
 *
 *      Filename:  topology_static.h
 *
 *      Description:  Header File for hardcoded cache information
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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

#ifndef TOPOLOGY_STATIC_H
#define TOPOLOGY_STATIC_H

CacheLevel caviumTX2_caches[3] = {
    {1, DATACACHE, 32, 4, 64, 32768, 2, 1},
    {2, DATACACHE, 1, 8, 64, 262144, 2, 1},
    {3, DATACACHE, 0, 8, 64, 29360128, 112, 1},
};

CacheLevel a64fx_caches[2] = {
    {1, DATACACHE, 4, 64, 256, 65536, 1, 1},
    {2, DATACACHE, 16, 2048, 256, 8388608, 12, 1},
};

CacheLevel apple_m1_caches[2] = {
    {1, DATACACHE, 4, 64, 256, 131072, 1, 1},
    {2, DATACACHE, 16, 2048, 256, 12582912, 4, 1},
};


#endif
