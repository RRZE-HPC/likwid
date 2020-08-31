/*
 * =======================================================================================
 *
 *      Filename:  topology_cavtx2.h
 *
 *      Description:  Header File for hardcoded cache information for Marvell/Cavium
 *                    Thunder X2.
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#ifndef TOPOLOGY_CAVTX2_H
#define TOPOLOGY_CAVTX2_H

CacheLevel caviumTX2_caches[3] = {
    {1, DATACACHE, 32, 4, 64, 32768, 2, 1},
    {2, DATACACHE, 1, 8, 64, 262144, 2, 1},
    {3, DATACACHE, 0, 8, 64, 29360128, 112, 1},
};




#endif
