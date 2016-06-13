/*
 * =======================================================================================
 *
 *      Filename:  allocator.h
 *
 *      Description:  Header File allocator Module. 
 *
 *      Version:   4.1
 *      Released:  13.6.2016
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  none
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

#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <stdint.h>
#include <bstrlib.h>
#include <test_types.h>

#define LLU_CAST (unsigned long long)

extern void allocator_init(int numVectors);
extern void allocator_finalize();
extern void allocator_allocateVector(void** ptr,
        int alignment,
        uint64_t size,
        int offset,
        DataType type,
        bstring domain);

#endif /*ALLOCATOR_H*/

