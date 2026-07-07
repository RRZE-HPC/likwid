/*
 * =======================================================================================
 *
 *      Filename:  bitUtil.c
 *
 *      Description:  Utility routines manipulating bit arrays.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <assert.h>
#include <stdlib.h>

#include <types.h>
#include <bitUtil.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
uint64_t
field64(uint64_t bitfield, int start, int length)
{
    assert(start >= 0 && start < 64);
    assert(length >= 0 && length <= 64);
    return (bitfield >> start) & (~0ULL >> (64 - length));
}

uint32_t
field32(uint32_t bitfield, int start, int length)
{
    assert(start >= 0 && start < 32);
    assert(length >= 0 && length <= 32);
    return (bitfield >> start) & (~0U >> (32 - length));
}

void
field64set(uint64_t* bitfield, int start, int length, uint64_t value)
{
    assert(start >= 0 && start < 64);
    assert(length >= 0 && length <= 64);
    const uint64_t mask = (~0ULL >> (64 - length)) << start;
    *bitfield = (*bitfield & ~mask) | ((value << start) & mask);
}

void
field32set(uint32_t* bitfield, int start, int length, uint32_t value)
{
    assert(start >= 0 && start < 32);
    assert(length >= 0 && length <= 32);
    const uint32_t mask = (~0ULL >> (32 - length)) << start;
    *bitfield = (*bitfield & ~mask) | ((value << start) & mask);
}

// Get the number of bits required for 'count' - 1.
// Why -1? Because it calculates the number of bits required for the number
// of combinations, not the number itself.
uint32_t fieldWidthForCount(uint64_t count)
{
    if (count == 0 || count == 1)
        return 0;

    // max representable number for N combinations
    count -= 1;

    const unsigned long long countLL = count;
    const uint64_t countLLWidth = 8 * sizeof(countLL);

    return (uint32_t)(countLLWidth - __builtin_clzll(countLL));
}
