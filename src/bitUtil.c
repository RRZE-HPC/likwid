/*
 * =======================================================================================
 *
 *      Filename:  bitUtil.c
 *
 *      Description:  Utility routines manipulating bit arrays.
 *
 *      Version:   4.3.0
 *      Released:  22.12.2017
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2017 RRZE, University Erlangen-Nuremberg
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

#include <stdlib.h>

#include <types.h>
#include <bitUtil.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
uint64_t
field64(uint64_t value, int start, int length)
{
    return (value >> start) & (~0ULL >> (64 - length));
}

uint32_t
field32(uint32_t value, int start, int length)
{
    return (value >> start) & (~0U >> (32 - length));
}

uint32_t
extractBitField(uint32_t inField, uint32_t width, uint32_t offset)
{
    uint32_t bitMask;
    uint32_t outField;

    if ((offset+width) == 32)
    {
        bitMask = (0xFFFFFFFF<<offset);
    }
    else
    {
        bitMask = (0xFFFFFFFF<<offset) ^ (0xFFFFFFFF<<(offset+width));

    }

    outField = (inField & bitMask) >> offset;
    return outField;
}

uint32_t
getBitFieldWidth(uint32_t number)
{
    uint32_t fieldWidth=0;

    number--;
    if (number == 0)
    {
        return 0;
    }
#ifdef __x86_64
    __asm__ volatile ( "bsr %%eax, %%ecx\n\t"
            : "=c" (fieldWidth)
            : "a"(number));
#endif

    return fieldWidth+1;  /* bsr returns the position, we want the width */
}

