/*
 * =======================================================================================
 *
 *      Filename:  bitUtil.h
 *
 *      Description:  Header File bitUtil Module.
 *                    Helper routines for dealing with bit manipulations
 *
 *      Version:   5.2.1
 *      Released:  11.11.2021
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2021 NHR@FAU, University Erlangen-Nuremberg
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
#ifndef BITUTIL_H
#define BITUTIL_H

#include <types.h>

extern uint64_t field64(uint64_t value, int start, int length);
extern uint32_t field32(uint32_t value, int start, int length);
extern uint32_t extractBitField(uint32_t inField, uint32_t width, uint32_t offset);
extern uint32_t getBitFieldWidth(uint32_t number);

#define setBit(reg,bit)  (reg) |= (1ULL<<(bit))
#define clearBit(reg,bit) (reg) &= ~(1ULL<<(bit))
#define toggleBit(reg,bit) (reg) ^= (1ULL<<(bit))
#define testBit(reg,bit) ((reg) & (1ULL<<(bit)))

#define bitMask_init(_mask)  \
    (_mask).mask[0] = 0ULL; \
    (_mask).mask[1] = 0ULL;

#define bitMask_set(_mask,_bit)  \
{ \
    int pos = _bit/64; \
    int shift = _bit-(pos*64); \
    setBit((_mask).mask[pos],shift); \
}

#define bitMask_test(_res,_mask,_bit)  \
{ \
    _res = 0LLU; \
    int pos = _bit/64; \
    int shift = _bit-(pos*64); \
    _res = testBit((_mask).mask[pos],shift); \
}

#define bitMask_toString(_string,_mask)  \
    sprintf(_string,"%llX %llX", LLU_CAST (_mask).mask[0], LLU_CAST (_mask).mask[1]);

#endif /*BITUTIL_H*/
