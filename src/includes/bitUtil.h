/*
 * =======================================================================================
 *
 *      Filename:  bitUtil.h
 *
 *      Description:  Header File bitUtil Module. 
 *                    Helper routines for dealing with bit manipulations
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

extern uint32_t extractBitField(uint32_t inField, uint32_t width, uint32_t offset);
extern uint32_t getBitFieldWidth(uint32_t number);

#define setBit(reg,bit)  (reg) |= (1ULL<<(bit))
#define clearBit(reg,bit) (reg) &= ~(1ULL<<(bit))
#define toggleBit(reg,bit) (reg) ^= (1ULL<<(bit))
#define testBit(reg,bit) (reg) & (1ULL<<(bit))

#endif /*BITUTIL_H*/
