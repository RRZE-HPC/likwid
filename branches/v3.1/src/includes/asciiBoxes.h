/*
 * =======================================================================================
 *
 *      Filename:  asciiBoxes.h
 *
 *      Description:  Module to draw nested ascii art boxes.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

#ifndef ASCIIBOXES_H
#define ASCIIBOXES_H

#include <types.h>
#include <bstrlib.h>

extern BoxContainer* asciiBoxes_allocateContainer(int numLines,int numColumns);
extern void asciiBoxes_addBox(BoxContainer* container, int line, int column, bstring label);
extern void asciiBoxes_addJoinedBox(BoxContainer* container, int line, int startColumn, int endColumn, bstring label);
extern void asciiBoxes_print(FILE* OUTSTREAM, BoxContainer* container);

#endif /*ASCIIBOXES_H*/
