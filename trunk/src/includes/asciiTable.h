/*
 * =======================================================================================
 *
 *      Filename:  asciiTable.h
 *
 *      Description:  Module to create and print a ascii table
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#ifndef ASCIITABLE_H
#define ASCIITABLE_H

#include <types.h>
#include <bstrlib.h>

extern TableContainer* asciiTable_allocate(int numRows,int numColumns, bstrList* headerLabels);
extern void asciiTable_free(TableContainer* container);
extern void asciiTable_insertRow(TableContainer* container, int row,  bstrList* fields);
extern void asciiTable_appendRow(TableContainer* container, bstrList* fields);
extern void asciiTable_setCurrentRow(TableContainer* container, int row);
extern void asciiTable_print(TableContainer* container);
extern void asciiTable_setOutput(FILE* stream);

#endif /*ASCIITABLE_H*/
