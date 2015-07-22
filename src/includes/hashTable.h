/*
 * =======================================================================================
 *
 *      Filename:  hashTable.h
 *
 *      Description:  Header File hashtable Module. 
 *                    Wrapper for HAshTable data structure holding thread
 *                    specific region information.
 *
 *      Version:   4.0
 *      Released:  16.6.2015
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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

#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <bstrlib.h>
#include <types.h>

extern void hashTable_init();
void hashTable_initThread(int coreID);
extern int hashTable_get(bstring regionTag, LikwidThreadResults** result);
extern void hashTable_finalize(int* numberOfThreads, int* numberOfRegions, LikwidResults** results);


#endif /*CPUID_H*/
