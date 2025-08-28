/*
 * =======================================================================================
 *
 *      Filename:  tlb-info.h
 *
 *      Description:  Header File of topology module that contains the TLB
 *                    describing strings. Not used currently.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
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
#ifndef TLB_INFO_H
#define TLB_INFO_H

static char *intel_tlb_info[256] = { [0] = NULL,
    [1]           = "Instruction TLB: 4 KByte pages, 4-way set associative, 32 entries",
    [2]           = "Instruction TLB: 4 MByte pages, fully associative, 2 entries",
    [3]           = "Data TLB: 4 KByte pages, 4-way set associative, 64 entries",
    [4]           = "Data TLB: 4 MByte pages, 4-way set associative, 8 entries",
    [5]           = "Data TLB1: 4 MByte pages, 4-way set associative, 32 entries",
    [6 ... 10]    = NULL,
    [11]          = "Instruction TLB: 4 MByte pages, 4-way set associative, 4 entries",
    [12 ... 78]   = NULL,
    [79]          = "Instruction TLB: 4 KByte pages, 32 entries",
    [80]          = "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 64 entries",
    [81]          = "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 128 entries",
    [82]          = "Instruction TLB: 4 KByte and 2-MByte or 4-MByte pages, 256 entries",
    [83 ... 84]   = NULL,
    [85]          = "Instruction TLB: 2-MByte or 4-MByte pages, fully associative, 7 entries",
    [86]          = "Data TLB0: 4 MByte pages, 4-way set associative, 16 entries",
    [87]          = "Data TLB0: 4 KByte pages, 4-way associative, 16 entries",
    [88]          = NULL,
    [89]          = "Data TLB0: 4 KByte pages, fully associative, 16 entries",
    [90]          = "Data TLB0: 2-MByte or 4 MByte pages, 4-way set associative, 32 entries",
    [91]          = "Data TLB: 4 KByte and 4 MByte pages, 64 entries",
    [92]          = "Data TLB: 4 KByte and 4 MByte pages,128 entries",
    [93]          = "Data TLB: 4 KByte and 4 MByte pages,256 entries",
    [94 ... 96]   = NULL,
    [97]          = "Instruction TLB: 4 KByte pages, fully associative, 48 entries",
    [98]          = NULL,
    [99]          = "Data TLB: 1 GByte pages, 4-way set associative, 4 entries",
    [100 ... 117] = NULL,
    [118]         = "Instruction TLB: 2M/4M pages, fully associative, 8 entries",
    [119 ... 159] = NULL,
    [160]         = "DTLB: 4k pages, fully associative, 32 entries",
    [161 ... 175] = NULL,
    [176]         = "Instruction TLB: 4 KByte pages, 4-way set associative, 128 entries",
    [177]         = "Instruction TLB: 2M pages, 4-way, 8 entries or 4M pages, 4-way, 4 entries",
    [178]         = "Instruction TLB: 4KByte pages, 4-way set associative, 64 entries",
    [179]         = "Data TLB: 4 KByte pages, 4-way set associative, 128 entries",
    [180]         = "Data TLB1: 4 KByte pages, 4-way associative, 256 entries",
    [181]         = "Instruction TLB: 4KByte pages, 8-way set associative, 64 entries",
    [182]         = "Instruction TLB: 4KByte pages, 8-way set associative, 128 entries",
    [183 ... 185] = NULL,
    [186]         = "Data TLB1: 4 KByte pages, 4-way associative, 64 entries",
    [187 ... 191] = NULL,
    [192]         = "Data TLB: 4 KByte and 4 MByte pages, 4-way associative, 8 entries",
    [193]         = "Shared 2nd-Level TLB: 4 KByte/2MByte pages, 8-way associative, 1024 entries",
    [194]         = "DTLB: 4 KByte/2 MByte pages, 4-way associative, 16 entries",
    [195 ... 201] = NULL,
    [202]         = "Shared 2nd-Level TLB: 4 KByte pages, 4-way associative, 512 entries",
    [203 ... 239] = NULL,
    [240]         = "64-Byte prefetching",
    [241]         = "128-Byte prefetching",
    [242 ... 255] = NULL };

#endif /* TLB_INFO_H */
