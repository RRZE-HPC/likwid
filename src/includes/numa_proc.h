/*
 * =======================================================================================
 *
 *      Filename:  numa_proc.h
 *
 *      Description:  Header File procfs/sysfs NUMA backend
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
#ifndef LIKWID_NUMA_PROC
#define LIKWID_NUMA_PROC

extern int proc_numa_init(void);
extern void proc_numa_membind(void* ptr, size_t size, int domainId);
extern void proc_numa_setInterleaved(const int* processorList, int numberOfProcessors);
extern void proc_numa_setMembind(const int* processorList, int numberOfProcessors);

#endif
