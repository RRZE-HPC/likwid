/*
 * =======================================================================================
 *
 *      Filename:  msr.h
 *
 *      Description:  Header File msr Module. 
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

#ifndef MSR_H
#define MSR_H

#include <types.h>

/* Initializes the MSR module, trying to open either the MSR files or
 * the connection to the msr daemon. */
extern void msr_init(int socket_fd);
extern void msr_finalize(void);
extern uint64_t msr_read(int cpu, uint32_t reg);
extern void msr_write(int cpu, uint32_t reg, uint64_t data);

/* variants for thread safe execution with a per thread socket */
extern uint64_t msr_tread(int socket_fd, int cpu, uint32_t reg);
extern void msr_twrite(int socket_fd, int cpu, uint32_t reg, uint64_t data);

#endif /* MSR_H */
