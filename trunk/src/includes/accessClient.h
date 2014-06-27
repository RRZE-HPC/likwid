/*
 * =======================================================================================
 *
 *      Filename:  accessClient.h
 *
 *      Description:  Header File accessClient Module. 
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

#ifndef ACCESSCLIENT_H
#define ACCESSCLIENT_H

#include <types.h>

extern int accessClient_mode;

/* This needs to be called BEFORE msr_init and
 * sets how the module tries to access the MSR registers. */
extern void accessClient_setaccessmode(int mode);

/* This needs to be called BEFORE msr_init and
 * sets the priority the module reports to the daemon.
 * This is a noop in any msr access mode except sysmsrd. */
extern void accessClient_setlowaccesspriority(void);

/* Initializes the MSR module, trying to open either the MSR files or
 * the connection to the msr daemon. */
extern void accessClient_init(int* socket_fd);
extern void accessClient_initThread(int* socket_fd);
extern void accessClient_finalize(int socket_fd);
extern int accessClient_read(int socket_fd, int cpu, int device, uint32_t reg, uint64_t *data);
extern int accessClient_write(int socket_fd, int cpu, int device, uint32_t reg, uint64_t data);

#endif /* ACCESSCLIENT_H */
