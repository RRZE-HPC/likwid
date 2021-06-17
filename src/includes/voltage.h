/*
 * =======================================================================================
 *
 *      Filename:  voltage.h
 *
 *      Description:  Header File Voltage Module.
 *                    Implements Intel Core Voltage Interface.
 *
 *      Version:   5.2
 *      Released:  17.6.2021
 *
 *      Author:   Jimmy Situ web@jimmystone.cn
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
#ifndef VOLTAGE_H
#define VOLTAGE_H

#include <types.h>
#include <registers.h>
#include <bitUtil.h>
#include <error.h>
#include <access.h>

int voltage_read(int cpuId, uint64_t *data);
int voltage_tread(int socket_fd, int cpuId, uint64_t *data);
double voltage_value(uint64_t raw_value);

#endif /*VOLTAGE_H*/
