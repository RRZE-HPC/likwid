/*
 * =======================================================================================
 *
 *      Filename:  access.h
 *
 *      Description:  Header File HPM access Module
 *
 *      Version:   4.3.3
 *      Released:  09.11.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
#ifndef ACCESS_H
#define ACCESS_H

void HPMmode(int mode);
int HPMinit(void);
int HPMinitialized(void);
int HPMaddThread(int cpu_id);
void HPMfinalize();
int HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data);
int HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data);
int HPMcheck(PciDeviceIndex dev, int cpu_id);

#endif /* ACCESS_H */
