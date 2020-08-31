/*
 * =======================================================================================
 *
 *      Filename:  nvmon_nvml.h
 *
 *      Description:  Header File of nvmon module (NVML backend).
 *
 *      Version:   5.0.2
 *      Released:  31.08.2020
 *
 *      Author:   Thomas Gruber (tg), thomas.gruber@googlemail.com
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
#ifndef LIKWID_NVMON_NVML_H
#define LIKWID_NVMON_NVML_H






NvmonFunctions nvmon_nvml_functions = {
    .freeDevice = NULL,
    .createDevice = NULL,
    .getEventList = NULL,
    .addEvents = NULL,
    .setupCounters = NULL,
    .startCounters = NULL,
    .readCounters = NULL,
};



#endif /* LIKWID_NVMON_NVML_H */
