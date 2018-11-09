/*
 * =======================================================================================
 *
 *      Filename:  frequency_acpi.c
 *
 *      Description:  Module implementing an interface for frequency manipulation, the
 *                    ACPI CPUFreq backend
 *
 *      Version:   4.3.3
 *      Released:  09.11.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
 *                Jan Treibig (jt), jan.treibig@gmail.com
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


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <bstrlib.h>
#include <likwid.h>
#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <registers.h>

#include <frequency_acpi.h>



uint64_t freq_acpi_getCpuClockMax(const int cpu_id )
{
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL;
    uint64_t clock = 0x0ULL;


    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_max_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        clock = strtoull(cmd, NULL, 10);
    }
    fclose(f);
    return clock *1E3;
}


uint64_t freq_acpi_getCpuClockMin(const int cpu_id )
{

    uint64_t clock = 0x0ULL;
    FILE *f = NULL;
    char cmd[256];
    char buff[256];
    char* eptr = NULL;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_min_freq", cpu_id);
    f = fopen(buff, "r");
    if (f == NULL) {
        fprintf(stderr, "Unable to open path %s for reading\n", buff);
        return 0;
    }
    eptr = fgets(cmd, 256, f);
    if (eptr != NULL)
    {
        clock = strtoull(cmd, NULL, 10);
    }
    fclose(f);
    return clock *1E3;
}



