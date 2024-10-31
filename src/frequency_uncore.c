/*
 * =======================================================================================
 *
 *      Filename:  frequency_uncore.c
 *
 *      Description:  Module implementing an interface for frequency manipulation
 *                    Module for manipuating Uncore frequencies (Intel only)
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *                Jan Treibig (jt), jan.treibig@gmail.com
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

#include <bstrlib.h>
#include <likwid.h>
#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <registers.h>
#include <lock.h>

#include <frequency.h>


static int _freq_getUncoreMinMax(const int socket_id, int *cpuId, double* min, double* max)
{
    int cpu = -1;
    *cpuId = -1;
    *min = 0;
    *max = 0;
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpu = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpu < 0)
    {
        fprintf(stderr, "Unknown socket ID %d\n", socket_id);
        return -ENODEV;
    }

    char* avail = freq_getAvailFreq(cpu);
    if (!avail)
    {
        avail = malloc(1000 * sizeof(char));
        if (avail)
        {
            int ret = snprintf(avail, 999, "%d %d", freq_getConfCpuClockMin(cpu)/1000000, freq_getConfCpuClockMax(cpu)/1000000);
            if (ret > 0)
            {
                avail[ret] = '\0';
            }
            else
            {
                free(avail);
                fprintf(stderr, "Failed to get available CPU frequencies\n");
                return -EINVAL;
            }
        }
        else
        {
            fprintf(stderr, "Failed to get available CPU frequencies\n");
            return -EINVAL;
        }
    }

    double dmin = 0.0;
    double dmax = 0.0;
    bstring bavail = bfromcstr(avail);
    free(avail);
    struct bstrList* bavail_list;
    bavail_list = bsplit(bavail, ' ');
    bdestroy(bavail);
    if (bavail_list->qty < 2)
    {
        fprintf(stderr, "Failed to read minimal and maximal frequencies\n");
        bstrListDestroy(bavail_list);
        return -EINVAL;
    }
    if (blength(bavail_list->entry[0]) > 0)
    {
        char* tptr = NULL;
        dmin = strtod(bdata(bavail_list->entry[0]), &tptr);
        if (bdata(bavail_list->entry[0]) != tptr)
        {
            dmin *= 1000;
        }
        else
        {
            fprintf(stderr, "Problem converting %s to double for comparison with given freq.\n", bdata(bavail_list->entry[0]));
            return -EINVAL;
        }
    }
    if (blength(bavail_list->entry[bavail_list->qty-1]) > 0)
    {
        char* tptr = NULL;
        dmax = strtod(bdata(bavail_list->entry[bavail_list->qty-1]), &tptr);
        if (bdata(bavail_list->entry[bavail_list->qty-1]) != tptr)
        {
            dmax *= 1000;
        }
        else
        {
            fprintf(stderr, "Problem converting %s to double for comparison with given freq.\n", bdata(bavail_list->entry[bavail_list->qty-1]));
            return -EINVAL;
        }
    }
    bstrListDestroy(bavail_list);

    *cpuId = cpu;
    if (dmin < dmax)
    {
        *min = dmin;
        *max = dmax;
    }
    else
    {
        *max = dmin;
        *min = dmax;
    }

    power_init(cpu);
    if (power_info.turbo.numSteps > 0)
    {
        if (power_info.turbo.steps[0] > *max)
        {
            *max = power_info.turbo.steps[0];
        }
    }

    return 0;
}


int freq_setUncoreFreqMin(const int socket_id, const uint64_t freq)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    uint64_t f = freq / 100;
    double fmin, fmax;
    if (!lock_check())
    {
        fprintf(stderr,"Access to frequency backend is locked.\n");
        return -EPERM;
    }
    if (isAMD())
    {
        return 0;
    }
    err = _freq_getUncoreMinMax(socket_id, &cpuId, &fmin, &fmax);
    if (err < 0)
    {
        return err;
    }
    if (freq < (uint64_t)fmin)
    {
        ERROR_PRINT(Given frequency %llu MHz lower than system limit of %.0f MHz, freq, fmin);
        return -EINVAL;
    }
    if (freq > (uint64_t)fmax)
    {
        ERROR_PRINT(Given frequency %llu MHz higher than system limit of %.0f MHz, freq, fmax);
        return -EINVAL;
    }
#ifdef LIKWID_USE_PERFEVENT
    fprintf(stderr,"Cannot manipulate Uncore frequency with ACCESSMODE=perf_event.\n");
    return 0;
#else
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
    }
    err = HPMaddThread(cpuId);
    if (err != 0)
    {
        ERROR_PLAIN_PRINT(Cannot get access to MSRs);
        return 0;
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }
    tmp &= ~(0xFF00);
    tmp |= (f<<8);
    err = HPMwrite(cpuId, MSR_DEV, MSR_UNCORE_FREQ, tmp);
    if (err)
    {
        ERROR_PRINT(Cannot write register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return 0;
#endif
}




uint64_t freq_getUncoreFreqMin(const int socket_id)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;

    if (!lock_check())
    {
        fprintf(stderr,"Access to frequency backend is locked.\n");
        return 0;
    }
    if (isAMD())
    {
        return 0;
    }
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpuId = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpuId < 0)
    {
        ERROR_PRINT(Unknown socket ID %d, socket_id);
        return 0;
    }
#ifdef LIKWID_USE_PERFEVENT
    fprintf(stderr,"Cannot manipulate Uncore frequency with ACCESSMODE=perf_event.\n");
    return 0;
#else
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
    }
    err = HPMaddThread(cpuId);
    if (err != 0)
    {
        ERROR_PLAIN_PRINT(Cannot get access to MSRs);
        return 0;
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return 0;
    }
    tmp = ((tmp>>8) & 0xFFULL) * 100;

    if (own_hpm)
        HPMfinalize();
    return tmp;
#endif
}

int freq_setUncoreFreqMax(const int socket_id, const uint64_t freq)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;
    uint64_t f = freq / 100;
    double fmin, fmax;
    if (!lock_check())
    {
        fprintf(stderr,"Access to frequency backend is locked.\n");
        return -EPERM;
    }
    if (isAMD())
    {
        return 0;
    }
    err = _freq_getUncoreMinMax(socket_id, &cpuId, &fmin, &fmax);
    if (err < 0)
    {
        return err;
    }
    if (freq < (uint64_t)fmin)
    {
        ERROR_PRINT(Given frequency %llu MHz lower than system limit of %.0f MHz, freq, fmin);
        return -EINVAL;
    }
    if (freq > (uint64_t)fmax)
    {
        ERROR_PRINT(Given frequency %llu MHz higher than system limit of %.0f MHz, freq, fmax);
        return -EINVAL;
    }
#ifdef LIKWID_USE_PERFEVENT
    fprintf(stderr,"Cannot manipulate Uncore frequency with ACCESSMODE=perf_event.\n");
    return -1;
#else
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
    }
    err = HPMaddThread(cpuId);
    if (err != 0)
    {
        ERROR_PLAIN_PRINT(Cannot get access to MSRs);
        return 0;
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }
    tmp &= ~(0xFFULL);
    tmp |= (f & 0xFFULL);
    err = HPMwrite(cpuId, MSR_DEV, MSR_UNCORE_FREQ, tmp);
    if (err)
    {
        ERROR_PRINT(Cannot write register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return err;
    }

    if (own_hpm)
        HPMfinalize();
    return 0;
#endif
}

uint64_t freq_getUncoreFreqMax(const int socket_id)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;

    if (!lock_check())
    {
        fprintf(stderr,"Access to frequency backend is locked.\n");
        return 0;
    }

    if (isAMD())
    {
        return 0;
    }
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpuId = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpuId < 0)
    {
        ERROR_PRINT(Unknown socket ID %d, socket_id);
        return 0;
    }
#ifdef LIKWID_USE_PERFEVENT
    fprintf(stderr,"Cannot manipulate Uncore frequency with ACCESSMODE=perf_event.\n");
    return 0;
#else
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
    }
    err = HPMaddThread(cpuId);
    if (err != 0)
    {
        ERROR_PLAIN_PRINT(Cannot get access to MSRs);
        return 0;
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ, cpuId);
        return 0;
    }
    tmp = (tmp & 0xFFULL) * 100;

    if (own_hpm)
        HPMfinalize();
    return tmp;
#endif
}

uint64_t freq_getUncoreFreqCur(const int socket_id)
{
    int err = 0;
    int own_hpm = 0;
    int cpuId = -1;

    if (!lock_check())
    {
        fprintf(stderr,"Access to frequency backend is locked.\n");
        return 0;
    }
    if (isAMD())
    {
        return 0;
    }
    for (int i=0; i<cpuid_topology.numHWThreads; i++)
    {
        if (cpuid_topology.threadPool[i].packageId == socket_id)
        {
            cpuId = cpuid_topology.threadPool[i].apicId;
            break;
        }
    }
    if (cpuId < 0)
    {
        ERROR_PRINT(Unknown socket ID %d, socket_id);
        return 0;
    }
#ifdef LIKWID_USE_PERFEVENT
    fprintf(stderr,"Cannot manipulate Uncore frequency with ACCESSMODE=perf_event.\n");
    return 0;
#else
    if (!HPMinitialized())
    {
        HPMinit();
        own_hpm = 1;
        err = HPMaddThread(cpuId);
        if (err != 0)
        {
            ERROR_PLAIN_PRINT(Cannot get access to MSRs);
            return 0;
        }
    }

    uint64_t tmp = 0x0ULL;
    err = HPMread(cpuId, MSR_DEV, MSR_UNCORE_FREQ_READ, &tmp);
    if (err)
    {
        //ERROR_PRINT(Cannot read register 0x%X on CPU %d, MSR_UNCORE_FREQ_READ, cpuId);
        return 0;
    }
    tmp = (tmp & 0xFFULL) * 100;

    if (own_hpm)
        HPMfinalize();
    return tmp;
#endif
}
