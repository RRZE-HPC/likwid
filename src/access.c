/*
 * =======================================================================================
 *
 *      Filename:  access.c
 *
 *      Description:  Interface for the different register access modules.
 *
 *      Version:   4.3.2
 *      Released:  12.04.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <configuration.h>
#include <perfmon.h>
#include <registers.h>
#include <access.h>
#include <access_client.h>
#include <access_x86.h>


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int registeredCpus = 0;
static int *registeredCpuList = NULL;
static int (*access_read)(PciDeviceIndex dev, const int cpu, uint32_t reg, uint64_t *data) = NULL;
static int (*access_write)(PciDeviceIndex dev, const int cpu, uint32_t reg, uint64_t data) = NULL;
static int (*access_init) (int cpu_id) = NULL;
static void (*access_finalize) (int cpu_id) = NULL;
static int (*access_check) (PciDeviceIndex dev, int cpu_id) = NULL;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
HPMmode(int mode)
{
    if ((mode == ACCESSMODE_DIRECT) || (mode == ACCESSMODE_DAEMON))
    {
        config.daemonMode = mode;
    }
}

int
HPMinit(void)
{
    int ret = 0;
    if (registeredCpuList == NULL)
    {
        registeredCpuList = malloc(cpuid_topology.numHWThreads* sizeof(int));
        memset(registeredCpuList, 0, cpuid_topology.numHWThreads* sizeof(int));
        registeredCpus = 0;
    }
    if (access_init == NULL)
    {
#if defined(__x86_64__) || defined(__i386__)
        if (config.daemonMode == -1)
        {
            config.daemonMode = ACCESSMODE_DAEMON;
        }
        if (config.daemonMode == ACCESSMODE_DAEMON)
        {
            DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, Adjusting functions for x86 architecture in daemon mode);
            access_init = &access_client_init;
            access_read = &access_client_read;
            access_write = &access_client_write;
            access_finalize = &access_client_finalize;
            access_check = &access_client_check;
        }
        else if (config.daemonMode == ACCESSMODE_DIRECT)
        {
            DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, Adjusting functions for x86 architecture in direct mode);
            access_init = &access_x86_init;
            access_read = &access_x86_read;
            access_write = &access_x86_write;
            access_finalize = &access_x86_finalize;
            access_check = &access_x86_check;
        }
#endif
    }

    return 0;
}

int
HPMinitialized(void)
{
    return registeredCpus;
}

int
HPMaddThread(int cpu_id)
{
    int ret;
    if (registeredCpuList[cpu_id] == 0)
    {
        if (access_init != NULL)
        {
            ret = access_init(cpu_id);
            if (ret == 0)
            {
                DEBUG_PRINT(DEBUGLEV_DETAIL, Adding CPU %d to access module, cpu_id);
                registeredCpus++;
                registeredCpuList[cpu_id] = 1;
            }
            else
            {
                return ret;
            }
        }
        else
        {
            return -ENODEV;
        }
    }
    return 0;
}

void
HPMfinalize()
{
    if (registeredCpus != 0)
    {
        for (int i=0; i<cpuid_topology.numHWThreads; i++)
        {
            if (i >= cpuid_topology.numHWThreads)
            {
                break;
            }
            if (registeredCpuList[i] == 1)
            {
                access_finalize(i);
                registeredCpus--;
                registeredCpuList[i] = 0;
            }
        }
        if (registeredCpuList && registeredCpus == 0)
        {
            free(registeredCpuList);
            registeredCpuList = NULL;
        }
    }
    if (access_init != NULL)
        access_init = NULL;
    if (access_finalize != NULL)
        access_finalize = NULL;
    if (access_read != NULL)
        access_read = NULL;
    if (access_write != NULL)
        access_write = NULL;
    if (access_check != NULL)
        access_check = NULL;
    return;
}

int
HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data)
{
    uint64_t tmp = 0x0ULL;
    *data = 0x0ULL;
    int err = 0;
    if ((dev >= MAX_NUM_PCI_DEVICES) || (data == NULL))
    {
        return -EFAULT;
    }
    if ((cpu_id < 0) || (cpu_id >= cpuid_topology.numHWThreads))
    {
        return -ERANGE;
    }
    if (registeredCpuList[cpu_id] == 0)
    {
        return -ENODEV;
    }
    err = access_read(dev, cpu_id, reg, &tmp);
    *data = tmp;
    return err;
}

int
HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data)
{
    int err = 0;
    if (dev >= MAX_NUM_PCI_DEVICES)
    {
        return -EFAULT;
    }
    if ((cpu_id < 0) || (cpu_id >= cpuid_topology.numHWThreads))
    {
        ERROR_PRINT(MSR WRITE C %d OUT OF RANGE, cpu_id);
        return -ERANGE;
    }
    if (registeredCpuList[cpu_id] == 0)
    {
        return -ENODEV;
    }
    err = access_write(dev, cpu_id, reg, data);
    return err;
}

int
HPMcheck(PciDeviceIndex dev, int cpu_id)
{
    if (registeredCpuList[cpu_id] == 0)
    {
        return -ENODEV;
    }
    return access_check(dev, cpu_id);
}

