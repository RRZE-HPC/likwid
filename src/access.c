/*
 * =======================================================================================
 *
 *      Filename:  access.c
 *
 *      Description:  Interface for the different register access modules.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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
#include <access.h>
#include <access_client.h>
#include <access_x86.h>



static int registeredCpus = 0;
static int registeredCpuList[MAX_NUM_THREADS] = { 0 };


int (*access_read)(PciDeviceIndex dev, const int cpu, uint32_t reg, uint64_t *data) = NULL;
int (*access_write)(PciDeviceIndex dev, const int cpu, uint32_t reg, uint64_t data) = NULL;
int (*access_init) (int cpu_id) = NULL;
void (*access_finalize) (int cpu_id) = NULL;
int (*access_check) (PciDeviceIndex dev, int cpu_id) = NULL;

void HPMmode(int mode)
{
    config.daemonMode = mode;
}

int HPMinit(void)
{
    int ret = 0;
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
/*#if defined(__powerpc__) || defined(__ppc__) || defined(__PPC__)
        DEBUG_PLAIN_PRINT(DEBUGLEV_DEVELOP, Adjusting functions for POWER architecture in direct mode);
        access_init = &access_power_init;
        access_read = &access_power_read;
        access_write = &access_power_write;
        access_finalize = &access_power_finalize;
        access_check = &access_power_check;
#endif*/
        for (int i=0; i<cpuid_topology.numHWThreads; i++)
        {
            ret = access_init(cpuid_topology.threadPool[i].apicId);
            if (ret == 0)
            {
                registeredCpus++;
                registeredCpuList[cpuid_topology.threadPool[i].apicId] = 1;
            }
        }
    }
    return 0;
}


int HPMinitialized(void)
{
    return registeredCpus;
}

int HPMaddThread(int cpu_id)
{
    return HPMinit();
}

void HPMfinalize()
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
        }
    }
    return;
}

int HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data)
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
    access_read(dev, cpu_id, reg, &tmp);
    *data = tmp;
    return err;
}

int HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data)
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
    access_write(dev, cpu_id, reg, data);
    return err;
}

int HPMcheck(PciDeviceIndex dev, int cpu_id)
{
    return access_check(dev, cpu_id);
}
