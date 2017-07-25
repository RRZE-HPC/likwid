/*
 * =======================================================================================
 *
 *      Filename:  access_x86_msr.c
 *
 *      Description:  Implementation of msr module.
 *                   Provides API to read and write values to the model
 *                   specific registers on x86 processors using the msr
 *                   sys interface of the Linux 2.6 kernel. This module
 *                   is based on the msr-util tools.
 *
 *      Version:   4.2
 *      Released:  22.12.2016
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com.
 *                Thomas Roehl (tr), thomas.roehl@googlemail.com
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <access_x86_msr.h>
#include <registers.h>
#ifdef LIKWID_PROFILE_COUNTER_READ
#include <timer.h>
#endif

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define MAX_LENGTH_MSR_DEV_NAME  20
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int FD[MAX_NUM_THREADS] = { [0 ... MAX_NUM_THREADS-1] = -1 };
static int rdpmc_works_pmc = -1;
static int rdpmc_works_fixed = -1;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static inline int
__rdpmc(int cpu_id, int counter, uint64_t* value)
{
    unsigned low, high;
    cpu_set_t cpuset, current;
    sched_getaffinity(0, sizeof(cpu_set_t), &current);
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    __asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));
    *value = ((low) | ((uint64_t )(high) << 32));
    sched_setaffinity(0, sizeof(cpu_set_t), &current);
    return 0;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

//Needed for rdpmc check
void
segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    exit(1);
}

int
test_rdpmc(int cpu_id, uint64_t value, int flag)
{
    int ret;
    int pid;

    pid = fork();

    if (pid < 0)
    {
        return -1;
    }
    if (!pid)
    {
        uint64_t tmp;
        struct sigaction sa;
        memset(&sa, 0, sizeof(struct sigaction));
        sigemptyset(&sa.sa_mask);
        sa.sa_sigaction = segfault_sigaction;
        sa.sa_flags   = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, NULL);
        if (flag == 0)
        {
            __rdpmc(cpu_id, value, &tmp);
            usleep(100);
        }
        exit(0);
    }
    else
    {
        int status = 0;
        int waiting = 0;
        waiting = waitpid(pid, &status, 0);
        if ((waiting < 0) || (WEXITSTATUS(status) != 0))
        {
            ret = 0;
        }
        else
        {
            ret = 1;
        }
    }
    return ret;
}

int
access_x86_msr_init(const int cpu_id)
{
    int fd = 0;
    int i = 0;

    char* msr_file_name;
    if (FD[cpu_id] > 0)
    {
        return 0;
    }
    msr_file_name = (char*) malloc(MAX_LENGTH_MSR_DEV_NAME * sizeof(char));
    if (!msr_file_name)
    {
        return -ENOMEM;
    }

    sprintf(msr_file_name,"/dev/msr%d", cpu_id);
    fd = open(msr_file_name, O_RDWR);
    if (fd < 0)
    {
        sprintf(msr_file_name,"/dev/cpu/%d/msr_safe", cpu_id);
        fd = open(msr_file_name, O_RDWR);
        if (fd < 0)
        {
            sprintf(msr_file_name,"/dev/cpu/%d/msr", cpu_id);
        }
        else
        {
            if(geteuid() != 0 && cpuid_info.supportUncore)
            {
                fprintf(stdout, "Using msr_safe kernel module. Currently, this deactivates the\n");
                fprintf(stdout, "PCI-based Uncore monitoring.\n");
                cpuid_info.supportUncore = 0;
            }
            close(fd);
        }
    }
    else
    {
        close(fd);
    }
    fd = open(msr_file_name, O_RDWR);
    if (fd < 0)
    {
        ERROR_PRINT(Cannot access MSR device file %s: %s.,msr_file_name , strerror(errno))
        ERROR_PLAIN_PRINT(Please check if 'msr' module is loaded and device files have correct permissions);
        ERROR_PLAIN_PRINT(Alternatively you might want to look into (sys)daemonmode);
        free(msr_file_name);
        return -EPERM;
    }
    else
    {
        close(fd);
    }
    if (rdpmc_works_pmc < 0)
    {
        rdpmc_works_pmc = test_rdpmc(cpu_id, 0, 0);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for PMC counters returned %d, rdpmc_works_pmc);
    }
    if (rdpmc_works_fixed < 0)
    {
        rdpmc_works_fixed = test_rdpmc(cpu_id, (1<<30), 0);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for FIXED counters returned %d, rdpmc_works_fixed);
    }

    sprintf(msr_file_name,"/dev/msr%d",cpu_id);
    fd = open(msr_file_name, O_RDWR);
    if (fd < 0)
    {
        sprintf(msr_file_name,"/dev/cpu/%d/msr_safe", cpu_id);
        fd = open(msr_file_name, O_RDWR);
        if (fd < 0)
        {
            sprintf(msr_file_name,"/dev/cpu/%d/msr", cpu_id);
        }
        else
        {
            close(fd);
        }
    }
    else
    {
        close(fd);
    }
    FD[cpu_id] = open(msr_file_name, O_RDWR);
    if ( FD[cpu_id] < 0 )
    {
        ERROR_PRINT(Cannot access MSR device file %s in direct mode, msr_file_name);
        free(msr_file_name);
        return -EPERM;
    }
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Opened MSR device %s for CPU %d,msr_file_name, cpu_id);
    free(msr_file_name);

    return 0;
}

void
access_x86_msr_finalize(const int cpu_id)
{
    int i = 0;

    if (FD[cpu_id] > 0)
    {
        close(FD[cpu_id]);
        FD[cpu_id] = 0;
    }
}

int
access_x86_msr_read( const int cpu_id, uint32_t reg, uint64_t *data)
{
    int ret;

    if ((rdpmc_works_pmc == 1) && (reg >= MSR_PMC0) && (reg <=MSR_PMC7))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Read PMC counter with RDPMC instruction with index %d, reg - MSR_PMC0);
        if (__rdpmc(cpu_id, reg - MSR_PMC0, data) )
        {
            rdpmc_works_pmc = 0;
            goto fallback;
        }
    }
    else if ((rdpmc_works_fixed == 1) && (reg >= MSR_PERF_FIXED_CTR0) && (reg <= MSR_PERF_FIXED_CTR2))
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED counter with RDPMC instruction with index %d, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
        if (__rdpmc(cpu_id, (1<<30) + (reg - MSR_PERF_FIXED_CTR0), data) )
        {
            rdpmc_works_fixed = 0;
            goto fallback;
        }
    }
    else
    {
fallback:
        if (FD[cpu_id] > 0)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read MSR counter 0x%X with RDMSR instruction on CPU %d, reg, cpu_id);
            ret = pread(FD[cpu_id], data, sizeof(*data), reg);
            if ( ret != sizeof(*data) )
            {
                return ret;
            }
        }
    }
    return 0;
}

int
access_x86_msr_write( const int cpu_id, uint32_t reg, uint64_t data)
{
    int ret;
    if (FD[cpu_id] > 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Write MSR counter 0x%X with WRMSR instruction on CPU %d data 0x%lX, reg, cpu_id, data);
        ret = pwrite(FD[cpu_id], &data, sizeof(data), reg);
        if (ret != sizeof(data))
        {
            return ret;
        }
    }
    return 0;
}

int access_x86_msr_check(PciDeviceIndex dev, int cpu_id)
{
    if (dev == MSR_DEV)
    {
        if (FD[cpu_id] > 0)
        {
            return 1;
        }
    }
    return 0;
}

