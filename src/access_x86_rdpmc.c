/*
 * =======================================================================================
 *
 *      Filename:  access_x86_rdpmc.c
 *
 *      Description:  Implementation of rdpmc module to bypass costly msr or accessdaemon
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2020 RRZE, University Erlangen-Nuremberg
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
#include <sys/types.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <pthread.h>


#include <access_x86_rdpmc.h>
#include <types.h>
#include <error.h>
#include <registers.h>
#include <signal.h>
#include <sched.h>
#include <cpuid.h>

static int rdpmc_works_pmc = -1;
static int rdpmc_works_fixed_inst = -1;
static int rdpmc_works_fixed_cyc = -1;
static int rdpmc_works_fixed_ref = -1;
static int rdpmc_works_fixed_slots = -1;
static int rdpmc_works_llc = -1;
static int rdpmc_works_mem = -1;
static pthread_mutex_t rdpmc_setup_lock = PTHREAD_MUTEX_INITIALIZER;


static inline int
__rdpmc(int cpu_id, int counter, uint64_t* value)
{
    unsigned low, high;
    int reset = 0;
    cpu_set_t cpuset, current;
    sched_getaffinity(0, sizeof(cpu_set_t), &current);
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    if (!CPU_EQUAL(&current, &cpuset))
    {
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        reset = 1;
    }
    __asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));
    *value = ((low) | ((uint64_t )(high) << 32));
    if (reset)
    {
        sched_setaffinity(0, sizeof(cpu_set_t), &current);
    }
    return 0;
}



//Needed for rdpmc check
void
segfault_sigaction_rdpmc(int signal, siginfo_t *si, void *arg)
{
    _exit(1);
}

static int
test_rdpmc(int cpu_id, uint64_t value, int flag)
{
    int ret = -1;
    int pid;

    pid = fork();

    if (pid < 0)
    {
        return -1;
    }
    if (!pid)
    {
        // Note: when exiting the child process we use _exit() instead of exit()
        // to avoid calling exit handlers registered by the parent process
        // (e.g. libuv functions from Julia), which can cause crashes.

        uint64_t tmp;
        struct sigaction sa;
        memset(&sa, 0, sizeof(struct sigaction));
        sigemptyset(&sa.sa_mask);
        sa.sa_sigaction = segfault_sigaction_rdpmc;
        sa.sa_flags   = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, NULL);
        if (flag == 0)
        {
            __rdpmc(cpu_id, value, &tmp);
            usleep(100);
        }
        _exit(0);
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

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_x86_rdpmc_init(const int cpu_id)
{
    unsigned eax,ebx,ecx,edx;
    if (cpuid_info.isIntel)
    {
        eax = 0xA;
        CPUID(eax, ebx, ecx, edx);
    }
    unsigned eventSupportedCount = (eax >> 24) & 0xff;
    pthread_mutex_lock(&rdpmc_setup_lock);
    if (rdpmc_works_pmc < 0)
    {
        rdpmc_works_pmc = test_rdpmc(cpu_id, 0, 0);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for PMC counters returned %d, rdpmc_works_pmc);
    }
    if (rdpmc_works_fixed_inst < 0 && cpuid_info.isIntel)
    {
        if (eventSupportedCount > 1 && (!(ebx & (1<<1))))
        {
            rdpmc_works_fixed_inst = test_rdpmc(cpu_id, (1<<30), 0);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for FIXED instruction counter returned %d, rdpmc_works_fixed_inst);
        }
    }
    if (rdpmc_works_fixed_cyc < 0 && cpuid_info.isIntel)
    {
        if (eventSupportedCount > 0 && (!(ebx & (1<<0))))
        {
            rdpmc_works_fixed_cyc = test_rdpmc(cpu_id, (1<<30) + 1, 0);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for FIXED core cycles counter returned %d, rdpmc_works_fixed_cyc);
        }
    }
    if (rdpmc_works_fixed_ref < 0 && cpuid_info.isIntel)
    {
        if (eventSupportedCount > 2 && (!(ebx & (1<<2))))
        {
            rdpmc_works_fixed_ref = test_rdpmc(cpu_id, (1<<30) + 2, 0);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for FIXED reference cycle counter returned %d, rdpmc_works_fixed_ref);
        }
    }
    if (rdpmc_works_fixed_slots < 0 && cpuid_info.isIntel)
    {
        if (eventSupportedCount > 7 && (!(ebx & (1<<7))))
        {
            rdpmc_works_fixed_slots = test_rdpmc(cpu_id, (1<<30) + 3, 0);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for FIXED slots counter returned %d, rdpmc_works_fixed_slots);
        }
    }
    if (rdpmc_works_llc < 0 && (!cpuid_info.isIntel))
    {
        switch (cpuid_info.family)
        {
            case 0x17:
                rdpmc_works_llc = test_rdpmc(cpu_id, 0xA, 0);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for L3 counters returned %d, rdpmc_works_llc);
                break;
            case 0x19:
                rdpmc_works_llc = test_rdpmc(cpu_id, 0xA, 0);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for L3 counters returned %d, rdpmc_works_llc);
                break;
            default:
                break;
        }
    }
    if (rdpmc_works_mem < 0)
    {
        switch (cpuid_info.family)
        {
            case 0x17:
                rdpmc_works_mem = test_rdpmc(cpu_id, 0x6, 0);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for DataFabric counters returned %d, rdpmc_works_mem);
                break;
            case 0x19:
                rdpmc_works_mem = test_rdpmc(cpu_id, 0x6, 0);
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for DataFabric counters returned %d, rdpmc_works_mem);
                break;
            default:
                break;
        }
    }
    pthread_mutex_unlock(&rdpmc_setup_lock);
    return 0;
}

void
access_x86_rdpmc_finalize(const int cpu_id)
{
    pthread_mutex_lock(&rdpmc_setup_lock);
    rdpmc_works_pmc = -1;
    rdpmc_works_fixed_inst = -1;
    rdpmc_works_fixed_cyc = -1;
    rdpmc_works_fixed_ref = -1;
    rdpmc_works_fixed_slots = -1;
    rdpmc_works_llc = -1;
    rdpmc_works_mem = -1;
    pthread_mutex_unlock(&rdpmc_setup_lock);
}

int
access_x86_rdpmc_read( const int cpu_id, uint32_t reg, uint64_t *data)
{
    int ret = -EAGAIN;

    switch(reg)
    {
        case MSR_PMC0:
        case MSR_PMC1:
        case MSR_PMC2:
        case MSR_PMC3:
        case MSR_PMC4:
        case MSR_PMC5:
        case MSR_PMC6:
        case MSR_PMC7:
            if (rdpmc_works_pmc == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read PMC counter with RDPMC instruction with index 0x%X, reg - MSR_PMC0);
                ret = __rdpmc(cpu_id, reg - MSR_PMC0, data);
                if (ret)
                {
                    rdpmc_works_pmc = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_AMD17_PMC0:
        case MSR_AMD17_PMC1:
        case MSR_AMD17_PMC2:
        case MSR_AMD17_PMC3:
            if (rdpmc_works_pmc == 1 && !cpuid_info.isIntel)
            {
                int index = (reg - MSR_AMD17_PMC0)/2;
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read PMC counter with RDPMC instruction with index 0x%X, index);
                ret = __rdpmc(cpu_id, index, data);
                if (ret)
                {
                    rdpmc_works_pmc = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_AMD16_PMC0:
        case MSR_AMD16_PMC1:
        case MSR_AMD16_PMC2:
        case MSR_AMD16_PMC3:
            if (rdpmc_works_pmc == 1 && !cpuid_info.isIntel)
            {
                int index = (reg - MSR_AMD16_PMC0)/2;
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read PMC counter with RDPMC instruction with index 0x%X, index);
                ret = __rdpmc(cpu_id, index, data);
                if (ret)
                {
                    rdpmc_works_pmc = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_PERF_FIXED_CTR0:
            if (rdpmc_works_fixed_inst == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED instruction counter with RDPMC instruction with index 0x%X, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
                ret = __rdpmc(cpu_id, (1<<30) + (reg - MSR_PERF_FIXED_CTR0), data);
                if (ret)
                {
                    rdpmc_works_fixed_inst = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_PERF_FIXED_CTR1:
            if (rdpmc_works_fixed_cyc == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED core cycle counter with RDPMC instruction with index 0x%X, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
                ret = __rdpmc(cpu_id, (1<<30) + (reg - MSR_PERF_FIXED_CTR0), data);
                if (ret)
                {
                    rdpmc_works_fixed_cyc = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_PERF_FIXED_CTR2:
            if (rdpmc_works_fixed_ref == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED reference cycle counter with RDPMC instruction with index 0x%X, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
                ret = __rdpmc(cpu_id, (1<<30) + (reg - MSR_PERF_FIXED_CTR0), data);
                if (ret)
                {
                    rdpmc_works_fixed_ref = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_PERF_FIXED_CTR3: //Fixed-purpose counter for TOPDOWN_SLOTS is not readable with RDPMC
            if (rdpmc_works_fixed_slots == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED slots counter with RDPMC instruction with index 0x%X, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
                ret = __rdpmc(cpu_id, (1<<30) + (reg - MSR_PERF_FIXED_CTR0), data);
                if (ret)
                {
                    rdpmc_works_fixed_slots = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_AMD17_L3_PMC0:
        case MSR_AMD17_L3_PMC1:
        case MSR_AMD17_L3_PMC2:
        case MSR_AMD17_L3_PMC3:
        case MSR_AMD17_L3_PMC4:
        case MSR_AMD17_L3_PMC5:
            if (rdpmc_works_llc == 1)
            {
                int index = (reg - MSR_AMD17_L3_PMC0)/2;

                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read AMD L3 counter with RDPMC instruction with index 0x%X, 0xA + index);
                ret = __rdpmc(cpu_id, 0xA + index, data);
                if (ret)
                {
                    rdpmc_works_llc = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        case MSR_AMD17_2_DF_PMC0:
        case MSR_AMD17_2_DF_PMC1:
        case MSR_AMD17_2_DF_PMC2:
        case MSR_AMD17_2_DF_PMC3:
            if (rdpmc_works_mem == 1)
            {
                int index = (reg - MSR_AMD17_2_DF_PMC0)/2;

                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read AMD DF counter with RDPMC instruction with index 0x%X, 0x6 + index);
                ret = __rdpmc(cpu_id, 0x6 + index, data);
                if (ret)
                {
                    rdpmc_works_mem = 0;
                    ret = -EAGAIN;
                }
            }
            break;
        default:
            ret = -EAGAIN;
    }
    return ret;
}

int
access_x86_rdpmc_write( const int cpu_id, uint32_t reg, uint64_t data)
{
    return -EPERM;
}

int access_x86_rdpmc_check(PciDeviceIndex dev, int cpu_id)
{
    int core_works = rdpmc_works_pmc;
    if (cpuid_info.isIntel)
    {
        core_works += rdpmc_works_fixed_ref + rdpmc_works_fixed_cyc;
        core_works += rdpmc_works_fixed_inst + rdpmc_works_fixed_slots; 
    }
    if (dev == MSR_DEV && (core_works > 0))
    {
        return 1;
    }
    if (cpuid_info.isIntel == 0 && dev == MSR_DEV && (rdpmc_works_mem > 0 || rdpmc_works_llc > 0))
    {
        return 1;
    }
    return 0;
}
