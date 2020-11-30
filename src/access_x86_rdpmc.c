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


#include <access_x86_rdpmc.h>
#include <types.h>
#include <error.h>
#include <registers.h>
#include <signal.h>
#include <sched.h>

static int rdpmc_works_pmc = -1;
static int rdpmc_works_fixed = -1;
static int rdpmc_works_llc = -1;
static int rdpmc_works_mem = -1;


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
    exit(1);
}

static int
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
        sa.sa_sigaction = segfault_sigaction_rdpmc;
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

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_x86_rdpmc_init(const int cpu_id)
{
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
    if (rdpmc_works_llc < 0)
    {
        rdpmc_works_llc = test_rdpmc(cpu_id, 0xA, 0);
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Test for RDPMC for L3 counters returned %d, rdpmc_works_llc);
    }
    return 0;
}

void
access_x86_rdpmc_finalize(const int cpu_id)
{
    rdpmc_works_pmc = -1;
    rdpmc_works_fixed = -1;
}

int
access_x86_rdpmc_read( const int cpu_id, uint32_t reg, uint64_t *data)
{
    int ret = 0;

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
        case MSR_PERF_FIXED_CTR0:
        case MSR_PERF_FIXED_CTR1:
        case MSR_PERF_FIXED_CTR2:
        case MSR_PERF_FIXED_CTR3:
            if (rdpmc_works_fixed == 1)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Read FIXED counter with RDPMC instruction with index 0x%X, (1<<30) + (reg - MSR_PERF_FIXED_CTR0));
                ret = __rdpmc(cpu_id, reg - MSR_PERF_FIXED_CTR0, data);
                if (ret)
                {
                    rdpmc_works_fixed = 0;
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
                int index = (reg - MSR_PERF_FIXED_CTR0)/2;

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
        case MSR_AMD17_2_DF_PMC4:
        case MSR_AMD17_2_DF_PMC5:
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
    if (dev == MSR_DEV && (rdpmc_works_pmc > 0 || rdpmc_works_fixed > 0))
    {
        return 1;
    }
    return 0;
}