/*
 * =======================================================================================
 *
 *      Filename:  perfmon_pm.h
 *
 *      Description:  Header File of perfmon module Pentium M.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
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

#include <perfmon_pm_events.h>
#include <perfmon_pm_counters.h>

#define NUM_GROUPS_PM 5

static int perfmon_numCounters_pm = NUM_COUNTERS_PM;
static int perfmon_numGroups_pm = NUM_GROUPS_PM;
static int perfmon_numArchEvents_pm = NUM_ARCH_EVENTS_PM;

static PerfmonGroupMap pm_group_map[NUM_GROUPS_PM] = {
	{"FLOPS_DP",FLOPS_DP,0,"Double Precision MFlops/s",
        "EMON_SSE_SSE2_COMP_INST_RETIRED_PACKED_DP:PMC0,EMON_SSE_SSE2_COMP_INST_RETIRED_SCALAR_DP:PMC1"},
	{"FLOPS_SP",FLOPS_SP,0,"Single Precision MFlops/s",
        "EMON_SSE_SSE2_COMP_INST_RETIRED_ALL_SP:PMC0,EMON_SSE_SSE2_COMP_INST_RETIRED_SCALAR_SP:PMC1"},
	{"L2",L2,0,"L2 cache bandwidth in MBytes/s",
        "L2_LINES_IN_ALL_ALL:PMC0,L2_LINES_OUT_ALL_ALL:PMC1"},
	{"BRANCH",BRANCH,0,"Branch prediction miss rate",
        "BR_INST_EXEC:PMC0,BR_INST_MISSP_EXEC:PMC1"},
	{"CPI",CPI,0,"Cycles per instruction","UOPS_RETIRED:PMC0"}
};

void perfmon_init_pm(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);

    /* Preinit of two PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<19);  /* pin control flag */
    //    flags |= (1<<22);  /* enable flag */

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
}

void perfmon_setupCounterThread_pm(
        int thread_id,
        PerfmonEvent* event,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = pm_counter_map[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    perfmon_threadData[thread_id].counters[index].init = TRUE;
    flags = msr_read(cpu_id,reg);
    flags &= ~(0xFFFFU); 

    /* Intel with standard 8 bit event mask: [7:0] */
    flags |= (event->umask<<8) + event->eventId;

    msr_write(cpu_id, reg , flags);

    if (perfmon_verbose)
    {
        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                cpu_id,
                LLU_CAST reg,
                LLU_CAST flags);
    }
}


void perfmon_startCountersThread_pm(int thread_id)
{
    uint64_t flags = 0ULL;
    int processorId = perfmon_threadData[thread_id].processorId;

    if (perfmon_threadData[thread_id].counters[0].init == TRUE)
    {
        msr_write(processorId, pm_counter_map[0].counterRegister , 0x0ULL);
        msr_write(processorId, pm_counter_map[1].counterRegister , 0x0ULL);

        /* on p6 only MSR_PERFEVTSEL0 has the enable bit
         * it enables both counters as long MSR_PERFEVTSEL1 
         * has a valid configuration */
        flags = msr_read(processorId, MSR_PERFEVTSEL0);
        flags |= (1<<22);  /* enable flag */

        if (perfmon_verbose)
        {
            printf("perfmon_start_counters: Write Register 0x%X , \
                    Flags: 0x%llX \n",MSR_PERFEVTSEL0, LLU_CAST flags);
        }

        msr_write(processorId, MSR_PERFEVTSEL0, flags);
    }

}

void perfmon_stopCountersThread_pm(int thread_id)
{
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);

    for (i=0;i<NUM_COUNTERS_PM;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            perfmon_threadData[thread_id].counters[i].counterData =
				msr_read(cpu_id, pm_counter_map[i].counterRegister);
        }
    }
}

void perfmon_printDerivedMetrics_pm(PerfmonGroup group)
{

    switch ( group )
    {
        case FLOPS_DP:

        case FLOPS_SP:

        case L2:

        case BRANCH:

        case _NOGROUP:
            fprintf (stderr, "The Pentium M supports only two counters. Therefore derived metrics are not computed due to missing runtime!\n" );
            break;

        default:
            fprintf (stderr, "perfmon_printDerivedMetricsCore2: Unknown group! Exiting!\n" );
            exit (EXIT_FAILURE);
            break;
    }
}


