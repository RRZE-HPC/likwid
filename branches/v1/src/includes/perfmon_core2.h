/*
 * ===========================================================================
 *
 *      Filename:  perfmon_core2.h
 *
 *      Description:  Core 2 specific subroutines
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */
#include <stdlib.h>
#include <stdio.h>

#include <bstrlib.h>
#include <types.h>
#include <registers.h>
#include <perfmon_core2_events.h>

#define NUM_COUNTERS_CORE2 4
#define NUM_GROUPS_CORE2 10
#define NUM_SETS_CORE2 8

static int perfmon_numCountersCore2 = NUM_COUNTERS_CORE2;
static int perfmon_numGroupsCore2 = NUM_GROUPS_CORE2;
static int perfmon_numArchEventsCore2 = NUM_ARCH_EVENTS_CORE2;

static PerfmonCounterMap core2_counter_map[NUM_COUNTERS_CORE2] = {
    {"FIXC0",PMC0},
    {"FIXC1",PMC1},
    {"PMC0",PMC2},
    {"PMC1",PMC3}
};

static PerfmonGroupMap core2_group_map[NUM_GROUPS_CORE2] = {
    {"FLOPS_DP",FLOPS_DP,"Double Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,SIMD_COMP_INST_RETIRED_PACKED_DOUBLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_DOUBLE:PMC1"},
    {"FLOPS_SP",FLOPS_SP,"Single Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,SIMD_COMP_INST_RETIRED_PACKED_SINGLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_SINGLE:PMC1"},
    {"FLOPS_X87",FLOPS_X87,"X87 MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,X87_OPS_RETIRED_ANY:PMC0"},
    {"L2",L2,"L2 cache bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L1D_REPL:PMC0,L1D_M_EVICT:PMC1"},
    {"MEM",MEM,"Main memory bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,BUS_TRANS_MEM_THIS_CORE_THIS_A:PMC0"},
    {"CACHE",CACHE,"Data cache miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L1D_REPL:PMC0,L1D_ALL_CACHE_REF:PMC1"},
    {"L2CACHE",L2CACHE,"L2 cache miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L2_RQSTS_THIS_CORE_ALL_MESI:PMC0,L2_RQSTS_SELF_I_STATE:PMC1"},
    {"DATA",DATA,"Load to store ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,INST_RETIRED_LOADS:PMC0,INST_RETIRED_STORES:PMC1"},
    {"BRANCH",BRANCH,"Branch prediction miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,BR_INST_RETIRED_ANY:PMC0,BR_INST_RETIRED_MISPRED:PMC1"},
    {"TLB",TLB,"TLB miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,DTLB_MISSES_ANY:PMC0,L1D_ALL_CACHE_REF:PMC1"}
};

#if 0
static char* core2_report_config[NUM_SETS_CORE2] = {
    "INST_RETIRED_LOADS:PMC0,INST_RETIRED_STORES:PMC1",
    "BR_INST_RETIRED_ANY:PMC0,BR_INST_RETIRED_MISPRED:PMC1",
    "SIMD_COMP_INST_RETIRED_PACKED_DOUBLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_DOUBLE:PMC1",
    "SIMD_COMP_INST_RETIRED_PACKED_SINGLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_SINGLE:PMC1",
    "INST_RETIRED_LOADS:PMC0,INST_RETIRED_STORES:PMC1",
    "MEM_LOAD_RETIRED_L1D_LINE_MISS:PMC0,L1D_ALL_REF:PMC1",
    "BUS_TRANS_MEM_THIS_CORE_THIS_A:PMC0,DTLB_MISSES_ANY:PMC1",
    "L1D_REPL:PMC0,L1D_M_EVICT:PMC1"};
#endif

void 
perfmon_init_core2(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    /* Fixed Counters: instructions retired, cycles unhalted core */
    thread->counters[PMC0].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC0].counterRegister = MSR_PERF_FIXED_CTR0;
    thread->counters[PMC0].type = FIXED;
    thread->counters[PMC1].configRegister = MSR_PERF_FIXED_CTR_CTRL;
    thread->counters[PMC1].counterRegister = MSR_PERF_FIXED_CTR1;
    thread->counters[PMC1].type = FIXED;

    /* PMC Counters: 2 40bit wide */
    thread->counters[PMC2].configRegister = MSR_PERFEVTSEL0;
    thread->counters[PMC2].counterRegister = MSR_PMC0;
    thread->counters[PMC2].type = PMC;
    thread->counters[PMC3].configRegister = MSR_PERFEVTSEL1;
    thread->counters[PMC3].counterRegister = MSR_PMC1;
    thread->counters[PMC3].type = PMC;

    /* Initialize registers */
    msr_write(cpu_id, MSR_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_PERFEVTSEL1, 0x0ULL);

    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x0ULL);
    msr_write(cpu_id, MSR_PEBS_ENABLE, 0x0ULL);

    /* always initialize fixed counters
     * FIXED 0: Instructions retired
     * FIXED 1: Clocks unhalted */
    msr_write(cpu_id, MSR_PERF_FIXED_CTR_CTRL, 0x22ULL);

    /* Preinit of PMC counters */
    flags |= (1<<16);  /* user mode flag */
    flags |= (1<<19);  /* pin control flag */
    flags |= (1<<22);  /* enable flag */

    msr_write(cpu_id, MSR_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_PERFEVTSEL1, flags);
}


void
perfmon_setupCounterThread_core2(int thread_id,
        uint32_t event, uint32_t umask,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    if (perfmon_threadData[thread_id].counters[index].type == PMC)
    {

        perfmon_threadData[thread_id].counters[index].init = TRUE;
        flags = msr_read(cpu_id,reg);
        flags &= ~(0xFFFFU); 

        /* Intel with standard 8 bit event mask: [7:0] */
        flags |= (umask<<8) + event;

        msr_write(cpu_id, reg , flags);

        if (perfmon_verbose)
        {
            printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
                    cpu_id,
                    LLU_CAST reg,
                    LLU_CAST flags);
        }
    }
    else if (perfmon_threadData[thread_id].counters[index].type == FIXED)
    {
        perfmon_threadData[thread_id].counters[index].init = TRUE;
    }
}

void
perfmon_startCountersThread_core2(int thread_id)
{
    int i;
    uint64_t flags = 0ULL;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    for (i=0;i<NUM_COUNTERS_CORE2;i++) {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) {
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);

            if (perfmon_threadData[thread_id].counters[i].type == PMC)
            {
                flags |= (1<<(i-2));  /* enable counter */
            }
            else if (perfmon_threadData[thread_id].counters[i].type == FIXED)
            {
                flags |= (1ULL<<(i+32));  /* enable fixed counter */
            }
        }
    }

    if (perfmon_verbose)
    {
        printf("perfmon_start_counters: Write Register 0x%X , Flags: 0x%llX \n",
                MSR_PERF_GLOBAL_CTRL, LLU_CAST flags);
    }

    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, flags);
    msr_write(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, 0x300000003ULL);
}

void 
perfmon_stopCountersThread_core2(int thread_id)
{
    uint64_t flags;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    /* stop counters */
    msr_write(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL);

    /* read out counter results */
    for (i=0;i<NUM_COUNTERS_CORE2;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            perfmon_threadData[thread_id].counters[i].counterData =
                msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }

    /* check overflow status */
    flags = msr_read(cpu_id,MSR_PERF_GLOBAL_STATUS);
    if((flags & 0x3) || (flags & (0x3ULL<<32)) ) 
    {
        printf ("Overflow occured \n");
        printf ("Status: 0x%llX \n", LLU_CAST flags);
    }
}

void
perfmon_printDerivedMetricsCore2(PerfmonGroup group)
{
    int threadId;
    double time = 0.0;
    double cpi = 0.0;
    double inverseClock = 1.0 /(double) timer_getCpuClock();
    PerfmonResultTable tableData;
    int numRows;
    int numColumns = perfmon_numThreads;
    bstrList* fc;
    bstring label;

    switch ( group ) 
    {
        case FLOPS_DP:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,DP MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")*2.0+
                            perfmon_getResult(threadId,"PMC1")) / time;
            }
            break;

        case FLOPS_SP:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,SP MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                   1.0E-06*(perfmon_getResult(threadId,"PMC0")*4.0+
                     perfmon_getResult(threadId,"PMC1")) / time;
            }
            break;

        case FLOPS_X87:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,X87 MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
            }
            break;

        case L2:
            numRows = 5;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,L2 Load [MBytes/s]);
            bstrListAdd(4,L2 Evict [MBytes/s]);
            bstrListAdd(5,L2 bandwidth [MBytes/s]);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")*64)/time;
                tableData.rows[3].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC1")*64)/time;
                tableData.rows[4].value[threadId] =
                1.0E-06*((perfmon_getResult(threadId,"PMC0")+
                            perfmon_getResult(threadId,"PMC1"))*64)/time;
            }
            break;

        case MEM:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,Memory bandwidth [MBytes/s]);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")*64)/time;
            }

            break;

        case CACHE:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Data cache misses);
            bstrListAdd(2,Data cache request rate);
            bstrListAdd(3,Data cache miss rate);
            bstrListAdd(4,Data cache miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                    perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[2].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[3].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"PMC1");

            }

            break;

        case L2CACHE:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,L2 request rate);
            bstrListAdd(2,L2 miss rate);
            bstrListAdd(3,L2 miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] =
                     perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[2].value[threadId] =
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"PMC0");
            }

            break;

        case DATA:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,Load to Store ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    perfmon_getResult(threadId,"PMC0")/perfmon_getResult(threadId,"PMC1");
            }

            break;

        case BRANCH:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Branch rate);
            bstrListAdd(2,Branch misprediction rate);
            bstrListAdd(3,Branch misprediction ratio);
            bstrListAdd(4,Instructions per branch);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = 
                    (perfmon_getResult(threadId,"PMC0")/ perfmon_getResult(threadId,"FIXC0"));
                tableData.rows[1].value[threadId] =
                    (perfmon_getResult(threadId,"PMC1")/ perfmon_getResult(threadId,"FIXC0"));
                tableData.rows[2].value[threadId] =
                    (perfmon_getResult(threadId,"PMC1")/ perfmon_getResult(threadId,"PMC0"));
                tableData.rows[3].value[threadId] =
                    (perfmon_getResult(threadId,"FIXC0")/ perfmon_getResult(threadId,"PMC0"));

            }
            break;

        case TLB:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,L1 DTLB request rate);
            bstrListAdd(2,L1 DTLB miss rate);
            bstrListAdd(3,L1 DTLB miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = 
                    perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[1].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[2].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"PMC1");
            }

            break;

        case NOGROUP:
            numRows = 2;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                if (perfmon_getResult(threadId,"FIXC0") < 1.0E-12)
                {
                    cpi  =  0.0;
                }
                else
                {
                    cpi  =  perfmon_getResult(threadId,"FIXC1")/
                        perfmon_getResult(threadId,"FIXC0");
                }

                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
            }

            break;

        default:
            fprintf (stderr, "perfmon_printDerivedMetricsCore2: Unknown group! Exiting!\n" );
            exit (EXIT_FAILURE);
            break;
    }

    printResultTable(&tableData);
    bdestroy(label);
    bstrListDestroy(fc);
}


#if 0
void
perfmon_setupReport_core2(MultiplexCollections* collections)
{
    collections->numberOfCollections = 6;
    collections->collections =
		(PerfmonEventSet*) malloc(collections->numberOfCollections
				* sizeof(PerfmonEventSet));

    /*  Load/Store */
    collections->collections[0].numberOfEvents = 2;

    collections->collections[0].events =
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[0].events[0].eventName =
		bfromcstr("INST_RETIRED_LOADS");

    collections->collections[0].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[0].events[1].eventName =
		bfromcstr("INST_RETIRED_STORES");

    collections->collections[0].events[1].reg =
		bfromcstr("PMC1");

    /*  Branches */
    collections->collections[1].numberOfEvents = 2;
    collections->collections[1].events =
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[1].events[0].eventName =
		bfromcstr("BR_INST_RETIRED_ANY");

    collections->collections[1].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[1].events[1].eventName =
		bfromcstr("BR_INST_RETIRED_MISPRED");

    collections->collections[1].events[1].reg =
		bfromcstr("PMC1");

    /*  SIMD Double */
    collections->collections[2].numberOfEvents = 2;
    collections->collections[2].events =
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[2].events[0].eventName =
		bfromcstr("SIMD_INST_RETIRED_PACKED_DOUBLE");

    collections->collections[2].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[2].events[1].eventName =
		bfromcstr("SIMD_INST_RETIRED_SCALAR_DOUBLE");

    collections->collections[2].events[1].reg =
		bfromcstr("PMC1");

    /*  L1 Utilization */
    collections->collections[3].numberOfEvents = 2;
    collections->collections[3].events = 
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[3].events[0].eventName =
		bfromcstr("MEM_LOAD_RETIRED_L1D_LINE_MISS");

    collections->collections[3].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[3].events[1].eventName =
		bfromcstr("L1D_ALL_REF");

    collections->collections[3].events[1].reg =
		bfromcstr("PMC1");

    /*  Memory transfers/TLB misses */
    collections->collections[4].numberOfEvents = 2;
    collections->collections[4].events =
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[4].events[0].eventName =
		bfromcstr("BUS_TRANS_MEM_THIS_CORE_THIS_A");

    collections->collections[4].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[4].events[1].eventName =
		bfromcstr("DTLB_MISSES_ANY");

    collections->collections[4].events[1].reg =
		bfromcstr("PMC1");

    /*  L2 bandwidth */
    collections->collections[5].numberOfEvents = 2;
    collections->collections[5].events =
		(PerfmonEventSetEntry*) malloc(2 * sizeof(PerfmonEventSetEntry));

    collections->collections[5].events[0].eventName =
		bfromcstr("L1D_REPL");

    collections->collections[5].events[0].reg =
		bfromcstr("PMC0");

    collections->collections[5].events[1].eventName =
		bfromcstr("L1D_M_EVICT");

    collections->collections[5].events[1].reg =
		bfromcstr("PMC1");
}

void
perfmon_printReport_core2(MultiplexCollections* collections)
{
    printf(HLINE);
    printf("PERFORMANCE REPORT\n");
    printf(HLINE);
    printf("\nRuntime  %.2f s\n\n",collections->time);

    /* Section 1 */
    printf("Code characteristics:\n");
    printf("\tLoad to store ratio %f \n",
			collections->collections[0].events[0].results[0]/
			collections->collections[0].events[1].results[0]);

    printf("\tPercentage SIMD vectorized double %.2f \n",
			(collections->collections[2].events[0].results[0]*100.0)/
            (collections->collections[2].events[0].results[0]+
			 collections->collections[2].events[1].results[0]));

    printf("\tPercentage mispredicted branches  %.2f \n",
			(collections->collections[1].events[1].results[0]*100.0)/
            collections->collections[1].events[0].results[0]);

    /* Section 2 */
    printf("\nCode intensity:\n");
    printf("\tDouble precision Flops/s  %.2f MFlops/s\n",
			1.0E-06*(collections->collections[2].events[0].results[0]*2.0+
				collections->collections[2].events[1].results[0] )/
			(double) (collections->time*0.16666666));

    printf("\nResource Utilization:\n");
    printf("\tL1 Ref per miss %.2f \n",
			(collections->collections[3].events[1].results[0]/
			 collections->collections[3].events[0].results[0]));

    printf("\tRefs per TLB miss  %.2f \n",
			(collections->collections[3].events[1].results[0]/
			 collections->collections[4].events[1].results[0]));

    /* Section 3 */
    printf("\nBandwidths:\n");
    printf("\tL2 bandwidth  %.2f MBytes/s\n",
			1.0E-06*((collections->collections[5].events[1].results[0]+
					collections->collections[5].events[0].results[0])*64.0)/
			(double) (collections->time*0.16666666));

    printf("\tMemory bandwidth  %.2f MBytes/s\n",
			1.0E-06*((collections->collections[4].events[0].results[0])*64.0)/
			(double) (collections->time*0.16666666));
    printf(HLINE);
}
#endif



