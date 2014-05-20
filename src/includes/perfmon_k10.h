/*
 * ===========================================================================
 *
 *      Filename:  perfmon_k10.h
 *
 *      Description:  AMD K10 specific subroutines
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
#include <perfmon_k10_events.h>

#define NUM_COUNTERS_K10 4
#define NUM_GROUPS_K10 13
#define NUM_SETS_K10 8

static int perfmon_numCountersK10 = NUM_COUNTERS_K10;
static int perfmon_numGroupsK10 = NUM_GROUPS_K10;
static int perfmon_numArchEventsK10 = NUM_ARCH_EVENTS_K10;

static PerfmonCounterMap k10_counter_map[NUM_COUNTERS_K10] = {
    {"PMC0",PMC0},
    {"PMC1",PMC1},
    {"PMC2",PMC2},
    {"PMC3",PMC3}
};

static PerfmonGroupMap k10_group_map[NUM_GROUPS_K10] = {
    {"FLOPS_DP",FLOPS_DP,"Double Precision MFlops/s","SSE_RETIRED_ADD_DOUBLE_FLOPS:PMC0,SSE_RETIRED_MULT_DOUBLE_FLOPS:PMC1,CPU_CLOCKS_UNHALTED:PMC2"},
    {"FLOPS_SP",FLOPS_SP,"Single Precision MFlops/s","SSE_RETIRED_ADD_SINGLE_FLOPS:PMC0,SSE_RETIRED_MULT_SINGLE_FLOPS:PMC1,CPU_CLOCKS_UNHALTED:PMC2"},
    {"FLOPS_X87",FLOPS_X87,"X87 MFlops/s","X87_FLOPS_RETIRED_ADD:PMC0,X87_FLOPS_RETIRED_MULT:PMC1,X87_FLOPS_RETIRED_DIV:PMC2,CPU_CLOCKS_UNHALTED:PMC3"},
    {"L2",L2,"L2 cache bandwidth in MBytes/s","DATA_CACHE_REFILLS_L2_ALL:PMC0,DATA_CACHE_EVICTED_ALL:PMC1,CPU_CLOCKS_UNHALTED:PMC2"},
/*    {"L3",L3,"L3 cache bandwidth in MBytes/s","L3_FILLS_ALL_ALL_CORES:PMC0,L3_READ_REQUEST_ALL_ALL_CORES:PMC1,CPU_CLOCKS_UNHALTED:PMC2"},*/
    {"MEM",MEM,"Main memory bandwidth in MBytes/s","NORTHBRIDGE_READ_RESPONSE_ALL:PMC0,OCTWORDS_WRITE_TRANSFERS:PMC1,DRAM_ACCESSES_DCTO_ALL:PMC2,DRAM_ACCESSES_DCT1_ALL:PMC3"},
    {"CACHE",CACHE,"Data cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,DATA_CACHE_ACCESSES:PMC1,DATA_CACHE_REFILLS_L2_ALL:PMC2,DATA_CACHE_REFILLS_NORTHBRIDGE_ALL:PMC3"},
    {"ICACHE",ICACHE,"Instruction cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,ICACHE_FETCHES:PMC1,ICACHE_REFILLS_L2:PMC2,ICACHE_REFILLS_MEM:PMC3"},
    {"L2CACHE",L2CACHE,"L2 cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,L2_REQUESTS_ALL:PMC1,L2_MISSES_ALL:PMC2,L2_FILL_ALL:PMC3"},
    {"L3CACHE",L3CACHE,"L3 cache miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,L3_READ_REQUEST_ALL_ALL_CORES:PMC1,L3_MISSES_ALL_ALL_CORES:PMC2"},
    {"BRANCH",BRANCH,"Branch prediction miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,BRANCH_RETIRED:PMC1,BRANCH_MISPREDICT_RETIRED:PMC2,BRANCH_TAKEN_RETIRED:PMC3"},
    {"CPI",CPI,"cycles per instruction","INSTRUCTIONS_RETIRED:PMC0,CPU_CLOCKS_UNHALTED:PMC1,UOPS_RETIRED:PMC2"},
    {"FPU_EXCEPTION",FPU_EXCEPTION,"Floating point exceptions","INSTRUCTIONS_RETIRED:PMC0,FP_INSTRUCTIONS_RETIRED_ALL:PMC1,FPU_EXCEPTIONS_ALL:PMC2"},
    {"TLB",TLB,"TLB miss rate/ratio","INSTRUCTIONS_RETIRED:PMC0,DATA_CACHE_ACCESSES:PMC1,DTLB_L2_HIT_ALL:PMC2,DTLB_L2_MISS_ALL:PMC3"}
};


void perfmon_init_k10(PerfmonThread *thread)
{
    uint64_t flags = 0x0ULL;
    int cpu_id = thread->processorId;

    thread->counters[PMC0].configRegister = MSR_AMD_PERFEVTSEL0;
    thread->counters[PMC0].counterRegister = MSR_AMD_PMC0;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC1].configRegister = MSR_AMD_PERFEVTSEL1;
    thread->counters[PMC1].counterRegister = MSR_AMD_PMC1;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC2].configRegister = MSR_AMD_PERFEVTSEL2;
    thread->counters[PMC2].counterRegister = MSR_AMD_PMC2;
    thread->counters[PMC0].type = PMC;
    thread->counters[PMC3].configRegister = MSR_AMD_PERFEVTSEL3;
    thread->counters[PMC3].counterRegister = MSR_AMD_PMC3;
    thread->counters[PMC0].type = PMC;

    msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, 0x0ULL);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, 0x0ULL);

    flags |= (1<<16);  /* user mode flag */

    msr_write(cpu_id, MSR_AMD_PERFEVTSEL0, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL1, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL2, flags);
    msr_write(cpu_id, MSR_AMD_PERFEVTSEL3, flags);
}


void
perfmon_setupCounterThread_k10(int thread_id,
        uint32_t event, uint32_t umask,
        PerfmonCounterIndex index)
{
    uint64_t flags;
    uint64_t reg = perfmon_threadData[thread_id].counters[index].configRegister;
    int cpu_id = perfmon_threadData[thread_id].processorId;
    perfmon_threadData[thread_id].counters[index].init = TRUE;

    flags = msr_read(cpu_id,reg);
    flags &= ~(0xFFFFU); 

	/* AMD uses a 12 bit Event mask: [35:32][7:0] */
	flags |= ((uint64_t)(event>>8)<<32) + (umask<<8) + (event & ~(0xF00U));

    if (perfmon_verbose)
    {
        printf("[%d] perfmon_setup_counter: Write Register 0x%llX , Flags: 0x%llX \n",
               cpu_id,
               LLU_CAST reg,
               LLU_CAST flags);
    }
    msr_write(cpu_id, reg , flags);
}


void
perfmon_startCountersThread_k10(int thread_id)
{
    int i;
    uint64_t flags;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    for (i=0;i<NUM_COUNTERS_K10;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister , 0x0ULL);
            flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
            flags |= (1<<22);  /* enable flag */
            if (perfmon_verbose) 
            {
                printf("perfmon_start_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                        LLU_CAST flags);
            }

            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
        }
    }
}

void 
perfmon_stopCountersThread_k10(int thread_id)
{
    uint64_t flags;
    int i;
    int cpu_id = perfmon_threadData[thread_id].processorId;

    for (i=0;i<NUM_COUNTERS_K10;i++) 
    {
        if (perfmon_threadData[thread_id].counters[i].init == TRUE) 
        {
            flags = msr_read(cpu_id,perfmon_threadData[thread_id].counters[i].configRegister);
            flags &= ~(1<<22);  /* clear enable flag */
            msr_write(cpu_id, perfmon_threadData[thread_id].counters[i].configRegister , flags);
            if (perfmon_verbose)
            {
                printf("perfmon_stop_counters: Write Register 0x%llX , Flags: 0x%llX \n",
                        LLU_CAST perfmon_threadData[thread_id].counters[i].configRegister,
                        LLU_CAST flags);
            }
            perfmon_threadData[thread_id].counters[i].counterData = msr_read(cpu_id, perfmon_threadData[thread_id].counters[i].counterRegister);
        }
    }
}



void perfmon_printDerivedMetrics_k10(PerfmonGroup group)
{
    int threadId;
    double time = 0.0;
    double inverseClock = 1.0 /(double) timer_getCpuClock();
    PerfmonResultTable tableData;
    int numRows;
    int numColumns = perfmon_numThreads;
    bstrList* fc;
    bstring label;

    switch ( group )
    {
        case FLOPS_DP:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,DP MFlops/s);
            bstrListAdd(3,DP Add MFlops/s);
            bstrListAdd(4,DP Mult MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC2") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")+ perfmon_getResult(threadId,"PMC1")) / time;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
                tableData.rows[3].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC1")) / time;
            }
            break;

        case FLOPS_SP:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,SP MFlops/s);
            bstrListAdd(3,SP Add MFlops/s);
            bstrListAdd(4,SP Mult MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC2") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")+ perfmon_getResult(threadId,"PMC1")) / time;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
                tableData.rows[3].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC1")) / time;
            }
            break;

        case FLOPS_X87:
            numRows = 5;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,X87 MFlops/s);
            bstrListAdd(3,X87 Add MFlops/s);
            bstrListAdd(4,X87 Mult MFlops/s);
            bstrListAdd(5,X87 Div MFlops/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC3") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")+
                             perfmon_getResult(threadId,"PMC1") +
                             perfmon_getResult(threadId,"PMC2")) / time;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
                tableData.rows[3].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC1")) / time;
                tableData.rows[4].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC2")) / time;
            }
            break;


        case L2:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,L2 bandwidth MBytes/s);
            bstrListAdd(3,L2 refill bandwidth MBytes/s);
            bstrListAdd(4,L2 evict MBytes/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC2") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")+ perfmon_getResult(threadId,"PMC1"))*64.0 / time;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")*64.0) / time;
                tableData.rows[3].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC1")*64.0) / time;
            }
            break;

#if 0
        case L3:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,L3 bandwidth MBytes/s);
            bstrListAdd(3,L3 evict bandwidth MBytes/s);
            bstrListAdd(4,L3 read MBytes/s);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC2") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")+ perfmon_getResult(threadId,"PMC1"))*64.0 / time;
                tableData.rows[2].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC0")*64.0) / time;
                tableData.rows[3].value[threadId] =
                     1.0E-06*(perfmon_getResult(threadId,"PMC1")*64.0) / time;
            }
            break;
#endif

        case MEM:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,Read data bandwidth (MBytes/s));
            bstrListAdd(3,Write data bandwidth (MBytes/s));
            bstrListAdd(4,DRAM bandwidth (MBytes/s));
            initResultTable(&tableData, fc, numRows, numColumns);
            printf("NOTE: Runtime is based on external cycles measurement and not on CPU_CLOCKS_UNHALTED!\n");

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = timer_printCyclesTime(&timeData);
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     1.0E-06*((perfmon_getResult(threadId,"PMC0")*64.0) / time);
                tableData.rows[2].value[threadId] =
                     1.0E-06*((perfmon_getResult(threadId,"PMC1")*8.0) / time);
                tableData.rows[3].value[threadId] =
                     1.0E-06*(((perfmon_getResult(threadId,"PMC2")+perfmon_getResult(threadId,"PMC3"))*64.0) / time);
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
                tableData.rows[0].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[2].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC0");
                tableData.rows[3].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC1");
            }

            break;

        case ICACHE:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Instruction cache misses);
            bstrListAdd(2,Instruction cache request rate);
            bstrListAdd(3,Instruction cache miss rate);
            bstrListAdd(4,Instruction cache miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[2].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC0");
                tableData.rows[3].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC1");
            }

            break;

        case L2CACHE:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,L2 request rate);
            bstrListAdd(2,L2 miss rate);
            bstrListAdd(3,L2 miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);
            printf("NOTE: Direct method with limited accuracy!\n");

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] =
                     (perfmon_getResult(threadId,"PMC1") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[2].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") / (perfmon_getResult(threadId,"PMC1") + perfmon_getResult(threadId,"PMC3"));
            }

            break;

        case L3CACHE:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,L3 request rate);
            bstrListAdd(2,L3 miss rate);
            bstrListAdd(3,L3 miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] =
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[2].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") / perfmon_getResult(threadId,"PMC1");
            }

            break;

        case BRANCH:
            numRows = 6;
            INIT_BASIC;
            bstrListAdd(1,Branch rate);
            bstrListAdd(2,Branch misprediction rate);
            bstrListAdd(3,Branch misprediction ratio);
            bstrListAdd(4,Branch taken rate);
            bstrListAdd(5,Branch taken ratio);
            bstrListAdd(6,Instructions per branch);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = 
                     (perfmon_getResult(threadId,"PMC1")/ perfmon_getResult(threadId,"PMC0"));
                tableData.rows[1].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2")/ perfmon_getResult(threadId,"PMC0"));
                tableData.rows[2].value[threadId] =
                     (perfmon_getResult(threadId,"PMC2")/ perfmon_getResult(threadId,"PMC1"));
                tableData.rows[3].value[threadId] =
                     (perfmon_getResult(threadId,"PMC3")/ perfmon_getResult(threadId,"PMC0"));
                tableData.rows[4].value[threadId] =
                     (perfmon_getResult(threadId,"PMC3")/ perfmon_getResult(threadId,"PMC1"));
                tableData.rows[5].value[threadId] =
                     (perfmon_getResult(threadId,"PMC0")/ perfmon_getResult(threadId,"PMC1"));
            }
            break;

        case CPI:
            numRows = 4;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,CPI (based on uops));
            bstrListAdd(4,IPC);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"PMC1") * inverseClock;
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] =
                     (perfmon_getResult(threadId,"PMC1")/ perfmon_getResult(threadId,"PMC0"));
                tableData.rows[2].value[threadId] =
                     (perfmon_getResult(threadId,"PMC1")/ perfmon_getResult(threadId,"PMC2"));
                tableData.rows[3].value[threadId] =
                     (perfmon_getResult(threadId,"PMC0")/ perfmon_getResult(threadId,"PMC1"));
            }
            break;

        case TLB:
            numRows = 6;
            INIT_BASIC;
            bstrListAdd(1,L1 DTLB request rate);
            bstrListAdd(2,L1 DTLB miss rate);
            bstrListAdd(3,L1 DTLB miss ratio);
            bstrListAdd(4,L2 DTLB request rate);
            bstrListAdd(5,L2 DTLB miss rate);
            bstrListAdd(6,L2 DTLB miss ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            printf("NOTE: The L2 metrics are only relevant if L2 DTLB request rate is equal to the L1 DTLB miss rate!\n");

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = 
                     perfmon_getResult(threadId,"PMC1") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                    (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC0");
                tableData.rows[2].value[threadId] =
                    (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC1");
                tableData.rows[3].value[threadId] =
                    (perfmon_getResult(threadId,"PMC2") + perfmon_getResult(threadId,"PMC3")) / perfmon_getResult(threadId,"PMC0");
                tableData.rows[4].value[threadId] = 
                     perfmon_getResult(threadId,"PMC3") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[5].value[threadId] = 
                     perfmon_getResult(threadId,"PMC3") / (perfmon_getResult(threadId,"PMC2")+perfmon_getResult(threadId,"PMC3"));
            }
            break;

        case FPU_EXCEPTION:
            numRows = 2;
            INIT_BASIC;
            bstrListAdd(1,Overall FP exception rate);
            bstrListAdd(2,FP exception rate);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = 
                     perfmon_getResult(threadId,"PMC2") / perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                     perfmon_getResult(threadId,"PMC2") / perfmon_getResult(threadId,"PMC1");
            }
            break;

        case NOGROUP:
            numRows = 1;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                if (perfmon_getResult(threadId,"PMC3") > 1.0E-6)
                {
                    time = perfmon_getResult(threadId,"PMC3") * inverseClock;
                    tableData.rows[0].value[threadId] = time;
                }
                else
                {
                    tableData.rows[0].value[threadId] = 0.0;
                }
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

