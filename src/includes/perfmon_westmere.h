/*
 * ===========================================================================
 *
 *      Filename:  perfmon_westmere.h
 *
 *      Description:  Header File of perfmon module for Westmere.
 *                    Configures and reads out performance counters
 *                    on x86 based architectures. Supports multi threading.
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

#include <cpuid.h>
#include <tree.h>
#include <bstrlib.h>
#include <types.h>
#include <registers.h>
#include <perfmon_westmere_events.h>

#define NUM_GROUPS_WESTMERE 12

static int perfmon_numGroupsWestmere = NUM_GROUPS_WESTMERE;
static int perfmon_numArchEventsWestmere = NUM_ARCH_EVENTS_WESTMERE;

static PerfmonGroupMap westmere_group_map[NUM_GROUPS_WESTMERE] = {
    {"FLOPS_DP",FLOPS_DP,"Double Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,FP_COMP_OPS_EXE_SSE_FP_PACKED:PMC0,FP_COMP_OPS_EXE_SSE_FP_SCALAR:PMC1,FP_COMP_OPS_EXE_SSE_SINGLE_PRECISION:PMC2,FP_COMP_OPS_EXE_SSE_DOUBLE_PRECISION:PMC3"},
    {"FLOPS_SP",FLOPS_SP,"Single Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,FP_COMP_OPS_EXE_SSE_FP_PACKED:PMC0,FP_COMP_OPS_EXE_SSE_FP_SCALAR:PMC1,FP_COMP_OPS_EXE_SSE_SINGLE_PRECISION:PMC2,FP_COMP_OPS_EXE_SSE_DOUBLE_PRECISION:PMC3"},
    {"FLOPS_X87",FLOPS_X87,"X87 MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,INST_RETIRED_X87:PMC0"},
    {"L2",L2,"L2 cache bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L1D_REPL:PMC0,L1D_M_EVICT:PMC1"},
    {"L3",L3,"L3 cache bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L2_LINES_IN_ANY:PMC0,L2_LINES_OUT_DEMAND_DIRTY:PMC1"},
    {"MEM",MEM,"Main memory bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,UNC_QMC_NORMAL_READS_ANY:UPMC0,UNC_QMC_WRITES_FULL_ANY:UPMC1"},
    {"CACHE",CACHE,"Data cache miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L1D_REPL:PMC0"},
    {"L2CACHE",L2CACHE,"L2 cache miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L2_DATA_RQSTS_DEMAND_ANY:PMC0,L2_RQSTS_MISS:PMC1"},
    {"L3CACHE",L3CACHE,"L3 cache miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,UNC_L3_HITS_ANY:UPMC0,UNC_L3_MISS_ANY:UPMC1,UNC_L3_LINES_IN_ANY:UPMC2,UNC_L3_LINES_OUT_ANY:UPMC3"},
    {"DATA",DATA,"Load to store ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,MEM_INST_RETIRED_LOADS:PMC0,MEM_INST_RETIRED_STORES:PMC1"},
    {"BRANCH",BRANCH,"Branch prediction miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,BR_INST_RETIRED_ALL_BRANCHES:PMC0,BR_MISP_RETIRED_ALL_BRANCHES:PMC1"},
    {"TLB",TLB,"TLB miss rate/ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,DTLB_MISSES_ANY:PMC0"}
};



void
perfmon_printDerivedMetricsWestmere(PerfmonGroup group)
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
            numRows = 7;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,DP MFlops/s (DP assumed));
            bstrListAdd(4,Packed MUOPS/s);
            bstrListAdd(5,Scalar MUOPS/s);
            bstrListAdd(6,SP MUOPS/s);
            bstrListAdd(7,DP MUOPS/s);
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
                tableData.rows[3].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
                tableData.rows[4].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC1")) / time;
                tableData.rows[5].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC2")) / time;
                tableData.rows[6].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC3")) / time;
            }
            break;

        case FLOPS_SP:
            numRows = 7;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,SP MFlops/s (SP assumed));
            bstrListAdd(4,Packed MUOPS/s);
            bstrListAdd(5,Scalar MUOPS/s);
            bstrListAdd(6,SP MUOPS/s);
            bstrListAdd(7,DP MUOPS/s);

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
                tableData.rows[3].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")) / time;
                tableData.rows[4].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC1")) / time;
                tableData.rows[5].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC2")) / time;
                tableData.rows[6].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC3")) / time;
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
            bstrListAdd(1,Runtime);
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
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")*64.0)/time;
                tableData.rows[3].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC1")*64.0)/time;
                tableData.rows[4].value[threadId] =
                    1.0E-06*((perfmon_getResult(threadId,"PMC0")+
                                perfmon_getResult(threadId,"PMC1"))*64.0)/time;
            }
            break;

        case L3:
            numRows = 5;
            INIT_BASIC;
            bstrListAdd(1,Runtime);
            bstrListAdd(2,CPI);
            bstrListAdd(3,L3 Load [MBytes/s]);
            bstrListAdd(4,L3 Evict [MBytes/s]);
            bstrListAdd(5,L3 bandwidth [MBytes/s]);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")*64.0)/time;
                tableData.rows[3].value[threadId] =
                    1.0E-06*(perfmon_getResult(threadId,"PMC1")*64.0)/time;
                tableData.rows[4].value[threadId] =
                    1.0E-06*((perfmon_getResult(threadId,"PMC0")+
                                perfmon_getResult(threadId,"PMC1"))*64.0)/time;
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
                    1.0E-06*(perfmon_getResult(threadId,"UPMC0")+
                            perfmon_getResult(threadId,"UPMC1")) * 64 / time;
            }
            break;

        case DATA:
            numRows = 3;
            INIT_BASIC;
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,Store to Load ratio);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
                    (perfmon_getResult(threadId,"PMC0")/perfmon_getResult(threadId,"PMC1"));
            }
            break;

        case CACHE:
            numRows = 2;
            INIT_BASIC;
            bstrListAdd(1,Data cache misses);
            bstrListAdd(2,Data cache miss rate);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] = perfmon_getResult(threadId,"PMC0");
                tableData.rows[1].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"FIXC0");

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
                    perfmon_getResult(threadId,"UPMC0") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[1].value[threadId] =
                    perfmon_getResult(threadId,"UPMC1") / perfmon_getResult(threadId,"FIXC0");
                tableData.rows[2].value[threadId] =
                    perfmon_getResult(threadId,"UPMC1") / perfmon_getResult(threadId,"UPMC0");
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
            numRows = 1;
            INIT_BASIC;
            bstrListAdd(1,L1 DTLB miss rate);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                tableData.rows[0].value[threadId] =
                    perfmon_getResult(threadId,"PMC0") / perfmon_getResult(threadId,"FIXC0");
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
            fprintf (stderr, "perfmon_printDerivedMetricsNehalem: Unknown group! Exiting!\n" );
            exit (EXIT_FAILURE);
            break;
    }

    printResultTable(&tableData);
    bdestroy(label);
    bstrListDestroy(fc);

}
