/*
 * ===========================================================================
 *
 *      Filename:  perfmon_atom.h
 *
 *      Description:  Atom specific subroutines
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
#include <perfmon_atom_events.h>

#define NUM_GROUPS_ATOM 7

static int perfmon_numGroupsAtom = NUM_GROUPS_ATOM;
static int perfmon_numArchEventsAtom = NUM_ARCH_EVENTS_ATOM;


static PerfmonGroupMap atom_group_map[NUM_GROUPS_ATOM] = {
    {"FLOPS_DP",FLOPS_DP,"Double Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,SIMD_COMP_INST_RETIRED_PACKED_DOUBLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_DOUBLE:PMC1"},
    {"FLOPS_SP",FLOPS_SP,"Single Precision MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,SIMD_COMP_INST_RETIRED_PACKED_SINGLE:PMC0,SIMD_COMP_INST_RETIRED_SCALAR_SINGLE:PMC1"},
    {"FLOPS_X87",FLOPS_X87,"X87 MFlops/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,X87_COMP_OPS_EXE_ANY_AR:PMC0"},
    {"MEM",MEM,"Main memory bandwidth in MBytes/s","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,BUS_TRANS_MEM_THIS_CORE_THIS_A:PMC0"},
    {"DATA",DATA,"Load to store ratio","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,L1D_CACHE_LD:PMC0,L1D_CACHE_ST:PMC1"},
    {"BRANCH",BRANCH,"Branch prediction miss rate","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,BR_INST_RETIRED_ANY:PMC0,BR_INST_RETIRED_MISPRED:PMC1"},
    {"TLB",TLB,"Translation lookaside buffer miss rate","INSTR_RETIRED_ANY:FIXC0,CPU_CLK_UNHALTED_CORE:FIXC1,DATA_TLB_MISSES_DTLB_MISS:PMC0"}
};


void
perfmon_printDerivedMetricsAtom(PerfmonGroup group)
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
                    1.0E-06*(perfmon_getResult(threadId,"PMC0")*4+
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
            bstrListAdd(1,Runtime [s]);
            bstrListAdd(2,CPI);
            bstrListAdd(3,L1 DTLB miss rate);
            initResultTable(&tableData, fc, numRows, numColumns);

            for(threadId=0; threadId < perfmon_numThreads; threadId++)
            {
                time = perfmon_getResult(threadId,"FIXC1") * inverseClock;
                cpi  =  perfmon_getResult(threadId,"FIXC1")/
                    perfmon_getResult(threadId,"FIXC0");
                tableData.rows[0].value[threadId] = time;
                tableData.rows[1].value[threadId] = cpi;
                tableData.rows[2].value[threadId] =
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



