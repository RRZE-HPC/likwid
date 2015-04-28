/*
 * =======================================================================================
 *
 *      Filename:  bench.c
 *
 *      Description:  Benchmarking framework for likwid-bench
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
/* #####   HEADER FILE INCLUDES   ######################################### */

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <string.h>
#include <sched.h>
//#include <types.h>
#include <unistd.h>

#include <allocator.h>
#include <threads.h>
#include <barrier.h>
#include <likwid.h>

/* #####   EXPORTED VARIABLES   ########################################### */


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

//#define BARRIER pthread_barrier_wait(&threads_barrier) 
#define BARRIER   barrier_synchronize(&barr)

#ifdef PERFMON
#define START_PERFMON likwid_markerStartRegion("bench");
#define STOP_PERFMON  likwid_markerStopRegion("bench");
#define LIKWID_THREAD_INIT  likwid_markerThreadInit();
#else
#define START_PERFMON
#define STOP_PERFMON
#define LIKWID_THREAD_INIT
#endif

#define EXECUTE(func)   \
    BARRIER; \
    if (data->globalThreadId == 0) \
    { \
        timer_start(&time); \
    } \
    START_PERFMON  \
    for (i=0; i<  myData->iter; i++) \
    {   \
        func; \
    } \
    BARRIER; \
    STOP_PERFMON  \
    if (data->globalThreadId == 0) \
    { \
        timer_stop(&time); \
        data->cycles = timer_printCycles(&time); \
    } \
    BARRIER




/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void* runTest(void* arg)
{
    int threadId;
    int offset;
    size_t size, allSize;
    size_t i;
    BarrierData barr;
    ThreadData* data;
    ThreadUserData* myData;
    TimerData time;
    FuncPrototype func;

    data = (ThreadData*) arg;
    myData = &(data->data);
    func = myData->test->kernel;
    threadId = data->threadId;
    barrier_registerThread(&barr, 0, data->globalThreadId);

    /* Prepare ptrs for thread */
    allSize = myData->size;
    size = myData->size / data->numberOfThreads;
    size -= (size%myData->test->stride);
    offset = data->threadId * size;
    myData->size = size;


    /* pin the thread */
    likwid_pinThread(myData->processors[threadId]);

    sleep(1);
    LIKWID_THREAD_INIT;
    BARRIER;
    printf("Group: %d Thread %d Global Thread %d running on core %d - Vector length %llu Offset %d\n",
            data->groupId,
            threadId,
            data->globalThreadId,
            affinity_threadGetProcessorId(),
            LLU_CAST allSize,
            offset);
    BARRIER;

    /* Up to 10 streams the following registers are used for Array ptr:
     * Size rdi
     * in Registers: rsi  rdx  rcx  r8  r9  
     * passed on stack, then: r10  r11  r12  r13  r14  r15
     * If more than 10 streams are used first 5 streams are in register, above 5 a macro must be used to
     * load them from stack 
     * */

    switch ( myData->test->streams ) {
        case STREAM_1:
            EXECUTE(func(size,myData->streams[0]));
            break;
        case STREAM_2:
            EXECUTE(func(size,myData->streams[0],myData->streams[1]));
            break;
        case STREAM_3:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2]));
            break;
        case STREAM_4:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3]));
            break;
        case STREAM_5:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4]));
            break;
        case STREAM_6:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5]));
            break;
        case STREAM_7:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6]));
            break;
        case STREAM_8:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7]));
            break;
        case STREAM_9:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8]));
            break;
        case STREAM_10:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9]));
            break;
        case STREAM_11:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10]));
            break;
        case STREAM_12:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11]));
            break;
        case STREAM_13:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12]));
            break;
        case STREAM_14:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13]));
            break;
        case STREAM_15:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14]));
            break;
        case STREAM_16:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15]));
            break;
        case STREAM_17:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16]));
            break;
        case STREAM_18:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17]));
            break;
        case STREAM_19:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18]));
            break;
        case STREAM_20:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19]));
            break;
        case STREAM_21:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20]));
            break;
        case STREAM_22:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21]));
            break;
        case STREAM_23:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22]));
            break;
        case STREAM_24:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23]));
            break;
        case STREAM_25:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24]));
            break;
        case STREAM_26:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25]));
            break;
        case STREAM_27:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26]));
            break;
        case STREAM_28:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27]));
            break;
        case STREAM_29:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28]));
            break;
        case STREAM_30:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29]));
            break;
        case STREAM_31:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30]));
            break;
        case STREAM_32:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31]));
            break;
        case STREAM_33:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32]));
            break;
        case STREAM_34:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33]));
            break;
        case STREAM_35:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34]));
            break;
        case STREAM_36:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35]));
            break;
        case STREAM_37:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35],
                        myData->streams[36]));
            break;
        case STREAM_38:
            EXECUTE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35],
                        myData->streams[36],myData->streams[37]));
            break;
        default:
            break;
    }
    pthread_exit(NULL);
}

#define MEASURE(func) \
    if (data->globalThreadId == 0) \
    { \
        timer_start(&time); \
        timer_stop(&time); \
        i = 0; \
        while (timer_print(&time) < data->data.min_runtime) \
        { \
            func; \
            timer_stop(&time); \
            iterations++; \
        } \
    } \
    BARRIER;


void* getIter(void* arg)
{
    int threadId;
    int offset;
    size_t size;
    size_t i;
    BarrierData barr;
    ThreadData* data;
    ThreadUserData* myData;
    TimerData time;
    FuncPrototype func;
    size_t iterations = 0;

    data = (ThreadData*) arg;
    myData = &(data->data);
    func = myData->test->kernel;
    threadId = data->threadId;
    barrier_registerThread(&barr, 0, data->globalThreadId);

    /* Prepare ptrs for thread */
    size = myData->size / data->numberOfThreads;
    size -= (size%myData->test->stride);
    offset = data->threadId * size;

    switch ( myData->test->type )
    {
        case SINGLE:
            {
                float* sptr;
                for (i=0; i <  myData->test->streams; i++)
                {
                    sptr = (float*) myData->streams[i];
                    sptr +=  offset;
              //      sptr +=  size;
                    myData->streams[i] = (float*) sptr;
                }
            }
            break;
        case DOUBLE:
            {
                double* dptr;
                for (i=0; i <  myData->test->streams; i++)
                {
                    dptr = (double*) myData->streams[i];
                    dptr +=  offset;
             //       dptr +=  size;
                    myData->streams[i] = (double*) dptr;
                }
            }
            break;
    }

    /* pint the thread */
    affinity_pinThread(myData->processors[threadId]);
    
    switch ( myData->test->streams ) {
        case STREAM_1:
            MEASURE(func(size,myData->streams[0]));
            break;
        case STREAM_2:
            MEASURE(func(size,myData->streams[0],myData->streams[1]));
            break;
        case STREAM_3:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2]));
            break;
        case STREAM_4:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3]));
            break;
        case STREAM_5:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4]));
            break;
        case STREAM_6:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5]));
            break;
        case STREAM_7:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6]));
            break;
        case STREAM_8:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7]));
            break;
        case STREAM_9:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8]));
            break;
        case STREAM_10:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9]));
            break;
        case STREAM_11:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10]));
            break;
        case STREAM_12:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11]));
            break;
        case STREAM_13:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12]));
            break;
        case STREAM_14:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13]));
            break;
        case STREAM_15:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14]));
            break;
        case STREAM_16:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15]));
            break;
        case STREAM_17:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16]));
            break;
        case STREAM_18:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17]));
            break;
        case STREAM_19:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18]));
            break;
        case STREAM_20:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19]));
            break;
        case STREAM_21:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20]));
            break;
        case STREAM_22:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21]));
            break;
        case STREAM_23:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22]));
            break;
        case STREAM_24:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23]));
            break;
        case STREAM_25:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24]));
            break;
        case STREAM_26:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25]));
            break;
        case STREAM_27:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26]));
            break;
        case STREAM_28:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27]));
            break;
        case STREAM_29:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28]));
            break;
        case STREAM_30:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29]));
            break;
        case STREAM_31:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30]));
            break;
        case STREAM_32:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31]));
            break;
        case STREAM_33:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32]));
            break;
        case STREAM_34:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33]));
            break;
        case STREAM_35:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34]));
            break;
        case STREAM_36:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35]));
            break;
        case STREAM_37:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35],
                        myData->streams[36]));
            break;
        case STREAM_38:
            MEASURE(func(size,myData->streams[0],myData->streams[1],myData->streams[2],myData->streams[3],
                        myData->streams[4],myData->streams[5],myData->streams[6],myData->streams[7],
                        myData->streams[8],myData->streams[9],myData->streams[10],myData->streams[11],
                        myData->streams[12],myData->streams[13],myData->streams[14],myData->streams[15],
                        myData->streams[16],myData->streams[17],myData->streams[18],myData->streams[19],
                        myData->streams[20],myData->streams[21],myData->streams[22],myData->streams[23],
                        myData->streams[24],myData->streams[25],myData->streams[26],myData->streams[27],
                        myData->streams[28],myData->streams[29],myData->streams[30],myData->streams[31],
                        myData->streams[32],myData->streams[33],myData->streams[34],myData->streams[35],
                        myData->streams[36],myData->streams[37]));
            break;
        default:
            break;
    }

    data->data.iter = iterations;
    pthread_exit(NULL);
}
