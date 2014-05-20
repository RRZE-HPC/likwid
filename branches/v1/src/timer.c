/*
 * ===========================================================================
 *
 *      Filename:  timer.c
 *
 *      Description:  Implementation of timer module
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


/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <types.h>
#include <timer.h>


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static uint64_t cpuClock = 0;
static uint64_t cyclesForCpuid = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static uint64_t
getCpuSpeed(void)
{
    int i;
    TscCounter start;
    TscCounter stop;
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    struct timeval tv1;
    struct timeval tv2;
    struct timezone tzp;
    struct timespec delay = { 0, 800000000 }; /* calibration time: 800 ms */

    for (i=0; i< 2; i++)
    {
        RDTSC(start);
        gettimeofday( &tv1, &tzp);
        nanosleep( &delay, NULL);
        RDTSC(stop);
        gettimeofday( &tv2, &tzp);

        result = MIN(result,(stop.int64 - start.int64 - cyclesForCpuid));
    }

    return (result) * 1000000 / 
        (((uint64_t)tv2.tv_sec * 1000000 + tv2.tv_usec) -
         ((uint64_t)tv1.tv_sec * 1000000 + tv1.tv_usec));
}


static uint64_t
getCpuidCycles(void)
{
    int i;
    TscCounter start;
    TscCounter stop;
    uint64_t result = 1000000000ULL;

    for (i=0; i< 5; i++)
    {
        RDTSC(start);
        RDTSC(stop);

        result = MIN(result,(stop.int64 - start.int64));
    }

    return result;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
timer_init(void )
{
    getCpuidCycles();
    getCpuSpeed();

    cyclesForCpuid = getCpuidCycles();
    cpuClock = getCpuSpeed();
}

void
timer_startCycles(CyclesData* time) 
{
    TscCounter start;
    TscCounter stop;
    getCpuidCycles();

    RDTSC(start);
    RDTSC(stop);

    time->base = cyclesForCpuid + (stop.int64 - start.int64 - cyclesForCpuid);
    RDTSC(time->start);
}

void
timer_stopCycles(CyclesData* time) 
{
    RDTSC(time->stop);
}

void
timer_start(TimerData* time)
{
    gettimeofday(&(time->before),NULL);

#ifdef DEBUG
    printf("Timer Start - Seconds: %ld \t uSeconds: %ld \n",
            before.tv_sec, before.tv_usec);
#endif
}

void
timer_stop(TimerData* time)
{
    gettimeofday(&(time->after),NULL);

#ifdef DEBUG
    printf("Timer Start - Seconds: %ld \t uSeconds: %ld \n",
            after.tv_sec, after.tv_usec);
#endif
}

float
timer_print(TimerData* time)
{
    long int sec;
    float timeDuration;

    sec = time->after.tv_sec - time->before.tv_sec;

    timeDuration = (((sec*1000000)+time->after.tv_usec)- time->before.tv_usec);

#ifdef VERBOSE
    printf("*******************************************\n");
    printf("TIME [ms]\t:\t %f \n", timeDuration );
    printf("*******************************************\n\n");
#endif

    return (timeDuration * 0.000001F);
}

uint64_t
timer_printCycles(CyclesData* time)
{
    return (time->stop.int64 - time->start.int64 - time->base);
}

float
timer_printCyclesTime(CyclesData* time)
{
    uint64_t cycles;
    float timeDuration;

    cycles = (time->stop.int64 - time->start.int64 - time->base);
    timeDuration   =  (float) cycles / (float) cpuClock;

#ifdef VERBOSE
    printf("*******************************************\n");
    printf("TIME [ms]\t:\t %f \n", timeDuration );
    printf("*******************************************\n\n");
#endif

    return timeDuration;
}

uint64_t
timer_getCpuClock(void)
{
    return cpuClock;
}

uint64_t
timer_getCpuidCycles(void)
{
    return cyclesForCpuid;
}


