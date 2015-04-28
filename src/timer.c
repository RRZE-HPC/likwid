/*
 * =======================================================================================
 *
 *      Filename:  timer.c
 *
 *      Description:  Implementation of timer module
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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include <types.h>
#include <likwid.h>
//#include <timer_types.h>

static uint64_t baseline = 0ULL;
static uint64_t cpuClock = 0ULL;


static uint64_t
getCpuSpeed(void)
{
#ifdef __x86_64
    int i;
    TimerData data;
    TscCounter start;
    TscCounter stop;
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    struct timeval tv1;
    struct timeval tv2;
    struct timezone tzp;
    struct timespec delay = { 0, 800000000 }; /* calibration time: 800 ms */

    for (i=0; i< 10; i++)
    {
        timer_start(&data);
        timer_stop(&data);
        result = MIN(result,timer_printCycles(&data));
    }

    baseline = result;
    result = 0xFFFFFFFFFFFFFFFFULL;

    for (i=0; i< 2; i++)
    {
        RDTSC(start);
        gettimeofday( &tv1, &tzp);
        nanosleep( &delay, NULL);
        RDTSC_STOP(stop);
        gettimeofday( &tv2, &tzp);

        result = MIN(result,(stop.int64 - start.int64));
    }

    return (result) * 1000000 /
        (((uint64_t)tv2.tv_sec * 1000000 + tv2.tv_usec) -
         ((uint64_t)tv1.tv_sec * 1000000 + tv1.tv_usec));
#endif
#ifdef _ARCH_PPC
    FILE *fpipe;
    char *command="grep timebase /proc/cpuinfo | awk '{ print $3 }'";
    char buff[256];

    if ( !(fpipe = (FILE*)popen(command,"r")) )
    {  // If fpipe is NULL
        perror("Problems with pipe");
        exit(1);
    }

    fgets(buff, 256, fpipe);

    return (uint64_t)   atoi(buff);
#endif
}


void timer_init( void )
{
    cpuClock = getCpuSpeed();
}

uint64_t timer_printCycles( TimerData* time )
{
    /* clamp to zero if something goes wrong */
    if ((time->stop.int64-baseline) < time->start.int64)
    {
        return 0ULL;
    }
    else
    {
        return (time->stop.int64 - time->start.int64 - baseline);
    }
}

/* Return time duration in seconds */
double timer_print( TimerData* time )
{
    uint64_t cycles;

    /* clamp to zero if something goes wrong */
    if ((time->stop.int64-baseline) < time->start.int64)
    {
        cycles = 0ULL;
    }
    else
    {
        cycles = time->stop.int64 - time->start.int64 - baseline;
    }

    return  ((double) cycles / (double) cpuClock);
}

uint64_t timer_getCpuClock( void )
{
    return cpuClock;
}

uint64_t timer_getBaseline( void )
{
    return baseline;
}

void timer_start( TimerData* time )
{
#ifdef __x86_64
    RDTSC(time->start);
#endif
#ifdef _ARCH_PPC
    uint32_t tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    time->start.int64 = (((uint64_t)tbu0) << 32) | tbl;
#endif
}


void timer_stop( TimerData* time )
{
#ifdef __x86_64
    RDTSC_STOP(time->stop)
#endif
#ifdef _ARCH_PPC
    uint32_t tbl, tbu0, tbu1;
    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    time->stop.int64 = (((uint64_t)tbu0) << 32) | tbl;
#endif
}

