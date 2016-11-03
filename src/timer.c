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
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#include <types.h>
#include <error.h>
#include <likwid.h>
#include <cpuid.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static uint64_t baseline = 0ULL;
static uint64_t cpuClock = 0ULL;
static uint64_t cyclesClock = 0ULL;
static uint64_t sleepbase = 0ULL;
static int timer_initialized = 0;

void (*TSTART)(TscCounter*) = NULL;
void (*TSTOP)(TscCounter*) = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
#if defined(__x86_64)
static void
fRDTSC(TscCounter* cpu_c)
{
    __asm__ volatile("xor %%eax,%%eax\n\t"           \
    "cpuid\n\t"           \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%ebx","%ecx","%edx");
}

static void
fRDTSC_CR(TscCounter* cpu_c)
{
    __asm__ volatile(   \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%ebx","%ecx","%edx");
}
#ifndef __MIC__
static void
fRDTSCP(TscCounter* cpu_c)
{
    __asm__ volatile(     \
    "rdtscp\n\t"          \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    "cpuid\n\t"           \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%ebx","%ecx","%edx");
}
#endif
#endif

#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__)
static void
fRDTSC(TscCounter* cpu_c)
{
    uint64_t tmp;
    __asm__ volatile( \
    "xchgl %%ebx, %2\n\t"  \
    "xor %%eax,%%eax\n\t" \
    "cpuid\n\t"           \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    "xchgl %2, %%ebx\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi), "=m" (tmp) \
    : : "%eax","%ecx","%edx");
}

static void
fRDTSC_CR(TscCounter* cpu_c)
{
    __asm__ volatile(     \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%edx");
}
#ifndef __MIC__
static void
fRDTSCP(TscCounter* cpu_c)
{
    uint64_t tmp;
    __asm__ volatile(     \
    "rdtscp\n\t"          \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    "xchgl %%ebx, %2\n\t"  \
    "cpuid\n\t"           \
    "xchgl %2, %%ebx\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi), "=m" (tmp) \
    : : "%eax","%ecx","%edx");
}
#endif
#endif

#if defined(_ARCH_PPC)
static void
TIMER(TscCounter* cpu_c)
{
    uint32_t tbl, tbu0, tbu1;

    do {
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
        __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
        __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
    } while (tbu0 != tbu1);

    (cpu_c)->int64 = (((uint64_t)tbu0) << 32) | tbl;
}
#endif


static void
_timer_start( TimerData* time )
{
    if (TSTART)
        TSTART(&(time->start));
}

static void
_timer_stop( TimerData* time )
{
    if (TSTOP)
        TSTOP(&(time->stop));
}

static uint64_t
_timer_printCycles( const TimerData* time )
{
    /* clamp to zero if something goes wrong */
    if (((time->stop.int64-baseline) < time->start.int64) ||
        (time->start.int64 == time->stop.int64))
    {
        return 0ULL;
    }
    else
    {
        return (time->stop.int64 - time->start.int64 - baseline);
    }
}

/* Return time duration in seconds */
static double
_timer_print( const TimerData* time )
{
    uint64_t cycles;
    /* clamp to zero if something goes wrong */
    if (((time->stop.int64-baseline) < time->start.int64) ||
        (time->start.int64 == time->stop.int64))
    {
        cycles = 0ULL;
    }
    else
    {
        cycles = time->stop.int64 - time->start.int64 - baseline;
    }
    return  ((double) cycles / (double) cyclesClock);
}

static void
getCpuSpeed(void)
{
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
    int i;
    TimerData data;
    TscCounter start;
    TscCounter stop;
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    struct timeval tv1;
    struct timeval tv2;
    struct timezone tzp;
    struct timespec delay = { 0, 500000000 }; /* calibration time: 500 ms */

    for (i=0; i< 10; i++)
    {
        _timer_start(&data);
        _timer_stop(&data);
        result = MIN(result,_timer_printCycles(&data));
    }

    baseline = result;
    result = 0xFFFFFFFFFFFFFFFFULL;
    data.stop.int64 = 0;
    data.start.int64 = 0;

    for (i=0; i< 2; i++)
    {
        _timer_start(&data);
        gettimeofday( &tv1, &tzp);
        nanosleep( &delay, NULL);
        _timer_stop(&data);
        gettimeofday( &tv2, &tzp);

        result = MIN(result,(data.stop.int64 - data.start.int64));
    }

    cpuClock = (result) * 1000000 /
        (((uint64_t)tv2.tv_sec * 1000000 + tv2.tv_usec) -
         ((uint64_t)tv1.tv_sec * 1000000 + tv1.tv_usec));
    cyclesClock = cpuClock;
#endif
#ifdef _ARCH_PPC
    FILE *fpipe = NULL;
    char *ptr = NULL;
    char *command="grep timebase /proc/cpuinfo | awk '{ print $3 }'";
    char *command2="grep clock /proc/cpuinfo | head -n 1 | awk '{ print $3 }'";
    char buff[256];

    if ( !(fpipe = (FILE*)popen(command,"r")) )
    {  // If fpipe is NULL
        perror("Problems with pipe");
        exit(1);
    }

    ptr = fgets(buff, 256, fpipe);

    cyclesClock = (uint64_t)   atoi(buff);
    if ( !(fpipe = (FILE*)popen(command2,"r")) )
    {  // If fpipe is NULL
        perror("Problems with pipe");
        exit(1);
    }

    ptr = fgets(buff, 256, fpipe);

    cpuClock = (uint64_t)   atoi(buff);
    cpuClock *= 1E6;
    pclose(fpipe);
#endif
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
init_sleep()
{
    int status;
    TimerData timer;
    struct timespec req = {0,1};
    struct timespec rem = {0,0};
    for (int i=0; i<10; ++i)
    {
        _timer_start(&timer);
        status = clock_nanosleep(CLOCK_REALTIME,0,&req, &rem);
        _timer_stop(&timer);
        if (_timer_print(&timer)*1E6 > sleepbase)
        {
            sleepbase = _timer_print(&timer)*1E6 + 2;
        }
    }
}

void
timer_init( void )
{
    uint32_t eax = 0x0,ebx = 0x0,ecx = 0x0,edx = 0x0;
    if (timer_initialized == 1)
    {
        return;
    }
    if ((!TSTART) && (!TSTOP))
    {
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) || defined(__x86_64)
        TSTART = fRDTSC;
        eax = 0x80000001;
        CPUID (eax, ebx, ecx, edx);
#ifndef __MIC__
        if (edx & (1<<27))
        {
            TSTOP = fRDTSCP;
        }
        else
        {
            TSTOP = fRDTSC_CR;
        }
#else
        TSTOP = fRDTSC_CR;
#endif
#endif
#ifdef _ARCH_PPC
        TSTART = TIMER;
        TSTOP = TIMER;
#endif
    }
    if (cpuClock == 0ULL)
    {
        getCpuSpeed();
    }
    timer_initialized = 1;
}

uint64_t
timer_printCycles( const TimerData* time )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return 0ULL;
    }
    return _timer_printCycles(time);
}

/* Return time duration in seconds */
double
timer_print( const TimerData* time )
{
    uint64_t cycles;
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return 0ULL;
    }
    return _timer_print(time);
}

uint64_t
timer_getCpuClock( void )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return 0ULL;
    }
    return cpuClock;
}

uint64_t
timer_getCpuClockCurrent( int cpu_id )
{
    int err;
    uint64_t clock = 0x0ULL;
    FILE *fpipe;
    char cmd[256];
    char buff[256];
    char* eptr, *rptr;

    sprintf(buff, "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu_id);
    if (access(buff, R_OK))
    {
        ERROR_PRINT(File %s not readable, buff);
        return clock;
    }
    sprintf(cmd, "cat %s", buff);
    if ( !(fpipe = (FILE*)popen(cmd,"r")) )
    {  // If fpipe is NULL
        ERROR_PRINT(Problems reading cpu frequency of CPU %d, cpu_id);
        return clock;
    }

    rptr = fgets(buff, 256, fpipe);
    if (rptr != NULL)
    {
        clock = strtoull(buff, &eptr, 10);
    }
    return clock *1E3;
}

uint64_t
timer_getCycleClock( void )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return 0ULL;
    }
    return cyclesClock;
}

uint64_t
timer_getBaseline( void )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return 0ULL;
    }
    return baseline;
}

void
timer_start( TimerData* time )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return;
    }
    _timer_start(time);
}

void
timer_stop( TimerData* time )
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return;
    }
    _timer_stop(time);
}

int
timer_sleep(unsigned long usec)
{
    int status = -1;
    struct timespec req;
    struct timespec rem = {0,0};
    if (sleepbase == 0x0ULL)
    {
        init_sleep();
    }
    if (usec >= 1000000)
    {
        status = sleep(usec / 1000000);
    }
    else
    {
        req.tv_sec = 0;
        req.tv_nsec = (usec-sleepbase)*1.E3;
        status = clock_nanosleep(CLOCK_REALTIME,0,&req, &rem);
        if ((status == -1) && (errno == EINTR))
        {
            status = (rem.tv_sec * 1E6) + (rem.tv_nsec * 1E-3);
        }
    }
    return status;
}

void
timer_finalize(void)
{
    if (timer_initialized != 1)
    {
        ERROR_PLAIN_PRINT(Timer module not properly initialized);
        return;
    }
    baseline = 0ULL;
    cpuClock = 0ULL;
    TSTART = NULL;
    TSTOP = NULL;
    timer_initialized = 0;
}

void
timer_reset( TimerData* time )
{
    time->start.int64 = 0;
    time->stop.int64 = 0;
}

