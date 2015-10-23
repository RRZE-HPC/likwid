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
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
/* #####   EXPORTED VARIABLES   ########################################### */
/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static uint64_t baseline = 0ULL;
static uint64_t cpuClock = 0ULL;
static uint64_t sleepbase = 0ULL;
static uint8_t fixedFreq = 0;

void (*TSTART)(TscCounter*) = NULL;
void (*TSTOP)(TscCounter*) = NULL;

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#if defined(__x86_64) || defined(__i386__)
#define CPUID                              \
    __asm__ volatile ("cpuid"                             \
            : "=a" (eax),     \
            "=b" (ebx),     \
            "=c" (ecx),     \
            "=d" (edx)      \
            : "0" (eax), "2" (ecx))
#endif
/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static int os_timer(TscCounter* time)
{
    int ret;
    struct timeval cur;
    ret = gettimeofday(&cur, NULL);
    if (!ret)
    {
        time->int64 = ((uint64_t)cur.tv_sec) * 1E6;
        time->int64 += cur.tv_usec;
    }
    return ret;
}

static void os_timer_start(TscCounter* time)
{
    os_timer(time);
}

static void os_timer_stop(TscCounter* time)
{
    os_timer(time);
}

#if defined(__x86_64) || defined(__i386__)
static void fRDTSC(TscCounter* cpu_c)
{
    __asm__ volatile("xor %%eax,%%eax\n\t"           \
    "cpuid\n\t"           \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%ebx","%ecx","%edx");
}

static void fRDTSC_CR(TscCounter* cpu_c)
{
    __asm__ volatile(   \
    "rdtsc\n\t"           \
    "movl %%eax, %0\n\t"  \
    "movl %%edx, %1\n\t"  \
    : "=r" ((cpu_c)->int32.lo), "=r" ((cpu_c)->int32.hi) \
    : : "%eax","%ebx","%ecx","%edx");
}

static void fRDTSCP(TscCounter* cpu_c)
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


static uint64_t
getCpuSpeed(void)
{
#if defined(__x86_64) || defined(__i386__)
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
        timer_start(&data);
        timer_stop(&data);
        result = MIN(result,timer_printCycles(&data));
    }

    baseline = result;
    result = 0xFFFFFFFFFFFFFFFFULL;

    for (i=0; i< 2; i++)
    {
        TSTART(&start);
        gettimeofday( &tv1, &tzp);
        nanosleep( &delay, NULL);
        TSTOP(&stop);
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

    return ((uint64_t)   atoi(buff)) * 1000;
#endif
#if defined(__arm__)
    uint64_t result = 0xFFFFFFFFFFFFFFFFULL;
    TimerData data;
    FILE *fpipe;
    char *command="cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
    char *command2="cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor";
    char buff[256];
    buff[0] = '\0';
    char* buffptr = NULL;
    if ( !(fpipe = (FILE*)popen(command2, "r")))
    {
        perror("Problems with pipe, cannot read /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
        exit(1);
    }
    buffptr = fgets(buff, 256, fpipe);
    fclose(fpipe);
    if ((strncmp(buff, "userspace", 9) == 0) || (strncmp(buff, "performance", 11) == 0))
    {
        fixedFreq = 1;
    }
    buff[0] = '\0';
    buffptr = NULL;

    if ( !(fpipe = (FILE*)popen(command, "r")))
    {
        perror("Problems with pipe, cannot read /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq");
        exit(1);
    }
    buffptr = fgets(buff, 256, fpipe);
    fclose(fpipe);
    for (int i=0;i<10;i++)
    {
        TSTART(&data.start);
        TSTOP(&data.stop);
        result = MIN(result,timer_printCycles(&data));
    }
    baseline = result;
    return ((uint64_t) atoi(buff)) * 1000;
#endif
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void init_sleep()
{
    int status;
    TimerData timer;
    struct timespec req = {0,1};
    struct timespec rem = {0,0};
    for (int i=0; i<10; ++i)
    {
        timer_start(&timer);
        status = clock_nanosleep(CLOCK_REALTIME,0,&req, &rem);
        timer_stop(&timer);
        if (timer_print(&timer)*1E6 > sleepbase)
        {
            sleepbase = timer_print(&timer)*1E6 + 2;
        }
    }
    sleepbase = 0;
}


void timer_init( void )
{
    uint32_t eax,ebx,ecx,edx;
    if ((!TSTART) && (!TSTOP))
    {
#if defined(__x86_64) || defined(__i386__)
        TSTART = fRDTSC;
        eax = 0x80000001;
        CPUID;
        if (edx & (1<<27))
        {
            TSTOP = fRDTSCP;
        }
        else
        {
            TSTOP = fRDTSC_CR;
        }
#endif
#if defined(__arm__)
        TSTART = os_timer_start;
        TSTOP = os_timer_stop;
#endif
    }
    /*if (cpuClock == 0ULL)
    {
        cpuClock = getCpuSpeed();
    }*/
}

uint64_t timer_printCycles( TimerData* time )
{
    /* clamp to zero if something goes wrong */
    uint64_t cycles = 0x0ULL;
    if ((time->stop.int64-baseline) >= time->start.int64)
    {
        cycles = (time->stop.int64 - time->start.int64 - baseline);
    }
#if defined(__arm__)
    if (fixedFreq == 1)
    {
        cycles *= 1E-6 * cpuClock;
    }
    else
    {
        cycles = 0x0ULL;
    }
#endif
    return cycles;
}

/* Return time duration in seconds */
double timer_print( TimerData* time )
{
    uint64_t cycles = 0x0ULL;
    if ((time->stop.int64-baseline) >= time->start.int64)
    {
        cycles = (time->stop.int64 - time->start.int64 - baseline);
    }
#if defined(__arm__)
    return ((double) cycles) * 1E-6;
#else
    return  ((double) cycles / (double) cpuClock);
#endif
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
#if defined(__x86_64) || defined(__i386__)
    if (TSTART)
        TSTART(&(time->start));
#endif
#if defined(__arm__)
    if (TSTART)
        TSTART(&(time->start));
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
#if defined(__x86_64) || defined(__i386__)
    if (TSTOP)
        TSTOP(&(time->stop));
#endif
#if defined(__arm__)
    if (TSTOP)
        TSTOP(&(time->stop));
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



int timer_sleep(unsigned long usec)
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

