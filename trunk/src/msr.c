/*
 * =======================================================================================
 *
 *      Filename:  msr.c
 *
 *      Description:  Implementation of msr module.
 *                   Provides API to read and write values to the model
 *                   specific registers on x86 processors using the msr
 *                   sys interface of the Linux 2.6 kernel. This module 
 *                   is based on the msr-util tools.
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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <accessClient.h>
#include <msr.h>
#include <registers.h>
#ifdef LIKWID_PROFILE_COUNTER_READ
#include <timer.h>
#endif
/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define MAX_LENGTH_MSR_DEV_NAME  20
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static int FD[MAX_NUM_THREADS] = { [0 ... MAX_NUM_THREADS-1] = -1 };
static int rdpmc_works = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static inline int __rdpmc(int counter, uint64_t* value)
{
    unsigned low, high;

    __asm__ volatile("rdpmc" : "=a" (low), "=d" (high) : "c" (counter));
    *value = ((low) | ((uint64_t )(high) << 32));
    return 0;
}

//Needed for rdpmc check
void segfault_sigaction(int signal, siginfo_t *si, void *arg)
{
    exit(1);
}

int test_rdpmc(int flag)
{
    int ret, waiting;
    int pid;
    int status = 0;
    uint64_t tmp;
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags   = SA_SIGINFO;
    
    
    pid = fork();
    
    if (pid < 0)
    {
        return -1;
    }
    if (!pid)
    {
        sigaction(SIGSEGV, &sa, NULL);
        if (flag == 0)
        {
            __rdpmc(0, &tmp);
        }
        exit(0);
    } else {
    
        waiting = waitpid(pid, &status, 0);
        if (waiting < 0 || status)
        {
            ret = 0;
        } else 
        {
            ret = 1;
        }
    }
    return ret;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */


int
msr_init(int initSocket_fd)
{
    int fd = 0;
    int i = 0;
    
    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        int fd;
        char* msr_file_name = (char*) malloc(MAX_LENGTH_MSR_DEV_NAME * sizeof(char));
        if (!msr_file_name)
        {    
            return -ENOMEM;
        }

        sprintf(msr_file_name,"/dev/msr0");
        fd = open(msr_file_name, O_RDWR);
        if (fd < 0)
        {
            sprintf(msr_file_name,"/dev/cpu/0/msr");
        }
        else
        {
            close(fd);
        }
        fd = open(msr_file_name, O_RDWR);   
        if (fd < 0)
        {
            ERROR_PRINT("Cannot access MSR device file %s: %s.\n"
                        "Please check if 'msr' module is loaded and device files have correct permissions\n"
                        "Alternatively you might want to look into (sys)daemonmode\n",msr_file_name , strerror(errno));
            free(msr_file_name);
            return -EPERM;
        }
        else
        {
            close(fd);
        }
        rdpmc_works = test_rdpmc(0);

        /* NOTICE: This assumes consecutive processor Ids! */
        for ( i=0; i < cpuid_topology.numHWThreads; i++ )
        {
            sprintf(msr_file_name,"/dev/msr%d",cpuid_topology.threadPool[i].apicId);
            fd = open(msr_file_name, O_RDWR); 
            if (fd < 0)
            {
                sprintf(msr_file_name,"/dev/cpu/%d/msr",cpuid_topology.threadPool[i].apicId);
            }
            else
            {
                close(fd);
            }
            FD[cpuid_topology.threadPool[i].apicId] = open(msr_file_name, O_RDWR);
            if ( FD[cpuid_topology.threadPool[i].threadId] < 0 )
            {
                ERROR_PRINT(Cannot access MSR device file %s in direct mode, msr_file_name);
                free(msr_file_name);
                return -EPERM;
            }
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Opened MSR for CPU %d FD %d,
                                        cpuid_topology.threadPool[i].apicId,
                                        FD[cpuid_topology.threadPool[i].apicId]);
        }
        free(msr_file_name);
    }
    else
    {
        fd = 1;
        socket_fd = initSocket_fd;
    }
    return 0;
}

void
msr_finalize(void)
{
    int i = 0;
    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        for (i=0; i < cpuid_topology.numHWThreads; i++ )
        {
            close(FD[i]);
        }
    }
    else
    {
        socket_fd = -1;
    }
}


int
msr_tread(const int tsocket_fd, const int cpu, uint32_t reg, uint64_t *data)
{
    int ret = 0;

    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        if (rdpmc_works && reg >= MSR_PMC0 && reg <=MSR_PMC7)
        {
            if (__rdpmc(reg - MSR_PMC0, data) )
            {
                //ERROR_PRINT(Cannot read MSR reg 0x%x with RDPMC instruction on CPU %d, reg, cpu);
                return -EIO;
            }
        }
        /*else if (rdpmc_works && reg >= MSR_PERF_FIXED_CTR0 && reg <= MSR_PERF_FIXED_CTR2)
        {
            if (__rdpmc(0x4000000ULL + (reg - MSR_PERF_FIXED_CTR0), data) )
            {
                ERROR_PRINT(Cannot read MSR reg 0x%x with RDPMC instruction on CPU %d,reg,cpu);
                return -EIO;
            }
        }*/
        else
        {
            if (FD[cpu] > 0)
            {
                ret = pread(FD[cpu], data, sizeof(*data), reg);
                if (ret  != sizeof(*data) )
                {
                    //ERROR_PRINT(Cannot read MSR reg 0x%x with RDMSR instruction on CPU %d, reg, cpu);
                    return -EIO;
                }
            }
            else
            {
                //ERROR_PRINT(MSR device for CPU %d not found, cpu);
                return -EBADFD;
            }
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (tsocket_fd != -1)
        {
            ret = accessClient_read(tsocket_fd, cpu, DAEMON_AD_MSR, reg, data);
            if (ret)
            {
                //ERROR_PRINT(Cannot read MSR reg 0x%x through accessDaemon on CPU %d, reg, cpu);
                return ret;
            }
        }
        else
        {
            //ERROR_PLAIN_PRINT(Bad socket to accessDaemon);
            return -EBADFD;
        }
    }
    return 0;
}


int 
msr_twrite(const int tsocket_fd, const int cpu, uint32_t reg, uint64_t data)
{
    int ret;
    if (accessClient_mode == ACCESSMODE_DIRECT) 
    {
        if (FD[cpu] > 0)
        {
            ret = pwrite(FD[cpu], &data, sizeof(data), reg);
            if (ret != sizeof(data))
            {
                //ERROR_PRINT(Cannot write MSR reg 0x%x with WRMSR instruction on CPU %d\n,
                //            reg, cpu);
                return -EIO;
            }
        }
        else
        {
            //ERROR_PRINT(MSR device for CPU %d not found, cpu);
            return -EBADFD;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (tsocket_fd != -1)
        {
            ret = accessClient_write(tsocket_fd, cpu, DAEMON_AD_MSR, reg, data);
            if (ret)
            {
                return ret;
            }
        }
        else
        {
            //ERROR_PLAIN_PRINT(Bad socket to accessDaemon);
            return -EBADFD;
        }
    }
    return 0;
}


int
msr_read( const int cpu, uint32_t reg, uint64_t *data)
{
    int ret;
    if (accessClient_mode == ACCESSMODE_DIRECT) 
    {
        if (rdpmc_works && reg >= MSR_PMC0 && reg <=MSR_PMC7)
        {
            if (__rdpmc(reg - MSR_PMC0, data) )
            {
                ERROR_PRINT(Cannot read MSR reg 0x%x with RDPMC instruction on CPU %d\n,
                        reg,cpu);
                return -EIO;
            }
        }
        else if (rdpmc_works && reg >= MSR_PERF_FIXED_CTR0 && reg <= MSR_PERF_FIXED_CTR2)
        {
            if (__rdpmc(0x4000000ULL + (reg - MSR_PERF_FIXED_CTR0), data) )
            {
                ERROR_PRINT(Cannot read MSR reg 0x%x with RDPMC instruction on CPU %d,reg,cpu);
                return -EIO;
            }
        }
        else
        {
            if ( pread(FD[cpu], data, sizeof(*data), reg) != sizeof(*data) )
            {
                ERROR_PRINT(Cannot read MSR reg 0x%x with RDMSR instruction on CPU %d,
                        reg, cpu);
                return -EIO;
            }
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (socket_fd != -1)
        {
            ret = accessClient_read(socket_fd, cpu, DAEMON_AD_MSR, reg, data);
            if (ret)
            {
                ERROR_PRINT(Cannot read MSR reg 0x%x through accessDaemon on CPU %d, reg, cpu);
                return ret;
            }
        }
        else
        {
            ERROR_PLAIN_PRINT(Bad socket to accessDaemon);
            return -EBADFD;
        }
    }
    return 0;
}


int
msr_write( const int cpu, uint32_t reg, uint64_t data)
{
    int ret;
    if (accessClient_mode == ACCESSMODE_DIRECT) 
    {
        ret = pwrite(FD[cpu], &data, sizeof(data), reg);
        if (ret != sizeof(data))
        {
            return ret;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (socket_fd != -1)
        {
            ret = accessClient_write(socket_fd, cpu, DAEMON_AD_MSR, reg, data);
            if (ret)
            {
                return ret;
            }
        }
        else
        {
            ERROR_PLAIN_PRINT(Bad socket to accessDaemon);
            return -EBADFD;
        }
    }
    return 0;
}


