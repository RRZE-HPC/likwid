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
 *      Copyright (C) 2012 Jan Treibig 
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
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <cpuid.h>
#include <accessClient.h>
#include <msr.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define MAX_LENGTH_MSR_DEV_NAME  20
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */
static int FD[MAX_NUM_THREADS];
static int socket_fd = -1;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */


void
msr_init(int initSocket_fd)
{
    if (accessClient_mode == DAEMON_AM_DIRECT) 
    {

        int  fd;
#ifdef __MIC
        char* msr_file_name = "/dev/msr0";
#else
        char* msr_file_name = "/dev/cpu/0/msr";
#endif

        fd = open(msr_file_name, O_RDWR);

        if (fd < 0)
        {
            fprintf(stderr, "ERROR\n");
            fprintf(stderr, "rdmsr: failed to open '%s': %s!\n",msr_file_name , strerror(errno));
            fprintf(stderr, "       Please check if the msr module is loaded and the device file has correct permissions.\n");
            fprintf(stderr, "       Alternatively you might want to look into (sys)daemonmode.\n\n");
            exit(127);
        }

        close(fd);

        /* NOTICE: This assumes consecutive processor Ids! */
        for ( uint32_t i=0; i < cpuid_topology.numHWThreads; i++ )
        {
            char* msr_file_name = (char*) malloc(MAX_LENGTH_MSR_DEV_NAME * sizeof(char));
#ifdef __MIC
            sprintf(msr_file_name,"/dev/msr%d",i);
#else
            sprintf(msr_file_name,"/dev/cpu/%d/msr",i);
#endif

            FD[i] = open(msr_file_name, O_RDWR);

            if ( FD[i] < 0 )
            {
                ERROR;
            }
        }
    }
    else
    {
        socket_fd = initSocket_fd;
    }
}

void
msr_finalize(void)
{
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        for ( uint32_t i=0; i < cpuid_topology.numHWThreads; i++ )
        {
            close(FD[i]);
        }
    }
    else
    {
        socket_fd = -1;
    }
}


uint64_t 
msr_tread(const int tsocket_fd, const int cpu, uint32_t reg)
{
    if (accessClient_mode == DAEMON_AM_DIRECT) 
    {
        uint64_t data;

        if ( pread(FD[cpu], &data, sizeof data, reg) != sizeof data ) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }

        return data;
    }
    else
    { /* daemon or sysdaemon-mode */
        return accessClient_read(tsocket_fd, cpu, DAEMON_AD_MSR, reg);
    }
}


void 
msr_twrite(const int tsocket_fd, const int cpu, uint32_t reg, uint64_t data)
{
    if (accessClient_mode == DAEMON_AM_DIRECT) 
    {
        if (pwrite(FD[cpu], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        accessClient_write(tsocket_fd, cpu, DAEMON_AD_MSR, reg, data);
    }
}


uint64_t 
msr_read( const int cpu, uint32_t reg)
{
    if (accessClient_mode == DAEMON_AM_DIRECT) 
    {
        uint64_t data;

        if ( pread(FD[cpu], &data, sizeof data, reg) != sizeof data ) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }

        return data;
    }
    else
    { /* daemon or sysdaemon-mode */
        return accessClient_read(socket_fd, cpu, DAEMON_AD_MSR, reg);
    }
}


void 
msr_write( const int cpu, uint32_t reg, uint64_t data)
{
    if (accessClient_mode == DAEMON_AM_DIRECT) 
    {
        if (pwrite(FD[cpu], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        accessClient_write(socket_fd, cpu, DAEMON_AD_MSR, reg, data);
    }
}


