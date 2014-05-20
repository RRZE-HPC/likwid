/*
 * ===========================================================================
 *
 *      Filename:  msr.c
 *
 *      Description:  Implementation of msr module.
 *                   Provides API to read and write values to the model
 *                   specific registers on x86 processors using the msr
 *                   sys interface of the Linux 2.6 kernel. This module 
 *                   is based on the msr-util tools.
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

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include <msr.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
void
msr_check()
{
    int  fd;
    char* msr_file_name = "/dev/cpu/0/msr";

    fd = open(msr_file_name, O_RDWR);

    if (fd < 0)
    {
        fprintf(stderr, "ERROR\n");
        fprintf(stderr, "rdmsr: failed to open '%s': %s!\n",msr_file_name , strerror(errno));
        fprintf(stderr, "       Please check if the msr module is loaded and the device file has correct permissions.\n\n");
        exit(127);
    }

    close(fd);
}


uint64_t 
msr_read(const int cpu, uint32_t reg)
{
    int  fd;
    uint64_t data;
    char msr_file_name[64];

    sprintf(msr_file_name, "/dev/cpu/%d/msr", cpu);
    fd = open(msr_file_name, O_RDONLY);
    if (fd < 0)
    {
        if (errno == ENXIO) 
        {
            fprintf(stderr, "rdmsr: No CPU %d\n", cpu);
            exit(2);
        }
        else if (errno == EIO) 
        {
            fprintf(stderr, "rdmsr: CPU %d doesn't support MSRs\n", cpu);
            exit(3);
        }
        else
        {
            perror("rdmsr: open");
            exit(127);
        }
    }

    if (pread(fd, &data, sizeof data, reg) != sizeof data) 
    {
        if (errno == EIO)
        {
            fprintf(stderr, "rdmsr: CPU %d cannot read MSR %x\n", cpu, reg);
            exit(4);
        }
        else
        {
            perror("rdmsr: pread");
            exit(127);
        }
    }

    close(fd);
    return data;
}

void 
msr_write(const int cpu, uint32_t reg, uint64_t data)
{
    int  fd;
    char msr_file_name[64];

    sprintf(msr_file_name, "/dev/cpu/%d/msr", cpu);
    fd = open(msr_file_name, O_WRONLY);
    if (fd < 0) 
    {
        if (errno == ENXIO)
        {
            fprintf(stderr, "wrmsr: No CPU %d\n", cpu);
            exit(2);
        }
        else if (errno == EIO)
        {
            fprintf(stderr, "wrmsr: CPU %d doesn't support MSRs\n", cpu);
            exit(3);
        } 
        else
        {
            perror("wrmsr: open");
            exit(127);
        }
    }

    if (pwrite(fd, &data, sizeof data, reg) != sizeof data) 
    {
        if (errno == EIO) 
        {
            fprintf(stderr, "wrmsr: CPU %d cannot set MSR %X to %llX\n",
                    cpu, reg, LLU_CAST data);
            exit(4);
        }
        else
        {
            perror("wrmsr: pwrite");
            exit(127);
        }
    }

    close(fd);
}


