/*
 * =======================================================================================
 *
 *      Filename:  access_x86_clientmem.c
 *
 *      Description:  Implementation of pci module.
 *                   Provides API to read and write values to the hardware
 *                   performance monitoring registers in PCI Cfg space
 *                   for Intel Sandy Bridge Processors.
 *
 *      Version:   4.3.1
 *      Released:  04.01.2018
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com,
 *                Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2018 RRZE, University Erlangen-Nuremberg
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
#include <sys/mman.h>


#include <types.h>
#include <bstrlib.h>
#include <error.h>
#include <topology.h>

#include <access_x86_clientmem.h>


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */


#define PCM_CLIENT_IMC_BAR_OFFSET       (0x0048)
#define PCM_CLIENT_IMC_DRAM_IO_REQUESTS  (0x5048)
#define PCM_CLIENT_IMC_DRAM_DATA_READS  (0x5050)
#define PCM_CLIENT_IMC_DRAM_DATA_WRITES (0x5054)
#define PCM_CLIENT_IMC_PP0_TEMP (0x597C)
#define PCM_CLIENT_IMC_PP1_TEMP (0x5980)
#define PCM_CLIENT_IMC_MMAP_SIZE (0x6000)


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int clientmem_handle = -1;
static char *clientmem_addr = NULL;
static int access_clientmem_initialized = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

static int
clientmem_getStartAddr(uint64_t* startAddr)
{
    uint64_t imcbar = 0;

    int pcihandle = open("/proc/bus/pci/00/00.0", O_RDONLY);
    if (pcihandle < 0)
    {
        ERROR_PLAIN_PRINT(Cannot get start address: failed to open /proc/bus/pci/00/00.0);
        return -1;
    }

    ssize_t ret = pread(pcihandle, &imcbar, sizeof(uint64_t), PCM_CLIENT_IMC_BAR_OFFSET);
    if (ret < 0)
    {
        ERROR_PLAIN_PRINT(Cannot get start address: mmap failed);
        close(pcihandle);
        return -1;
    }
    if (!imcbar)
    {
        ERROR_PLAIN_PRINT(Cannot get start address: imcbar is zero);
        close(pcihandle);
        return -1;
    }

    close(pcihandle);
    if (startAddr)
        *startAddr = imcbar & (~(4096 - 1));
    return 1;
}


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_x86_clientmem_init(const int socket)
{
    uint64_t startAddr = 0;

    if (!access_clientmem_initialized)
    {
        int ret = clientmem_getStartAddr(&startAddr);
        if (ret < 0)
        {
            ERROR_PLAIN_PRINT(Failed to get clientmem start address);
            return -1;
        }
        
        clientmem_handle = open("/dev/mem", O_RDONLY);
        if (clientmem_handle < 0)
        {
            ERROR_PLAIN_PRINT(Unable to open /dev/mem for clientmem);
            return -1;
        }

        clientmem_addr = (char *)mmap(NULL, PCM_CLIENT_IMC_MMAP_SIZE, PROT_READ, MAP_SHARED, clientmem_handle, startAddr);
        if (clientmem_addr == MAP_FAILED)
        {
            close(clientmem_handle);
            ERROR_PLAIN_PRINT(Mapping of clientmem device failed);
            clientmem_addr = NULL;
            return -1;
        }
        access_clientmem_initialized = 1;
    }
    return 0;
}

void
access_x86_clientmem_finalize(const int socket)
{
    if (access_clientmem_initialized)
    {
        if (clientmem_handle >= 0)
        {
            if (clientmem_addr)
            {
                munmap(clientmem_addr, PCM_CLIENT_IMC_MMAP_SIZE);
            }
            close(clientmem_handle);
        }
        access_clientmem_initialized = 0;
    }
}

int
access_x86_clientmem_read(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t *data)
{
    uint64_t d = 0;
    if (dev != PCI_IMC_DEVICE_0_CH_0)
    {
        return -ENODEV;
    }
    if (clientmem_handle < 0 || !clientmem_addr)
    {
        *data = 0ULL;
        return -ENODEV;
    }
    switch (reg)
    {
        case 0x00:
            d = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_IO_REQUESTS));
            break;
        case 0x01:
            d = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_DATA_READS));
            break;
        case 0x02:
            d = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_DRAM_DATA_WRITES));
            break;
        case 0x03:
            d = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_PP0_TEMP));
            break;
        case 0x04:
            d = (uint64_t)*((uint32_t *)(clientmem_addr + PCM_CLIENT_IMC_PP1_TEMP));
            break;
        default:
            ERROR_PRINT(Read from clientmem device at reg 0x%X failed, reg);
            break;
    }
    *data = d;
    return 0;
}



int
access_x86_clientmem_write(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t data)
{
    return -EACCES;
}

int
access_x86_clientmem_check(PciDeviceIndex dev, int socket)
{
    if (dev != PCI_IMC_DEVICE_0_CH_0)
    {
        return 0;
    }
    else if (clientmem_handle >= 0 && clientmem_addr)
    {
        return 1;
    }
    return 0;
}

