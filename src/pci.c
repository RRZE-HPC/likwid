/*
 * =======================================================================================
 *
 *      Filename:  pci.c
 *
 *      Description:  Implementation of pci module.
 *                   Provides API to read and write values to the hardware
 *                   performance monitoring registers in PCI Cfg space
 *                   for Intel Sandy Bridge Processors.
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
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <accessClient.h>
#include <bstrlib.h>
#include <error.h>
#include <pci.h>
#include <topology.h>
#include <affinity.h>
#ifdef LIKWID_USE_HWLOC
#include <pci_hwloc.h>
#else
#include <pci_proc.h>
#endif

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define PCI_ROOT_PATH  "/proc/bus/pci/"

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int socket_fd = -1;
static int FD[MAX_NUM_NODES][MAX_NUM_DEVICES];

static char* pci_DevicePath[MAX_NUM_DEVICES] = {
 "13.5",   /* PCI_R3QPI_DEVICE_LINK_0 */
 "13.6",   /* PCI_R3QPI_DEVICE_LINK_1 */
 "13.1",   /* PCI_R2PCIE_DEVICE */
 "10.0",   /* PCI_IMC_DEVICE_CH_0 */
 "10.1",   /* PCI_IMC_DEVICE_CH_1 */
 "10.4",   /* PCI_IMC_DEVICE_CH_2 */
 "10.5",   /* PCI_IMC_DEVICE_CH_3 */
 "0e.1",   /* PCI_HA_DEVICE */
 "08.2",   /* PCI_QPI_DEVICE_PORT_0 */
 "09.2",   /* PCI_QPI_DEVICE_PORT_1 */
 "08.6",   /* PCI_QPI_MASK_DEVICE_PORT_0 */
 "09.6",   /* PCI_QPI_MASK_DEVICE_PORT_1 */
 "08.0",   /* PCI_QPI_MISC_DEVICE_PORT_0 */
 "09.0" }; /* PCI_QPI_MISC_DEVICE_PORT_1 */
 

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */
static char* socket_bus[MAX_NUM_NODES];

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
pci_init(int initSocket_fd)
{
    uint16_t testDevice;
    int nr_sockets = 0;
    int i=0;
    int j=0;
    int ret = 0;

    for (i=0; i<MAX_NUM_NODES; i++ )
    {
        socket_bus[i] = "N-A";
        for(j=0;j<MAX_NUM_DEVICES;j++)
        {
            FD[i][j] = -2;
        }
    }

    /* PCI is only provided by Intel systems */
    if (!cpuid_info.isIntel)
    {
        return;
    }

    switch (cpuid_info.model)
    {
        case SANDYBRIDGE_EP:
            testDevice = 0x3c44;
            break;
        case IVYBRIDGE_EP:
            testDevice = 0x0e36;
            break;
        default:
            return;
    }
    
#ifdef LIKWID_USE_HWLOC
    ret = hwloc_pci_init(testDevice, socket_bus, &nr_sockets);
#else
    ret = proc_pci_init(testDevice, socket_bus, &nr_sockets);
#endif
    if (ret)
    {
        fprintf(stderr, "Uncore not supported on this system\n");
        return;
    }
    

    for(i=0;i<nr_sockets;i++)
    {
        for(j=0;j<MAX_NUM_DEVICES;j++)
        {
            bstring filepath = bformat("%s%s%s",PCI_ROOT_PATH,
                                                socket_bus[i],
                                                pci_DevicePath[j]);
            if (!access(bdata(filepath), F_OK))
            {
                FD[i][j] = 0;
            }
        }
    }

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        if(geteuid() != 0)
        {
            fprintf(stderr, "WARNING\n");
            fprintf(stderr, "       Direct access to the PCI Cfg Adressspace is only allowed for uid root!\n");
            fprintf(stderr, "       This means you can use performance groups as MEM only as root in direct mode.\n");
            fprintf(stderr, "       Alternatively you might want to look into (sys)daemonmode.\n\n");
        }
    }
    else /* daemon or sysdaemon-mode */
    {
        socket_fd = initSocket_fd;
    }
}


void
pci_finalize()
{
    int i=0;
    int j=0;
    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        for (i=0; i<MAX_NUM_NODES; i++)
        {
            for (j=0; j<MAX_NUM_DEVICES; j++)
            {
                if (FD[i][j] > 0)
                {
                    close(FD[i][j]);
                }
            }
        }
    }
    else
    {
        socket_fd = -1;
    }
}


int
pci_read(int cpu, PciDeviceIndex device, uint32_t reg, uint32_t* data)
{
    int socketId = affinity_core2node_lookup[cpu];
    bstring filepath = NULL;
    uint64_t tmp;
    
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        if (FD[socketId][device] < 0)
        {
            *data = 0;
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_read:\nFailed to open PCI device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                *data = 0;
                return -EACCES;
            }
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &tmp, sizeof(tmp), reg) != sizeof(tmp) ) 
        {
            fprintf(stderr,"ERROR in pci_read:\nCannot read from PCI device %s: %s\n",
                    bdata(filepath),strerror(errno));
            *data = 0;
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (accessClient_read(socket_fd, socketId, device, reg, &tmp))
        {
            return -EIO;
        } 
    }
    *data = tmp;
    return 0;
}



int
pci_write(int cpu, PciDeviceIndex device, uint32_t reg, uint32_t data)
{
    int socketId = affinity_core2node_lookup[cpu];
    bstring filepath = NULL;
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath = bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            
            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_write:\nFailed to open PCI device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                return -EACCES;
            }
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            fprintf(stderr,"ERROR in pci_write:\nCannot write to PCI device %s: %s\n",
                    bdata(filepath),strerror(errno));
            return -EIO;
        }    
    }
    else
    { /* daemon or sysdaemon-mode */
        if (accessClient_write(socket_fd, socketId, device, reg, (uint64_t) data))
        {
            return -EIO;
        }
    }
    return 0;
}

int
pci_tread(const int tsocket_fd, const int cpu, PciDeviceIndex device, uint32_t reg, uint32_t *data)
{
    int socketId = affinity_core2node_lookup[cpu];
    bstring filepath = NULL;
    uint64_t tmp;
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        *data = 0;

        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );

            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_tread:\nFailed to open PCI device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                return -EACCES;
            }
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &tmp, sizeof(tmp), reg) != sizeof(tmp) ) 
        {
            fprintf(stderr,"ERROR in pci_tread:\nCannot read from PCI device %s: %s\n",
                    bdata(filepath),strerror(errno));
            *data = 0;
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (accessClient_read(tsocket_fd, socketId, device, reg, &tmp))
        {
            return -EIO;
        }
        
    }
    *data = tmp;
    return 0;
}

int
pci_twrite( const int tsocket_fd, const int cpu, PciDeviceIndex device, uint32_t reg, uint32_t data)
{
    int socketId = affinity_core2node_lookup[cpu];
    bstring filepath = NULL;
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {

        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );

            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_twrite:\n    failed to open pci device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                return -EACCES;
            }
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            fprintf(stderr,"ERROR in pci_twrite:\nCannot write to pci device %s: %s\n",
                    bdata(filepath),strerror(errno));
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (accessClient_write(tsocket_fd, socketId, device, reg, data))
        {
            return -EIO;
        }
    }
    return 0;
}

int pci_checkDevice(PciDeviceIndex index, int cpu)
{
    int socketId = affinity_core2node_lookup[cpu];
    if (FD[socketId][index] > 0)
    {
        return 1;
    }
    return 0;
}

