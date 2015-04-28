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
#define PCM_PCI_CLASS  0x1101

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int FD[MAX_NUM_NODES][MAX_NUM_PCI_DEVICES];

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
/* Dirty hack to avoid nonull warnings */
int (*ownaccess)(const char*, int);
int (*ownopen)(const char*, int, ...);

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
pci_init(int initSocket_fd)
{
    uint16_t testDevice;
    int nr_sockets = 0;
    int i=0;
    int j=0;
    int ret = 0;
    int access_flags = 0;
    ownaccess = &access;
    ownopen = &open;

    for (i=0; i<MAX_NUM_NODES; i++ )
    {
        socket_bus[i] = "N-A";
        for(j=1;j<MAX_NUM_PCI_DEVICES;j++)
        {
            FD[i][j] = -2;
        }
    }
    /* PCI is only provided by Intel systems */
    if (!cpuid_info.isIntel)
    {
        DEBUG_PLAIN_PRINT(DEBUGLEV_DETAIL, PCI based Uncore performance monitoring only supported on Intel systems);
        return -ENODEV;
    }

    switch (cpuid_info.model)
    {
        case SANDYBRIDGE_EP:
            testDevice = 0x3c44;
            break;
        case IVYBRIDGE_EP:
            testDevice = 0x0e36;
            break;
        case HASWELL_EP:
            testDevice = 0x2f30;
            break;
        default:
            DEBUG_PRINT(DEBUGLEV_INFO,CPU model %s does not support PCI based Uncore performance monitoring, cpuid_info.name);
            return -ENODEV;
            break;
    }

#ifdef LIKWID_USE_HWLOC
    DEBUG_PLAIN_PRINT(DEBUGLEV_DETAIL, Using hwloc to find pci devices);
    ret = hwloc_pci_init(testDevice, socket_bus, &nr_sockets);
    if (ret)
    {
        ERROR_PLAIN_PRINT(Using hwloc to find pci devices failed);
        return -ENODEV;
    }
#else
    DEBUG_PLAIN_PRINT(DEBUGLEV_DETAIL, Using procfs to find pci devices);
    ret = proc_pci_init(testDevice, socket_bus, &nr_sockets);
    if (ret)
    {
        ERROR_PLAIN_PRINT(Using procfs to find pci devices failed);
        return -ENODEV;
    }
#endif

    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        access_flags = R_OK|W_OK;
    }
    else
    {
        access_flags = F_OK;
    }

    for(i=0;i<nr_sockets;i++)
    {
        for(j=1;j<MAX_NUM_PCI_DEVICES;j++)
        {
            if (pci_devices[j].path != NULL)
            {
                bstring filepath = bformat("%s%s%s",PCI_ROOT_PATH,
                                                    socket_bus[i],
                                                    pci_devices[j].path);
                if (!ownaccess(bdata(filepath),access_flags))
                {
                    FD[i][j] = 0;
                    pci_devices[j].online = 1;
                    if (i==0)
                    {
                        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCI device %s (%d) online for socket %d at path %s, pci_devices[j].name,j, i,bdata(filepath));
                    }
                }
                else
                {
                    pci_devices[j].online = 0;
                }
            }
        }
    }

    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        if(geteuid() != 0)
        {
            fprintf(stderr, "WARNING\n");
            fprintf(stderr, "Direct access to the PCI Cfg Adressspace is only allowed for uid root!\n");
            fprintf(stderr, "This means you can use performance groups as MEM only as root in direct mode.\n");
            fprintf(stderr, "Alternatively you might want to look into (sys)daemonmode.\n\n");
        }
    }
    else /* daemon or sysdaemon-mode */
    {
        socket_fd = initSocket_fd;
    }
    return 0;
}


void
pci_finalize()
{
    int i=0;
    int j=0;
    if (accessClient_mode != ACCESSMODE_DIRECT)
    {
        for (i=0; i<MAX_NUM_NODES; i++)
        {
            for (j=1; j<MAX_NUM_PCI_DEVICES; j++)
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
    int err;

    if (device == MSR_DEV)
    {
        return -ENODEV;
    }

    if (accessClient_mode == ACCESSMODE_DIRECT)
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
            bcatcstr(filepath, pci_devices[device].path);
            FD[socketId][device] = ownopen( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                ERROR_PRINT(Failed to open PCI device %s at path %s\n, 
                                pci_devices[device].name,
                                bdata(filepath));
                *data = 0;
                return -EACCES;
            }
            DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s, pci_devices[device].name);
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &tmp, sizeof(tmp), reg) != sizeof(tmp) ) 
        {
            ERROR_PRINT(Read from PCI device %s at register 0x%x failed,pci_devices[device].name, reg);
            *data = 0;
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCI READ [%d] SOCKET %d DEV %s(%d) REG 0x%x, cpu, socket_fd, pci_devices[device].name,device, reg);
        err = accessClient_read(socket_fd, socketId, device, reg, &tmp);
        if (err)
        {
            ERROR_PRINT(Read from PCI device %s at register 0x%x through access daemon failed,pci_devices[device].name, reg);
            return err;
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
    int err;

    if (device == MSR_DEV)
    {
        return -ENODEV;
    }
    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath = bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_devices[device].path );
            
            FD[socketId][device] = ownopen( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                ERROR_PRINT(Failed to open PCI device %s at path %s\n, 
                                    pci_devices[device].name,
                                    bdata(filepath));
                return -EACCES;
            }
            DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s, pci_devices[device].name);
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT(Write to PCI device %s at register 0x%x failed,pci_devices[device].name, reg);
            return -EIO;
        }    
    }
    else
    { /* daemon or sysdaemon-mode */
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, PCI WRITE [%d] SOCKET %d DEV %s REG 0x%x, cpu, socket_fd, pci_devices[device].name, reg);
        err = accessClient_write(socket_fd, socketId, device, reg, (uint64_t) data);
        if (err)
        {
            ERROR_PRINT(Write to PCI device %s at register 0x%x through access daemon failed,pci_devices[device].name, reg);
            return err;
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
    int err;

    if (accessClient_mode == ACCESSMODE_DIRECT)
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
            bcatcstr(filepath, pci_devices[device].path );

            FD[socketId][device] = ownopen( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                ERROR_PRINT(Failed to open PCI device %s at path %s\n, 
                                pci_devices[device].name,
                                bdata(filepath));
                return -EACCES;
            }
            DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s, pci_devices[device].name);
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &tmp, sizeof(tmp), reg) != sizeof(tmp) ) 
        {
            ERROR_PRINT(Read from PCI device %s at register 0x%x failed,pci_devices[device].name, reg);
            *data = 0;
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        err = accessClient_read(tsocket_fd, socketId, device, reg, &tmp);
        if (err)
        {
            ERROR_PRINT(Read from PCI device %s at register 0x%x through access daemon failed,pci_devices[device].name, reg);
            return err;
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
    int err;

    if (accessClient_mode == ACCESSMODE_DIRECT)
    {
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        else if ( !FD[socketId][device] )
        {
            filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_devices[device].path );

            FD[socketId][device] = ownopen( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                ERROR_PRINT(Failed to open PCI device %s at path %s\n, 
                                pci_devices[device].name,
                                bdata(filepath));
                return -EACCES;
            }
            DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s, pci_devices[device].name);
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT(Write to PCI device %s at register 0x%x failed,pci_devices[device].name, reg);
            return -EIO;
        }
    }
    else
    { /* daemon or sysdaemon-mode */
        if (FD[socketId][device] < 0)
        {
            return -ENODEV;
        }
        err = accessClient_write(tsocket_fd, socketId, device, reg, data);
        if (err)
        {
            ERROR_PRINT(Write to PCI device %s at register 0x%x through access daemon failed,pci_devices[device].name, reg);
            return -EIO;
        }
    }
    return 0;
}

int pci_checkDevice(PciDeviceIndex index, int cpu)
{
    int socketId = affinity_core2node_lookup[cpu];
    if (index == MSR_DEV)
    {
        return 1;
    }
    else if ((pci_devices[index].online == 1) || (FD[socketId][index] >= 0))
    {
        return 1;
    }
    return 0;
}

