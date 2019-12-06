/*
 * =======================================================================================
 *
 *      Filename:  access_x86_pci.c
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

#include <types.h>
#include <bstrlib.h>
#include <error.h>
#include <topology.h>

#include <access_x86_pci.h>

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
static int access_x86_initialized = 0;
static int nr_sockets = 0;

/* Socket to bus mapping -- will be determined at runtime;
 * typical mappings are:
 * Socket  Bus (2S)  Bus (4s)
 *   0        0xff      0x3f
 *   1        0x7f      0x7f
 *   2                  0xbf
 *   3                  0xff
 */
static char* socket_bus[MAX_NUM_NODES] = { [0 ... (MAX_NUM_NODES-1)] = "N-A"};

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */

/* Dirty hack to avoid nonull warnings */
int (*ownaccess)(const char*, int);
int (*ownopen)(const char*, int, ...);

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
access_x86_pci_init(const int socket)
{
    int ret = 0;


    if (access_x86_initialized == 0)
    {
        uint16_t testDevice;
        ownaccess = &access;
        ownopen = &open;

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
            case BROADWELL_D:
                testDevice = 0x6f30;
                break;
            case BROADWELL_E:
                testDevice = 0x6f30;
                break;
            case XEON_PHI_KNL:
            case XEON_PHI_KML:
                testDevice = 0x7843;
                break;
            case SKYLAKEX:
                testDevice = 0x2042;
                break;
            default:
                DEBUG_PRINT(DEBUGLEV_INFO,CPU model %s does not support PCI based Uncore performance monitoring, cpuid_info.name);
                return -ENODEV;
                break;
        }
        if(geteuid() != 0)
        {
            fprintf(stderr, "WARNING\n");
            fprintf(stderr, "Direct access to the PCI Cfg Adressspace is only allowed for uid root!\n");
            fprintf(stderr, "This means you can use performance groups as MEM only as root in direct mode.\n");
            fprintf(stderr, "Alternatively you might want to look into (sys)daemonmode.\n\n");
            return -EPERM;
        }

        for(int i=0; i<MAX_NUM_NODES; i++)
        {
            for(int j=1;j<MAX_NUM_PCI_DEVICES;j++)
            {
                FD[i][j] = -2;
            }
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
    }

    for(int j=1;j<MAX_NUM_PCI_DEVICES;j++)
    {
        if ((pci_devices[j].path != NULL) && (FD[socket][j] == -2))
        {
            bstring filepath = bformat("%s%s%s",PCI_ROOT_PATH,
                                                socket_bus[socket],
                                                pci_devices[j].path);
            if (!ownaccess(bdata(filepath),F_OK))
            {
                FD[socket][j] = 0;
                pci_devices[j].online = 1;
                if (access_x86_initialized == 0)
                {
                    DEBUG_PRINT(DEBUGLEV_DETAIL,
                            PCI device %s (%d) online for socket %d at path %s, pci_devices[j].name,j, socket,bdata(filepath));
                    if (ownaccess(bdata(filepath),R_OK|W_OK))
                    {
                        ERROR_PRINT(PCI device %s (%d) online for socket %d at path %s but not accessible, pci_devices[j].name,j, socket,bdata(filepath));
                    }
                }
            }
            else
            {
                pci_devices[j].online = 0;
            }
        }
    }

    access_x86_initialized = 1;
    return 0;
}

void
access_x86_pci_finalize(const int socket)
{
    if (access_x86_initialized)
    {
        for (int j=1; j<MAX_NUM_PCI_DEVICES; j++)
        {
            if (FD[socket][j] > 0)
            {
                close(FD[socket][j]);
                FD[socket][j] = -2;
                pci_devices[j].online = 0;
            }
        }
        access_x86_initialized = 0;
    }
}

int
access_x86_pci_read(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t *data)
{
    bstring filepath = NULL;
    uint32_t tmp;

    if (dev == MSR_DEV)
    {
        return -ENODEV;
    }

    if (FD[socket][dev] < 0)
    {
        *data = 0ULL;
        return -ENODEV;
    }
    else if ( !FD[socket][dev] )
    {
        filepath =  bfromcstr ( PCI_ROOT_PATH );
        bcatcstr(filepath, socket_bus[socket]);
        bcatcstr(filepath, pci_devices[dev].path);
        FD[socket][dev] = ownopen( bdata(filepath), O_RDWR);

        if ( FD[socket][dev] < 0)
        {
            ERROR_PRINT(Failed to open PCI device %s at path %s\n,
                            pci_devices[dev].name,
                            bdata(filepath));
            *data = 0ULL;
            return -EACCES;
        }
        DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s: %s, pci_devices[dev].name, bdata(filepath));
    }

    if ( FD[socket][dev] > 0 &&
         pread(FD[socket][dev], &tmp, sizeof(tmp), reg) != sizeof(tmp) )
    {
        ERROR_PRINT(Read from PCI device %s at register 0x%x failed, pci_devices[dev].name, reg);
        *data = 0ULL;
        return -EIO;
    }
    *data = (uint64_t)tmp;
    return 0;
}

int
access_x86_pci_write(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t data)
{
    bstring filepath = NULL;
    uint32_t tmp = (uint32_t)data;

    if (dev == MSR_DEV)
    {
        return -ENODEV;
    }
    if (FD[socket][dev] < 0)
    {
        return -ENODEV;
    }
    else if ( !FD[socket][dev] )
    {
        filepath = bfromcstr ( PCI_ROOT_PATH );
        bcatcstr(filepath, socket_bus[socket]);
        bcatcstr(filepath, pci_devices[dev].path );
        FD[socket][dev] = ownopen( bdata(filepath), O_RDWR);

        if ( FD[socket][dev] < 0)
        {
            ERROR_PRINT(Failed to open PCI device %s at path %s\n,
                                pci_devices[dev].name,
                                bdata(filepath));
            return -EACCES;
        }
        DEBUG_PRINT(DEBUGLEV_DETAIL, Opened PCI device %s: %s, pci_devices[dev].name, bdata(filepath));
    }

    if ( FD[socket][dev] > 0 &&
         pwrite(FD[socket][dev], &tmp, sizeof tmp, reg) != sizeof tmp)
    {
        ERROR_PRINT(Write to PCI device %s at register 0x%x failed, pci_devices[dev].name, reg);
        return -EIO;
    }
    return 0;
}

int
access_x86_pci_check(PciDeviceIndex dev, int socket)
{
    if (dev == MSR_DEV)
    {
        return 1;
    }
    else if ((pci_devices[dev].online == 1) || (FD[socket][dev] > 0))
    {
        return 1;
    }
    return 0;
}

