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
 *      Copyright (C) 2014 Jan Treibig
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
#include <cpuid.h>
#include <affinity.h>

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
static int socket_count = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
pci_init(int initSocket_fd)
{
    FILE *fptr;
    char buf[1024];
    uint32_t testDevice;
    uint32_t sbus, sdevfn, svend;
    int cntr = 0;
    int active_devs = 0;

    for ( int j=0; j<MAX_NUM_NODES; j++ )
    {
        socket_bus[j] = "N-A";
        for (int i=0; i<MAX_NUM_DEVICES; i++)
        {
            FD[j][i] = 0;
        }
    }

    if (cpuid_info.model == SANDYBRIDGE_EP)
    {
        testDevice = 0x80863c44;
    }
    else if (cpuid_info.model == IVYBRIDGE_EP)
    {
        testDevice = 0x80860e36;
    }
    else
    {
        /*
        fprintf(stderr, "Unsupported architecture for pci based uncore. \
                Thus, no support for PCI based Uncore counters.\n");
                */
        return;
    }

    if ( (fptr = fopen( "/proc/bus/pci/devices", "r")) == NULL )
    {
        fprintf(stderr, "Unable to open /proc/bus/pci/devices. \
                Thus, no support for PCI based Uncore counters.\n");
        return;
    }

    while( fgets(buf, sizeof(buf)-1, fptr) )
    {
        if ( sscanf(buf, "%2x%2x %8x", &sbus, &sdevfn, &svend) == 3 &&
             svend == testDevice )
        {
            socket_bus[cntr] = (char*)malloc(4);
            sprintf(socket_bus[cntr++], "%02x/", sbus);
        }
    }
    fclose(fptr);

    if ( cntr == 0 )
    {
        fprintf(stderr, "Uncore not supported on this system\n");
        return;
    }

	socket_count = cntr;
	
			bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
    bcatcstr(filepath, socket_bus[0]);
    bcatcstr(filepath, pci_DevicePath[0] );


    if (access(bdata(filepath),F_OK))
    {
        fprintf(stderr, "INFO\n");
        fprintf(stderr, "       This system has no support for PCI based Uncore counters.\n");
        fprintf(stderr, "       This means you cannot use performance groups as MEM, which require Uncore counters.\n\n");
        return;
    }
    bdestroy(filepath);

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        if(geteuid() != 0)
        {
            fprintf(stderr, "WARNING\n");
            fprintf(stderr, "       Direct access to the PCI Cfg Adressspace is only allowed for uid root!\n");
            fprintf(stderr, "       This means you can use performance groups as MEM only as root in direct mode.\n");
            fprintf(stderr, "       Alternatively you might want to look into (sys)daemonmode.\n\n");
        }

        for (int j=0; j<socket_count; j++)
        {
            for (int i=0; i<MAX_NUM_DEVICES; i++)
            {

                bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
				bcatcstr(filepath, socket_bus[j]);
				bcatcstr(filepath, pci_DevicePath[i] );
				
				if (!access(bdata(filepath),R_OK|W_OK))
				{
					FD[j][i] = 0;
				}
				else
				{
					fprintf(stderr, "Device %s not found, excluded it from device list\n",bdata(filepath));
					FD[j][i] = -2;
				}
				bdestroy(filepath);
            }
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
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        for (int j=0; j<socket_count; j++)
        {
            for (int i=0; i<MAX_NUM_DEVICES; i++)
            {
                if (FD[j][i] > 0)
                {
                    close(FD[j][i]);
                }
            }
        }
    }
    else
    {
        socket_fd = -1;
    }
}


uint32_t
pci_read(int cpu, PciDeviceIndex device, uint32_t reg)
{
    int socketId = affinity_core2node_lookup[cpu];

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        uint32_t data = 0;
		if ( FD[socketId][device] == -2)
		{
			return data;
		}
        else if ( !FD[socketId][device] )
        {
            bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_read:\n    failed to open pci device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                // exit(127);
            }
            bdestroy(filepath);
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &data, sizeof data, reg) != sizeof data ) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }

        return data;
    }
    else
    { /* daemon or sysdaemon-mode */
        return (uint32_t) accessClient_read(socket_fd, socketId, device, reg);
    }
}



void
pci_write(int cpu, PciDeviceIndex device, uint32_t reg, uint32_t data)
{
    int socketId = affinity_core2node_lookup[cpu];

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
		if ( FD[socketId][device] == -2)
	{
		return;
	}
        else if ( !FD[socketId][device] )
        {
            bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_write:\n    failed to open pci device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                // exit(127);
            }
            bdestroy(filepath);
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }

        //    printf("WRITE Device %s cpu %d reg 0x%x data 0x%x \n",bdata(filepath), cpu, reg, data);
    }
    else
    { /* daemon or sysdaemon-mode */
        accessClient_write(socket_fd, socketId, device, reg, (uint64_t) data);
    }
}

uint32_t
pci_tread(const int tsocket_fd, const int cpu, PciDeviceIndex device, uint32_t reg)
{
    int socketId = affinity_core2node_lookup[cpu];

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        uint32_t data = 0;
		if ( FD[socketId][device] == -2)
		{
			return data;
		}
        else if ( !FD[socketId][device] )
        {
            bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            //        printf("Generate PATH = %s \n",bdata(filepath));

            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_tread:\n    failed to open pci device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                // exit(127);
            }
            bdestroy(filepath);
        }

        if ( FD[socketId][device] > 0 &&
             pread(FD[socketId][device], &data, sizeof data, reg) != sizeof data ) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }
        //    printf("READ Device %s cpu %d reg 0x%x data 0x%x \n",bdata(filepath), cpu, reg, data);

        return data;
    }
    else
    { /* daemon or sysdaemon-mode */
        return accessClient_read(tsocket_fd, socketId, device, reg);
    }
}

void
pci_twrite( const int tsocket_fd, const int cpu, PciDeviceIndex device, uint32_t reg, uint32_t data)
{
    int socketId = affinity_core2node_lookup[cpu];
	
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
		if ( FD[socketId][device] == -2)
	{
		return;
	}
        else if ( !FD[socketId][device] )
        {
            bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
            bcatcstr(filepath, socket_bus[socketId]);
            bcatcstr(filepath, pci_DevicePath[device] );
            //        printf("Generate PATH = %s \n",bdata(filepath));

            FD[socketId][device] = open( bdata(filepath), O_RDWR);

            if ( FD[socketId][device] < 0)
            {
                fprintf(stderr, "ERROR in pci_twrite:\n    failed to open pci device %s: %s!\n",
                        bdata(filepath), strerror(errno));
                //exit(127);
            }
            bdestroy(filepath);
        }

        if ( FD[socketId][device] > 0 &&
             pwrite(FD[socketId][device], &data, sizeof data, reg) != sizeof data) 
        {
            ERROR_PRINT("cpu %d reg %x",cpu, reg);
        }

        //    printf("WRITE Device %s cpu %d reg 0x%x data 0x%x \n",bdata(filepath), cpu, reg, data);
    }
    else
    { /* daemon or sysdaemon-mode */
        accessClient_write(tsocket_fd, socketId, device, reg, data);
    }
}



