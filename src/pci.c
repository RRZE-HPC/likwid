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
#include <cpuid.h>
#include <affinity.h>
#ifdef LIKWID_USE_HWLOC
#include <cpuid-hwloc.h>
#endif

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define PCI_ROOT_PATH  "/proc/bus/pci/"

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int socket_fd = -1;
static int FD[MAX_NUM_NODES][MAX_NUM_DEVICES] = { [0 ... MAX_NUM_NODES-1] = { 0 } };

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
static char* socket_bus[MAX_NUM_NODES] = { "N-A" };

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
pci_init(int initSocket_fd)
{
    FILE *fptr;
    char buf[1024];
    uint16_t testDevice;
    uint16_t testVendor = 0x8086;
    uint32_t sbus, sdevfn, svend;
    int cntr = 0;
#ifdef LIKWID_USE_HWLOC
	hwloc_obj_t obj;
	int flags;
	int i,j ;
	char testString[5];
#endif

    /*for ( int i=0; i<MAX_NUM_NODES; i++ )
    {
        socket_bus[i] = "N-A";
    }*/

    if (cpuid_info.model == SANDYBRIDGE_EP)
    {
        testDevice = 0x3c44;
    }
    else if (cpuid_info.model == IVYBRIDGE_EP)
    {
        testDevice = 0x0e36;
    }
    else
    {
        /*
        fprintf(stderr, "Unsupported architecture for pci based uncore. \
                Thus, no support for PCI based Uncore counters.\n");
                */
        return;
    }
#ifdef LIKWID_USE_HWLOC
	if (!hwloc_topology)
	{
		fprintf(stderr, "HwLoc Topology not initialized\n");
		hwloc_topology_init(&hwloc_topology);
		flags = hwloc_topology_get_flags(hwloc_topology);
        hwloc_topology_set_flags(hwloc_topology, flags | HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
        hwloc_topology_load(hwloc_topology);
	}
	
	for(i=0;i<hwloc_get_nbobjs_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE);i++)
	{
		obj = hwloc_get_obj_by_type(hwloc_topology, HWLOC_OBJ_PCI_DEVICE, i);
		if (obj->attr->pcidev.vendor_id != testVendor)
			continue;
		if (obj->attr->pcidev.vendor_id == testVendor && obj->attr->pcidev.device_id == testDevice)
		{
			socket_bus[cntr] = (char*)malloc(4);
            sprintf(socket_bus[cntr++], "%02x/", obj->attr->pcidev.bus);
		}
		
		bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
		sprintf(testString,"%02x/", obj->attr->pcidev.bus);
		bcatcstr(filepath, testString);
		sprintf(testString, "%02x.%01x", obj->attr->pcidev.dev, obj->attr->pcidev.func);
		bcatcstr(filepath, testString );
		

        for(j=0;j<MAX_NUM_DEVICES;j++)
        {
        	if (strcmp(testString, pci_DevicePath[j]) == 0 &&
        		!access(bdata(filepath),F_OK))
        	{
        		pci_DevicePresent[j] = 1;
        		break;
        	}
        }
	}
	for(j=0;j<MAX_NUM_DEVICES;j++)
    {
		if (pci_DevicePresent[j])
			printf("Device exists %s\n",pci_DevicePath[j]);
		else
			printf("Device does not exist %s\n",pci_DevicePath[j]);
    }
	
	
#endif
    /*if ( (fptr = fopen( "/proc/bus/pci/devices", "r")) == NULL )
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

    bstring filepath =  bfromcstr ( PCI_ROOT_PATH );
    bcatcstr(filepath, socket_bus[0]);
    bcatcstr(filepath, pci_DevicePath[0] );

    int fd = open( bdata(filepath), O_RDONLY);

    if (fd < 0)
    {
        fprintf(stderr, "INFO\n");
        fprintf(stderr, "       This system has no support for PCI based Uncore counters.\n");
        fprintf(stderr, "       This means you cannot use performance groups as MEM, which require Uncore counters.\n\n");
        return;
    }

    close(fd);*/

    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        if(geteuid() != 0)
        {
            fprintf(stderr, "WARNING\n");
            fprintf(stderr, "       Direct access to the PCI Cfg Adressspace is only allowed for uid root!\n");
            fprintf(stderr, "       This means you can use performance groups as MEM only as root in direct mode.\n");
            fprintf(stderr, "       Alternatively you might want to look into (sys)daemonmode.\n\n");
        }

        /*for (int j=0; j<MAX_NUM_NODES; j++)
        {
            for (int i=0; i<MAX_NUM_DEVICES; i++)
            {
                FD[j][i] = 0;
            }
        }*/
    }
    else /* daemon or sysdaemon-mode */
    {
        socket_fd = initSocket_fd;
    }
}


void
pci_finalize()
{
    if (accessClient_mode != DAEMON_AM_DIRECT)
    {
        for (int j=0; j<MAX_NUM_NODES; j++)
        {
            for (int i=0; i<MAX_NUM_DEVICES; i++)
            {
                if (FD[j][i])
                {
                    close(FD[i][i]);
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
    if (!pci_DevicePresent[device])
    {
    	return 0;
	}
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        uint32_t data = 0;

        if ( !FD[socketId][device] )
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

        if ( !FD[socketId][device] )
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
	if (!pci_DevicePresent[device])
    	return 0;
    if (accessClient_mode == DAEMON_AM_DIRECT)
    {
        uint32_t data = 0;

        if ( !FD[socketId][device] )
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

        if ( !FD[socketId][device] )
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



