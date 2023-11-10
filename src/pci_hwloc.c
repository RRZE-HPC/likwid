/*
 * =======================================================================================
 *
 *      Filename:  pci_hwloc.c
 *
 *      Description:  Interface to hwloc for PCI device lookup
 *
 *      Version:   5.3
 *      Released:  10.11.2023
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2023 RRZE, University Erlangen-Nuremberg
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


#include <hwloc.h>
#include <types.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <topology_hwloc.h>
#include <error.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
hwloc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    hwloc_obj_t obj;
    int flags;
    int i;

    if (!hwloc_topology)
    {
        LIKWID_HWLOC_NAME(topology_init)(&hwloc_topology);
#if HWLOC_API_VERSION > 0x00020000
        LIKWID_HWLOC_NAME(topology_set_flags)(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM );
#else
        LIKWID_HWLOC_NAME(topology_set_flags)(hwloc_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_SYSTEM|HWLOC_TOPOLOGY_FLAG_WHOLE_IO );
#endif
        LIKWID_HWLOC_NAME(topology_load)(hwloc_topology);
    }

    for(i = 0; i < LIKWID_HWLOC_NAME(get_nbobjs_by_type)(hwloc_topology, HWLOC_OBJ_PCI_DEVICE); i++)
    {
        obj = LIKWID_HWLOC_NAME(get_obj_by_type)(hwloc_topology, HWLOC_OBJ_PCI_DEVICE, i);
        if (obj->attr->pcidev.vendor_id != testVendor)
        {
            continue;
        }
        if ((obj->attr->pcidev.vendor_id == testVendor) && (obj->attr->pcidev.device_id == testDevice))
        {
            hwloc_obj_t walk = obj->parent;
            while (walk->type != HWLOC_OBJ_SOCKET) walk = walk->parent;
            if (socket_bus[walk->os_index] == NULL)
            {
                socket_bus[walk->os_index] = (char*)malloc(5);
                snprintf(socket_bus[walk->os_index], 4, "%02x/", obj->attr->pcidev.bus);
                cntr++;
            }
        }
    }
    *nrSockets = cntr;

    if (cntr == 0)
    {
        return -ENODEV;
    }

    return 0;
}

#define SKYLAKE_SERVER_SOCKETID_MBOX_DID 0x2042

int sysfs_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    struct dirent *pDirent, *pDirentInner;
    DIR *pDir, *pDirInner;
    pDir = opendir ("/sys/devices");
    FILE* fp = NULL;
    char iPath[200], iiPath[200], buff[100];
    char testDev[50];
    size_t ret = 0;
    int nrSocks = 0;
    if (pDir == NULL)
    {
        fprintf(stderr, "Cannot read /sys/devices\n");
        return 1;
    }

    while ((pDirent = readdir(pDir)) != NULL)
    {
        //printf ("[%s]\n", pDirent->d_name);
        if (strncmp(pDirent->d_name, "pci0", 4) == 0)
        {
            sprintf(iPath, "/sys/devices/%s", pDirent->d_name);
            char bus[4];
            strncpy(bus, &(pDirent->d_name[strlen(pDirent->d_name)-2]), 2);
            bus[2] = '/';
            bus[3] = '\0';
            //printf("PATH %s\n", iPath);
            pDirInner = opendir (iPath);
            if (pDir == NULL)
            {
                fprintf(stderr, "Cannot read %s\n", iPath);
                return 1;
            }
            while ((pDirentInner = readdir(pDirInner)) != NULL)
            {
                if (strncmp(pDirentInner->d_name, "0000", 4) == 0)
                {
                    uint32_t dev_id = 0x0;
                    int numa_node = 0;
                    sprintf(iiPath, "/sys/devices/%s/%s/device", pDirent->d_name, pDirentInner->d_name);
                    fp = fopen(iiPath,"r");
                    if( fp == NULL )
                    {
                        continue;
                    }
                    ret = fread(buff, sizeof(char), 99, fp);
                    dev_id = strtoul(buff, NULL, 16);
                    if (dev_id == SKYLAKE_SERVER_SOCKETID_MBOX_DID)
                    {
                        fclose(fp);
                        iiPath[0] = '\0';
                        sprintf(iiPath, "/sys/devices/%s/%s/numa_node", pDirent->d_name, pDirentInner->d_name);
                        fp = fopen(iiPath,"r");
                        if( fp == NULL )
                        {
                            continue;
                        }
                        ret = fread(buff, sizeof(char), 99, fp);
                        numa_node = atoi(buff);
                        socket_bus[numa_node] = (char*)malloc(4);
                        sprintf(socket_bus[numa_node], "%02x/", bus);
                        nrSocks++;
                    }
                    fclose(fp);
                    iiPath[0] = '\0';
                    buff[0] = '\0';
                }
            }
            closedir (pDirInner);
            iPath[0] = '\0';
        }
    }
    closedir (pDir);
    *nrSockets = nrSocks;
    return 0;
}


