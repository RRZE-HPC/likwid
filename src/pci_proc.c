/*
 * =======================================================================================
 *
 *      Filename:  pci_proc.c
 *
 *      Description:  Interface to procfs/sysfs for PCI device lookup
 *
 *      Version:   4.3.2
 *      Released:  12.04.2018
 *
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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

#include <types.h>
#include <bstrlib.h>
#include <affinity.h>
#include <topology.h>
#include <error.h>

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
getBusFromSocket(const uint32_t socket)
{
    int cur_bus = 0;
    uint32_t cur_socket = 0;
    char pci_filepath[1024];
    int fp;
    int ret = 0;
    while(cur_socket <= socket)
    {
        sprintf(pci_filepath, "/proc/bus/pci/%02x/05.0", cur_bus);
        fp = open(pci_filepath, O_RDONLY);
        if (fp < 0)
        {
            return -1;
        }
        uint32_t cpubusno = 0;
        ret = pread(fp, &cpubusno, sizeof(uint32_t), 0x108);
        if (ret != sizeof(uint32_t))
        {
            close(fp);
            return -1;
        }
        cur_bus = (cpubusno >> 8) & 0x0ff;
        close(fp);
        if(socket == cur_socket)
            return cur_bus;
        ++cur_socket;
        ++cur_bus;
        if(cur_bus > 0x0ff)
           return -1;
    }

    return -1;
}

#define PCI_SLOT(devfn)         (((devfn) >> 3) & 0x1f)
#define PCI_FUNC(devfn)         ((devfn) & 0x07)


/* This code gets the PCI device using the given devid in pcidev. It assumes that the
 * PCI busses are sorted like: if sock_id1 < sock_id2 then bus1 < bus2 end
 * This code is only the fallback if a device is not found using the combination of
 * filename and devid
 */
typedef struct {
    uint32_t bus;
    uint32_t devfn;
} PciCandidate;

static int getBusFromSocketByDevid(const uint32_t socket, uint16_t testDevice)
{
    int ret = 0;
    int cur_socket = (int)socket;
    int out_bus_id = -1;
    uint32_t out_devfn = 0x0;
    int bufflen = 1024;
    char buff[1024];
    FILE* fp = NULL;
    uint32_t bus, devfn, vendor, devid;
    PciCandidate candidates[10];
    int candidate = -1;
    int cand_idx = 0;

    fp = fopen("/proc/bus/pci/devices", "r");
    if (fp)
    {
        while (fgets(buff, bufflen, fp) != NULL)
        {
            ret = sscanf((char*)buff, "%02x%02x\t%04x%04x", &bus, &devfn, &vendor, &devid);
            if (ret == 4 && devid == testDevice)
            {
                candidates[cand_idx].bus = bus;
                candidates[cand_idx].devfn = devfn;
                cand_idx++;
            }
        }
        fclose(fp);
    }
    else
    {
        ERROR_PLAIN_PRINT(Failed read file /proc/bus/pci/devices);
    }

    while (cur_socket >= 0)
    {
        int min_idx = 0;
        uint32_t min = 0xFFF;
        for (ret = 0; ret < cand_idx; ret++)
        {
            if (candidates[ret].bus < min)
            {
                min = candidates[ret].bus;
                min_idx = ret;
            }
        }
        if (cur_socket > 0)
        {
            candidates[min_idx].bus = 0xFFF;
            cur_socket--;
        }
        else
        {
            if (candidates[min_idx].bus <= 0xff)
            {
                candidate = min_idx;
            }
            cur_socket = -1;
            break;
        }
    }

    if (candidate >= 0 && candidates[candidate].bus > 0 && candidates[candidate].devfn > 0)
        return candidates[candidate].bus;
    return -1;
}


int
proc_pci_init(uint16_t testDevice, char** socket_bus, int* nrSockets)
{
    FILE *fptr;
    char buf[1024];
    int cntr = 0;
    uint16_t testVendor = 0x8086;
    uint32_t sbus, sdevfn, svend, sdev;
    int busID;

    if ( (fptr = fopen( "/proc/bus/pci/devices", "r")) == NULL )
    {
        fprintf(stderr, "Unable to open /proc/bus/pci/devices. \
                Thus, no support for PCI based Uncore counters.\n");
        return -ENODEV;
    }

    while( fgets(buf, sizeof(buf)-1, fptr) )
    {
        if ( sscanf(buf, "%02x%02x\t%04x%04x", &sbus, &sdevfn, &svend, &sdev) == 4 &&
             svend == testVendor && sdev == testDevice )
        {
            socket_bus[cntr] = (char*)malloc(4);
            busID = getBusFromSocketByDevid(cntr, testDevice);
            if (busID == sbus)
            {
                sprintf(socket_bus[cntr], "%02x/", sbus);
            }
            else
            {
                sprintf(socket_bus[cntr], "%02x/", busID);
            }
            cntr++;
        }
    }
    fclose(fptr);
    *nrSockets = cntr;
    if ( cntr == 0 )
    {
        //fprintf(stderr, "Uncore not supported on this system\n");
        return -ENODEV;
    }
    return 0;
}

