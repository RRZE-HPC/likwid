#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>


#include <types.h>
#include <error.h>
#include <topology.h>
#include <access.h>
#include <access_x86.h>
#include <access_x86_msr.h>
#include <access_x86_pci.h>
#include <affinity.h>



int access_x86_init(int cpu_id)
{
    int ret = access_x86_msr_init(cpu_id);
    if (ret == 0)
    {
        if (cpuid_info.supportUncore)
        {
            ret = access_x86_pci_init(affinity_core2node_lookup[cpu_id]);
        }
    }
    return ret;
}

int access_x86_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data)
{
    int err;
    uint64_t tmp = 0x0ULL;
    if (dev == MSR_DEV)
    {
        err = access_x86_msr_read(cpu_id, reg, &tmp);
        *data = tmp;
    }
    else
    {
        if (access_x86_pci_check(dev, affinity_core2node_lookup[cpu_id]))
        {
            err = access_x86_pci_read(affinity_core2node_lookup[cpu_id], dev, reg, &tmp);
            *data = tmp;
        }
    }
    return err;
}

int access_x86_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data)
{
    int err;
    if (dev == MSR_DEV)
    {
        err = access_x86_msr_write(cpu_id, reg, data);
    }
    else
    {
        if (access_x86_pci_check(dev, affinity_core2node_lookup[cpu_id]))
        {
            err = access_x86_pci_write(affinity_core2node_lookup[cpu_id], dev, reg, data);
        }
    }
    return err;
}

void access_x86_finalize(int cpu_id)
{
    access_x86_msr_finalize(cpu_id);
    if (cpuid_info.supportUncore)
    {
        access_x86_pci_finalize(affinity_core2node_lookup[cpu_id]);
    }
}

int access_x86_check(PciDeviceIndex dev, int cpu_id)
{
    if (dev == MSR_DEV)
    {
        access_x86_msr_check(dev, cpu_id);
    }
    else
    {
        access_x86_pci_check(dev, affinity_core2node_lookup[cpu_id]);
    }
}
