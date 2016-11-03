#ifndef LIKWID_ACCESS_POWER_H
#define LIKWID_ACCESS_POWER_H

#include <types.h>

int access_power_init(int cpu_id);
int access_power_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_power_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
void access_power_finalize(int cpu_id);
int access_power_check(PciDeviceIndex dev, int cpu_id);


#endif
