#ifndef LIKWID_ACCESS_X86_H
#define LIKWID_ACCESS_X86_H

#include <types.h>

int access_x86_init(int cpu_id);
int access_x86_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_x86_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
void access_x86_finalize(int cpu_id);
int access_x86_check(PciDeviceIndex dev, int cpu_id);


#endif
