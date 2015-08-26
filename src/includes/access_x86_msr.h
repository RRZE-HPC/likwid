#ifndef LIKWID_ACCESS_X86_MSR_H
#define LIKWID_ACCESS_X86_MSR_H

#include <types.h>

int access_x86_msr_init(const int cpu_id);
void access_x86_msr_finalize(const int cpu_id);
int access_x86_msr_read(const int cpu, uint32_t reg, uint64_t *data);
int access_x86_msr_write(const int cpu, uint32_t reg, uint64_t data);
int access_x86_msr_check(PciDeviceIndex dev, int cpu_id);

#endif
