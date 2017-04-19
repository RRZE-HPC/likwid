#ifndef LIKWID_ACCESS_ARMV8_H
#define LIKWID_ACCESS_ARMV8_H

int access_armv8_init(int cpu_id);
void access_armv8_finalize(int cpu_id);
int access_armv8_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_armv8_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
int access_armv8_check(PciDeviceIndex dev, int cpu_id);



#endif
