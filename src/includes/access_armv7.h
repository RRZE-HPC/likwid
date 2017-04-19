#ifndef LIKWID_ACCESS_ARMV7_H
#define LIKWID_ACCESS_ARMV7_H

int access_armv7_init(int cpu_id);
void access_armv7_finalize(int cpu_id);
int access_armv7_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_armv7_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
int access_armv7_check(PciDeviceIndex dev, int cpu_id);



#endif
