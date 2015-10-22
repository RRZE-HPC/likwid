#ifndef LIKWID_ACCESS_ARM_H
#define LIKWID_ACCESS_ARM_H

int access_arm_init(int cpu_id);
void access_arm_finalize(int cpu_id);
int access_arm_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_arm_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
int access_arm_check(PciDeviceIndex dev, int cpu_id);



#endif
