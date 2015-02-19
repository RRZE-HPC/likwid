#ifndef ACCESS_H
#define ACCESS_H

int HPMinit(void);
int HPMinitialized(void);
int HPMaddThread(int cpu_id);
void HPMfinalize(void);
int HPMread(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t* data);
int HPMwrite(int cpu_id, PciDeviceIndex dev, uint32_t reg, uint64_t data);

#endif
