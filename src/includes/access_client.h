#ifndef LIKWID_ACCESS_CLIENT_H
#define LIKWID_ACCESS_CLIENT_H


int access_client_init(int cpu_id);
int access_client_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_client_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
void access_client_finalize(int cpu_id);
int access_client_check(PciDeviceIndex dev, int cpu_id);

#endif
