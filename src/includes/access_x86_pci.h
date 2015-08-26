#ifndef LIKWID_ACCESS_X86_PCI_H
#define LIKWID_ACCESS_X86_PCI_H

#include <types.h>

int access_x86_pci_init(const int socket);
void access_x86_pci_finalize(const int socket);
int access_x86_pci_read(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t *data);
int access_x86_pci_write(PciDeviceIndex dev, const int socket, uint32_t reg, uint64_t data);
int access_x86_pci_check(PciDeviceIndex dev, int socket);

#endif
