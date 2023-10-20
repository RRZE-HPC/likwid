#ifndef ACCESS_X86_TRANSLATE_H
#define ACCESS_X86_TRANSLATE_H


int access_x86_translate_init(const int cpu_id);
int access_x86_translate_check(PciDeviceIndex dev, int cpu_id);
int access_x86_translate_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data);
int access_x86_translate_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data);
int access_x86_translate_finalize(int cpu_id);


#endif /* ACCESS_X86_TRANSLATE_H */
