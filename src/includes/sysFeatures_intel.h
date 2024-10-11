#ifndef HWFEATURES_X86_INTEL_H
#define HWFEATURES_X86_INTEL_H

#include <sysFeatures_types.h>
#include <likwid.h>
#include <stdbool.h>

int intel_cpu_msr_register_getter(const LikwidDevice_t device, uint32_t reg, int bitoffset, int width, bool invert, char** value);
int intel_cpu_msr_register_setter(const LikwidDevice_t device, uint32_t reg, int bitoffset, int width, bool invert, const char* value);
int likwid_sysft_init_x86_intel(_SysFeatureList* list);

#endif
