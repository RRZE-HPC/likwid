#ifndef HWFEATURES_COMMON_H
#define HWFEATURES_COMMON_H

#include <sysFeatures.h>

int register_features(_SysFeatureList *features, const _SysFeatureList* in);
int sysFeatures_init_generic(const _HWArchFeatures* infeatures, _SysFeatureList *list);

int sysFeatures_uint64_to_string(uint64_t value, char** str);
int sysFeatures_string_to_uint64(const char* str, uint64_t* value);
int sysFeatures_double_to_string(double value, char **str);
int sysFeatures_string_to_double(const char* str, double *value);

int likwid_sysft_foreach_hwt_testmsr(uint64_t reg);
int likwid_sysft_foreach_socket_testmsr(uint64_t reg);
int likwid_sysft_readmsr(const LikwidDevice_t device, uint64_t reg, uint64_t *msrData);

#endif
