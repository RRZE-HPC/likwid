#ifndef HWFEATURES_COMMON_H
#define HWFEATURES_COMMON_H

#include <sysFeatures.h>

int register_features(_SysFeatureList *features, _SysFeatureList* in);
int sysFeatures_init_generic(_HWArchFeatures* infeatures, _SysFeatureList *list);

int _uint64_to_string(uint64_t value, char** str);
int _string_to_uint64(char* str, uint64_t* value);

#endif
