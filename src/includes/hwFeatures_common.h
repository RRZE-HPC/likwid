#ifndef HWFEATURES_COMMON_H
#define HWFEATURES_COMMON_H

#include <hwFeatures.h>

int register_features(_HWFeatureList *features, _HWFeatureList* in);
int hwFeatures_init_generic(_HWArchFeatures* infeatures, _HWFeatureList *list);

int _uint64_to_string(uint64_t value, char** str);
int _string_to_uint64(char* str, uint64_t* value);

#endif
