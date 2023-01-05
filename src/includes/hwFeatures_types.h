#ifndef HWFEATURES_TYPES_H
#define HWFEATURES_TYPES_H

#include <stdint.h>


typedef int (*hwfeature_getter_function)(int hwthread, uint64_t *value);
typedef int (*hwfeature_setter_function)(int hwthread, uint64_t value);

typedef struct {
    char* name;
    char* description;
    hwfeature_getter_function getter;
    hwfeature_setter_function setter;
    HWFeatureScope scope;
} _HWFeature;

typedef struct {
    int num_features;
    _HWFeature* features;
} _HWFeatureList;

typedef struct {
    int family;
    int model;
    _HWFeatureList** features;
} _HWArchFeatures;

#endif /* HWFEATURES_TYPES_H */
