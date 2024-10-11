#ifndef HWFEATURES_TYPES_H
#define HWFEATURES_TYPES_H

#include <stdint.h>
#include <likwid.h>
#include <likwid_device.h>

typedef enum {
    HWFEATURES_TYPE_UINT64 = 0,
    HWFEATURES_TYPE_DOUBLE,
    HWFEATURES_TYPE_STRING
} HWFEATURES_VALUE_TYPES;


typedef int (*hwfeature_getter_function)(const LikwidDevice_t device, char** value);
typedef int (*hwfeature_setter_function)(const LikwidDevice_t device, const char* value);
typedef int (*hwfeature_test_function)();

typedef struct {
    const char* name;
    const char* category;
    const char* description;
    hwfeature_getter_function getter;
    hwfeature_setter_function setter;
    LikwidDeviceType type;
    hwfeature_test_function tester;
    const char* unit;
} _SysFeature;

typedef struct {
    int num_features;
    _SysFeature* features;
    hwfeature_test_function tester;
} _SysFeatureList;

typedef struct {
    int family;
    int model;
    const _SysFeatureList** features;
    int max_stepping;
} _HWArchFeatures;

#define IS_VALID_DEVICE_TYPE(scope) (((scope) >= MIN_DEVICE_TYPE) && ((scope) < MAX_DEVICE_TYPE))


#endif /* HWFEATURES_TYPES_H */
