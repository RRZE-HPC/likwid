#ifndef LIKWID_DEVICE_H
#define LIKWID_DEVICE_H

#include <likwid_device_types.h>

int likwid_device_create(LikwidDeviceType scope, int id, LikwidDevice_t* device) __attribute__ ((visibility ("default") ));
void likwid_device_destroy(LikwidDevice_t device) __attribute__ ((visibility ("default") ));

char* device_type_name(LikwidDeviceType type) __attribute__ ((visibility ("default") ));

#endif /* LIKWID_DEVICE_H */
