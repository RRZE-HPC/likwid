#ifndef DEVSTRING_H
#define DEVSTRING_H

#include <stddef.h>

#include <likwid.h>

int likwid_devstr_to_devlist(const char *str, LikwidDeviceList_t *dev_list);

#endif // DEVSTRING_H
