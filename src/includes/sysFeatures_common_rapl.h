#ifndef HWFEATURES_COMMON_RAPL_H
#define HWFEATURES_COMMON_RAPL_H

#include <likwid.h>

typedef struct {
    double powerUnit;   // unit W
    double energyUnit;  // unit J
    double timeUnit;    // unit s
} RaplDomainInfo;

static inline double timeWindow_to_seconds(const RaplDomainInfo *info, uint64_t timeWindow)
{
    /* see Intel SDM: Package RAPL Domain: MSR_PKG_POWER_LIMIT */
    const uint64_t y = field64(timeWindow, 0, 5);
    const uint64_t z = field64(timeWindow, 5, 2);
    return (1 << y) * (1.0 + z / 4.0) * info->timeUnit;
}

static inline uint64_t seconds_to_timeWindow(const RaplDomainInfo *info, double seconds)
{
    /* see Intel SDM: Package RAPL Domain: MSR_PKG_POWER_LIMIT */
    const uint64_t timeInHwTime = (uint64_t)(seconds / info->timeUnit);
    uint64_t y = (uint64_t)(log2(timeInHwTime));
    if (y > 0x1F)
        y = 0x7F;
    const uint64_t o = (1 << y);
    const uint64_t z = (4 * (timeInHwTime - o)) / o;
    return (y & 0x1F) | ((z & 0x3) << 5);
}

static inline int getset_check(const LikwidDevice_t device, const void *value, LikwidDeviceType type)
{
    if (!device || !value || (device->type != type))
    {
        return -EINVAL;
    }
    return 0;
}

static inline int getset_unusedinfo_check(const LikwidDevice_t device, const void *value, const RaplDomainInfo *info, LikwidDeviceType type)
{
    (void)info;
    return getset_check(device, value, type);
}

static inline int getset_info_check(const LikwidDevice_t device, const void *value, const RaplDomainInfo *info, LikwidDeviceType type)
{
    if (!info)
    {
        return -EINVAL;
    }
    return getset_unusedinfo_check(device, value, info, type);
}

#endif /* HWFEATURES_COMMON_RAPL_H */
