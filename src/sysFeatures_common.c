#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <registers.h>
#include <cpuid.h>
#include <pci_types.h>
#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <sysFeatures_common.h>

int register_features(_SysFeatureList *features, const _SysFeatureList* in)
{
    if (in->tester)
    {
        if (!in->tester())
        {
            return -ENOTSUP;
        }
    }
    for (int i = 0; i < in->num_features; i++)
    {
        _SysFeature *f = &in->features[i];
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering feature %s.%s, f->category, f->name);
        if (f->tester)
        {
            if (f->tester())
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Running test for feature %s.%s, f->category, f->name);
                int err = _add_to_feature_list(features, f);
                if (err < 0)
                {
                    ERROR_PRINT(Failed to add HW feature %s.%s to feature list, f->category, f->name);
                }
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Test function for feature %s.%s failed, f->category, f->name);
            }
        }
        else
        {
            int err = _add_to_feature_list(features, f);
            if (err < 0)
            {
                ERROR_PRINT(Failed to add HW feature %s.%s to feature list, f->category, f->name);
            }
        }
    }
    return 0;
}

int sysFeatures_init_generic(const _HWArchFeatures* infeatures, _SysFeatureList *list)
{
    int err = topology_init();
    if (err < 0)
    {
        ERROR_PRINT(Failed to initialize topology module);
        return err;
    }
    CpuInfo_t cpuinfo = get_cpuInfo();

    _SysFeatureList** feature_list = NULL;
    for (unsigned c = 0; infeatures[c].family >= 0 && infeatures[c].model >= 0; c++)
    {
        if ((unsigned)infeatures[c].family == cpuinfo->family && (unsigned)infeatures[c].model == cpuinfo->model)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Using feature list for CPU family 0x%X and model 0x%X, cpuinfo->family, cpuinfo->model);
            feature_list = infeatures[c].features;
            break;
        }
    }
    if (!feature_list)
    {
        ERROR_PRINT(No feature list found for current architecture);
        return -ENOTSUP;
    }

    for (unsigned j = 0; feature_list[j] != NULL; j++)
    {
        register_features(list, feature_list[j]);
    }
    return 0;
}

int sysFeatures_uint64_to_string(uint64_t value, char** str)
{
    char s[HWFEATURES_MAX_STR_LENGTH];
    const int len = snprintf(s, sizeof(s), "%llu", value);
    if (len < 0)
    {
        ERROR_PRINT(Conversion of uint64_t %lld failed: %s, value, strerror(errno));
        return -errno;
    }
    char *newstr = realloc(*str, len+1);
    if (!newstr)
    {
        return -ENOMEM;
    }
    *str = newstr;
    strcpy(*str, s);
    return 0;
}

int sysFeatures_string_to_uint64(const char* str, uint64_t* value)
{
    char* ptr = NULL;
    if ((strncmp(str, "true", 4) == 0) || (strncmp(str, "TRUE", 4) == 0))
    {
        *value = 0x1ULL;
        return 0;
    }
    else if ((strncmp(str, "false", 5) == 0) || (strncmp(str, "FALSE", 5) == 0))
    {
        *value = 0x0ULL;
        return 0;
    }
    errno = 0;
    uint64_t v = strtoull(str, &ptr, 10);
    if (v == 0 && errno != 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Conversion of string '%s' to uint64_t failed %d: %s, str, v, strerror(errno));
        return -errno;
    }
    *value = v;
    return 0;
}

int sysFeatures_double_to_string(double value, char **str)
{
    char s[HWFEATURES_MAX_STR_LENGTH];
    const int len = snprintf(s, sizeof(s), "%f", value);
    if (len < 0)
    {
        ERROR_PRINT(Conversion of double %f failed: %s, value, strerror(errno));
        return -errno;
    }
    char* newstr = realloc(*str, len+1);
    if (!newstr)
    {
        return -ENOMEM;
    }
    *str = newstr;
    strcpy(*str, s);
    return 0;
}

int sysFeatures_string_to_double(const char* str, double *value)
{
    errno = 0;
    char* endptr = NULL;
    const double result = strtod(str, &endptr);
    if (!endptr)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Conversion of string '%s' to double failed: %s, str, strerror(errno));
        return -errno;
    }
    if (errno != 0)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Conversion of string '%s' to double failed: %s, str, result, strerror(errno));
        return -errno;
    }
    *value = result;
    return 0;
}
