#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <sysFeatures_types.h>
#include <likwid.h>
#include <error.h>
#include <topology.h>

#include <bstrlib.h>
#include <bstrlib_helper.h>
#include <sysFeatures_cpufreq.h>
#include <sysFeatures_common.h>


static int cpufreq_sysfs_getter(const LikwidDevice_t device, char** value, const char* sysfs_filename)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfs_filename) || (device->type != DEVICE_TYPE_HWTHREAD))
    {
        return -EINVAL;
    }
    bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);
    if (!access(bdata(filename), R_OK))
    {
        bstring content = read_file(bdata(filename));
        if (blength(content) > 0)
        {
            btrimws(content);
            char* v = malloc((blength(content)+1) * sizeof(char));
            if (v)
            {
                strncpy(v, bdata(content), blength(content));
                v[blength(content)] = '\0';
                *value = v;
            }
            else
            {
                err = -ENOMEM;
            }
            bdestroy(content);
        }
    }
    else
    {
        err = errno;
    }
    bdestroy(filename);
    return err;
}

static int cpufreq_sysfs_setter(const LikwidDevice_t device, const char* value, const char* sysfs_filename)
{
    int err = 0;
    if ((!device) || (!value) || (!sysfs_filename) || (device->type != DEVICE_TYPE_HWTHREAD))
    {
        return -EINVAL;
    }
    int vallen = strlen(value);
    bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/%s", device->id.simple.id, sysfs_filename);
    errno = 0;
    if (!access(bdata(filename), W_OK))
    {
        FILE* fp = NULL;
        errno = 0;
        fp = fopen(bdata(filename), "w");
        if (fp == NULL) {
            err = -errno;
            ERROR_PRINT("Failed to open file '%s' for writing: %s", bdata(filename), strerror(errno))
        }
        else
        {
            int ret = fwrite(value, sizeof(char), vallen, fp);
            if (ret != (sizeof(char) * vallen))
            {
                ERROR_PRINT("Failed to open file '%s' for writing: %s", bdata(filename), strerror(errno))
            }
            fclose(fp);
        }
    }
    else
    {
        err = -errno;
    }
    bdestroy(filename);
    return err;
}


static int cpufreq_driver_test(const char* testgovernor)
{
    int err = 0;
    bstring btest = bfromcstr(testgovernor);
    topology_init();
    CpuTopology_t topo = get_cpuTopology();

    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = &topo->threadPool[i];
        if (t->inCpuSet)
        {
            bstring filename = bformat("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_driver", t->apicId);
            if (!access(bdata(filename), R_OK))
            {
                bstring content = read_file(bdata(filename));
                btrimws(content);
                err = (bstrncmp(content, btest, blength(btest)) == BSTR_OK);
                bdestroy(content);
            }
            bdestroy(filename);
            break;
        }
    }
    return err;
}

/* ACPI CPUfreq driver */

int cpufreq_acpi_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

int cpufreq_acpi_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

int cpufreq_acpi_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

int cpufreq_acpi_avail_cpu_freqs_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_frequencies");
}

int cpufreq_acpi_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

int cpufreq_acpi_governor_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_governor");
}

int cpufreq_acpi_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

int cpufreq_acpi_test()
{
    return cpufreq_driver_test("acpi_cpufreq");
}


/* Intel Pstate driver */

int cpufreq_intel_pstate_base_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    int err = cpufreq_sysfs_getter(device, value, "base_frequency");
    if (err == 0) return 0;
    return cpufreq_sysfs_getter(device, value, "bios_limit");
}

int cpufreq_intel_pstate_cur_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_cur_freq");
}

int cpufreq_intel_pstate_min_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_min_freq");
}

int cpufreq_intel_pstate_min_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_min_freq");
}

int cpufreq_intel_pstate_max_cpu_freq_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_max_freq");
}

int cpufreq_intel_pstate_max_cpu_freq_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_max_freq");
}

int cpufreq_intel_pstate_governor_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_governor");
}

int cpufreq_intel_pstate_governor_setter(const LikwidDevice_t device, const char* value)
{
    return cpufreq_sysfs_setter(device, value, "scaling_governor");
}

int cpufreq_intel_pstate_avail_governors_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "scaling_available_governors");
}

int cpufreq_intel_pstate_test()
{
    return cpufreq_driver_test("intel_pstate");
}


/* Energy Performance Preference */

int cpufreq_epp_test()
{
    if ((!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference", R_OK)) &&
        (!access("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_available_preferences", R_OK)))
    {
        return 1;
    }
    return 0;
}

int cpufreq_intel_pstate_epp_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_preference");
}

int cpufreq_intel_pstate_avail_epps_getter(const LikwidDevice_t device, char** value)
{
    return cpufreq_sysfs_getter(device, value, "energy_performance_available_preferences");
}

/* Init function */

int sysFeatures_init_cpufreq(_SysFeatureList* out)
{
    int err = 0;
    if (cpufreq_intel_pstate_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Intel Pstate knobs for cpufreq)
        err = register_features(out, &cpufreq_pstate_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    else if (cpufreq_acpi_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering ACPI cpufreq knobs for cpufreq)
        register_features(out, &cpufreq_acpi_feature_list);
        if (err < 0)
        {
            return err;
        }
    }

    if (cpufreq_epp_test())
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Registering Energy Performance Preference knobs for cpufreq)
        err = register_features(out, &cpufreq_epp_feature_list);
        if (err < 0)
        {
            return err;
        }
    }
    return 0;
}
