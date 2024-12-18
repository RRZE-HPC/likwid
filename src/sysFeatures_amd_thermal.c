#include <sysFeatures_amd_thermal.h>

#include <assert.h>
#include <dirent.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>

#include <bstrlib.h>
#include <error.h>
#include <sysFeatures_common.h>
#include <types.h>

struct sysfs_ccd {
    /* e.g. /sys/bus/pci/drivers/k10temp/0000:12:3.4/hwmon/hwmon5/temp3_input */
    bstring temp_path;
    /* e.g. Tccd8 */
    bstring label;
};

struct sysfs_socket {
    /* 'socket' may be slightly misleading, as we actually refer to a PCI device.
     * However, we will later associate each socket with one PCI device. */
    struct sysfs_ccd *ccds;
    size_t count;
    /* e.g. /sys/bus/pci/drivers/k10temp/0000:12:3.4 */
    bstring pci_path;
    /* e.g. /sys/bus/pci/drivers/k10temp/0000:12:3.4/hwmon/hwmon5 */
    bstring hwmon_path;
    /* e.g. /sys/bus/pci/drivers/k10temp/0000:12:3.4/hwmon/hwmon5/temp1_label */
    bstring temp_path;
    /* e.g. Tctl */
    bstring label;
};

struct sysfs_info {
    struct sysfs_socket *sockets;
    size_t count;
    bool init;
} info;

__attribute__((destructor)) static void free_paths(void)
{
    for (size_t i = 0; i < info.count; i++)
    {
        struct sysfs_socket *socket = &info.sockets[i];

        for (size_t j = 0; j < socket->count; j++)
        {
            struct sysfs_ccd *ccd = &socket->ccds[j];

            bdestroy(ccd->temp_path);
            bdestroy(ccd->label);
        }

        free(socket->ccds);
        bdestroy(socket->pci_path);
        bdestroy(socket->hwmon_path);
        bdestroy(socket->temp_path);
        bdestroy(socket->label);
    }

    free(info.sockets);
    memset(&info, 0, sizeof(info));
}

static int read_sysfs_file(const char *path, char *buf, size_t size)
{
    if (size == 0 || !buf || !path)
        return -EINVAL;

    FILE *file = fopen(path, "r");
    if (!file)
        return -errno;
    const size_t len = fread(buf, sizeof(buf[0]), size - 1, file);
    buf[len] = '\0';
    fclose(file);
    return 0;
}

static int ccd_sort(const void *a, const void *b)
{
    const struct sysfs_ccd *ca = a;
    const struct sysfs_ccd *cb = b;

    const char *sa = bdata(ca->label);
    const char *sb = bdata(cb->label);

    static const size_t digit_pos = strlen("Tccd");
    assert(strncmp(sa, "Tccd", digit_pos) == 0);
    assert(strncmp(sb, "Tccd", digit_pos) == 0);

    const int ia = atoi(&sa[digit_pos]);
    const int ib = atoi(&sb[digit_pos]);
    if (ia < ib)
        return -1;
    if (ia > ib)
        return 1;
    return 0;
}

static int socket_sort(const void *a, const void *b)
{
    const struct sysfs_socket *sa = a;
    const struct sysfs_socket *sb = b;
    return bstrcmp(sa->pci_path, sb->pci_path);
}

static int create_paths(void)
{
    if (info.init)
        return 0;

    /* Enumerate PCI devices associated with k10temp driver. */
    const char *k10temp_base = "/sys/bus/pci/drivers/k10temp";
    DIR *k10temp_dir = opendir(k10temp_base);
    if (!k10temp_dir)
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, "%s not found. Not initializing k10temp", k10temp_base);
        return -errno;
    }

    struct dirent *pcidevice_file;
    while (errno = 0, (pcidevice_file = readdir(k10temp_dir)))
    {
        char d_name_tokenized[sizeof(pcidevice_file->d_name)];
        snprintf(d_name_tokenized, sizeof(d_name_tokenized), "%s", pcidevice_file->d_name); // <-- manual strlcpy
        /* Read all entries in the k10temp directory.
         * Find all entries, which look like a PCI address in order to determine
         * which devices are associated with the k10temp driver.
         * A PCI address looks like 0000:00:00:0 */
        char *saveptr = NULL;
        const char *domain = strtok_r(d_name_tokenized, ":", &saveptr);
        const char *bus = strtok_r(NULL, ":", &saveptr);
        const char *dev = strtok_r(NULL, ".", &saveptr);
        const char *func = strtok_r(NULL, "", &saveptr);

        /* Incase not all tokens are in the file name, just continue to next file. */
        if (!domain || !bus || !dev || !func)
            continue;

        void *new_sockets = realloc(info.sockets, (info.count + 1) * sizeof(info.sockets[0]));
        if (!new_sockets)
            break;

        info.sockets = new_sockets;
        struct sysfs_socket *s = &info.sockets[info.count];
        info.count += 1;

        memset(s, 0, sizeof(*s));
        s->pci_path = bformat("%s/%s", k10temp_base, pcidevice_file->d_name);
        if (!s->pci_path)
        {
            errno = ENOMEM;
            break;
        }
    }

    if (errno != 0)
    {
        const int errno_save = errno;
        closedir(k10temp_dir);
        free_paths();
        return -errno_save;
    }

    closedir(k10temp_dir);

    /* Sort enumerated PCI devices in ascending order.
     *
     * IMPORTANT: This is an attempt to hopefully match the device with
     * the lowest device ID to socket 0, and so on.
     * There is no guarantee this is actually correct, but we otherwise do not
     * have the ability to know which socket the temperature PCI device belongs to. */
    qsort(info.sockets, info.count, sizeof(info.sockets[0]), socket_sort);

    /* Populate hwmon_path for each socket. */
    for (size_t socket_id = 0; socket_id < info.count; socket_id++)
    {
        /* Determine hwmon path */
        struct sysfs_socket *s = &info.sockets[socket_id];

        char hwmon_base[PATH_MAX];
        snprintf(hwmon_base, sizeof(hwmon_base), "%s/hwmon", bdata(s->pci_path));
        DIR *hwmon_base_dir = opendir(hwmon_base);
        if (!hwmon_base_dir)
        {
            const int errno_save = errno;
            DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "k10temp: Unable to read dir %s", hwmon_base);
            free_paths();
            return -errno_save;
        }

        struct dirent *hwmon_candidate_dir;
        while (errno = 0, (hwmon_candidate_dir = readdir(hwmon_base_dir)))
        {
            /* only allow hwmon subdirectories */
            if (strncmp(hwmon_candidate_dir->d_name, "hwmon", 5) != 0)
                continue;

            /* Check if the current dirent is actually a directory */
            char hwmon_candidate_path[PATH_MAX];
            snprintf(
                hwmon_candidate_path,
                sizeof(hwmon_candidate_path),
                "%s/%s",
                bdata(s->hwmon_path), 
                hwmon_candidate_dir->d_name
            );

            s->hwmon_path = bformat("%s/%s", hwmon_base, hwmon_candidate_dir->d_name);
            if (!s->hwmon_path)
                errno = ENOMEM;
            break;
        }

        if (errno != 0)
        {
            const int errno_save = errno;
            closedir(hwmon_base_dir);
            free_paths();
            return -errno_save;
        }

        closedir(hwmon_base_dir);

        /* Crawl hwmon dir for tempX_input and tempX_label and populate arrays accordingly. */
        DIR *hwmon_dir = opendir(bdata(s->hwmon_path));
        if (!hwmon_dir)
        {
            const int errno_save = errno;
            DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "k10temp: Unable to read dir %s", bdata(s->hwmon_path));
            free_paths();
            return -errno_save;
        }

        struct dirent *temp_dirent;
        while (errno = 0, (temp_dirent = readdir(hwmon_dir)))
        {
            /* check if file name is of form  'temp\d+_label'. */
            if (strncmp(temp_dirent->d_name, "temp", strlen("temp")) != 0)
                continue;

            const char *numstart = &temp_dirent->d_name[strlen("temp")];
            char *numend;
            errno = 0;
            unsigned long no = strtoul(numstart, &numend, 10);
            if (numstart == numend || errno != 0)
                continue;

            if (strcmp(numend, "_label") != 0)
                continue;

            /* Okay, now we have made sure out file name is of the right format.
             * Next, we read the label name from the file to decide if it's a CCD
             * temperature or a Tctl temperature. */

            /* temporarily store path to e.g. /sys/...../temp3_label */
            char label_path[PATH_MAX];
            snprintf(label_path, sizeof(label_path), "%s/%s", bdata(s->hwmon_path), temp_dirent->d_name);

            /* read e.g. temp3_label to string and store it. */
            char label_string[64];
            int err = read_sysfs_file(label_path, label_string, sizeof(label_string));
            if (err < 0)
            {
                errno = -err;
                break;
            }

            /* temporarilty store path to e.g. /sys/...../temp3_input */
            char temp_path[PATH_MAX];
            snprintf(temp_path, sizeof(temp_path), "%s/temp%lu_input", bdata(s->hwmon_path), no);

            /* We now have to differentiate between CCD temeprature sensors and CTL temperature
             * sensors. The CCD temperatures are stored in the 'ccds' array, while we should
             * hopefully only find a single Tctl temperature. The latter one will be stored
             * only once per socket. */
            if (strncmp(label_string, "Tccd", strlen("Tccd")) == 0)
            {
                void *new_ccds = realloc(s->ccds, (s->count + 1) * sizeof(s->ccds[0]));
                if (!new_ccds)
                    break;

                s->ccds = new_ccds;
                struct sysfs_ccd *ccd = &s->ccds[s->count];
                s->count += 1;

                memset(ccd, 0, sizeof(*ccd));
                ccd->temp_path = bfromcstr(temp_path);
                ccd->label = bfromcstr(label_string);
                if (!ccd->temp_path || !ccd->label)
                {
                    errno = ENOMEM;
                    break;
                }
            } else {
                if (s->label)
                {
                    /* If s->label is alreay set, we have encountered more then one non-CCD temperature.
                     * We only support one sensors per socket, so issue a warning but continue regardless. */
                    DEBUG_PRINT(DEBUGLEV_ONLY_ERROR, "Found more than one non-Tccd. current=%s new=%s", bdata(s->label), label_string);
                    bdestroy(s->label);
                    bdestroy(s->temp_path);
                }

                s->label = bfromcstr(label_string);
                s->temp_path = bfromcstr(temp_path);
                if (!s->label || !s->temp_path)
                {
                    errno = ENOMEM;
                    break;
                }
            }
        }

        if (errno != 0)
        {
            const int errno_save = errno;
            closedir(hwmon_dir);
            free_paths();
            return -errno_save;
        }

        closedir(hwmon_dir);

        /* Fail if no socket sensor has been found. CCD sensors are not mandatory, since
         * some CPUs do not have any. */
        if (!s->label)
            return -ENODEV;

        /* Last, we have to sort each CCD entry according to their label. */
        qsort(s->ccds, s->count, sizeof(s->ccds[0]), ccd_sort);
    }

    info.init = true;
    return 0;
}

static int temp_getter(const char *file, char **value)
{
    /* read temperature */
    char temp_data[64];
    int err = read_sysfs_file(file, temp_data, sizeof(temp_data));
    if (err < 0)
        return err;

    /* parse to value */
    char *endptr;
    errno = 0;
    long temp = strtol(temp_data, &endptr, 10);
    if (temp_data == endptr)
        return -EIO;
    if (errno != 0)
        return -errno;

    return likwid_sysft_double_to_string((double)temp / 1000.0, value);
}

static int amd_thermal_temperature_ccd_getter(LikwidDevice_t device, char **value)
{
    int err = create_paths();
    if (err < 0)
        return err;

    err = topology_init();
    if (err < 0)
        return err;

    /* determine CCD to read from */
    CpuTopology_t topo = get_cpuTopology();

    const uint32_t local_die_id = device->id.simple.id % (topo->numDies / topo->numSockets);
    const uint32_t socket_id = device->id.simple.id / (topo->numDies / topo->numSockets);

    if (socket_id >= info.count)
        return -EINVAL;

    if (local_die_id >= info.sockets[socket_id].count)
        return -EINVAL;

    return temp_getter(bdata(info.sockets[socket_id].ccds[local_die_id].temp_path), value);
}

static int amd_thermal_temperature_ctl_getter(LikwidDevice_t device, char **value)
{
    int err = create_paths();
    if (err < 0)
        return err;

    if (device->id.simple.id >= info.count)
        return -EINVAL;

    return temp_getter(bdata(info.sockets[device->id.simple.id].temp_path), value);
}

static int amd_thermal_tester(void)
{
    int err = create_paths();
    if (err < 0)
        return 0;

    /* We need at least one socket in order to detect the thermal sensor as valid. */
    return info.count > 0;
}

static _SysFeature amd_thermal_features[] = {
    {"ccd_temp", "thermal", "Current CPU CCD temperature (Tccd)", amd_thermal_temperature_ccd_getter, NULL, DEVICE_TYPE_DIE, NULL, "degrees C"},
    {"pkg_temp", "thermal", "Current CPU socket temperature (Tctl)", amd_thermal_temperature_ctl_getter, NULL, DEVICE_TYPE_SOCKET, NULL, "degrees C"},
};

const _SysFeatureList likwid_sysft_amd_k10_cpu_thermal_feature_list = {
    .num_features = ARRAY_COUNT(amd_thermal_features),
    .tester = amd_thermal_tester,
    .features = amd_thermal_features,
};
