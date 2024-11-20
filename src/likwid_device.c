/*
 * =======================================================================================
 *
 *      Filename:  likwid_device.c
 *
 *      Description:  Interface for LIKWID's device handling
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Authors:  Thomas Gruber (tg), thomas.roehl@googlemail.com
 *                Michael Panzlaff, michael.panzlaff@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

#include <likwid.h>
#include <topology.h>
#include <access.h>
#include <error.h>

static int parse_pci_addr(const char *id, uint16_t *domain, uint8_t *bus, uint8_t *dev, uint8_t *func)
{
    /* Try to parse PCI address string of type: 00000000:00:00.0
     * Leading zeroes may be omitted in input string. */
    int err = 0;

    char *id_tokenized = strdup(id);
    if (!id_tokenized)
        return -ENOMEM;

    char *saveptr = NULL;

    const char *domain_token = strtok_r(id_tokenized, ":", &saveptr);
    const char *bus_token = strtok_r(NULL, ":", &saveptr);
    const char *dev_token = strtok_r(NULL, ".", &saveptr);
    const char *func_token = strtok_r(NULL, "", &saveptr);

    if (!domain_token || !bus_token || !dev_token || !func_token)
    {
        err = -EINVAL;
        goto cleanup;
    }

    char *endptr = NULL;
    unsigned long domain_ul = strtoul(domain_token, &endptr, 16);
    if (domain_token == endptr || *endptr != '\0')
    {
        err = -EINVAL;
        goto cleanup;
    }
    unsigned long bus_ul = strtoul(bus_token, &endptr, 16);
    if (bus_token == endptr || *endptr != '\0')
    {
        err = -EINVAL;
        goto cleanup;
    }
    unsigned long dev_ul = strtoul(dev_token, &endptr, 16);
    if (dev_token == endptr || *endptr != '\0')
    {
        err = -EINVAL;
        goto cleanup;
    }
    unsigned long func_ul = strtoul(func_token, &endptr, 16);
    if (func_token == endptr || *endptr != '\0')
    {
        err = -EINVAL;
        goto cleanup;
    }

    *domain = (uint16_t)domain_ul;
    *bus = (uint8_t)bus_ul;
    *dev = (uint8_t)dev_ul;
    *func = (uint8_t)func_ul;

cleanup:
    free(id_tokenized);
    return err;
}

static int device_create_simple(LikwidDeviceType type, int id, LikwidDevice_t* device)
{
    LikwidDevice_t dev = malloc(sizeof(_LikwidDevice));
    if (!dev)
        return -ENOMEM;

    dev->type = type;
    dev->id.simple.id = id;
    dev->internal_id = id;
    *device = dev;
    return 0;
}

static int device_create_pci(LikwidDeviceType type, int id, uint16_t domain, uint8_t bus, uint8_t dev, uint8_t func, LikwidDevice_t *device)
{
    int err = 0;
    LikwidDevice_t lw_dev = malloc(sizeof(_LikwidDevice));
    if (!lw_dev)
        return -ENOMEM;

    lw_dev->type = type;
    lw_dev->id.pci.pci_domain = domain;
    lw_dev->id.pci.pci_bus = bus;
    lw_dev->id.pci.pci_dev = dev;
    lw_dev->id.pci.pci_func = func;
    lw_dev->internal_id = id;
    *device = lw_dev;
    return 0;
}

static int device_create_hwthread(int id, LikwidDevice_t *device)
{
    CpuTopology_t topo = get_cpuTopology();

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = & topo->threadPool[i];
        if (t->apicId == (unsigned)id && t->inCpuSet)
        {
            return device_create_simple(DEVICE_TYPE_HWTHREAD, id, device);
        }
    }

    return -ENODEV;
}

static int device_create_node(int id, LikwidDevice_t *device)
{
    if (id != 0)
        return -ENODEV;

    return device_create_simple(DEVICE_TYPE_NODE, 0, device);
}

static int device_create_core(int id, LikwidDevice_t *device)
{
    CpuTopology_t topo = get_cpuTopology();

    if (id < 0 || (unsigned)id >= topo->numSockets * topo->numCoresPerSocket)
        return -ENODEV;

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = & topo->threadPool[i];
        if (t->coreId == (unsigned)id && t->inCpuSet)
            return device_create_simple(DEVICE_TYPE_CORE, id, device);
    }

    return -ENODEV;
}

static int device_create_socket(int id, LikwidDevice_t *device)
{
    CpuTopology_t topo = get_cpuTopology();

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = & topo->threadPool[i];
        if (t->packageId == (unsigned)id && t->inCpuSet)
            return device_create_simple(DEVICE_TYPE_SOCKET, id, device);
    }

    return -ENODEV;
}

static int device_create_numa(int id, LikwidDevice_t *device)
{
    int err = numa_init();
    if (err != 0)
        return err;

    NumaTopology_t numatopo = get_numaTopology();
    for (unsigned i = 0; i < numatopo->numberOfNodes; i++)
    {
        NumaNode* node = &numatopo->nodes[i];
        if (node->id == (unsigned)id)
            return device_create_simple(DEVICE_TYPE_NUMA, id, device);
    }

    return -ENODEV;
}

static int device_create_die(int id, LikwidDevice_t *device)
{
    CpuTopology_t topo = get_cpuTopology();

    for (unsigned i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = & topo->threadPool[i];
        if (t->dieId == (unsigned)id && t->inCpuSet)
            return device_create_simple(DEVICE_TYPE_DIE, id, device);
    }

    return -ENODEV;
}

#ifdef LIKWID_WITH_NVMON
static int device_create_nvgpu_by_index(int id, LikwidDevice_t *device)
{
    CudaTopology_t topo = get_cudaTopology();

    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].devid != id)
            continue;

        // TODO we do not have the function ID, but it should usually be 0
        return device_create_pci(DEVICE_TYPE_NVIDIA_GPU, topo->devices[i].devid, topo->devices[i].pciDom, topo->devices[i].pciBus, topo->devices[i].pciDev, 0, device);
    }

    return -ENODEV;
}

static int device_create_nvgpu_by_pciaddr(uint16_t dom, uint8_t bus, uint8_t dev, uint8_t func, LikwidDevice_t *device)
{
    CudaTopology_t topo = get_cudaTopology();

    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].pciDom != dom)
            continue;
        if (topo->devices[i].pciBus != bus)
            continue;
        if (topo->devices[i].pciDev != dev)
            continue;

        return device_create_pci(DEVICE_TYPE_NVIDIA_GPU, topo->devices[i].devid, dom, bus, dev, func, device);
    }

    return -ENODEV;
}
#endif

#ifdef LIKWID_WITH_ROCMON
static int device_create_amdgpu_by_index(int id, LikwidDevice_t *device)
{
    RocmTopology_t topo = get_rocmTopology();

    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].devid != id)
            continue;

        // TODO we do not have the function ID, but it should usually be 0
        return device_create_pci(DEVICE_TYPE_AMD_GPU, topo->devices[i].pciDom, topo->devices[i].pciBus, topo->devices[i].pciDev, 0, device);
    }

    return -ENODEV;
}

static int device_create_amdgpu_by_pciaddr(uint16_t dom, uint8_t bus, uint8_t dev, uint8_t func, LikwidDevice_t *device)
{
    RocmTopology_t topo = get_rocmTopology();

    for (int i = 0; i < topo->numDevices; i++)
    {
        if (topo->devices[i].pciDom != domain)
            continue;
        if (topo->devices[i].pciBus != bus)
            continue;
        if (topo->devices[i].pciDev != dev)
            continue;

        return device_create_pci(DEVICE_TYPE_AMD_GPU, domain, bus, dev, func, device);
    }

    return -ENODEV;
}
#endif

static int topo_init(void)
{
    int err = topology_init();
    if (err < 0)
        return err;

    err = numa_init();
    if (err < 0)
        return err;

#ifdef LIKWID_WITH_NVMON
    /* The topology init functions currently return both positive and negative
     * error numbers :-/, so use this workaround for now. */
    err = topology_cuda_init();
    if (err != 0)
        return -EPERM;
#endif
#ifdef LIKWID_WITH_ROCMON
    err = topology_rocm_init();
    if (err != 0)
        return -EPERM;
#endif
    return 0;
}

int likwid_device_create(LikwidDeviceType type, int id, LikwidDevice_t* device)
{
    if ((type <= DEVICE_TYPE_INVALID) || (type >= MAX_DEVICE_TYPE) || (id < 0) || (!device))
        return -EINVAL;

    int err = topo_init();
    if (err < 0)
        return err;

    switch (type) {
        case DEVICE_TYPE_HWTHREAD:
            return device_create_hwthread(id, device);
        case DEVICE_TYPE_NODE:
            return device_create_node(id, device);
        case DEVICE_TYPE_CORE:
            return device_create_core(id, device);
        case DEVICE_TYPE_SOCKET:
            return device_create_socket(id, device);
        case DEVICE_TYPE_NUMA:
            return device_create_numa(id, device);
        case DEVICE_TYPE_DIE:
            return device_create_die(id, device);
#ifdef LIKWID_WITH_NVMON
        case DEVICE_TYPE_NVIDIA_GPU:
            return device_create_nvgpu_by_index(id, device);
#endif
#ifdef LIKWID_WITH_ROCMON
        case DEVICE_TYPE_AMD_GPU:
            return device_create_amdgpu_by_index(id, device);
#endif
        default:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Unimplemented device type: %d, type);
            break;
    }
    return -ENODEV;
}

int likwid_device_create_from_string(LikwidDeviceType type, const char *id, LikwidDevice_t *device)
{
    if ((type < DEVICE_TYPE_INVALID) || (type >= MAX_DEVICE_TYPE) || (!id) || (!device))
        return -EINVAL;

    int err = topo_init();
    if (err < 0)
        return err;

    char *tokenized_string = strdup(id);
    if (!tokenized_string)
        return -ENOMEM;

    char *saveptr = NULL;
    const char *type_token = strtok_r(tokenized_string, "=", &saveptr);
    const char *id_token = strtok_r(NULL, "", &saveptr);
    if (id_token)
    {
        /* id string looks like this TYPE_NAME-TYPE_ID (type name and value combined) */
        type = DEVICE_TYPE_INVALID;

        for (int i = 0; i < MAX_DEVICE_TYPE; i++)
        {
            if (strcmp(type_token, LikwidDeviceTypeNames[i]) == 0)
            {
                type = i;
                break;
            }
        }

        if (type == DEVICE_TYPE_INVALID)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Cannot create device from string type: %s, type_token);
            err = -EINVAL;
            goto cleanup;
        }
    }
    else
    {
        /* id string looks like this: TYPE_ID (just the ID value) */
        id_token = tokenized_string;

        /* This allows an ID string of just "node", which should imply node 0 */
        if (strcmp(tokenized_string, "node") == 0)
            type = DEVICE_TYPE_NODE;
    }

    /* Parsing the PCI address is stricly only necessary for GPUs, but we only
     * do it once here for simplicity. */
    uint16_t dom;
    uint8_t bus, dev, func;

    char *endptr;
    long long_id;

    if (parse_pci_addr(id_token, &dom, &bus, &dev, &func) == 0)
    {
        switch (type)
        {
#ifdef LIKWID_WITH_NVMON
            case DEVICE_TYPE_NVIDIA_GPU:
                err = device_create_nvgpu_by_pciaddr(dom, bus, dev, func, device);
                break;
#endif
#ifdef LIKWID_WITH_ROCMON
            case DEVICE_TYPE_NVIDIA_GPU:
                err = device_create_amdgpu_by_pciaddr(dom, bus, dev, func, device);
                break;
#endif
            default:
                err = -EINVAL;
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Unable to use PCI address to create device type: %d, type);
                break;
        }
    }
    else if (long_id = strtol(id_token, &endptr, 0), id_token != endptr && *endptr == '\0')
    {
        const int int_id = (int)long_id;
        switch (type)
        {
            case DEVICE_TYPE_HWTHREAD:
                err = device_create_hwthread(int_id, device);
                break;
            case DEVICE_TYPE_NODE:
                err = device_create_node(int_id, device);
                break;
            case DEVICE_TYPE_CORE:
                err = device_create_core(int_id, device);
                break;
            case DEVICE_TYPE_SOCKET:
                err = device_create_socket(int_id, device);
                break;
            case DEVICE_TYPE_NUMA:
                err = device_create_numa(int_id, device);
                break;
            case DEVICE_TYPE_DIE:
                err = device_create_die(int_id, device);
                break;
#ifdef LIKWID_WITH_NVMON
            case DEVICE_TYPE_NVIDIA_GPU:
                err = device_create_nvgpu_by_index(int_id, device);
                break;
#endif
#ifdef LIKWID_WITH_ROCMON
            case DEVICE_TYPE_AMD_GPU:
                err = device_create_amdgpu_by_index(id, device);
                break;
#endif
            default:
                err = -EPERM;
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Unimplemented device type: %d, type);
                break;
        }
    }
    else if (type == DEVICE_TYPE_NODE)
    {
        /* This allows an ID string of just "node", which should imply node 0 */
        err = device_create_node(0, device);
    }
    else
    {
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Unable to parse '%s' as valid PCI address or integer, id_token);
        err = -EINVAL;
    }

cleanup:
    free(tokenized_string);
    return err;
}

void likwid_device_destroy(LikwidDevice_t device)
{
    if (device)
    {
        free(device);
        device = NULL;
    }
}

const char *likwid_device_type_name(LikwidDeviceType type)
{
    if (type < DEVICE_TYPE_INVALID || type >= MAX_DEVICE_TYPE)
        return "unsupported";
    return LikwidDeviceTypeNames[type];
}

void likwid_device_fmt_pci(char *buf, size_t size, LikwidDevice_t device)
{
    const uint16_t dom = device->id.pci.pci_domain;
    const uint8_t bus = device->id.pci.pci_bus;
    const uint8_t dev = device->id.pci.pci_dev;
    const uint8_t func = device->id.pci.pci_func;
    snprintf(buf, size, "%08x:%02x:%02x.%01x", dom, bus, dev, func);
}

static bool device_in_cpuset(LikwidDeviceType type, size_t id)
{
#ifdef LIKWID_WITH_NVMON
    assert(type != DEVICE_TYPE_NVIDIA_GPU);
#endif
#ifdef LIKWID_WITH_ROCMON
    assert(type != DEVICE_TYPE_AMD_GPU);
#endif

    CpuTopology_t cpu_topo = get_cpuTopology();
    NumaTopology_t numa_topo = get_numaTopology();

    for (size_t i = 0; i < cpu_topo->numHWThreads; i++)
    {
        const HWThread* t = &cpu_topo->threadPool[i];
        switch (type)
        {
            case DEVICE_TYPE_HWTHREAD:
                if (t->apicId == id)
                    return t->inCpuSet;
                break;
            case DEVICE_TYPE_CORE:
                if (t->coreId == id && t->inCpuSet)
                    return true;
                break;
            case DEVICE_TYPE_NUMA:
                assert(id < numa_topo->numberOfNodes);
                const NumaNode *n = &numa_topo->nodes[id];
                for (size_t j = 0; j < n->numberOfProcessors; j++)
                {
                    if (n->processors[j] == t->apicId && t->inCpuSet)
                        return true;
                }
                break;
            case DEVICE_TYPE_DIE:
                if (t->dieId == id && t->inCpuSet)
                    return true;
                break;
            case DEVICE_TYPE_SOCKET:
                if (t->packageId == id && t->inCpuSet)
                    return true;
                break;
            case DEVICE_TYPE_NODE:
                return true;
            default:
                DEBUG_PRINT(DEBUGLEV_DEVELOP, Unimplemented device type: %d, type);
                return false;
        }
    }

    return false;
}

static int likwid_device_get_list(LikwidDeviceType type, char ***id_list, size_t *id_list_count, bool cpuset_only)
{
    if (type <= DEVICE_TYPE_INVALID || type >= MAX_DEVICE_TYPE || !id_list)
        return -EINVAL;

    int err = topo_init();
    if (err < 0)
        return err;

    CpuTopology_t cpu_topo = get_cpuTopology();
    NumaTopology_t numa_topo = get_numaTopology();
#ifdef LIKWID_WITH_NVMON
    CudaTopology_t cuda_topo = get_cudaTopology();
#endif
#ifdef LIKWID_WITH_ROCMON
    RocmTopology_t rocm_topo = get_rocmTopology();
#endif

    size_t id_count = 0;

    switch (type)
    {
        case DEVICE_TYPE_HWTHREAD:
            id_count = cpu_topo->numHWThreads;
            break;
        case DEVICE_TYPE_CORE:
            id_count = cpu_topo->numCoresPerSocket * cpu_topo->numSockets;
            break;
        case DEVICE_TYPE_NUMA:
            id_count = numa_topo->numberOfNodes;
            break;
        case DEVICE_TYPE_DIE:
            id_count = cpu_topo->numDies;
            break;
        case DEVICE_TYPE_SOCKET:
            id_count = cpu_topo->numSockets;
            break;
        case DEVICE_TYPE_NODE:
            id_count = 1;
            break;
#ifdef LIKWID_WITH_NVMON
        case DEVICE_TYPE_NVIDIA_GPU:
            // no need to set id_prefix, hardcoded below
            id_count = cuda_topo->numDevices;
            break;
#endif
#ifdef LIKWID_WITH_ROCMON
        case DEVICE_TYPE_AMD_GPU:
            // no need to set id_prefix, hardcoded below
            id_count = rocm_topo->numDevices;
            break;
#endif
        default:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Unimplemented device type: %d, type);
            return -EINVAL;
    }

    char **name_list = calloc(id_count, sizeof(name_list[0]));
    if (!name_list)
        return -errno;

    for (size_t i = 0; i < id_count; i++)
    {
        char id_str[64];
        switch (type)
        {
#ifdef LIKWID_WITH_NVMON
            case DEVICE_TYPE_NVIDIA_GPU:
                snprintf(id_str, sizeof(id_str), "%08x:%02x:%02x.%01x",
                        cuda_topo->devices[i].pciDom,
                        cuda_topo->devices[i].pciBus,
                        cuda_topo->devices[i].pciDev,
                        cuda_topo->devices[i].pciFunc);
                break;
#endif
#ifdef LIKWID_WITH_ROCMON
            case DEVICE_TYPE_AMD_GPU:
                snprintf(id_str, sizeof(id_str), "%08x:%02x:%02x.0",
                        rocm_topo->devices[i].pciDom,
                        rocm_topo->devices[i].pciBus,
                        rocm_topo->devices[i].pciDev);
                break;
#endif
            default:
                if (cpuset_only && !device_in_cpuset(type, i))
                    continue;
                snprintf(id_str, sizeof(id_str), "%zu", i);
                break;
        }

        name_list[i] = strdup(id_str);
        if (!name_list[i])
        {
            err = -errno;
            break;
        }
    }

    if (err < 0)
    {
        for (size_t i = 0; i < id_count; i++)
            free(name_list[i]);
        free(name_list);
        return err;
    }

    if (cpuset_only)
    {
        /* Shrink list again and remove all entries which were left blank. */
        size_t write_index = 0;
        for (size_t read_index = 0; read_index < id_count; read_index++)
        {
            if (!name_list[read_index])
                continue;

            name_list[write_index++] = name_list[read_index];
        }

        for (size_t i = write_index; i < id_count; i++)
            name_list[i] = NULL;

        char **shrink_name_list = realloc(name_list, write_index * sizeof(name_list[0]));
        if (!shrink_name_list)
        {
            for (size_t i = 0; i < id_count; i++)
                free(name_list[i]);
            free(name_list);
            return -errno;
        }

        name_list = shrink_name_list;
        id_count = write_index;
    }

    *id_list = name_list;
    *id_list_count = id_count;
    return 0;
}

int likwid_device_get_available(LikwidDeviceType type, char ***id_list, size_t *id_list_count)
{
    return likwid_device_get_list(type, id_list, id_list_count, true);
}

int likwid_device_get_all(LikwidDeviceType type, char ***id_list, size_t *id_list_count)
{
    return likwid_device_get_list(type, id_list, id_list_count, false);
}
