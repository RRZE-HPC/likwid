#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <likwid.h>
#include <topology.h>
#include <access.h>
#include <error.h>



static int likwid_device_create_simple(LikwidDeviceType type, int id, LikwidDevice_t* device)
{
    LikwidDevice_t dev = malloc(sizeof(_LikwidDevice));
    if (!dev)
    {
        return -ENOMEM;
    }
    dev->type = type;
    dev->id.simple.id = id;
    dev->internal_id = id;
    *device = dev;
    return 0;
}

int likwid_device_create(LikwidDeviceType type, int id, LikwidDevice_t* device)
{
    if ((type <= DEVICE_TYPE_INVALID) || (type >= MAX_DEVICE_TYPE) || (id < 0) || (!device))
    {
        return -EINVAL;
    }
    CpuTopology_t topo = NULL;
    int err = topology_init();
    if (err < 0)
    {
        return err;
    }
    topo = get_cpuTopology();
    int maxHWThreadNum = 0;
    for (int i = 0; i < topo->numHWThreads; i++)
    {
        HWThread* t = & topo->threadPool[i];
        if (t->apicId > maxHWThreadNum)
        {
            maxHWThreadNum = t->apicId;
        }
    }

    switch (type) {
        case DEVICE_TYPE_HWTHREAD:
            if (id >= 0 && id < maxHWThreadNum)
            {
                for (int i = 0; i < topo->numHWThreads; i++)
                {
                    HWThread* t = & topo->threadPool[i];
                    if (t->apicId == id && t->inCpuSet)
                    {
                        return likwid_device_create_simple(type, id, device);
                    }
                }
            }
            break;
        case DEVICE_TYPE_NODE:
            return likwid_device_create_simple(type, 0, device);
            break;
        case DEVICE_TYPE_CORE:
            if (id > 0 && id < maxHWThreadNum)
            {
                for (int i = 0; i < topo->numHWThreads; i++)
                {
                    HWThread* t = & topo->threadPool[i];
                    if (t->coreId == id && t->inCpuSet)
                    {
                        return likwid_device_create_simple(type, id, device);
                    }
                }
            }
            break;
        case DEVICE_TYPE_SOCKET:
            for (int i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = & topo->threadPool[i];
                if (t->packageId == id && t->inCpuSet)
                {
                    return likwid_device_create_simple(type, id, device);
                }
            }
            break;
        case DEVICE_TYPE_LLC:
            ERROR_PRINT(Not implemented);
            break;
        case DEVICE_TYPE_NUMA:
            err = numa_init();
            if (err == 0)
            {
                NumaTopology_t numatopo = get_numaTopology();
                for (int i = 0; i < numatopo->numberOfNodes; i++)
                {
                    NumaNode* node = &numatopo->nodes[i];
                    if (node->id == id)
                    {
                        return likwid_device_create_simple(type, id, device);
                    }
                }
            }
            break;
        case DEVICE_TYPE_DIE:
            for (int i = 0; i < topo->numHWThreads; i++)
            {
                HWThread* t = & topo->threadPool[i];
                if (t->dieId == id && t->inCpuSet)
                {
                    return likwid_device_create_simple(type, id, device);
                }
            }
            break;
        default:
            break;
    }
    return -ENODEV;
}

void likwid_device_destroy(LikwidDevice_t device)
{
    if (device)
    {
        free(device);
        device = NULL;
    }
}

char* device_type_name(LikwidDeviceType type)
{
    if ((type < DEVICE_TYPE_INVALID) || (type >= MAX_DEVICE_TYPE))
    {
        return "unsupported";
    }
    return LikwidDeviceTypeNames[type];
}
