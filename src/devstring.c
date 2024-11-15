/*
 * =======================================================================================
 *
 *      Filename:  devstring.c
 *
 *      Description:  Code to resolve a LIKWID device string to a list of devices
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Michael Panzlaff, michael.panzlaff@fau.de
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

#include <devstring.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#include <error.h>

static int btoi(const bstring bstr, int *i)
{
    const char *cstr = bdata(bstr);
    char *endptr;
    errno = 0;
    unsigned long long result = strtoull(cstr, &endptr, 0);
    if (cstr == endptr || *endptr != '\0')
        return -EINVAL;
    if (errno != 0)
        return errno;
    if (result > INT_MAX)
        return -ERANGE;
    *i = (int)result;
    return 0;
}

typedef struct
{
    int start;  // inclusive
    int end;    // exclusive
} range_t;

typedef struct
{
    int count;
    range_t ranges[];
} range_list_t;

static int range_parse(const bstring range_str, range_t *range)
{
    /* Parse a string of style '3' (single integer) or '5-7'. */

    /* First try to parse as single integer, if that doesn't work, split
     * by '-'. */
    int start, end;
    int err = btoi(range_str, &start);
    if (err == 0)
    {
        range->start = start;
        range->end = start + 1;
        return 0;
    }

    bstring start_str = NULL;
    bstring end_str = NULL;

    int sep_index = bstrchr(range_str, '-');
    if (sep_index == BSTR_ERR)
    {
        err = -EINVAL;
        goto cleanup;
    }

    start_str = bmidstr(range_str, 0, sep_index);
    end_str = bmidstr(range_str, sep_index + 1, INT_MAX);
    if (!start_str || !end_str)
    {
        err = -ENOMEM;
        goto cleanup;
    }

    err = btoi(start_str, &start);
    if (err < 0)
        goto cleanup;

    err = btoi(end_str, &end);
    if (err < 0)
        goto cleanup;

    range->start = start;
    range->end = end + 1;

cleanup:
    bdestroy(start_str);
    bdestroy(end_str);
    return err;
}

static int range_create(const bstring range_list_str, range_list_t **rlist)
{
    /* Create a list of ranges from a string like '1-4,5,6,7-20' */
    struct bstrList *slist = bsplit(range_list_str, ',');
    if (!slist)
        return -ENOMEM;

    int err = 0;
    range_list_t *new_rlist = malloc(sizeof(range_list_t) + sizeof(range_t) * slist->qty);
    if (!new_rlist)
    {
        err = -ENOMEM;
        goto cleanup;
    }

    new_rlist->count = slist->qty;
    for (int i = 0; i < slist->qty; i++)
    {
        err = range_parse(slist->entry[i], &new_rlist->ranges[i]);
        if (err < 0)
            goto cleanup;
    }

    *rlist = new_rlist;
    
cleanup:
    bstrListDestroy(slist);
    return err;
}

static void range_destroy(range_list_t *rlist)
{
    if (!rlist)
        return;
    free(rlist);
}

static int misc_init(void)
{
    int err = topology_init();
    if (err < 0)
        return err;

#ifdef LIKWID_WITH_NVMON
    /* The topology init functions currently return both positive and negative
     * error numbers :-/, so use this workaround for now. */
    if (topology_cuda_init() != 0)
        return -EPERM;
#endif
#ifdef LIKWID_WITH_ROCMON
    if (topology_rocm_init() != 0)
        return -EPERM;
#endif
    if (affinity_init())
        return -EPERM;

    return 0;
}

static int device_create_from_string_and_append(LikwidDeviceType type, const char *id, LikwidDeviceList_t dev_list)
{
    LikwidDevice_t dev;
    int err = likwid_device_create_from_string(type, id, &dev);
    if (err < 0)
        return err;

    const int new_num_devices = dev_list->num_devices + 1;
    LikwidDevice_t *new_devices = realloc(dev_list->devices, new_num_devices * sizeof(dev_list->devices[0]));
    if (!new_devices)
    {
        likwid_device_destroy(dev);
        return -ENOMEM;
    }

    dev_list->num_devices = new_num_devices;
    dev_list->devices = new_devices;
    dev_list->devices[new_num_devices - 1] = dev;
    return 0;
}

static int device_create_and_append(LikwidDeviceType type, int id, LikwidDeviceList_t dev_list)
{
    LikwidDevice_t dev;
    int err = likwid_device_create(type, id, &dev);
    if (err < 0)
        return err;

    const int new_num_devices = dev_list->num_devices + 1;
    LikwidDevice_t *new_devices = realloc(dev_list->devices, new_num_devices * sizeof(dev_list->devices[0]));
    if (!new_devices)
    {
        likwid_device_destroy(dev);
        return -ENOMEM;
    }

    dev_list->num_devices = new_num_devices;
    dev_list->devices = new_devices;
    dev_list->devices[new_num_devices - 1] = dev;
    return 0;
}

static int parse_node(const bstring domain_selector, LikwidDeviceList_t dev_list)
{
    /* A node string must not have a domain_selector, as only domain 0 is allowed.
     * Optionally domain 0 is allowed. */
    const char *ds = bdata(domain_selector);
    if (domain_selector && strcmp(ds, "0") != 0)
    {
        ERROR_PRINT(If specified node domain may only sppecify node '0' found: '%s', bdata(domain_selector));
        return -EINVAL;
    }

    return device_create_and_append(DEVICE_TYPE_NODE, 0, dev_list);
}

static int parse_simple(const bstring domain_selector, LikwidDeviceType type, LikwidDeviceList_t dev_list)
{
    /* All others (except GPUs) have a domain_selector like this: '0,1,4-5,7' */
    range_list_t *range_list;
    int err = range_create(domain_selector, &range_list);
    if (err < 0)
    {
        ERROR_PRINT(Unable to parse malformed range: %s, bdata(domain_selector));
        return err;
    }

    for (int range_index = 0; range_index < range_list->count; range_index++)
    {
        range_t *range = &range_list->ranges[range_index];
        for (int i = range->start; i < range->end; i++)
        {
            err = device_create_and_append(type, i, dev_list);
            if (err < 0)
                goto cleanup;
        }
    }

cleanup:
    range_destroy(range_list);
    return err;
}

static int parse_gpu(const bstring domain_selector, LikwidDeviceType type, LikwidDeviceList_t dev_list)
{
    /* GPUs differ once again in their domain_selector, as they also allow a list of PCI adresses.
     * So it may look like '0,1,4-5,7' or '00000000:e0:1f:1,00000000:30:13:0'. */
    range_list_t *range_list;
    int err = range_create(domain_selector, &range_list);
    if (err == 0)
    {
        for (int range_index = 0; range_index < range_list->count; range_index++)
        {
            range_t *range = &range_list->ranges[range_index];
            for (int i = range->start; i < range->end; i++)
            {
                err = device_create_and_append(type, i, dev_list);
                if (err < 0)
                    goto cleanup;
            }
        }

cleanup:
        range_destroy(range_list);
        return err;
    }

    /* If the previous if wasn't entered, check for the PCI address format here. */

    struct bstrList *slist = bsplit(domain_selector, ',');
    if (!slist)
        return -ENOMEM;

    for (int i = 0; i < slist->qty; i++)
    {
        err = device_create_from_string_and_append(type, bdata(slist->entry[i]), dev_list);
        if (err < 0)
            break;
    }

    bstrListDestroy(slist);
    return err;
}

static int process_domain(const bstring domain_type, const bstring domain_selector, LikwidDeviceList_t dev_list)
{
    const char *dt = bdata(domain_type);
    if (strcmp(dt, "N") == 0)
        return parse_node(domain_selector, dev_list);
    else if (strcmp(dt, "S") == 0)
        return parse_simple(domain_selector, DEVICE_TYPE_SOCKET, dev_list);
    else if (strcmp(dt, "C") == 0)
        return parse_simple(domain_selector, DEVICE_TYPE_CORE, dev_list);
    else if (strcmp(dt, "M") == 0)
        return parse_simple(domain_selector, DEVICE_TYPE_NUMA, dev_list);
    else if (strcmp(dt, "D") == 0)
        return parse_simple(domain_selector, DEVICE_TYPE_DIE, dev_list);
#ifdef LIKWID_WITH_NVMON
    else if (strcmp(dt, "GN") == 0)
        return parse_gpu(domain_selector, DEVICE_TYPE_NVIDIA_GPU, dev_list);
#endif
#ifdef LIKWID_WITH_ROCMON
    else if (strcmp(dt, "GA") == 0)
        return parse_gpu(domain_selector, DEVICE_TYPE_AMD_GPU, dev_list);
#endif
    else if (strcmp(dt, "T") == 0)
        return parse_simple(domain_selector, DEVICE_TYPE_HWTHREAD, dev_list);
    /* If no domain prefix matches, assume legacy behavior and create hardware threads. */
    return parse_simple(domain_type, DEVICE_TYPE_HWTHREAD, dev_list);
}

static int parse_domain(const bstring dev_bstr, LikwidDeviceList_t dev_list)
{
    /* Parse a domain name like 'M:0-4,7' to 'M' and '0-4,7'.
     * We call this the domain_type and domain_selector. */
    int err;
    int sep_index = bstrchr(dev_bstr, ':');
    if (sep_index == BSTR_ERR)
    {
        err = process_domain(dev_bstr, NULL, dev_list);
    }
    else
    {
        bstring domain_name = bmidstr(dev_bstr, 0, sep_index);
        if (!domain_name)
            return -ENOMEM;
        bstring domain_selector = bmidstr(dev_bstr, sep_index + 1, INT_MAX);
        if (!domain_selector)
        {
            bdestroy(domain_name);
            return -ENOMEM;
        }
        err = process_domain(domain_name, domain_selector, dev_list);
        bdestroy(domain_selector);
        bdestroy(domain_name);
    }
    return err;
}

int likwid_devstr_to_devlist(const char *str, LikwidDeviceList_t *dev_list)
{
    /* Init */
    int err = misc_init();
    if (err < 0)
        return err;

    /* Alloc list */
    LikwidDeviceList_t new_list = calloc(1, sizeof(_LikwidDeviceList));
    if (!new_list)
        return -ENOMEM;

    /* Split input string by '@' */
    bstring bstr = bfromcstr(str);
    struct bstrList *bstr_list = bsplit(bstr, '@');

    if (!bstr_list)
    {
        free(new_list);
        return -ENOMEM;
    }

    /* Iterate over all devices and add them to the list */
    for (int i = 0; i < bstr_list->qty; i++)
    {
        err = parse_domain(bstr_list->entry[i], new_list);
        if (err < 0)
            break;
    }

    *dev_list = new_list;
    bstrListDestroy(bstr_list);
    return err;
}
