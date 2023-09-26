/*
 * =======================================================================================
 *
 *      Filename:  perfmon_perfevent.h
 *
 *      Description:  Header File of perfmon module for perf_event kernel interface.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Thomas Gruber (tr), thomas.roehl@googlemail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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

#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>
#include <perfmon.h>
#include <linux/perf_event.h>
#include <linux/version.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <string.h>
#include <bstrlib.h>
#include <bstrlib_helper.h>

extern char** translate_types;
static int** cpu_event_fds = NULL;
static int active_cpus = 0;
static int perf_event_initialized = 0;
/*static int informed_paranoid = 0;*/
static int running_group = -1;
static int perf_event_num_cpus = 0;
static int perf_disable_uncore = 0;
static int perf_event_paranoid = -1;

static char* perfEventOptionNames[] = {
    [EVENT_OPTION_EDGE] = "edge",
    [EVENT_OPTION_ANYTHREAD] = "any",
    [EVENT_OPTION_THRESHOLD] = "cmask",
    [EVENT_OPTION_INVERT] = "inv",
    [EVENT_OPTION_IN_TRANS] = "in_tx",
    [EVENT_OPTION_IN_TRANS_ABORT] = "in_tx_cp",
    [EVENT_OPTION_MATCH0] = "offcore_rsp",
    [EVENT_OPTION_MATCH1] = "offcore_rsp",
    [EVENT_OPTION_TID] = "tid_en",
    [EVENT_OPTION_CID] = "cid",
    [EVENT_OPTION_SLICE] = "slice",
    [EVENT_OPTION_STATE] = "filter_state",
    [EVENT_OPTION_NID] = "filter_nid",
    [EVENT_OPTION_OPCODE] = "filter_opc",
    [EVENT_OPTION_OCCUPANCY] = "occ_sel",
    [EVENT_OPTION_OCCUPANCY_FILTER] = "occ_band0",
    [EVENT_OPTION_OCCUPANCY_EDGE] = "occ_edge",
    [EVENT_OPTION_OCCUPANCY_INVERT] = "occ_inv",
#ifdef _ARCH_PPC
    [EVENT_OPTION_GENERIC_CONFIG] = "pmcxsel",
    [EVENT_OPTION_UNCORE_CONFIG] = "event",
#else
    [EVENT_OPTION_GENERIC_CONFIG] = "event",
#endif
    [EVENT_OPTION_GENERIC_UMASK] = "umask",
#ifdef _ARCH_PPC
    [EVENT_OPTION_PMC] = "pmc",
    [EVENT_OPTION_PMCXSEL] = "pmcxsel",
#endif
};

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                int cpu, int group_fd, unsigned long flags)
{
    int ret;

    ret = syscall(__NR_perf_event_open, hw_event, pid, cpu,
                   group_fd, flags);
    return ret;
}

int perfevent_paranoid_value()
{
    FILE* fd;
    int paranoid = 3;
    char buff[100];
    fd = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
    if (fd == NULL)
    {
        errno = EPERM;
        ERROR_PRINT(Linux kernel has no perf_event support. Cannot access /proc/sys/kernel/perf_event_paranoid);
        return paranoid;
    }
    size_t read = fread(buff, sizeof(char), 100, fd);
    if (read > 0)
    {
        paranoid = atoi(buff);
    }
    fclose(fd);
    return paranoid;
}

int perfmon_init_perfevent(int cpu_id)
{
    perf_event_paranoid = perfevent_paranoid_value();
    if (getuid() != 0 && perf_event_paranoid > 2)
    {
        errno = EPERM;
        ERROR_PRINT(Cannot use performance monitoring with perf_event_paranoid = %d, perf_event_paranoid);
        return -(cpu_id+1);
    }

    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &numa_lock[affinity_thread2numa_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &core_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    if (cpu_event_fds == NULL)
    {
        cpu_event_fds = malloc(cpuid_topology.numHWThreads * sizeof(int*));
        for (int i=0; i < cpuid_topology.numHWThreads; i++)
            cpu_event_fds[i] = NULL;
    }
    if (cpu_event_fds[cpu_id] == NULL)
    {
        cpu_event_fds[cpu_id] = (int*) malloc(perfmon_numCounters * sizeof(int));
        if (cpu_event_fds[cpu_id] == NULL)
        {
            return -ENOMEM;
        }
        memset(cpu_event_fds[cpu_id], -1, perfmon_numCounters * sizeof(int));
        active_cpus += 1;
    }
    perf_event_num_cpus = cpuid_topology.numHWThreads;
    if (cpuid_info.family == ZEN3_FAMILY && (cpuid_info.model == ZEN4_RYZEN || cpuid_info.model == ZEN4_EPYC))
    {
        perfEventOptionNames[EVENT_OPTION_TID] = "threadmask";
        perfEventOptionNames[EVENT_OPTION_CID] = "coreid";
        perfEventOptionNames[EVENT_OPTION_SLICE] = "sliceid";
    }
    perf_event_initialized = 1;
    return 0;
}


struct perf_event_config_format {
    enum {
        INVALID = 0,
        CONFIG,
        CONFIG1,
        CONFIG2
    } reg;
    int start;
    int end;
};



int parse_event_config(char* base, char* option, int* num_formats, struct perf_event_config_format **formats)
{
    int err = 0;
    struct tagbstring config_reg = bsStatic("config:");
    struct tagbstring config1_reg = bsStatic("config1:");
    struct tagbstring config2_reg = bsStatic("config2:");
    if (!base || !option || !num_formats || !formats)
    {
        return -EINVAL;
    }
    *num_formats = 0;
    *formats = NULL;
    if (strlen(base) > 0 && strlen(option) > 0)
    {
        bstring path = bformat("%s/format/%s", base, option);
        FILE *fp = fopen(bdata(path), "r");
        if (fp)
        {
            bstring src = bread ((bNread) fread, fp);
            struct bstrList* formatList = bsplit(src, ',');
            struct perf_event_config_format *flist = malloc(formatList->qty * sizeof(struct perf_event_config_format));
            if (flist)
            {
                int nf = 0;
                for (int i = 0; i < formatList->qty; i++)
                {
                    flist[nf].reg = INVALID;
                    flist[nf].start = -1;
                    flist[nf].end = -1;
                    if (bstrncmp(formatList->entry[i], &config_reg, blength(&config_reg)) == BSTR_OK)
                    {
                        flist[nf].reg = CONFIG;
                        bdelete(formatList->entry[i], 0, blength(&config_reg));
                    }
                    else if (bstrncmp(formatList->entry[i], &config1_reg, blength(&config1_reg)) == BSTR_OK)
                    {
                        flist[nf].reg = CONFIG1;
                        bdelete(formatList->entry[i], 0, blength(&config1_reg));
                    }
                    else if (bstrncmp(formatList->entry[i], &config2_reg, blength(&config2_reg)) == BSTR_OK)
                    {
                        flist[nf].reg = CONFIG2;
                        bdelete(formatList->entry[i], 0, blength(&config2_reg));
                    }
                    else
                    {
                        if (nf > 0)
                        {
                            flist[nf].reg = flist[0].reg;
                        }
                    }
                    int s = -1;
                    int e = -1;
                    int c = sscanf(bdata(formatList->entry[i]), "%d-%d", &s, &e);
                    flist[nf].start = (c >= 1 ? s : -1);
                    flist[nf].end = (c == 2 ? e : -1);
                    nf++;
                }
                *formats = flist;
                *num_formats = nf;
            }
            else
            {
                err = -ENOMEM;
            }
            bstrListDestroy(formatList);
            bdestroy(src);
            fclose(fp);
        }
        bdestroy(path);
    }
    else
    {
        err = -ENOENT;
    }
    return err;
}

uint64_t create_mask(uint64_t value, int start, int end)
{
    if (end < 0)
    {
        return (value<<start);
    }
    else
    {
        uint64_t mask = 0x0ULL;
        for (int i = start; i <=end; i++)
            mask |= (1ULL<<i);
        return (value << start ) & mask;
    }
    return 0x0ULL;
}

int read_perf_event_type(char* folder)
{
    if (!folder || strlen(folder) == 0)
    {
        return -1;
    }
    int type = -1;
    bstring path = bformat("%s/type", folder);
    FILE *fp = fopen(bdata(path), "r");
    if (fp)
    {
        int t = -1;
        bstring src = bread ((bNread) fread, fp);
        int c = sscanf(bdata(src), "%d", &t);
        type = (c == 1 ? t : -1);
        bdestroy(src);
        fclose(fp);
    }
    bdestroy(path);
    return type;
}

int apply_event_config(struct perf_event_attr *attr, uint64_t optval, int num_formats, struct perf_event_config_format *formats)
{
    if (!attr || num_formats <= 0 || !formats)
    {
        return -EINVAL;
    }
    for (int i = 0; i < num_formats && optval != 0x0; i++)
    {
        switch(formats[i].reg)
        {
            case CONFIG:
                attr->config |= create_mask(optval, formats[i].start, formats[i].end);
                break;
            case CONFIG1:
                attr->config1 |= create_mask(optval, formats[i].start, formats[i].end);
                break;
            case CONFIG2:
                attr->config2 |= create_mask(optval, formats[i].start, formats[i].end);
                break;
        }
        optval = (optval >> ((formats[i].end - formats[i].start)+1));
    }
    return 0;
}

int parse_and_apply_event_config(char* base, char* optname, uint64_t optval, struct perf_event_attr *attr)
{
    int num_formats = 0;
    struct perf_event_config_format *formats = NULL;
    int ret = parse_event_config(base, optname, &num_formats, &formats);
    if (ret == 0)
    {
        ret = apply_event_config(attr, optval, num_formats, formats);
        free(formats);
    }
    return ret;
}


int perf_fixed_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    int err = -1;
    int ret = 0;
    int num_formats = 0;
    struct perf_event_config_format* formats = NULL;
    attr->type = PERF_TYPE_HARDWARE;
    attr->disabled = 1;
    attr->inherit = 1;
    attr->pinned = 1;
    if (translate_types[FIXED] != NULL &&
        strcmp(translate_types[PMC], translate_types[FIXED]) == 0)
    {
        attr->exclude_kernel = 1;
        attr->exclude_hv = 1;
        if (strncmp(event->name, "INSTR_RETIRED_ANY", 18) == 0)
        {
            attr->config = PERF_COUNT_HW_INSTRUCTIONS;
            err = 0;
        }
        if (strncmp(event->name, "CPU_CLK_UNHALTED_CORE", 22) == 0 ||
            strncmp(event->name, "ACTUAL_CPU_CLOCK", 17) == 0 ||
            strncmp(event->name, "APERF", 5) == 0)
        {
            attr->config = PERF_COUNT_HW_CPU_CYCLES;
            err = 0;
        }
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,3,0)
        if (strncmp(event->name, "CPU_CLK_UNHALTED_REF", 21) == 0)
        {
            attr->config = PERF_COUNT_HW_REF_CPU_CYCLES;
            err = 0;
        }
#endif
        if (cpuid_info.isIntel)
        {
            switch(cpuid_info.model)
            {
                case ICELAKE1:
                case ICELAKE2:
                case ICELAKEX1:
                case ICELAKEX2:
                case ROCKETLAKE:
                case COMETLAKE1:
                case COMETLAKE2:
                case TIGERLAKE1:
                case TIGERLAKE2:
                case SNOWRIDGEX:
                case SAPPHIRERAPIDS:
                    if (strncmp(event->name, "TOPDOWN_SLOTS", 13) == 0)
                    {
                        attr->config = 0x0400;
                        attr->type = PERF_TYPE_RAW;
                        err = 0;
                    }
            }
        }
    }
    else
    {
        int perf_type = read_perf_event_type(translate_types[FIXED]);
        if (perf_type >= 0)
        {
            attr->type = perf_type;
            ret = parse_event_config(translate_types[FIXED], perfEventOptionNames[EVENT_OPTION_GENERIC_CONFIG], &num_formats, &formats);
            if (ret == 0)
            {
                uint64_t eventConfig = event->eventId;
                for (int i = 0; i < num_formats && eventConfig != 0x0; i++)
                {
                    switch(formats[i].reg)
                    {
                        case CONFIG:
                            attr->config |= create_mask(eventConfig, formats[i].start, formats[i].end);
                            break;
                        case CONFIG1:
                            attr->config1 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                            break;
                        case CONFIG2:
                            attr->config2 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                            break;
                    }
                    eventConfig = (eventConfig >> ((formats[i].end - formats[i].start)+1));
                }
                free(formats);
                err = 0;
            }
        }
        else
        {
            err = perf_type;
        }
    }

    return err;
}

int perf_perf_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    attr->type = PERF_TYPE_HARDWARE;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;
    attr->config = event->eventId;
    return 0;
}

int perf_metrics_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    attr->type = 4;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->exclude_guest = 1;
    //attr->disabled = 1;
    //attr->inherit = 1;
    attr->config = (event->umask<<8) + event->eventId;
    return 0;
}

int perf_pmc_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    int ret = 0;
    uint64_t offcore_flags = 0x0ULL;
    int num_formats = 0;
    struct perf_event_config_format* formats = NULL;

    attr->type = PERF_TYPE_RAW;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;

    num_formats = 0;
    formats = NULL;
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    if (cpuid_info.vendor == FUJITSU_ARM && cpuid_info.part == FUJITSU_A64FX)
    {
        formats = malloc(sizeof(struct perf_event_config_format));
        if (formats)
        {
            formats[0].reg = CONFIG;
            formats[0].start = 0;
            formats[0].end = 31;
            num_formats = 1;
        }
    }
    else
    {
        ret = parse_event_config(translate_types[PMC], perfEventOptionNames[EVENT_OPTION_GENERIC_CONFIG], &num_formats, &formats);
    }
#else
    ret = parse_event_config(translate_types[PMC], perfEventOptionNames[EVENT_OPTION_GENERIC_CONFIG], &num_formats, &formats);
#endif
    if (ret == 0)
    {
        uint64_t eventConfig = event->eventId;
        for (int i = 0; i < num_formats && eventConfig != 0x0; i++)
        {
            switch(formats[i].reg)
            {
                case CONFIG:
                    attr->config |= create_mask(eventConfig, formats[i].start, formats[i].end);
                    break;
                case CONFIG1:
                    attr->config1 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                    break;
                case CONFIG2:
                    attr->config2 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                    break;
            }
            eventConfig = (eventConfig >> ((formats[i].end - formats[i].start)+1));
        }
        free(formats);
    }

    ret = 0;
    num_formats = 0;
    formats = NULL;
#ifdef _ARCH_PPC
    formats = malloc(sizeof(struct perf_event_config_format));
    if (formats)
    {
        formats[0].reg = CONFIG;
        formats[0].start = 0;
        formats[0].end = 31;
        num_formats = 1;
    }
#else
    ret = parse_event_config(translate_types[PMC], perfEventOptionNames[EVENT_OPTION_GENERIC_UMASK], &num_formats, &formats);
#endif
    if (ret == 0)
    {
        uint64_t umask = event->umask;
        for (int i = 0; i < num_formats && umask != 0x0; i++)
        {
            switch(formats[i].reg)
            {
                case CONFIG:
                    attr->config |= create_mask(umask, formats[i].start, formats[i].end);
                    break;
                case CONFIG1:
                    attr->config1 |= create_mask(umask, formats[i].start, formats[i].end);
                    break;
                case CONFIG2:
                    attr->config2 |= create_mask(umask, formats[i].start, formats[i].end);
                    break;
            }
            umask = (umask >> ((formats[i].end - formats[i].start)+1));
        }
        free(formats);
    }

    if (event->numberOfOptions > 0)
    {
        for(int j = 0; j < event->numberOfOptions; j++)
        {

            switch (event->options[j].type)
            {
                case EVENT_OPTION_COUNT_KERNEL:
                    attr->exclude_kernel = 0;
                    break;
                case EVENT_OPTION_EDGE:
                case EVENT_OPTION_ANYTHREAD:
                case EVENT_OPTION_THRESHOLD:
                case EVENT_OPTION_INVERT:
                case EVENT_OPTION_IN_TRANS:
                case EVENT_OPTION_IN_TRANS_ABORT:
                    ret = 0;
                    num_formats = 0;
                    formats = NULL;
                    ret = parse_event_config(translate_types[PMC], perfEventOptionNames[event->options[j].type], &num_formats, &formats);
                    if (ret == 0)
                    {
                        uint64_t optval = event->options[j].value;
                        for (int i = 0; i < num_formats && optval != 0x0; i++)
                        {
                            switch(formats[i].reg)
                            {
                                case CONFIG:
                                    attr->config |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                                case CONFIG1:
                                    attr->config1 |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                                case CONFIG2:
                                    attr->config2 |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                            }
                            optval = (optval >> ((formats[i].end - formats[i].start)+1));
                        }
                        free(formats);
                    }
                    break;
                case EVENT_OPTION_MATCH0:
                    if (cpuid_info.isIntel && (event->eventId == 0xB7 || event->eventId == 0xBB))
                    {
                        offcore_flags |= (event->options[j].value & 0xFFFFULL);
                    }
                    break;
                case EVENT_OPTION_MATCH1:
                    if (cpuid_info.isIntel && (event->eventId == 0xB7 || event->eventId == 0xBB))
                    {
                        offcore_flags |= (event->options[j].value & 0x3FFFFFFFULL)<<16;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    if (cpuid_info.isIntel && (event->eventId == 0xB7 || event->eventId == 0xBB))
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        ret = 0;
        num_formats = 0;
        formats = NULL;
        ret = parse_event_config(translate_types[PMC], perfEventOptionNames[EVENT_OPTION_MATCH0], &num_formats, &formats);
        if (ret == 0)
        {
            uint64_t optval = offcore_flags;
            for (int i = 0; i < num_formats && optval != 0x0; i++)
            {
                switch(formats[i].reg)
                {
                    case CONFIG:
                        attr->config |= create_mask(optval, formats[i].start, formats[i].end);
                        break;
                    case CONFIG1:
                        attr->config1 |= create_mask(optval, formats[i].start, formats[i].end);
                        break;
                    case CONFIG2:
                        attr->config2 |= create_mask(optval, formats[i].start, formats[i].end);
                        break;
                }
                optval = (optval >> ((formats[i].end - formats[i].start)+1));
            }
            free(formats);
        }
    }
#ifdef _ARCH_PPC
    ret = 0;
    num_formats = 0;
    formats = NULL;
    ret = parse_event_config(translate_types[PMC], perfEventOptionNames[EVENT_OPTION_PMC], &num_formats, &formats);
    if (ret == 0)
    {
        uint64_t optval = getCounterTypeOffset(index)+1;
        for (int i = 0; i < num_formats && offcore_flags != 0x0; i++)
        {
            switch(formats[i].reg)
            {
                case CONFIG:
                    attr->config |= create_mask(optval, formats[i].start, formats[i].end);
                    break;
                case CONFIG1:
                    attr->config1 |= create_mask(optval, formats[i].start, formats[i].end);
                    break;
                case CONFIG2:
                    attr->config2 |= create_mask(optval, formats[i].start, formats[i].end);
                    break;
            }
            optval = (optval >> ((formats[i].end - formats[i].start)+1));
        }
        free(formats);
    }
#endif
    return 0;
}

int perf_uncore_setup(struct perf_event_attr *attr, RegisterType type, PerfmonEvent *event)
{
    int num_formats = 0;
    struct perf_event_config_format* formats = NULL;
    char checkfolder[1024];
    int ret = 0;
    FILE* fp = NULL;
    int perf_type = 0;
    uint64_t eventConfig = 0x0;
    if (perf_event_paranoid > 0 && getuid() != 0)
    {
        return EPERM;
    }
    attr->type = 0;
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Get information for uncore counters from folder %s, translate_types[type]);
    perf_type = read_perf_event_type(translate_types[type]);
    if (perf_type < 0)
    {
        if ((type == UBOX)||(type == UBOXFIX))
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Get information for uncore counters from folder /sys/bus/event_source/devices/uncore_arb);
            perf_type = read_perf_event_type("/sys/bus/event_source/devices/uncore_arb");
        }
    }
    if (perf_type < 0)
    {
        return 1;
    }
    attr->type = perf_type;
    attr->disabled = 1;
    attr->inherit = 1;
#ifdef _ARCH_PPC
    eventConfig = (event->umask<<8)|event->eventId;
#else
    eventConfig = event->eventId;
#endif
    ret = parse_and_apply_event_config(translate_types[type], perfEventOptionNames[EVENT_OPTION_GENERIC_CONFIG], eventConfig, attr);
/*    num_formats = 0;*/
/*    formats = NULL;*/
/*    ret = parse_event_config(translate_types[type], perfEventOptionNames[EVENT_OPTION_GENERIC_CONFIG], &num_formats, &formats);*/
/*    if (ret == 0)*/
/*    {*/
/*        apply_event_config(attr, eventConfig, num_formats, formats);*/
/*        for (int i = 0; i < num_formats && eventConfig != 0x0; i++)*/
/*        {*/
/*            switch(formats[i].reg)*/
/*            {*/
/*                case CONFIG:*/
/*                    attr->config |= create_mask(eventConfig, formats[i].start, formats[i].end);*/
/*                    break;*/
/*                case CONFIG1:*/
/*                    attr->config1 |= create_mask(eventConfig, formats[i].start, formats[i].end);*/
/*                    break;*/
/*                case CONFIG2:*/
/*                    attr->config2 |= create_mask(eventConfig, formats[i].start, formats[i].end);*/
/*                    break;*/
/*            }*/
/*            eventConfig = (eventConfig >> ((formats[i].end - formats[i].start)+1));*/
/*        }*/
/*        free(formats);*/
/*    }*/
#ifdef _ARCH_PPC
    if (ret == 0 && reg == INVALID)
    {
        num_formats = 0;
        formats = NULL;
        ret = parse_event_config(translate_types[type], perfEventOptionNames[EVENT_OPTION_UNCORE_CONFIG], &num_formats, &formats);
        if (ret == 0)
        {
            for (int i = 0; i < num_formats && umask != 0x0; i++)
            {
                switch(formats[i].reg)
                {
                    case CONFIG:
                        attr->config |= create_mask(eventConfig, formats[i].start, formats[i].end);
                        break;
                    case CONFIG1:
                        attr->config1 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                        break;
                    case CONFIG2:
                        attr->config2 |= create_mask(eventConfig, formats[i].start, formats[i].end);
                        break;
                }
                eventConfig = (eventConfig >> ((formats[i].end - formats[i].start)+1));
            }
            free(formats);
        }
    }
#else
    if (event->umask != 0x0)
    {
        num_formats = 0;
        formats = NULL;
        ret = parse_event_config(translate_types[type], perfEventOptionNames[EVENT_OPTION_GENERIC_UMASK], &num_formats, &formats);
        if (ret == 0)
        {
            uint64_t umask = event->umask;
            for (int i = 0; i < num_formats && umask != 0x0; i++)
            {
                switch(formats[i].reg)
                {
                    case CONFIG:
                        attr->config |= create_mask(umask, formats[i].start, formats[i].end);
                        break;
                    case CONFIG1:
                        attr->config1 |= create_mask(umask, formats[i].start, formats[i].end);
                        break;
                    case CONFIG2:
                        attr->config2 |= create_mask(umask, formats[i].start, formats[i].end);
                        break;
                }
                umask = (umask >> ((formats[i].end - formats[i].start)+1));
            }
            free(formats);
        }
    }
#endif

    if (event->numberOfOptions > 0)
    {
        for(int j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_COUNT_KERNEL:
                    attr->exclude_kernel = 0;
                    break;
                case EVENT_OPTION_EDGE:
                case EVENT_OPTION_ANYTHREAD:
                case EVENT_OPTION_THRESHOLD:
                case EVENT_OPTION_INVERT:
                case EVENT_OPTION_IN_TRANS:
                case EVENT_OPTION_IN_TRANS_ABORT:
                case EVENT_OPTION_MATCH0:
                case EVENT_OPTION_MATCH1:
                case EVENT_OPTION_TID:
                case EVENT_OPTION_CID:
                case EVENT_OPTION_SLICE:
                    num_formats = 0;
                    formats = NULL;
                    ret = parse_event_config(translate_types[type], perfEventOptionNames[event->options[j].type], &num_formats, &formats);
                    if (ret == 0)
                    {
                        uint64_t optval = event->options[j].value;
                        for (int i = 0; i < num_formats && optval != 0x0; i++)
                        {
                            switch(formats[i].reg)
                            {
                                case CONFIG:
                                    attr->config |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                                case CONFIG1:
                                    attr->config1 |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                                case CONFIG2:
                                    attr->config2 |= create_mask(optval, formats[i].start, formats[i].end);
                                    break;
                            }
                            optval = (optval >> ((formats[i].end - formats[i].start)+1));
                        }
                        free(formats);
                    }
                    if (cpuid_info.model == SAPPHIRERAPIDS && event->options[j].type == EVENT_OPTION_MATCH0)
                    {
                        attr->config |= (((uint64_t)event->options[j].value) & 0x3ffffff) << 32;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    if (type != POWER && cpuid_info.family == ZEN3_FAMILY && (cpuid_info.model == ZEN4_RYZEN || cpuid_info.model == ZEN4_EPYC))
    {
        int got_cid = 0;
        int got_slices = 0;
        int got_tid = 0;
        for(int j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_CID:
                    got_cid = 1;
                    break;
                case EVENT_OPTION_TID:
                    got_tid = 1;
                    break;
                case EVENT_OPTION_SLICE:
                    got_slices = 1;
                    break;
            }
        }
        if (!got_cid)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, AMD Zen4 L3: activate counting for all cores);
            attr->config |= (1ULL<<47);
        }
        if (!got_tid)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, AMD Zen4 L3: activate counting for all SMT threads);
            attr->config |= (0x3ULL<<56);
        }
        if (!got_slices)
        {
            DEBUG_PRINT(DEBUGLEV_DEVELOP, AMD Zen4 L3: activate counting for all L3 slices);
            attr->config |= (1ULL<<46);
        }
    }
    return 0;
}


int perfmon_setupCountersThread_perfevent(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int ret;
    int cpu_id = groupSet->threads[thread_id].processorId;
    struct perf_event_attr attr;
    int group_fd = -1;
    int is_uncore = 0;
    pid_t allpid = -1;
    unsigned long allflags = 0;
    char* env_perf_pid = getenv("LIKWID_PERF_PID");

    if (!perf_event_initialized)
    {
        return -(thread_id+1);
    }
    if (env_perf_pid != NULL)
    {
        allpid = (pid_t)atoi(env_perf_pid);
    }
    if (perf_event_paranoid > 0 && getuid() != 0)
    {
        if (allpid == -1)
        {
            DEBUG_PRINT(DEBUGLEV_INFO, PID of application required. Use LIKWID_PERF_PID env variable or likwid-perfctr options);
            return -EPERM;
        }
        DEBUG_PRINT(DEBUGLEV_DEVELOP, Using PID %d for perf_event measurements, allpid);
    }

    if (getenv("LIKWID_PERF_FLAGS") != NULL)
    {
        allflags = strtoul(getenv("LIKWID_PERF_FLAGS"), NULL, 16);
    }
    if (groupSet->activeGroup >= 0)
    {
        for (int j = 0; j < perfmon_numCounters; j++)
        {
            if (cpu_event_fds[cpu_id][j] != -1)
            {
                close(cpu_event_fds[cpu_id][j]);
                cpu_event_fds[cpu_id][j] = -1;
            }
        }
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        int has_lock = 0;
        int pmc_lock = 0;
        is_uncore = 0;
        ret = 1;
        RegisterIndex index = eventSet->events[i].index;
        if (cpu_event_fds[cpu_id][index] != -1)
        {
            continue;
        }
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        PerfmonEvent *event = &(eventSet->events[i].event);
        memset(&attr, 0, sizeof(struct perf_event_attr));
        attr.size = sizeof(struct perf_event_attr);
        switch (type)
        {
            case FIXED:
                ret = perf_fixed_setup(&attr, index, event);
                if (ret < 0)
                {
                    continue;
                }
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_FIXED);
                break;
            case PERF:
                ret = perf_perf_setup(&attr, index, event);
                if (ret < 0)
                {
                    continue;
                }
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_PERF);
                break;
            case METRICS:
                ret = perf_metrics_setup(&attr, index, event);
                if (ret < 0)
                {
                    continue;
                }
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_METRICS);
                break;
            case PMC:
                pmc_lock = 1;
#if defined(__ARM_ARCH_8A)
                if (cpuid_info.vendor == FUJITSU_ARM && cpuid_info.part == FUJITSU_A64FX)
                {
                    if (event->eventId == 0x3E8 ||
                        event->eventId == 0x3E0 ||
                        event->eventId == 0x308 ||
                        event->eventId == 0x309 ||
                        (event->eventId >= 0x314 &&  event->eventId <= 0x31E))
                    {
                        if (numa_lock[affinity_thread2numa_lookup[cpu_id]] != cpu_id)
                        {
                            pmc_lock = 0;
                        }
                    }
                }
#endif
                if (pmc_lock)
                {
                    ret = perf_pmc_setup(&attr, index, event);
                    VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_PMC);
                }
                break;
            case POWER:
                if (cpuid_info.isIntel && socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
                {
                    has_lock = 1;
                    VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
                    ret = perf_uncore_setup(&attr, type, event);
                }
                else if ((cpuid_info.family == ZEN_FAMILY || cpuid_info.family == ZEN3_FAMILY))
                {
                    if (event->eventId == 0x01 && core_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                        VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
                        ret = perf_uncore_setup(&attr, type, event);
                    }
                    else if (event->eventId == 0x02 && socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                        VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
                        ret = perf_uncore_setup(&attr, type, event);
                    }
                    else if (event->eventId == 0x03 && sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                        VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
                        ret = perf_uncore_setup(&attr, type, event);
                    }
                }
                is_uncore = 1;
                break;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,5,0)
            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case MBOX4:
            case MBOX5:
            case MBOX6:
            case MBOX7:
            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
            case CBOX8:
            case CBOX9:
            case CBOX10:
            case CBOX11:
            case CBOX12:
            case CBOX13:
            case CBOX14:
            case CBOX15:
            case CBOX16:
            case CBOX17:
            case CBOX18:
            case CBOX19:
            case CBOX20:
            case CBOX21:
            case CBOX22:
            case CBOX23:
            case CBOX24:
            case CBOX25:
            case CBOX26:
            case CBOX27:
            case CBOX28:
            case CBOX29:
            case CBOX30:
            case CBOX31:
            case CBOX32:
            case CBOX33:
            case CBOX34:
            case CBOX35:
            case CBOX36:
            case CBOX37:
            case CBOX38:
            case CBOX39:
            case CBOX40:
            case CBOX41:
            case CBOX42:
            case CBOX43:
            case CBOX44:
            case CBOX45:
            case CBOX46:
            case CBOX47:
            case CBOX48:
            case CBOX49:
            case CBOX50:
            case CBOX51:
            case CBOX52:
            case CBOX53:
            case CBOX54:
            case CBOX55:
            case CBOX56:
            case CBOX57:
            case CBOX58:
            case CBOX59:
            case MDF0:
            case MDF1:
            case MDF2:
            case MDF3:
            case MDF4:
            case MDF5:
            case MDF6:
            case MDF7:
            case MDF8:
            case MDF9:
            case MDF10:
            case MDF11:
            case MDF12:
            case MDF13:
            case MDF14:
            case MDF15:
            case MDF16:
            case MDF17:
            case MDF18:
            case MDF19:
            case MDF20:
            case MDF21:
            case MDF22:
            case MDF23:
            case MDF24:
            case MDF25:
            case MDF26:
            case MDF27:
            case MDF28:
            case MDF29:
            case MDF30:
            case MDF31:
            case MDF32:
            case MDF33:
            case MDF34:
            case MDF35:
            case MDF36:
            case MDF37:
            case MDF38:
            case MDF39:
            case MDF40:
            case MDF41:
            case MDF42:
            case MDF43:
            case MDF44:
            case MDF45:
            case MDF46:
            case MDF47:
            case MDF48:
            case MDF49:
            case UBOX:
            case UBOXFIX:
            case SBOX0:
            case SBOX1:
            case SBOX2:
            case SBOX3:
            case QBOX0:
            case QBOX1:
            case QBOX2:
            case QBOX3:
            case WBOX:
            case PBOX:
            case RBOX0:
            case RBOX1:
            case RBOX2:
            case RBOX3:
            case BBOX0:
            case BBOX1:
            case BBOX2:
            case BBOX3:
            case BBOX4:
            case BBOX5:
            case BBOX6:
            case BBOX7:
            case BBOX8:
            case BBOX9:
            case BBOX10:
            case BBOX11:
            case BBOX12:
            case BBOX13:
            case BBOX14:
            case BBOX15:
            case BBOX16:
            case BBOX17:
            case BBOX18:
            case BBOX19:
            case BBOX20:
            case BBOX21:
            case BBOX22:
            case BBOX23:
            case BBOX24:
            case BBOX25:
            case BBOX26:
            case BBOX27:
            case BBOX28:
            case BBOX29:
            case BBOX30:
            case BBOX31:
            case EDBOX0:
            case EDBOX1:
            case EDBOX2:
            case EDBOX3:
            case EDBOX4:
            case EDBOX5:
            case EDBOX6:
            case EDBOX7:
            case EUBOX0:
            case EUBOX1:
            case EUBOX2:
            case EUBOX3:
            case EUBOX4:
            case EUBOX5:
            case EUBOX6:
            case EUBOX7:
            case IBOX0:
            case IBOX1:
            case IBOX2:
            case IBOX3:
            case IBOX4:
            case IBOX5:
            case IBOX6:
            case IBOX7:
            case IBOX8:
            case IBOX9:
            case IBOX10:
            case IBOX11:
            case IBOX12:
            case IBOX13:
            case IBOX14:
            case IBOX15:
            case IRP0:
            case IRP1:
            case IRP2:
            case IRP3:
            case IRP4:
            case IRP5:
            case IRP6:
            case IRP7:
            case IRP8:
            case IRP9:
            case IRP10:
            case IRP11:
            case IRP12:
            case IRP13:
            case IRP14:
            case IRP15:
            case HBM0:
            case HBM1:
            case HBM2:
            case HBM3:
            case HBM4:
            case HBM5:
            case HBM6:
            case HBM7:
            case HBM8:
            case HBM9:
            case HBM10:
            case HBM11:
            case HBM12:
            case HBM13:
            case HBM14:
            case HBM15:
            case HBM16:
            case HBM17:
            case HBM18:
            case HBM19:
            case HBM20:
            case HBM21:
            case HBM22:
            case HBM23:
            case HBM24:
            case HBM25:
            case HBM26:
            case HBM27:
            case HBM28:
            case HBM29:
            case HBM30:
            case HBM31:
                if (cpuid_info.family == ZEN_FAMILY && type == MBOX0)
                {
                    if (numa_lock[affinity_thread2numa_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                    }
                }
                else if (cpuid_info.family == ZEN_FAMILY && type == CBOX0)
                {
                    if (sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                    }
                }
                else if (cpuid_info.family == P6_FAMILY &&
                         cpuid_info.model == SKYLAKEX &&
                         cpuid_info.stepping >= 5 &&
                         cpuid_topology.numDies > cpuid_topology.numSockets)
                {
                    if (die_lock[affinity_thread2die_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                    }
                }
                else
                {
                    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
                    {
                        has_lock = 1;
                    }
                }
                if (has_lock)
                {
                    ret = perf_uncore_setup(&attr, type, event);
                    is_uncore = 1;
                    VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_UNCORE);
                }
                break;
#endif
            default:
                break;
        }
        if (ret == 0)
        {
            pid_t curpid = allpid;
            if (is_uncore && curpid >= 0)
            {
                curpid = -1;
            }

            if (!is_uncore)
            {
                DEBUG_PRINT(DEBUGLEV_DEVELOP, perf_event_open: cpu_id=%d pid=%d flags=%d, cpu_id, curpid, allflags);
                cpu_event_fds[cpu_id][index] = perf_event_open(&attr, curpid, cpu_id, -1, allflags);
            }
            else if ((perf_disable_uncore == 0) && (has_lock))
            {
                if (perf_event_paranoid > 0 && getuid() != 0)
                {
                    DEBUG_PRINT(DEBUGLEV_INFO, Cannot measure Uncore with perf_event_paranoid value = %d, perf_event_paranoid);
                    perf_disable_uncore = 1;
                }
                DEBUG_PRINT(DEBUGLEV_DEVELOP, perf_event_open: cpu_id=%d pid=%d flags=%d type=%d config=0x%llX disabled=%d inherit=%d exclusive=%d config1=0x%llX config2=0x%llX, cpu_id, curpid, allflags, attr.type, attr.config, attr.disabled, attr.inherit, attr.exclusive, attr.config1, attr.config2);
                cpu_event_fds[cpu_id][index] = perf_event_open(&attr, curpid, cpu_id, -1, allflags);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_INFO, Unknown perf_event_paranoid value = %d, perf_event_paranoid);
            }
            if (cpu_event_fds[cpu_id][index] < 0)
            {
                ERROR_PRINT(Setup of event %s on CPU %d failed: %s, event->name, cpu_id, strerror(errno));
                DEBUG_PRINT(DEBUGLEV_DEVELOP, open error: cpu_id=%d pid=%d flags=%d type=%d config=0x%llX disabled=%d inherit=%d exclusive=%d config1=0x%llX config2=0x%llX, cpu_id, curpid, allflags, attr.type, attr.config, attr.disabled, attr.inherit, attr.exclusive, attr.config1, attr.config2);
            }
            if (group_fd < 0)
            {
                group_fd = cpu_event_fds[cpu_id][index];
                running_group = group_fd;
            }
            eventSet->events[i].threadCounter[thread_id].init = TRUE;
        }
        else if (ret == EPERM)
        {
            if (is_uncore && perf_disable_uncore == 0 && perf_event_paranoid > 0 && getuid() != 0)
            {
                DEBUG_PRINT(DEBUGLEV_INFO, Cannot measure Uncore with perf_event_paranoid value = %d, perf_event_paranoid);
                perf_disable_uncore = 1;
            }
        }
    }
    return 0;
}

int perfmon_startCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
    int ret = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    if (!perf_event_initialized)
    {
        return -(thread_id+1);
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, 0x0, 0x0, RESET_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_RESET, 0);
            PerfmonCounter *c = &eventSet->events[i].threadCounter[thread_id];
            c->startData = 0x0ULL;
            c->counterData = 0x0ULL;
            if (eventSet->events[i].type == POWER)
            {
                ret = read(cpu_event_fds[cpu_id][index],
                        &eventSet->events[i].threadCounter[thread_id].startData,
                        sizeof(long long));
            }
            VERBOSEPRINTREG(cpu_id, 0x0,
                            c->startData,
                            START_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    return 0;
}

int perfmon_stopCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
    int ret;
    int cpu_id = groupSet->threads[thread_id].processorId;
    long long tmp = 0;
    if (!perf_event_initialized)
    {
        return -(thread_id+1);
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, FREEZE_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_DISABLE, 0);
            tmp = 0x0LL;
            ret = read(cpu_event_fds[cpu_id][index], &tmp, sizeof(long long));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], tmp, READ_COUNTER);
            if (ret == sizeof(long long))
            {
#if defined(__ARM_ARCH_8A)
                if (cpuid_info.vendor == FUJITSU_ARM && cpuid_info.part == FUJITSU_A64FX)
                {
                    switch (eventSet->events[i].event.eventId) {
                        case 0x3E8:
                            tmp *= 256;
                            break;
                        case 0x3E0:
                            if (cpuid_topology.numCoresPerSocket == 24)
                                tmp *= 36;
                            else
                                tmp *= 32;
                            break;
                        default:
                            break;
                    }
                }
#endif
                eventSet->events[i].threadCounter[thread_id].counterData = tmp;

            }
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_RESET, 0);
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, RESET_COUNTER);
        }
    }
    return 0;
}

int perfmon_readCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
    int ret;
    int cpu_id = groupSet->threads[thread_id].processorId;
    long long tmp = 0;
    if (!perf_event_initialized)
    {
        return -(thread_id+1);
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, FREEZE_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_DISABLE, 0);
            tmp = 0x0LL;
            ret = read(cpu_event_fds[cpu_id][index], &tmp, sizeof(long long));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], tmp, READ_COUNTER);
            if (ret == sizeof(long long))
            {
#if defined(__ARM_ARCH_8A)
                if (cpuid_info.vendor == FUJITSU_ARM && cpuid_info.part == FUJITSU_A64FX)
                {
                    switch (eventSet->events[i].event.eventId) {
                        case 0x3E8:
                            tmp *= 256;
                            break;
                        case 0x3E0:
                            if (cpuid_topology.numCoresPerSocket == 24)
                                tmp *= 36;
                            else
                                tmp *= 32;
                            break;
                        default:
                            break;
                    }
                }
#endif
                eventSet->events[i].threadCounter[thread_id].counterData = tmp;
            }
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, UNFREEZE_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    return 0;
}

int perfmon_finalizeCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    if (cpu_event_fds != NULL)
    {
        if (cpu_event_fds[cpu_id] != NULL)
        {
            for (int j = 0; j < perfmon_numCounters; j++)
            {
                if (cpu_event_fds[cpu_id][j] > 0)
                {
                    ioctl(cpu_event_fds[cpu_id][j], PERF_EVENT_IOC_DISABLE, 0);
                    ioctl(cpu_event_fds[cpu_id][j], PERF_EVENT_IOC_RESET, 0);
                    close(cpu_event_fds[cpu_id][j]);
                    cpu_event_fds[cpu_id][j] = -1;
                    if (j < eventSet->numberOfEvents && eventSet->events[j].threadCounter[thread_id].init == TRUE)
                    {
                        eventSet->events[j].threadCounter[thread_id].init = FALSE;
                    }
                }
            }
            free(cpu_event_fds[cpu_id]);
            cpu_event_fds[cpu_id] = NULL;
            active_cpus--;
        }
    }
    return 0;
}

void __attribute__((destructor (101))) close_perfmon_perfevent(void)
{
    if (cpu_event_fds != NULL)
    {
        for (int i = 0; i < perf_event_num_cpus; i++)
        {
            if (cpu_event_fds[i] != NULL)
            {
                free(cpu_event_fds[i]);
                cpu_event_fds[i] = NULL;
                active_cpus--;
            }
        }
        free(cpu_event_fds);
        cpu_event_fds = NULL;
    }
}
