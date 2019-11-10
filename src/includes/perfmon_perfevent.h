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

extern char** translate_types;
static int** cpu_event_fds = NULL;
static int active_cpus = 0;
static int paranoid_level = -1;
static int informed_paranoid = 0;
static int running_group = -1;

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
        fprintf(stderr, "ERROR: Linux kernel has no perf_event support\n");
        fprintf(stderr, "ERROR: Cannot open file /proc/sys/kernel/perf_event_paranoid\n");
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
    int paranoid = -1;
    if (!informed_paranoid)
    {
        paranoid_level = perfevent_paranoid_value();
#if defined(__x86_64__) || defined(__i386__) || defined(_ARCH_PPC)
        if (paranoid_level > 0 && getuid() != 0)
        {
            fprintf(stderr, "WARN: Linux kernel configured with paranoid level %d\n", paranoid_level);
            fprintf(stderr, "WARN: Paranoid level 0 or root access is required to measure Uncore counters\n");
        }
#endif
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
        if (paranoid_level > 1 && getuid() != 0)
        {
            fprintf(stderr, "WARN: Linux kernel configured with paranoid level %d\n", paranoid_level);
        }
#endif
        informed_paranoid = 1;
    }
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &numa_lock[affinity_thread2numa_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &sharedl3_lock[affinity_thread2sharedl3_lookup[cpu_id]], cpu_id);
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
    return 0;
}

int perf_fixed_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    int ret = -1;
    attr->type = PERF_TYPE_HARDWARE;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;
    //attr->exclusive = 1;
    if (strcmp(event->name, "INSTR_RETIRED_ANY") == 0)
    {
        attr->config = PERF_COUNT_HW_INSTRUCTIONS;
        ret = 0;
    }
    if (strcmp(event->name, "CPU_CLK_UNHALTED_CORE") == 0)
    {
        attr->config = PERF_COUNT_HW_CPU_CYCLES;
        ret = 0;
    }
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,3,0)
    if (strcmp(event->name, "CPU_CLK_UNHALTED_REF") == 0)
    {
        attr->config = PERF_COUNT_HW_REF_CPU_CYCLES;
        ret = 0;
    }
#endif

    return ret;
}

typedef enum {
    PERF_EVENT_INVAL_REG = 0,
    PERF_EVENT_CONFIG_REG,
    PERF_EVENT_CONFIG1_REG,
    PERF_EVENT_CONFIG2_REG,
} PERF_EVENT_PMC_OPT_REGS;

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
    [EVENT_OPTION_STATE] = "filter_state",
    [EVENT_OPTION_NID] = "filter_nid",
    [EVENT_OPTION_OPCODE] = "filter_opc",
    [EVENT_OPTION_OCCUPANCY] = "occ_sel",
    [EVENT_OPTION_OCCUPANCY_FILTER] = "occ_band0",
    [EVENT_OPTION_OCCUPANCY_EDGE] = "occ_edge",
    [EVENT_OPTION_OCCUPANCY_INVERT] = "occ_inv",
#ifdef _ARCH_PPC
    [EVENT_OPTION_PMC] = "pmc",
    [EVENT_OPTION_PMCXSEL] = "pmcxsel",
#endif
};

int getEventOptionConfig(char* base, EventOptionType type, PERF_EVENT_PMC_OPT_REGS *reg, int* start, int* end)
{
    PERF_EVENT_PMC_OPT_REGS r;
    int s = 0;
    int e = 0;
    if (!base || !reg || !start || !end)
    {
        return -EINVAL;
    }
    if (strlen(base) > 0 && strlen(perfEventOptionNames[type]) > 0)
    {
        char path[1024];
        char buff[1024];
        int ret = snprintf(path, 1023, "%s/format/%s", base, perfEventOptionNames[type]);
        FILE *fp = fopen(path, "r");
        if (fp)
        {
            ret = fread(buff, sizeof(char), 1023, fp);
            buff[ret] = '\0';
            if (strncmp(buff, "config:", 7) == 0)
            {
                r = PERF_EVENT_CONFIG_REG;
            }
            else if (strncmp(buff, "config1", 7) == 0)
            {
                r = PERF_EVENT_CONFIG1_REG;
            }
            else if (strncmp(buff, "config2", 7) == 0)
            {
                r = PERF_EVENT_CONFIG2_REG;
            }
            while(buff[s] != ':' && e < strlen(buff)) {
                s++;
            }
            s++;
            e = s;
            while(buff[e] != '-' && e < strlen(buff)) {
                e++;
            }
            e++;
            sscanf(&buff[s], "%d", &s);
            if (e < strlen(buff))
            {
                sscanf(&buff[e], "%d", &e);
            }
            else
            {
                e = -1;
            }
            *reg = r;
            *start = s;
            *end = e;
            fclose(fp);
        }
        else
        {
            *reg = PERF_EVENT_INVAL_REG;
            *start = -1;
            *end = -1;
        }
    }
    return 0;
}

uint64_t create_mask(uint32_t value, int start, int end)
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

int perf_pmc_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t offcore_flags = 0x0ULL;
    PERF_EVENT_PMC_OPT_REGS reg = PERF_EVENT_INVAL_REG;
    int start = 0, end = -1;
    attr->type = PERF_TYPE_RAW;
    attr->config = (event->umask<<8) + event->eventId;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;
    //attr->exclusive = 1;
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
                    getEventOptionConfig("/sys/devices/cpu", event->options[j].type, &reg, &start, &end);
                    switch(reg)
                    {
                        case PERF_EVENT_CONFIG_REG:
                            attr->config |= create_mask(event->options[j].value, start, end);
                            break;
                        case PERF_EVENT_CONFIG1_REG:
                            attr->config1 |= create_mask(event->options[j].value, start, end);
                            break;
                        case PERF_EVENT_CONFIG2_REG:
                            attr->config2 |= create_mask(event->options[j].value, start, end);
                            break;
                    }
                    break;
                case EVENT_OPTION_MATCH0:
                    if (event->eventId == 0xB7 || event->eventId == 0xBB)
                    {
                        offcore_flags |= (event->options[j].value & 0xFFFFULL);
                    }
                    break;
                case EVENT_OPTION_MATCH1:
                    if (event->eventId == 0xB7 || event->eventId == 0xBB)
                    {
                        offcore_flags |= (event->options[j].value & 0x3FFFFFFFULL)<<16;
                    }
                    break;
                default:
                    break;
            }
        }
    }
    if (event->eventId == 0xB7 || event->eventId == 0xBB)
    {
        if ((event->cfgBits != 0xFF) && (event->cmask != 0xFF))
        {
            offcore_flags = (1ULL<<event->cfgBits)|(1ULL<<event->cmask);
        }
        getEventOptionConfig("/sys/devices/cpu", EVENT_OPTION_MATCH0, &reg, &start, &end);
        switch(reg)
        {
            case PERF_EVENT_CONFIG_REG:
                attr->config |= create_mask(offcore_flags, start, end);
                break;
            case PERF_EVENT_CONFIG1_REG:
                attr->config1 |= create_mask(offcore_flags, start, end);
                break;
            case PERF_EVENT_CONFIG2_REG:
                attr->config2 |= create_mask(offcore_flags, start, end);
                break;
        }
    }
#ifdef _ARCH_PPC
    getEventOptionConfig("/sys/devices/cpu", EVENT_OPTION_PMC, &reg, &start, &end);
    switch(reg)
    {
        case PERF_EVENT_CONFIG_REG:
            attr->config |= create_mask(getCounterTypeOffset(index)+1,start, end);
            break;
        case PERF_EVENT_CONFIG1_REG:
            attr->config1 |= create_mask(getCounterTypeOffset(index)+1,start, end);
            break;
        case PERF_EVENT_CONFIG2_REG:
            attr->config2 |= create_mask(getCounterTypeOffset(index)+1,start, end);
            break;
        default:
            break;
    }
#endif
    return 0;
}

int perf_uncore_setup(struct perf_event_attr *attr, RegisterType type, PerfmonEvent *event)
{

    char checkfolder[1024];
    int ret = 0;
    FILE* fp = NULL;
    int perf_type = 0;
    PERF_EVENT_PMC_OPT_REGS reg = PERF_EVENT_INVAL_REG;
    int start = 0, end = -1;
    if (paranoid_level > 0)
    {
        return 1;
    }
    attr->type = 0;
    ret = sprintf(checkfolder, "%s", translate_types[type]);
    if (access(checkfolder, F_OK))
    {
        if ((type == UBOX)||(type == UBOXFIX))
        {
            ret = sprintf(checkfolder, "%s", "/sys/bus/event_source/devices/uncore_arb");
            if (access(checkfolder, F_OK))
            {
                return 1;
            }
        }
        else
        {
            return 1;
        }
    }
    ret = sprintf(&(checkfolder[ret]), "/type");
    fp = fopen(checkfolder, "r");
    if (fp == NULL)
    {
        return 1;
    }
    ret = fread(checkfolder, sizeof(char), 1024, fp);
    perf_type = atoi(checkfolder);
    fclose(fp);
    attr->type = perf_type;
    attr->config = (event->umask<<8) + event->eventId;
    attr->disabled = 1;
    attr->inherit = 1;

    //attr->exclusive = 1;
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
                    getEventOptionConfig(translate_types[type], event->options[j].type, &reg, &start, &end);
                    switch(reg)
                    {
                        case PERF_EVENT_CONFIG_REG:
                            attr->config |= create_mask(event->options[j].value, start, end);
                            break;
                        case PERF_EVENT_CONFIG1_REG:
                            attr->config1 |= create_mask(event->options[j].value, start, end);
                            break;
                        case PERF_EVENT_CONFIG2_REG:
                            attr->config2 |= create_mask(event->options[j].value, start, end);
                            break;
                    }
                    break;
                default:
                    break;
            }
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

    if (getenv("LIKWID_PERF_PID") != NULL)
    {
        allpid = (pid_t)atoi(getenv("LIKWID_PERF_PID"));
    }
    else if (paranoid_level > 0)
    {
        fprintf(stderr, "Cannot setup events without PID of executed application because perf_event_paranoid > 0\n");
        fprintf(stderr, "You can use either --execpid to track the started application or --perfpid <pid> to monitor another application\n");
        return -((int)thread_id+1);
    }
    if (getenv("LIKWID_PERF_FLAGS") != NULL)
    {
        allflags = strtoul(getenv("LIKWID_PERF_FLAGS"), NULL, 16);
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        int has_lock = 0;
        is_uncore = 0;
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
            case PMC:
                ret = perf_pmc_setup(&attr, index, event);
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_PMC);
                break;
            case POWER:
                ret = perf_uncore_setup(&attr, type, event);
                is_uncore = 1;
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
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
            case UBOX:
            case SBOX0:
            case SBOX1:
            case SBOX2:
            case SBOX3:
            case QBOX0:
            case QBOX1:
            case QBOX2:
            case WBOX:
            case PBOX:
            case RBOX0:
            case RBOX1:
            case BBOX0:
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
            if (!is_uncore || has_lock)
            {
                pid_t curpid = allpid;
                if (is_uncore && curpid >= 0)
                    curpid = -1;
                DEBUG_PRINT(DEBUGLEV_DEVELOP, perf_event_open: cpu_id=%d pid=%d flags=%d, cpu_id, curpid, allflags);
                cpu_event_fds[cpu_id][index] = perf_event_open(&attr, curpid, cpu_id, -1, allflags);
                if (cpu_event_fds[cpu_id][index] < 0)
                {
                    fprintf(stderr, "Setup of event %s on CPU %d failed: %s\n", event->name, cpu_id, strerror(errno));
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, perf_event_open: cpu_id=%d pid=%d flags=%d, cpu_id, curpid, allflags);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, type %d, attr.type);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, size %d, attr.size);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, config 0x%llX, attr.config);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, read_format %d, attr.read_format);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, disabled %d, attr.disabled);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, inherit %d, attr.inherit);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, pinned %d, attr.pinned);
                    DEBUG_PRINT(DEBUGLEV_DEVELOP, exclusive %d, attr.exclusive);
                    continue;
                }
                if (group_fd < 0)
                {
                    group_fd = cpu_event_fds[cpu_id][index];
                    running_group = group_fd;
                }
                eventSet->events[i].threadCounter[thread_id].init = TRUE;
            }
        }
    }
    return 0;
}

int perfmon_startCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
    int ret = 0;
    int cpu_id = groupSet->threads[thread_id].processorId;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, 0x0, 0x0, RESET_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_RESET, 0);
            eventSet->events[i].threadCounter[thread_id].startData = 0x0ULL;
            if (eventSet->events[i].type == POWER)
            {
                ret = read(cpu_event_fds[cpu_id][index],
                        &eventSet->events[i].threadCounter[thread_id].startData,
                        sizeof(long long));
            }
            VERBOSEPRINTREG(cpu_id, 0x0,
                            eventSet->events[i].threadCounter[thread_id].startData,
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
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, FREEZE_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_DISABLE, 0);
            tmp = 0;
            ret = read(cpu_event_fds[cpu_id][index], &tmp, sizeof(long long));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], tmp, READ_COUNTER);
            if (ret == sizeof(long long))
            {
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
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            if (cpu_event_fds[cpu_id][index] < 0)
                continue;
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], 0x0, FREEZE_COUNTER);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_DISABLE, 0);
            tmp = 0;
            ret = read(cpu_event_fds[cpu_id][index], &tmp, sizeof(long long));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], tmp, READ_COUNTER);
            if (ret == sizeof(long long))
            {
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
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterIndex index = eventSet->events[i].index;
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_DISABLE, 0);
            ioctl(cpu_event_fds[cpu_id][index], PERF_EVENT_IOC_RESET, 0);
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
            close(cpu_event_fds[cpu_id][index]);
            cpu_event_fds[cpu_id][index] = -1;
        }
    }
    if (cpu_event_fds[cpu_id] != NULL)
    {
        free(cpu_event_fds[cpu_id]);
        cpu_event_fds[cpu_id] = NULL;
        active_cpus--;
    }
    if (active_cpus == 0)
    {
        free(cpu_event_fds);
        cpu_event_fds = NULL;
    }
    return 0;
}
