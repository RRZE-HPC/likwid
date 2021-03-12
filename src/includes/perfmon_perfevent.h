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


typedef struct __attribute__((packed)) {
    uint64_t value;
    uint64_t enabled;
    uint64_t running;
} PerfEventResult;

extern char** translate_types;
static int** cpu_event_fds = NULL;
static int active_cpus = 0;
static int perf_event_initialized = 0;
/*static int informed_paranoid = 0;*/
static int running_group = -1;
static int perf_event_num_cpus = 0;
static int perf_disable_uncore = 0;
static int perf_event_paranoid = -1;

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
    perf_event_initialized = 1;
    return 0;
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
#else
    [EVENT_OPTION_GENERIC_CONFIG] = "event",
#endif
    [EVENT_OPTION_GENERIC_UMASK] = "umask",
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

int perf_fixed_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    int ret = -1;
    attr->type = PERF_TYPE_HARDWARE;
    attr->disabled = 1;
    attr->inherit = 1;
    attr->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    if (translate_types[FIXED] != NULL &&
        strcmp(translate_types[PMC], translate_types[FIXED]) == 0)
    {
        attr->exclude_kernel = 1;
        attr->exclude_hv = 1;
        if (strcmp(event->name, "INSTR_RETIRED_ANY") == 0)
        {
            attr->config = PERF_COUNT_HW_INSTRUCTIONS;
            ret = 0;
        }
        if (strcmp(event->name, "CPU_CLK_UNHALTED_CORE") == 0 ||
            strcmp(event->name, "ACTUAL_CPU_CLOCK") == 0 ||
            strcmp(event->name, "APERF") == 0)
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
    }
    else
    {
        char checkfolder[1024];
        int perf_type = PERF_TYPE_HARDWARE;
        int start = 0, end = 0;
        PERF_EVENT_PMC_OPT_REGS reg = PERF_EVENT_INVAL_REG;
        int err = snprintf(checkfolder, 1023, "%s/type", translate_types[FIXED]);
        if (err > 0)
        {
            checkfolder[err] = '\0';
        }
        FILE* fp = fopen(checkfolder, "r");
        if (fp != NULL)
        {
            ret = fread(checkfolder, sizeof(char), 1024, fp);
            fclose(fp);
            perf_type = atoi(checkfolder);
            if (perf_type >= 0)
            {
                attr->type = perf_type;
                getEventOptionConfig(translate_types[FIXED], EVENT_OPTION_GENERIC_CONFIG, &reg, &start, &end);
                switch(reg)
                {
                    case PERF_EVENT_CONFIG_REG:
                        attr->config |= create_mask(event->eventId, start, end);
                        ret = 0;
                        break;
                    case PERF_EVENT_CONFIG1_REG:
                        attr->config1 |= create_mask(event->eventId, start, end);
                        ret = 0;
                        break;
                    case PERF_EVENT_CONFIG2_REG:
                        attr->config2 |= create_mask(event->eventId, start, end);
                        ret = 0;
                        break;
                    default:
                        ret = -1;
                        break;
                }
            }
        }
    }

    return ret;
}

int perf_perf_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    attr->type = PERF_TYPE_HARDWARE;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;
    attr->config = event->eventId;
    attr->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return 0;
}

int perf_pmc_setup(struct perf_event_attr *attr, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t offcore_flags = 0x0ULL;
    PERF_EVENT_PMC_OPT_REGS reg = PERF_EVENT_INVAL_REG;
    int start = 0, end = -1;
    attr->type = PERF_TYPE_RAW;
    //attr->config = (event->umask<<8) + event->eventId;
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
    attr->disabled = 1;
    attr->inherit = 1;
    attr->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    //attr->exclusive = 1;
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    if (cpuid_info.vendor == FUJITSU_ARM && cpuid_info.part == FUJITSU_A64FX)
    {
        reg = PERF_EVENT_CONFIG_REG;
        start = 0;
        end = 31;
    }
    else
    {
#endif
        getEventOptionConfig(translate_types[PMC], EVENT_OPTION_GENERIC_CONFIG, &reg, &start, &end);
#if defined(__ARM_ARCH_8A) || defined(__ARM_ARCH_7A__)
    }
#endif
    switch(reg)
    {
        case PERF_EVENT_CONFIG_REG:
            attr->config |= create_mask(event->eventId, start, end);
            break;
        case PERF_EVENT_CONFIG1_REG:
            attr->config1 |= create_mask(event->eventId, start, end);
            break;
        case PERF_EVENT_CONFIG2_REG:
            attr->config2 |= create_mask(event->eventId, start, end);
            break;
    }
#ifdef _ARCH_PPC
    reg = PERF_EVENT_CONFIG_REG;
    start = 8;
    end = 15;
#else
    getEventOptionConfig(translate_types[PMC], EVENT_OPTION_GENERIC_UMASK, &reg, &start, &end);
#endif
    switch(reg)
    {
        case PERF_EVENT_CONFIG_REG:
            attr->config |= create_mask(event->umask, start, end);
            break;
        case PERF_EVENT_CONFIG1_REG:
            attr->config1 |= create_mask(event->umask, start, end);
            break;
        case PERF_EVENT_CONFIG2_REG:
            attr->config2 |= create_mask(event->umask, start, end);
            break;
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
                    getEventOptionConfig(translate_types[PMC], event->options[j].type, &reg, &start, &end);
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
        getEventOptionConfig(translate_types[PMC], EVENT_OPTION_MATCH0, &reg, &start, &end);
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
    getEventOptionConfig(translate_types[PMC], EVENT_OPTION_PMC, &reg, &start, &end);
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
    if (perf_event_paranoid > 0 && getuid() != 0)
    {
        return EPERM;
    }
    attr->type = 0;
    attr->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
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
    DEBUG_PRINT(DEBUGLEV_DEVELOP, Get information for uncore counters from folder %s, checkfolder);
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
    attr->disabled = 1;
    attr->inherit = 1;
    //attr->config = (event->umask<<8) + event->eventId;
    getEventOptionConfig(translate_types[type], EVENT_OPTION_GENERIC_CONFIG, &reg, &start, &end);
    switch(reg)
    {
        case PERF_EVENT_CONFIG_REG:
            attr->config |= create_mask(event->eventId, start, end);
            break;
        case PERF_EVENT_CONFIG1_REG:
            attr->config1 |= create_mask(event->eventId, start, end);
            break;
        case PERF_EVENT_CONFIG2_REG:
            attr->config2 |= create_mask(event->eventId, start, end);
            break;
    }
    getEventOptionConfig(translate_types[type], EVENT_OPTION_GENERIC_UMASK, &reg, &start, &end);
    switch(reg)
    {
        case PERF_EVENT_CONFIG_REG:
            attr->config |= create_mask(event->umask, start, end);
            break;
        case PERF_EVENT_CONFIG1_REG:
            attr->config1 |= create_mask(event->umask, start, end);
            break;
        case PERF_EVENT_CONFIG2_REG:
            attr->config2 |= create_mask(event->umask, start, end);
            break;
    }


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
                if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
                {
                    has_lock = 1;
                    VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_POWER);
                    ret = perf_uncore_setup(&attr, type, event);
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
            case UBOX:
            case UBOXFIX:
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
                DEBUG_PRINT(DEBUGLEV_DEVELOP, perf_event_open: cpu_id=%d pid=%d flags=%d, cpu_id, curpid, allflags);
                cpu_event_fds[cpu_id][index] = perf_event_open(&attr, curpid, cpu_id, -1, allflags);
            }
            else
            {
                DEBUG_PRINT(DEBUGLEV_INFO, Unknown perf_event_paranoid value = %d, perf_event_paranoid);
            }
            if (cpu_event_fds[cpu_id][index] < 0)
            {
                ERROR_PRINT(Setup of event %s on CPU %d failed: %s, event->name, cpu_id, strerror(errno));
                DEBUG_PRINT(DEBUGLEV_DEVELOP, open error: cpu_id=%d pid=%d flags=%d type=%d config=0x%llX disabled=%d inherit=%d exclusive=%d config2=0x%llX, cpu_id, curpid, allflags, attr.type, attr.config, attr.disabled, attr.inherit, attr.exclusive);
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
            eventSet->events[i].threadCounter[thread_id].startData = 0x0ULL;
            if (eventSet->events[i].type == POWER)
            {
                PerfEventResult res = {0ULL, 0ULL, 0ULL};
                ret = read(cpu_event_fds[cpu_id][index], &res, sizeof(PerfEventResult));
                VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, START_POWER);
                if (ret == sizeof(PerfEventResult))
                {
                    if (res.value > 0 && res.enabled > 0 && res.enabled != res.running)
                    {
                        double value = (double)res.value;
                        double enabled = (double)res.enabled;
                        double running = (double)res.running;
                        value *= (enabled/running);
                        res.value = (uint64_t)value;
                        VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, SCALE_POWER);
                    }
                    eventSet->events[i].threadCounter[thread_id].startData = res.value;
                }
                else
                {
                    ERROR_PRINT(Failed to read FD %d HW thread %d RegIdx %d, cpu_event_fds[cpu_id][index], cpu_id, index);
                }
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
            PerfEventResult res = {0ULL, 0ULL, 0ULL};
            ret = read(cpu_event_fds[cpu_id][index], &res, sizeof(PerfEventResult));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, READ_COUNTER);
            if (ret == sizeof(PerfEventResult))
            {
                if (res.value > 0 && res.enabled > 0 && res.enabled != res.running)
                {
                    double value = (double)res.value;
                    double enabled = (double)res.enabled;
                    double running = (double)res.running;
                    value *= (enabled/running);
                    res.value = (uint64_t)value;
                    VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, SCALE_COUNTER);
                }
                eventSet->events[i].threadCounter[thread_id].counterData = res.value;
            }
            else
            {
                ERROR_PRINT(Failed to read FD %d HW thread %d RegIdx %d, cpu_event_fds[cpu_id][index], cpu_id, index);
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
    long long tmp[3] = {0};
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
            PerfEventResult res = {0ULL, 0ULL, 0ULL};
            ret = read(cpu_event_fds[cpu_id][index], &res, sizeof(PerfEventResult));
            VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, READ_COUNTER);
            if (ret == sizeof(PerfEventResult))
            {
                if (res.value > 0 && res.enabled > 0 && res.enabled != res.running)
                {
                    double value = (double)res.value;
                    double enabled = (double)res.enabled;
                    double running = (double)res.running;
                    value *= (enabled/running);
                    res.value = (uint64_t)value;
                    VERBOSEPRINTREG(cpu_id, cpu_event_fds[cpu_id][index], res.value, SCALE_COUNTER);
                }
                eventSet->events[i].threadCounter[thread_id].counterData = res.value;
            }
            else
            {
                ERROR_PRINT(Failed to read FD %d HW thread %d RegIdx %d, cpu_event_fds[cpu_id][index], cpu_id, index);
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
