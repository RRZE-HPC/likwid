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
 *      Author:   Thomas Roehl (tr), thomas.roehl@googlemail.com
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


static int* cpu_event_fds[MAX_NUM_THREADS] = { NULL };
static int paranoid_level = -1;
static int informed_paranoid = 0;
static int running_group = -1;

static char* translate_types[NUM_UNITS] = {
    [FIXED] = "/sys/bus/event_source/devices/cpu",
    [PMC] = "/sys/bus/event_source/devices/cpu",
    [MBOX0] = "/sys/bus/event_source/devices/uncore_imc_0",
    [MBOX1] = "/sys/bus/event_source/devices/uncore_imc_1",
    [MBOX2] = "/sys/bus/event_source/devices/uncore_imc_2",
    [MBOX3] = "/sys/bus/event_source/devices/uncore_imc_3",
    [CBOX0] = "/sys/bus/event_source/devices/uncore_cbox_0",
    [CBOX1] = "/sys/bus/event_source/devices/uncore_cbox_1",
    [CBOX2] = "/sys/bus/event_source/devices/uncore_cbox_2",
    [CBOX3] = "/sys/bus/event_source/devices/uncore_cbox_3",
    [CBOX4] = "/sys/bus/event_source/devices/uncore_cbox_4",
    [CBOX5] = "/sys/bus/event_source/devices/uncore_cbox_5",
    [CBOX6] = "/sys/bus/event_source/devices/uncore_cbox_6",
    [CBOX7] = "/sys/bus/event_source/devices/uncore_cbox_7",
    [CBOX8] = "/sys/bus/event_source/devices/uncore_cbox_8",
    [CBOX9] = "/sys/bus/event_source/devices/uncore_cbox_9",
    [CBOX10] = "/sys/bus/event_source/devices/uncore_cbox_10",
    [CBOX11] = "/sys/bus/event_source/devices/uncore_cbox_11",
    [CBOX12] = "/sys/bus/event_source/devices/uncore_cbox_12",
    [CBOX13] = "/sys/bus/event_source/devices/uncore_cbox_13",
    [CBOX14] = "/sys/bus/event_source/devices/uncore_cbox_14",
    [CBOX15] = "/sys/bus/event_source/devices/uncore_cbox_15",
    [CBOX16] = "/sys/bus/event_source/devices/uncore_cbox_16",
    [CBOX17] = "/sys/bus/event_source/devices/uncore_cbox_17",
    [CBOX18] = "/sys/bus/event_source/devices/uncore_cbox_18",
    [CBOX19] = "/sys/bus/event_source/devices/uncore_cbox_19",
    [CBOX20] = "/sys/bus/event_source/devices/uncore_cbox_20",
    [CBOX21] = "/sys/bus/event_source/devices/uncore_cbox_21",
    [CBOX22] = "/sys/bus/event_source/devices/uncore_cbox_22",
    [CBOX23] = "/sys/bus/event_source/devices/uncore_cbox_23",
    [BBOX0] = "/sys/bus/event_source/devices/uncore_ha",
    [WBOX] = "/sys/bus/event_source/devices/uncore_pcu",
    [SBOX0] = "/sys/bus/event_source/devices/uncore_qpi_0",
    [SBOX1] = "/sys/bus/event_source/devices/uncore_qpi_1",
    [PBOX] = "/sys/bus/event_source/devices/uncore_r2pcie",
    [RBOX0] = "/sys/bus/event_source/devices/uncore_r3qpi_0",
    [RBOX1] = "/sys/bus/event_source/devices/uncore_r3qpi_1",
    [UBOX] = "/sys/bus/event_source/devices/uncore_ubox",
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

int perfmon_init_perfevent(int cpu_id)
{
    size_t read;
    int paranoid = -1;
    char buff[100];
    FILE* fd;
    if (!informed_paranoid)
    {
        fd = fopen("/proc/sys/kernel/perf_event_paranoid", "r");
        if (fd == NULL)
        {
            fprintf(stderr, "ERROR: Linux kernel has no perf_event support\n");
            fprintf(stderr, "ERROR: Cannot open file /proc/sys/kernel/perf_event_paranoid\n");
            fclose(fd);
            exit(EXIT_FAILURE);
        }
        read = fread(buff, sizeof(char), 100, fd);
        if (read > 0)
        {
            paranoid_level = atoi(buff);
        }
        fclose(fd);
        if (paranoid_level > 0)
        {
            fprintf(stderr, "WARN: Linux kernel configured with paranoid level %d\n", paranoid_level);
#if defined(__x86_64__) || defined(__i386__)
            fprintf(stderr, "WARN: Paranoid level 0 is required to measure Uncore counters\n");
#endif
        }
        informed_paranoid = 1;
    }
    if (cpu_event_fds[cpu_id] == NULL)
    {
        cpu_event_fds[cpu_id] = (int*) malloc(perfmon_numCounters * sizeof(int));
        if (cpu_event_fds[cpu_id] == NULL)
        {
            return -ENOMEM;
        }
        memset(cpu_event_fds[cpu_id], -1, perfmon_numCounters * sizeof(int));
    }
    return 0;
}

int perf_fixed_setup(struct perf_event_attr *attr, PerfmonEvent *event)
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

int perf_pmc_setup(struct perf_event_attr *attr, PerfmonEvent *event)
{
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
                default:
                    break;
            }
        }
    }
    
    return 0;
}

int perf_uncore_setup(struct perf_event_attr *attr, RegisterType type, PerfmonEvent *event)
{

    char checkfolder[1024];
    int ret;
    FILE* fp;
    int perf_type;
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
    attr->exclude_kernel = 1;
    attr->exclude_hv = 1;
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
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        if (cpu_event_fds[cpu_id][index] != -1)
        {
            continue;
        }
        RegisterType type = eventSet->events[i].type;
        PerfmonEvent *event = &(eventSet->events[i].event);
        memset(&attr, 0, sizeof(struct perf_event_attr));
        attr.size = sizeof(struct perf_event_attr);
        switch (type)
        {
            case FIXED:
                ret = perf_fixed_setup(&attr, event);
                if (ret < 0)
                {
                    continue;
                }
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_FIXED);
                break;
            case PMC:
                ret = perf_pmc_setup(&attr, event);
                VERBOSEPRINTREG(cpu_id, index, attr.config, SETUP_PMC);
                break;
#if LINUX_VERSION_CODE >= KERNEL_VERSION(3,5,0)
            case MBOX0:
            case MBOX1:
            case MBOX2:
            case MBOX3:
            case CBOX0:
            case CBOX1:
            case CBOX2:
            case CBOX3:
            case CBOX4:
            case CBOX5:
            case CBOX6:
            case CBOX7:
            case UBOX:
            case SBOX0:
            case SBOX1:
            case WBOX:
            case PBOX:
            case RBOX0:
            case RBOX1:
            case BBOX0:
                ret = perf_uncore_setup(&attr, type, event);
                break;
#endif
            default:
                break;
        }
        if (ret == 0)
        {
            cpu_event_fds[cpu_id][index] = perf_event_open(&attr, 0, cpu_id, -1, 0);
            if (cpu_event_fds[cpu_id][index] < 0)
            {
                fprintf(stderr, "Setup of event %s on CPU %d failed: %s\n", event->name, cpu_id, strerror(errno));
                fprintf(stderr, "Config of event 0x%X\n", attr.config);
                fprintf(stderr, "Type of event 0x%X\n", attr.type);
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
    return 0;
}

int perfmon_startCountersThread_perfevent(int thread_id, PerfmonEventSet* eventSet)
{
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
            VERBOSEPRINTREG(cpu_id, 0x0, 0x0, START_COUNTER);
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
            DEBUG_PRINT(DEBUGLEV_DEVELOP, READ CPU %d COUNTER %d VALUE %llu, cpu_id, index, tmp);
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
    free(cpu_event_fds[cpu_id]);
    return 0;
}
