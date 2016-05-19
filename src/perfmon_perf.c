/*
 * =======================================================================================
 *
 *      Filename:  perfmon_perf.c
 *
 *      Description:  Example perfmon module for software events through perf_event
 *                    Currently not integrated in perfmon.
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

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#include <topology.h>
#include <error.h>
#include <perfmon.h>
#include <perfmon_perf.h>

static int* cpu_event_fds[MAX_NUM_THREADS] = { NULL };

const uint64_t configList[MAX_SW_EVENTS] = {
    [0x00] = PERF_COUNT_SW_CPU_CLOCK,
    [0x01] = PERF_COUNT_SW_TASK_CLOCK,
    [0x02] = PERF_COUNT_SW_PAGE_FAULTS,
    [0x03] = PERF_COUNT_SW_CONTEXT_SWITCHES,
    [0x04] = PERF_COUNT_SW_CPU_MIGRATIONS,
    [0x05] = PERF_COUNT_SW_PAGE_FAULTS_MIN,
    [0x06] = PERF_COUNT_SW_PAGE_FAULTS_MAJ,
    [0x07] = PERF_COUNT_SW_ALIGNMENT_FAULTS,
    [0x08] = PERF_COUNT_SW_EMULATION_FAULTS,
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

int init_perf_event(int cpu_id)
{
    if (cpu_event_fds[cpu_id] == NULL)
    {
        cpu_event_fds[cpu_id] = (int*) malloc(MAX_SW_EVENTS * sizeof(int));
        if (cpu_event_fds[cpu_id] == NULL)
        {
            return -ENOMEM;
        }
        memset(cpu_event_fds[cpu_id], -1, MAX_SW_EVENTS * sizeof(int));
    }
    return 0;
}

int setup_perf_event(int cpu_id, PerfmonEvent* event)
{
    struct perf_event_attr attr;
    if (event == NULL)
    {
        return -EINVAL;
    }
    if (cpu_event_fds[cpu_id] == NULL)
    {
        return -EFAULT;
    }
    if (cpu_event_fds[cpu_id][event->umask] != -1)
    {
        return 0;
    }
    memset(&attr, 0, sizeof(struct perf_event_attr));
    attr.type = PERF_TYPE_SOFTWARE;
    attr.size = sizeof(struct perf_event_attr);
    attr.config = configList[event->umask];
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.disabled = 1;
    attr.inherit = 1;
    if (event->numberOfOptions > 0)
    {
        for(int j = 0; j < event->numberOfOptions; j++)
        {
            switch (event->options[j].type)
            {
                case EVENT_OPTION_COUNT_KERNEL:
                    attr.exclude_kernel = 0;
                    break;
                default:
                    break;
            }
        }
    }
    cpu_event_fds[cpu_id][event->umask] = perf_event_open(&attr, 0, cpu_id, -1, 0);
    if (cpu_event_fds[cpu_id][event->umask] < 0)
    {
        printf("Setup of event %llu failed\n", event->umask);
        return -EFAULT;
    }
    return 0;
}

int read_perf_event(int cpu_id, uint64_t eventID, uint64_t *data)
{
    int ret = 0;
    long long tmp = 0;
    *data = 0x0ULL;
    if ((cpu_event_fds[cpu_id] != NULL) && (cpu_event_fds[cpu_id][eventID] != -1))
    {
        ret = read(cpu_event_fds[cpu_id][eventID], &tmp, sizeof(long long));
        if (ret == sizeof(long long))
        {
            *data = (uint64_t) tmp;
        }
    }
    else
    {
        printf("FD for event %llu not initialized\n", eventID);
        return -ENODEV;
    }
    return 0;
}

int stop_perf_event(int cpu_id, uint64_t eventID)
{
    if ((cpu_event_fds[cpu_id] != NULL) && (cpu_event_fds[cpu_id][eventID] != -1))
    {
        ioctl(cpu_event_fds[cpu_id][eventID], PERF_EVENT_IOC_DISABLE, 0);
    }
    else
    {
        return -ENODEV;
    }
    return 0;
}

int stop_all_perf_event(int cpu_id)
{
    if (cpu_event_fds[cpu_id] != NULL)
    {
        for (int i = 0; i< MAX_SW_EVENTS; i++)
        {
            if (cpu_event_fds[cpu_id][i] != -1)
            {
                stop_perf_event(cpu_id, i);
            }
        }
    }
    return 0;
}

int clear_perf_event(int cpu_id, uint64_t eventID)
{
    if ((cpu_event_fds[cpu_id] != NULL) && (cpu_event_fds[cpu_id][eventID] != -1))
    {
        ioctl(cpu_event_fds[cpu_id][eventID], PERF_EVENT_IOC_RESET, 0);
    }
    else
    {
        return -ENODEV;
    }
    return 0;
}

int clear_all_perf_event(int cpu_id)
{
    if (cpu_event_fds[cpu_id] != NULL)
    {
        for (int i = 0; i< MAX_SW_EVENTS; i++)
        {
            if (cpu_event_fds[cpu_id][i] != -1)
            {
                clear_perf_event(cpu_id, i);
            }
        }
    }
    return 0;
}

int start_perf_event(int cpu_id, uint64_t eventID)
{
    if ((cpu_event_fds[cpu_id] != NULL) && (cpu_event_fds[cpu_id][eventID] != -1))
    {
        ioctl(cpu_event_fds[cpu_id][eventID], PERF_EVENT_IOC_ENABLE, 0);
    }
    else
    {
        return -ENODEV;
    }
    return 0;
}

int start_all_perf_event(int cpu_id)
{
    if (cpu_event_fds[cpu_id] != NULL)
    {
        for (int i = 0; i< MAX_SW_EVENTS; i++)
        {
            if (cpu_event_fds[cpu_id][i] != -1)
            {
                start_perf_event(cpu_id, i);
            }
        }
    }
    return 0;
}

int close_perf_event(int cpu_id, uint64_t eventID)
{
    if ((cpu_event_fds[cpu_id] != NULL) && (cpu_event_fds[cpu_id][eventID] != -1))
    {
        close(cpu_event_fds[cpu_id][eventID]);
        cpu_event_fds[cpu_id][eventID] = -1;
    }
    return 0;
}

int finalize_perf_event(int cpu_id)
{
    if (cpu_event_fds[cpu_id] != NULL)
    {
        for (int i = 0; i< MAX_SW_EVENTS; i++)
        {
            if (cpu_event_fds[cpu_id][i] != -1)
            {
                close_perf_event(cpu_id, i);
            }
        }
        free(cpu_event_fds[cpu_id]);
    }
    
    return 0;
}
