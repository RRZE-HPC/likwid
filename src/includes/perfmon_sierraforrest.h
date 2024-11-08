/*
 * =======================================================================================
 *
 *      Filename:  perfmon_sierraforrest.h
 *
 *      Description:  Header File of perfmon module for Intel SierraForrest.
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


#include <perfmon_sierraforrest_counters.h>
#include <perfmon_sierraforrest_events.h>


static int perfmon_numCountersSierraForrest = NUM_COUNTERS_SIERRAFORREST;
static int perfmon_numCoreCountersSierraForrest = NUM_COUNTERS_CORE_SIERRAFORREST;
static int perfmon_numArchEventsSierraForrest = NUM_ARCH_EVENTS_SIERRAFORREST;

int perfmon_init_sierraforrest(int cpu_id)
{
    int ret = 0;
    lock_acquire((int*) &tile_lock[affinity_thread2core_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_thread2socket_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &die_lock[affinity_thread2die_lookup[cpu_id]], cpu_id);

/*    uint64_t misc_enable = 0x0;*/
/*    ret = HPMread(cpu_id, MSR_DEV, MSR_IA32_MISC_ENABLE, &misc_enable);*/
/*    if (ret == 0 && (testBit(misc_enable, 7) == 1) && (testBit(misc_enable, 12) == 0))*/
/*    {*/
/*        ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_ENABLE, 0x0ULL);*/
/*        if (ret != 0)*/
/*        {*/
/*            ERROR_PRINT(Cannot zero %s (0x%X), str(MSR_PEBS_ENABLE), MSR_PEBS_ENABLE);*/
/*        }*/
/*        ret = HPMwrite(cpu_id, MSR_DEV, MSR_PEBS_FRONTEND, 0x0ULL);*/
/*        if (ret != 0)*/
/*        {*/
/*            ERROR_PRINT(Cannot zero %s (0x%X), str(MSR_PEBS_FRONTEND), MSR_PEBS_FRONTEND);*/
/*        }*/
/*    }*/
    return 0;
}

uint64_t srf_fixed_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    return spr_fixed_setup(thread_id, index, event, data);
}

uint64_t srf_fixed_start(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    return spr_fixed_start(thread_id, index, event, data);
}

uint64_t srf_pmc_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    return spr_pmc_setup(thread_id, index, event, data);
}

uint64_t srf_uncore_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    return spr_setup_uncore(thread_id, index, event);
}

uint64_t srf_uncore_fixed_setup(int thread_id, RegisterIndex index, PerfmonEvent *event, PerfmonCounter* data)
{
    return spr_setup_uncore_fixed(thread_id, index, event);
}

uint64_t srf_pmc_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_pmc_start(thread_id, index, event, data);
}

uint64_t srf_power_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_power_start(thread_id, index, event, data);
}

uint64_t srf_metrics_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_metrics_start(thread_id, index, event, data);
}

uint64_t srf_uncore_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_start_uncore(thread_id, index, event, data);
}

uint64_t srf_uncore_fixed_start(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_start_uncore_fixed(thread_id, index, event, data);
}

uint64_t srf_fixed_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_fixed_stop(thread_id, index, event, data);
}

uint64_t srf_pmc_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_pmc_stop(thread_id, index, event, data);
}

uint64_t srf_power_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_power_stop(thread_id, index, event, data);
}

uint64_t srf_thermal_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_thermal_stop(thread_id, index, event, data);
}

uint64_t srf_voltage_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_voltage_stop(thread_id, index, event, data);
}

uint64_t srf_metrics_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_metrics_stop(thread_id, index, event, data);
}

uint64_t srf_uncore_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_stop_uncore(thread_id, index, event, data);
}

uint64_t srf_uncore_fixed_stop(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_stop_uncore_fixed(thread_id, index, event, data);
}

uint64_t srf_fixed_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_fixed_read(thread_id, index, event, data);
}

uint64_t srf_pmc_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_pmc_read(thread_id, index, event, data);
}

uint64_t srf_power_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_power_read(thread_id, index, event, data);
}

uint64_t srf_thermal_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_thermal_read(thread_id, index, event, data);
}

uint64_t srf_voltage_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_voltage_read(thread_id, index, event, data);
}

uint64_t srf_metrics_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_metrics_read(thread_id, index, event, data);
}

uint64_t srf_uncore_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_read_uncore(thread_id, index, event, data);
}

uint64_t srf_uncore_fixed_read(int thread_id, RegisterIndex index, PerfmonEvent* event, PerfmonCounter* data)
{
    return spr_read_uncore_fixed(thread_id, index, event, data);
}

static PerfmonFuncs SrfUnitFuncs[NUM_UNITS] = {
    [PMC] = {srf_pmc_setup, srf_pmc_start, srf_pmc_stop, srf_pmc_read, 0},
    [FIXED] = {srf_fixed_setup, srf_fixed_start, srf_fixed_stop, srf_fixed_read, 0},
    [POWER] = {NULL, srf_power_start, srf_power_stop, srf_power_read, PERFMON_LOCK_SOCKET},
    [THERMAL] = {NULL, NULL, srf_thermal_stop, srf_thermal_read, 0},
    [VOLTAGE] = {NULL, NULL, srf_voltage_stop, srf_voltage_read, 0},
    [METRICS] = {NULL, srf_metrics_start, srf_metrics_stop, srf_metrics_read, 0},
    [MBOX0 ... MBOX15] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    //[MBOX0FIX ... MBOX15FIX] = {srf_uncore_fixed_setup, srf_uncore_fixed_start, srf_uncore_fixed_stop, srf_uncore_fixed_read, PERFMON_LOCK_SOCKET},
    [UBOX] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [MDF0 ... MDF49] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [QBOX0 ... QBOX3] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [WBOX0 ... WBOX3] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [CBOX0 ... CBOX127] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [BBOX0 ... BBOX31] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [PBOX0 ... PBOX31] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [RBOX0 ... RBOX3] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [IRP0 ... IRP15] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
    [IBOX0 ... IBOX15] = {srf_uncore_setup, srf_uncore_start, srf_uncore_stop, srf_uncore_read, PERFMON_LOCK_SOCKET},
};

int perfmon_setupCounterThread_sierraforrest(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int err = 0;
    int haveLock = 0;
    uint64_t fixed_flags = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, 0xC00000070000000F));
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        PerfmonCounter* data = eventSet->events[i].threadCounter;
        uint64_t reg = counter_map[index].configRegister;
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        
        PerfmonFuncs *unitFuncs = &SrfUnitFuncs[type];
        if (unitFuncs && unitFuncs->setup != NULL)
        {
            haveLock = 0;
            switch (unitFuncs->requiresLock)
            {
                case PERFMON_LOCK_HWTHREAD:
                    haveLock = 1;
                    break;
                case PERFMON_LOCK_SOCKET:
                    haveLock = (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id);
                    break;
            }
            if (haveLock)
            {
                uint64_t ret = unitFuncs->setup(thread_id, index, event, data);
                if (type == FIXED)
                {
                    fixed_flags |= ret;
                }
            }
        }
    }
    if ((fixed_flags > 0x0ULL))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_FIXED_CTR_CTRL, LLU_CAST fixed_flags, SETUP_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_FIXED_CTR_CTRL, fixed_flags));
    }
    return 0;
}

int perfmon_startCountersThread_sierraforrest(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t uflags = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            tmp = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            uint64_t counter1 = counter_map[index].counterRegister;

            PciDeviceIndex dev = counter_map[index].device;
            data[thread_id].startData = 0;
            data[thread_id].counterData = 0;
            
            PerfmonFuncs *unitFuncs = &SrfUnitFuncs[type];
            if (unitFuncs && unitFuncs->start != NULL)
            {
                haveLock = 0;
                switch (unitFuncs->requiresLock)
                {
                    case PERFMON_LOCK_HWTHREAD:
                        haveLock = 1;
                        break;
                    case PERFMON_LOCK_SOCKET:
                        haveLock = (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id);
                        break;
                }
                if (haveLock)
                {
                    uint64_t ret = unitFuncs->start(thread_id, index, event, data);
                    if (type == FIXED || type == PMC)
                    {
                        flags |= ret;
                    }
                }
            }
            data[thread_id].counterData = data[thread_id].startData;
        }
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST 0x0ULL, UNFREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, 0x0ULL);
    }
    if (MEASURE_CORE(eventSet))
    {
        if (flags & (1ULL << 48))
        {
            VERBOSEPRINTREG(cpu_id, MSR_PERF_METRICS, 0x0ULL, CLEAR_METRICS)
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_METRICS, 0x0ULL));
        }
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST (1ULL<<63)|(1ULL<<62)|flags, CLEAR_PMC_AND_FIXED_OVERFLOW)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, (1ULL<<63)|(1ULL<<62)|flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, UNFREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }
    return 0;
}


int perfmon_stopCountersThread_sierraforrest(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int coffset = 0;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, FREEZE_PMC_AND_FIXED)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST (1ULL<<0), FREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, (1ULL<<0));
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
    }

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            tmp = 0x0ULL;
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            PerfmonFuncs *unitFuncs = &SrfUnitFuncs[type];
            if (unitFuncs && unitFuncs->stop != NULL)
            {
                haveLock = 0;
                switch (unitFuncs->requiresLock)
                {
                    case PERFMON_LOCK_HWTHREAD:
                        haveLock = 1;
                        break;
                    case PERFMON_LOCK_SOCKET:
                        haveLock = (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id);
                        break;
                }
                if (haveLock)
                {
                    unitFuncs->stop(thread_id, index, event, data);
                }
            }
        }
    }
    return 0;
}



int perfmon_readCountersThread_sierraforrest(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int coffset = 0;
    uint64_t flags = 0x0ULL;
    uint64_t counter_result = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, &flags));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, SAFE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, 0x0ULL, RESET_PMC_FLAGS)
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST (1ULL<<0), FREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, (1ULL<<0));
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST (1ULL<<0), FREEZE_UNCORE);
        HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, (1ULL<<0));
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        if (eventSet->events[i].threadCounter[thread_id].init == TRUE)
        {
            RegisterType type = eventSet->events[i].type;
            if (!TESTTYPE(eventSet, type))
            {
                continue;
            }
            tmp = 0x0ULL;
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            PerfmonCounter* data = eventSet->events[i].threadCounter;
            
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            PerfmonFuncs *unitFuncs = &SrfUnitFuncs[type];
            if (unitFuncs && unitFuncs->read != NULL)
            {
                haveLock = 0;
                switch (unitFuncs->requiresLock)
                {
                    case PERFMON_LOCK_HWTHREAD:
                        haveLock = 1;
                        break;
                    case PERFMON_LOCK_SOCKET:
                        haveLock = (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id);
                        break;
                }
                if (haveLock)
                {
                    unitFuncs->read(thread_id, index, event, data);
                }
            }
        }
    }
    if (haveLock && MEASURE_UNCORE(eventSet))
    {
        for (int i = MSR_DEV + 1; i < MAX_NUM_PCI_DEVICES; i++)
        {
            if (TESTTYPE(eventSet, i) && box_map[i].device != MSR_DEV)
            {
                VERBOSEPRINTPCIREG(cpu_id, box_map[i].device, box_map[i].ctrlRegister, LLU_CAST 0x0ULL, UNFREEZE_UNIT);
                HPMwrite(cpu_id, box_map[i].device, box_map[i].ctrlRegister, 0x0ULL);
            }
        }
        VERBOSEPRINTPCIREG(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, LLU_CAST 0x0ULL, UNFREEZE_UNCORE);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_UBOX_DEVICE, FAKE_UNC_GLOBAL_CTRL, 0x0ULL));
    }
    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST flags, RESTORE_PMC_FLAGS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, flags));
    }
    return 0;
}




int perfmon_finalizeCountersThread_sierraforrest(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    int haveTileLock = 0;
    int clearPBS = 0;
    uint64_t ovf_values_core = (1ULL<<63)|(1ULL<<62);
    uint64_t ovf_values_uncore = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

    if (socket_lock[affinity_thread2socket_lookup[cpu_id]] == cpu_id)
    {
        haveLock = 1;
    }
    if (tile_lock[affinity_thread2core_lookup[cpu_id]] == cpu_id)
    {
        haveTileLock = 1;
    }
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PciDeviceIndex dev = counter_map[index].device;
        uint64_t reg = counter_map[index].configRegister;
        switch (type)
        {
            case FIXED:
                ovf_values_core |= (1ULL<<(index+32));
                break;
            case PMC:
                ovf_values_core |= (1ULL<<(getCounterTypeOffset(index)));
                break;
            default:
                break;
        }
        if ((reg) && (((type == PMC)||(type == FIXED))||(type == METRICS)|| ((type >= UNCORE && type < NUM_UNITS) && (haveLock))))
        {
            VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            if ((type >= SBOX0) && (type <= SBOX3))
            {
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
            }
            VERBOSEPRINTPCIREG(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL, CLEAR_CTR);
            CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter_map[index].counterRegister, 0x0ULL));
            if (box_map[type].filterRegister1 != 0x0)
            {
                VERBOSEPRINTPCIREG(cpu_id, dev, box_map[type].filterRegister1, 0x0ULL, CLEAR_FILTER);
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, box_map[type].filterRegister1, 0x0ULL));
            }
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    if (MEASURE_CORE(eventSet))
    {
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_OVF_CTRL, LLU_CAST ovf_values_core, CLEAR_GLOBAL_OVF)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_OVF_CTRL, ovf_values_core));
        VERBOSEPRINTREG(cpu_id, MSR_PERF_GLOBAL_CTRL, LLU_CAST 0x0ULL, CLEAR_GLOBAL_CTRL)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, MSR_PERF_GLOBAL_CTRL, 0x0ULL));
    }
    return 0;
}
