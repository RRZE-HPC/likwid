








#include <perfmon_a15_events.h>
#include <perfmon_a15_counters.h>

#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>

static int perfmon_numCountersA15 = NUM_COUNTERS_A15;
static int perfmon_numArchEventsA15 = NUM_ARCH_EVENTS_A15;

int perfmon_init_a15(int cpu_id)
{
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    return 0;
}

int a15_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent *event)
{
    uint64_t flags = 0x0ULL;
    flags |= event->eventId;
    //flags |= (1ULL << 31);
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].configRegister, flags));
    VERBOSEPRINTREG(cpu_id, counter_map[index].configRegister, flags, SETUP_PMC)
    return 0;

}

int perfmon_setupCounterThread_a15(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;

    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
        {
            case PMC:
                a15_pmc_setup(cpu_id, index, event);
                break;
        }
    }

    return 0;
}


int perfmon_startCountersThread_a15(int thread_id, PerfmonEventSet* eventSet)
{
    uint32_t flags = 0x0U;
    int cpu_id = groupSet->threads[thread_id].processorId;


    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        switch (type)
        {
            case PMC:
                CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter_map[index].counterRegister, 0x0ULL));
                flags |= (1ULL<<index);
                break;
        }
    }
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_COUNT_ENABLE, flags|(1ULL<<31)));
    VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, flags, START_COUNTERS)
    return 0;
}

int perfmon_stopCountersThread_a15(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0U;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t pmc_overflows = 0x0ULL;

    if MEASURE_CORE(eventSet)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, A15_COUNT_ENABLE, &flags));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, flags, SAFE_COUNTERS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_COUNT_ENABLE, (1ULL<<31)));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, (1ULL<<31), CLEAR_COUNTERS)
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, A15_OVERFLOW_FLAGS, &pmc_overflows));
        VERBOSEPRINTREG(cpu_id, A15_OVERFLOW_FLAGS, pmc_overflows, READ_OVERFLOWS)
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
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, counter_result, READ_PMC)
                    if (pmc_overflows & index)
                    {
                        eventSet->events[i].threadCounter[thread_id].overflows++;
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_OVERFLOW_FLAGS, (1ULL<<index)));
                    }
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, counter1, 0x0ULL));
                    break;
                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    return 0;
}

int perfmon_readCountersThread_a15(int thread_id, PerfmonEventSet* eventSet)
{
    uint64_t flags = 0x0U;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t pmc_overflows = 0x0ULL;

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, A15_COUNT_ENABLE, &flags));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, flags, SAFE_COUNTERS)
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_COUNT_ENABLE, (1ULL<<31)));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, (1ULL<<31), CLEAR_COUNTERS)
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, A15_OVERFLOW_FLAGS, &pmc_overflows));
        VERBOSEPRINTREG(cpu_id, A15_OVERFLOW_FLAGS, pmc_overflows, READ_OVERFLOWS)
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
            counter_result = 0x0ULL;
            RegisterIndex index = eventSet->events[i].index;
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, counter_result, READ_PMC)
                    if (pmc_overflows & index)
                    {
                        eventSet->events[i].threadCounter[thread_id].overflows++;
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_OVERFLOW_FLAGS, (1ULL<<index)));
                    }
                    break;
                default:
                    break;
            }
            *current = field64(counter_result, 0, box_map[type].regWidth);
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }

    if (MEASURE_CORE(eventSet))
    {
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, A15_COUNT_ENABLE, flags));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, flags, START_COUNTERS)
    }

    return 0;
}


int perfmon_finalizeCountersThread_a15(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (type == NOTYPE || !TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        uint64_t reg = counter_map[index].configRegister;
        switch (type)
        {
            case PMC:
                VERBOSEPRINTREG(cpu_id, reg, 0x0ULL, CLEAR_PMC)
                HPMwrite(cpu_id, MSR_DEV, reg, 0x0ULL);
                break;
            default:
                break;
        }
        eventSet->events[i].threadCounter[thread_id].init = FALSE;
    }
    if (MEASURE_CORE(eventSet))
    {
        HPMwrite(cpu_id, MSR_DEV, A15_COUNT_ENABLE, (1ULL<<31));
        VERBOSEPRINTREG(cpu_id, A15_COUNT_ENABLE, (1ULL<<31), CLEAR_PMC_GLOBAL)
    }
    return 0;
}
