#include <error.h>
#include <affinity.h>
#include <limits.h>
#include <topology.h>
#include <access.h>
#include <perfmon_power8_counters.h>
#include <perfmon_power8_events.h>

static int perfmon_numCountersPower8 = NUM_COUNTERS_POWER8;
static int perfmon_numCoreCountersPower8 = NUM_COUNTERS_POWER8;
static int perfmon_numArchEventsPower8 = NUM_ARCH_EVENTS_POWER8;

//#define MMCR1_UNIT_SHIFT(pmc)           ((4 * (pmc)))
//#define MMCR1_PMCSEL_SHIFT(pmc)         (32 + ((pmc) * 8))
#define MMCR1_UNIT_SHIFT(pmc)           (60 - (4 * ((pmc))))
#define MMCR1_UNIT_MASK			0xFULL
#define MMCR1_COMBINE_SHIFT(pmc)        (35 - ((pmc)))
#define MMCR1_PMCSEL_SHIFT(pmc)         (24 - (((pmc))) * 8)
#define MMCR1_PMCSEL_MASK		0xFFULL
#define MMCR1_FAB_SHIFT                 36
#define MMCR1_FAB_MASK			0xFFULL
#define MMCR1_DC_QUAL_SHIFT             47
#define MMCR1_IC_QUAL_SHIFT             46

#define MMCR0_FREEZE_VALUE		0x86006000ULL

#define   MMCRA_SDSYNC  0x80000000UL /* SDAR synced with SIAR */
#define   MMCRA_SDAR_DCACHE_MISS 0x40000000UL
#define   MMCRA_SDAR_ERAT_MISS   0x20000000UL
#define   MMCRA_SIHV    0x10000000UL /* state of MSR HV when SIAR set */
#define   MMCRA_SIPR    0x08000000UL /* state of MSR PR when SIAR set */
#define   MMCRA_SLOT    0x07000000UL /* SLOT bits (37-39) */
#define   MMCRA_SLOT_SHIFT      24
#define   MMCRA_SAMPLE_ENABLE 0x00000001UL /* enable sampling */


int perfmon_init_power8(int cpu_id)
{   
    lock_acquire((int*) &tile_lock[affinity_thread2tile_lookup[cpu_id]], cpu_id);
    lock_acquire((int*) &socket_lock[affinity_core2node_lookup[cpu_id]], cpu_id);
    //CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR0, (1ULL<<44)|(1ULL<<45)|(1ULL<<15)|(1ULL<<25)|(1ULL<<26)|(1ULL<<31)));
    //CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR0, MMCR0_FREEZE_VALUE));
    //CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR1, 0x0ULL));
    //CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCRA, 0x40000000000ULL));
    return 0;
}



	/* bitmap for cfgBits
	 * |7   6   5   4|3   2    1    0|
	 * 0: Currently unused and cannot be used, some events set this bit
	 * 1: IC_RLD_QUAL bit in MMCR1
	 * 2: DC_RLD_QUAL bit in MMCR1
	 * 3: Currently unused and cannot be used, some events set this bit
	 * 4: Set the combine flag for the current counter
	 * 5: Set the mark flag for current counter
	 * 6: Cmask value is used for thresholds 
	 */

	/* bitmap for cmask
	 * if bit 6 in cfgBits is 0 the cmask is the value for FAB matching
         *    [7:0] : FAB match value
	 * if bit 6 in cfgBits is 1 the cmask is used for thresholds
	 *    [2:0] : Threshold select
	 *    [3] : must be 0
	 *    [11:4] : Threshold control
         *    [21:12] : Threshold compare
	 */

uint64_t power8_pmc_setup(int cpu_id, RegisterIndex index, PerfmonEvent* event)
{
    int j = 0;
    uint64_t flags = 0x0ULL;
    uint64_t pmcsel = event->eventId;
    /*if (event->umask >= 6 && event->umask <= 9 && event->umask & 0x7)
    {
	return flags;
    }*/
    int pmc_num = getCounterTypeOffset(index); 
    if (pmc_num > 3)
	return 0;
    /*printf("Counter PMC%d\n", pmc_num);
    printf("EventID: 0x%X\n", event->eventId);
    printf("Umask: 0x%X\n", event->umask);
    printf("CfgBits: 0x%X\n", event->cfgBits);
    printf("Cmask: 0x%X\n", event->cmask);*/
    for(j=0;j<event->numberOfOptions;j++)
    {
        switch (event->options[j].type)
        {
	    case EVENT_OPTION_EDGE:
		pmcsel |= (1<<7);
		DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flag EDGE_DETECT, pmc_num);
		break;
	}
    }
    flags |= ((pmcsel & MMCR1_PMCSEL_MASK) << MMCR1_PMCSEL_SHIFT(pmc_num));
    flags |= (((uint64_t)event->umask & MMCR1_UNIT_MASK) << MMCR1_UNIT_SHIFT(pmc_num)); 
    if (event->cfgBits != 0x0)
    {
    
	if (event->cfgBits & 0x4)
	{
	    DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flag DC_RLD_QUAL, pmc_num);
	    flags |= 1ULL<<MMCR1_DC_QUAL_SHIFT;
	}
	if (event->cfgBits & 0x2)
	{
	    DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flag IC_RLD_QUAL, pmc_num);
	    flags |= 1ULL<<MMCR1_IC_QUAL_SHIFT;
	}
	if (event->cfgBits & 0x10 && pmc_num >= 0 && pmc_num <=3)
	{
	    DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flag COMBINE, pmc_num);
	    flags |= 1ULL << MMCR1_COMBINE_SHIFT(pmc_num);
	}
    }
    // The cmask contains the FAB match bits (MMCR1 bits 20-27) 
    if (event->cmask != 0x0 && (event->cmask >> 63) == 0)
    {
	DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flag FAB_MATCH, pmc_num);
	flags |= (event->cmask & MMCR1_FAB_MASK) << MMCR1_FAB_SHIFT;
    }
    else if (event->cmask != 0x0)
    {
	DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with thresholds. Currently noop,  pmc_num);		
    } 
    DEBUG_PRINT(DEBUGLEV_DETAIL, Setup PMC %d with flags 0x%lx, pmc_num, flags);
    return flags;
}

//    flags = 0x0ULL; \
//    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, IBM_MMCR0, &flags)); \
//    VERBOSEPRINTREG(cpu_id, IBM_MMCR0, LLU_CAST flags, BEFORE_FREEZE); \
//    flags |= (1ULL<<15)|(1ULL<<25)|(1ULL<<26)|(1ULL<<31); \
//    flags |= (1ULL<<44)|(1ULL<<45); \

#define POWER8_FREEZE \
    flags = MMCR0_FREEZE_VALUE; \
    VERBOSEPRINTREG(cpu_id, IBM_MMCR0, LLU_CAST flags, FREEZE); \
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR0, flags)); 

    
//    flags = 0x0ULL; \
//    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, IBM_MMCR0, &flags)); \
//    VERBOSEPRINTREG(cpu_id, IBM_MMCR0, LLU_CAST flags, BEFORE_UNFREEZE); \
//    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, IBM_MMCR0, &flags)); \
//    VERBOSEPRINTREG(cpu_id, IBM_MMCR0, LLU_CAST flags, BEFORE_UNFREEZE); \

#define POWER8_UNFREEZE \
    flags = 0x0ULL; \
    VERBOSEPRINTREG(cpu_id, IBM_MMCR0, LLU_CAST flags, UNFREEZE); \
    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR0, flags)); 

#define POWER8_CHECK_CORE_OVERFLOW(index) \
    if (*current < eventSet->events[i].threadCounter[thread_id].startData) \
    { \
	DEBUG_PRINT(DEBUGLEV_DETAIL, OVERFLOW_IN_REGISTER %s, counter_map[index].key); \
	eventSet->events[i].threadCounter[thread_id].overflows++; \
    } 
	
    

int perfmon_setupCounterThread_power8(
        int thread_id,
        PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t fixed_flags = 0x0ULL;
    uint64_t mask = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;


    POWER8_FREEZE;
    flags = 0x0ULL;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterType type = eventSet->events[i].type;
        if (!TESTTYPE(eventSet, type))
        {
            continue;
        }
        RegisterIndex index = eventSet->events[i].index;
        PerfmonEvent *event = &(eventSet->events[i].event);
        uint64_t reg = counter_map[index].configRegister;
        int pmc_num = getCounterTypeOffset(index);
        eventSet->events[i].threadCounter[thread_id].init = TRUE;
        switch (type)
        {
            case PMC:
                flags |= power8_pmc_setup(cpu_id, index, event);
                break;
            default:
                break;
        }
    }
    if (flags != 0x0ULL)
    {
        CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, IBM_MMCR1, &fixed_flags));
        fixed_flags |= flags;
        VERBOSEPRINTREG(cpu_id, IBM_MMCR1, LLU_CAST fixed_flags, SETUP_PMC_ALL);
        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, MSR_DEV, IBM_MMCR1, fixed_flags));
    }
    return 0;
}


int perfmon_startCountersThread_power8(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t tmp = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;

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
            int pmc_num = getCounterTypeOffset(index); 
            uint64_t counter1 = counter_map[index].counterRegister;
            PciDeviceIndex dev = counter_map[index].device;
            switch (type)
            {
                case PMC:
                    if (pmc_num <= 5)
                    {
                        CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, counter1, 0x0ULL));
                    }
                    else
                    {
                        CHECK_MSR_READ_ERROR(HPMread(cpu_id, dev, counter1, &flags));
                        VERBOSEPRINTREG(cpu_id, counter1, flags, START_PMC)
                        eventSet->events[i].threadCounter[thread_id].startData = field64(flags, 0, box_map[type].regWidth);
                    }
                    break;
            }
        }
    }

    POWER8_UNFREEZE;
    return 0;
}

int perfmon_stopCountersThread_power8(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags = 0x0ULL;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;


    POWER8_FREEZE;
    flags = 0x0ULL;

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
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            int pmc_num = getCounterTypeOffset(index); 
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST field64(counter_result, 0, box_map[type].regWidth), STOP_PMC)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    POWER8_CHECK_CORE_OVERFLOW(index);
                    break;
                default:
                    break;
            }
            eventSet->events[i].threadCounter[thread_id].init = FALSE;
        }
    }
    return 0;
}


int perfmon_readCountersThread_power8(int thread_id, PerfmonEventSet* eventSet)
{
    int haveLock = 0;
    uint64_t flags;
    uint64_t counter_result = 0x0ULL;
    int cpu_id = groupSet->threads[thread_id].processorId;


    POWER8_FREEZE;


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
            PerfmonEvent *event = &(eventSet->events[i].event);
            PciDeviceIndex dev = counter_map[index].device;
            int pmc_num = getCounterTypeOffset(index); 
            uint64_t counter1 = counter_map[index].counterRegister;
            uint64_t* current = &(eventSet->events[i].threadCounter[thread_id].counterData);
            int* overflows = &(eventSet->events[i].threadCounter[thread_id].overflows);
            int ovf_offset = box_map[type].ovflOffset;
            switch (type)
            {
                case PMC:
                    CHECK_MSR_READ_ERROR(HPMread(cpu_id, MSR_DEV, counter1, &counter_result));
                    VERBOSEPRINTREG(cpu_id, counter1, LLU_CAST field64(counter_result, 0, box_map[type].regWidth), READ_PMC)
                    *current = field64(counter_result, 0, box_map[type].regWidth);
                    POWER8_CHECK_CORE_OVERFLOW(index);
                    break;
                default:
                    break;
            }
        }
    }
    POWER8_UNFREEZE;
    return 0;
}



int perfmon_finalizeCountersThread_power8(int thread_id, PerfmonEventSet* eventSet)
{
    int cpu_id = groupSet->threads[thread_id].processorId;
    uint64_t lastreg = -1;
    for (int i=0;i < eventSet->numberOfEvents;i++)
    {
        RegisterIndex index = eventSet->events[i].index;
        PciDeviceIndex dev = counter_map[index].device;
        uint64_t reg = counter_map[index].configRegister;
        RegisterType type = eventSet->events[i].type;
        if (type == NOTYPE)
        {
            continue;
        }
        switch (type)
        {
            case PMC:
                if (reg != lastreg)
                {
                    VERBOSEPRINTPCIREG(cpu_id, dev, reg, 0x0ULL, CLEAR_CTL);
                    CHECK_MSR_WRITE_ERROR(HPMwrite(cpu_id, dev, reg, 0x0ULL));
                    lastreg = reg;
                }
                break;
            default:
                break;
        }
    }
    return 0;
}

