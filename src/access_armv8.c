#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <registers.h>
#include <access_armv8.h>

#define gettid() syscall(SYS_gettid)

int access_armv8_init(int cpu_id)
{

    return 0;
}

void access_armv8_finalize(int cpu_id)
{
    return;
}

int access_armv8_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data)
{
#if defined(__ARM_ARCH_8A)

    int ret = 0;
    uint64_t tmp = 0x0ULL;
    uint32_t tmp32 = 0x0ULL;
    uint32_t reg_idx = 0;
    uint32_t hi = 0, lo = 0;
    cpu_set_t cpuset, current;
    ret = sched_getaffinity(gettid(), sizeof(cpu_set_t), &current);
    if (ret < 0)
        return errno;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
    if (ret < 0)
        return errno;
#if defined(__ARM_32BIT_STATE)
    switch(reg)
    {
        case A57_PMC0:
        case A57_PMC1:
        case A57_PMC2:
        case A57_PMC3:
        case A57_PMC4:
        case A57_PMC5:
            reg_idx = (reg - A57_PMC0)/8;
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"(reg_idx) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Select counter register %d, reg_idx);
            __asm__ volatile("mrc p15, 0, %0, c9, c13, 2" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, reg_idx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL0:
        case A57_PERFEVTSEL1:
        case A57_PERFEVTSEL2:
        case A57_PERFEVTSEL3:
        case A57_PERFEVTSEL4:
        case A57_PERFEVTSEL5:
            reg_idx = (reg - A57_PERFEVTSEL0)/8;
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"(reg_idx) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Select config register %d, reg_idx);
            __asm__ volatile("mrc p15, 0, %0, c9, c13, 1" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read config register %d: %lx, reg_idx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_CYCLES:
            //reg_idx = 0x1FU;
            hi = 0x0U, lo = 0x0U;
            //__asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"(reg_idx));
            __asm__ volatile("mrrc p15, 0, %0, %1, c9" : "=r" (lo), "=r" (hi) :: "memory");

            tmp = ((uint64_t)hi << 32) | lo;
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read cycles register %llx, tmp);
            break;
        case A57_COUNT_ENABLE:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 1" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read count_enable register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_COUNT_CLEAR:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 2" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read count_clear register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_OVERFLOW_FLAGS:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 3" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read overflow_flags register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_OVERFLOW_STATUS:
            __asm__ volatile("mrc p15, 0, %0, c9, c14, 3" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read overflow_status register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERF_CONTROL_CTRL:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read global_control register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_EVENTS0:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 6" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read events0 register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_EVENTS1:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 7" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read events1 register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        default:
            tmp = 0x0;
            *data = tmp;
            return -EINVAL;
            break;
    }

#else
    switch(reg)
    {
        case A57_PMC0:
            __asm__ volatile("MRS %0, PMEVCNTR0_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC0, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PMC1:
            __asm__ volatile("MRS %0, PMEVCNTR1_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC1, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PMC2:
            __asm__ volatile("MRS %0, PMEVCNTR2_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC2, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PMC3:
            __asm__ volatile("MRS %0, PMEVCNTR3_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC3, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PMC4:
            __asm__ volatile("MRS %0, PMEVCNTR4_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC4, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PMC5:
            __asm__ volatile("MRS %0, PMEVCNTR5_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PMC5, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL0:
            __asm__ volatile("MRS %0, pmevtyper0_el0" : "=&r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL0, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL1:
            __asm__ volatile("MRS %0, PMEVTYPER1_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL1, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL2:
            __asm__ volatile("MRS %0, PMEVTYPER2_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL2, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL3:
            __asm__ volatile("MRS %0, PMEVTYPER3_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL3, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL4:
            __asm__ volatile("MRS %0, PMEVTYPER4_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL4, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERFEVTSEL5:
            __asm__ volatile("MRS %0, PMEVTYPER5_EL0" : "=r"(tmp32) ::);
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read counter register %d: %lx, A57_PERFEVTSEL5, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_CYCLES:
            __asm__ volatile("MRS %0, PMCCNTR_EL0": "=r" (tmp));
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read cycles register %llx, tmp);
            break;
        case A57_COUNT_ENABLE:
            __asm__ volatile("MRS %0, PMCNTENSET_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read count_enable register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_COUNT_CLEAR:
            __asm__ volatile("MRS %0, PMCNTENCLR_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read count_clear register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_OVERFLOW_FLAGS:
            __asm__ volatile("MRS %0, PMOVSSET_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read overflow_flags register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_OVERFLOW_STATUS:
            __asm__ volatile("MRS %0, PMOVSCLR_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read overflow_status register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_PERF_CONTROL_CTRL:
            __asm__ volatile("MRS %0, PMCR_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read global_control register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_EVENTS0:
            __asm__ volatile("MRS %0, PMCEID0_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read events0 register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        case A57_EVENTS1:
            __asm__ volatile("MRS %0, PMCEID1_EL0" : "=r" (tmp32) :: "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Read events1 register %lx, tmp32);
            tmp = (uint64_t)tmp32;
            break;
        default:
            tmp = 0x0;
            *data = tmp;
            return -EINVAL;
            break;
    }
#endif
    *data = tmp;
    sched_setaffinity(gettid(), sizeof(cpu_set_t), &current);
    return 0;
#endif
    return 1;
}

int access_armv8_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data)
{
#if defined(__ARM_ARCH_8A)

    int ret = 0;
    uint64_t tmp = data;
    uint32_t tmp32 = (uint32_t) data;
    uint32_t reg_idx = 0;
    uint32_t hi = 0, lo = 0;
    cpu_set_t cpuset, current;
    ret = sched_getaffinity(gettid(), sizeof(cpu_set_t), &current);
    if (ret < 0)
        return errno;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &cpuset);
    if (ret < 0)
        return errno;
#if defined(__ARM_32BIT_STATE)
    switch(reg)
    {
        case A57_PMC0:
        case A57_PMC1:
        case A57_PMC2:
        case A57_PMC3:
        case A57_PMC4:
        case A57_PMC5:
            reg_idx = (reg - A57_PMC0)/8;
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"(reg_idx) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Select counter register %d, reg_idx);
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 2" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register %d: %lx, reg_idx, tmp32);
            break;
        case A57_PERFEVTSEL0:
        case A57_PERFEVTSEL1:
        case A57_PERFEVTSEL2:
        case A57_PERFEVTSEL3:
        case A57_PERFEVTSEL4:
        case A57_PERFEVTSEL5:
            reg_idx = (reg - A57_PERFEVTSEL0)/8;
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r" (reg_idx) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Select config register %d, reg_idx);
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 1" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register %d: %lx, reg_idx, tmp32);
            break;
            /*if ((reg >= ) && (reg <= 0x414))
            {
                __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"((reg & 0xFF)/8) );
            }
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 1" : : "r" (tmp));
            break;*/
        case A57_CYCLES:
            //reg_idx = 0x1FU;
            hi = ((data>>32) & 0xFFFFFFFF), lo = data & 0xFFFFFFFF;
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write cycles register %llx, data);
            //__asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"(reg_idx));
            __asm__ volatile("mcrr p15, 0, %0, %1, c9" : : "r" (lo), "r" (hi) : "memory");
            break;
        case A57_COUNT_ENABLE:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write count_enable register %lx, tmp32);
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 1" : : "r" (tmp32) : "memory");
            break;
        case A57_COUNT_CLEAR:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write count_clear register %lx, tmp32);
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 2" : : "r" (tmp32) : "memory");
            break;
        case A57_OVERFLOW_FLAGS:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write overflow_flags register %lx, tmp32);
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 3" : : "r" (tmp32) : "memory");
            break;
        case A57_OVERFLOW_STATUS:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write overflow_status register %lx, tmp32);
            __asm__ volatile("mcr p15, 0, %0, c9, c14, 3" : : "r" (tmp32) : "memory");
            break;
        case A57_PERF_CONTROL_CTRL:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write global_control register %lx, tmp32);
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 0" : : "r" (tmp32) : "memory");
            break;
        case A57_EVENTS0:
        case A57_EVENTS1:
        default:
            return -EINVAL;
            break;
    }
#else
    switch(reg)
    {
        case A57_PMC0:
            __asm__ volatile("MSR PMEVCNTR0_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC0, tmp32);
            break;
        case A57_PMC1:
            __asm__ volatile("MSR PMEVCNTR1_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC1, tmp32);
            break;
        case A57_PMC2:
            __asm__ volatile("MSR PMEVCNTR2_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC2, tmp32);
            break;
        case A57_PMC3:
            __asm__ volatile("MSR PMEVCNTR3_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC3, tmp32);
            break;
        case A57_PMC4:
            __asm__ volatile("MSR PMEVCNTR4_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC4, tmp32);
            break;
        case A57_PMC5:
            __asm__ volatile("MSR PMEVCNTR5_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write counter register 0x%X: %lx, A57_PMC5, tmp32);
            break;
        case A57_PERFEVTSEL0:
            __asm__ volatile("MSR PMEVTYPER0_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL0, tmp32);
            break;
        case A57_PERFEVTSEL1:
            __asm__ volatile("MSR PMEVTYPER1_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL1, tmp32);
            break;
        case A57_PERFEVTSEL2:
            __asm__ volatile("MSR PMEVTYPER2_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL2, tmp32);
            break;
        case A57_PERFEVTSEL3:
            __asm__ volatile("MSR PMEVTYPER3_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL3, tmp32);
            break;
        case A57_PERFEVTSEL4:
            __asm__ volatile("MSR PMEVTYPER4_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL4, tmp32);
            break;
        case A57_PERFEVTSEL5:
            __asm__ volatile("MSR PMEVTYPER5_EL0, %0" : : "r" (tmp32) : "memory");
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write config register 0x%X: %lx, A57_PERFEVTSEL5, tmp32);
            break;
        case A57_CYCLES:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write cycles register %llx, tmp);
            __asm__ volatile("MSR PMCCNTR_EL0, %0": : "r" (tmp));
            break;
        case A57_COUNT_ENABLE:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write count_enable register %lx, tmp32);
            __asm__ volatile("MSR PMCNTENSET_EL0, %0": : "r" (tmp32) : "memory");
            break;
        case A57_COUNT_CLEAR:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write count_clear register %lx, tmp32);
            __asm__ volatile("MSR PMCNTENCLR_EL0, %0": : "r" (tmp32) : "memory");
            break;
        case A57_OVERFLOW_FLAGS:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write overflow_flags register %lx, tmp32);
            __asm__ volatile("MSR PMOVSSET_EL0, %0": : "r" (tmp32) : "memory");
            break;
        case A57_OVERFLOW_STATUS:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write overflow_status register %lx, tmp32);
            __asm__ volatile("MSR PMOVSCLR_EL0, %0": : "r" (tmp32) : "memory");
            break;
        case A57_PERF_CONTROL_CTRL:
            DEBUG_PRINT(DEBUGLEV_DEVELOP, Write global_control register %lx, tmp32);
            __asm__ volatile("MSR PMCR_EL0, %0": : "r" (tmp32) : "memory");
            break;
        case A57_EVENTS0:
        case A57_EVENTS1:
        default:
            return -EINVAL;
            break;
    }
#endif
    ret = sched_setaffinity(gettid(), sizeof(cpu_set_t), &current);

#endif
    return 0;
}

int access_armv8_check(PciDeviceIndex dev, int cpu_id)
{
    return 1;
}

