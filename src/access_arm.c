#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <types.h>
#include <error.h>
#include <topology.h>
#include <registers.h>
#include <access_arm.h>


int access_arm_init(int cpu_id)
{

    return 0;
}

void access_arm_finalize(int cpu_id)
{
    return;
}

int access_arm_read(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t *data)
{
#if defined(__ARM_ARCH_7A__)
    uint32_t tmp = 0x0ULL;
    uint32_t copro_op = 0;
    uint32_t copro_reg = 0;
    cpu_set_t cpuset, current;
    sched_getaffinity(0, sizeof(cpu_set_t), &current);
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
    switch(reg)
    {
        case A15_PMC0:
        case A15_PMC1:
        case A15_PMC2:
        case A15_PMC3:
        case A15_PMC4:
        case A15_PMC5:
            if ((reg >= 0x000) && (reg <= 0x014))
            {
                __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"((reg & 0xFF)/4) );
            }
            __asm__ volatile("mrc p15, 0, %0, c9, c13, 2" : "=r" (tmp));
            break;
        case A15_PERFEVTSEL0:
        case A15_PERFEVTSEL1:
        case A15_PERFEVTSEL2:
        case A15_PERFEVTSEL3:
        case A15_PERFEVTSEL4:
        case A15_PERFEVTSEL5:
            if ((reg >= 0x400) && (reg <= 0x414))
            {
                __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"((reg & 0xFF)/4) );
            }
            __asm__ volatile("mrc p15, 0, %0, c9, c13, 1" : "=r" (tmp));
            break;
        case A15_CYCLES:
            __asm__ volatile("mrc p15, 0, %0, c9, c13, 0" : "=r" (tmp));
            break;
        case A15_COUNT_ENABLE:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 1" : "=r" (tmp));
            break;
        case A15_COUNT_CLEAR:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 2" : "=r" (tmp));
            break;
        case A15_OVERFLOW_FLAGS:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 3" : "=r" (tmp));
            break;
        case A15_OVERFLOW_STATUS:
            __asm__ volatile("mrc p15, 0, %0, c9, c14, 3" : "=r" (tmp));
            break;
        case A15_PERF_CONTROL_CTRL:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 0" : "=r" (tmp));
            break;
        case A15_EVENTS0:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 6" : "=r" (tmp));
            break;
        case A15_EVENTS1:
            __asm__ volatile("mrc p15, 0, %0, c9, c12, 7" : "=r" (tmp));
            break;
        default:
            tmp = 0x0;
            *data = tmp;
            return -EINVAL;
            break;
    }
    *data = (uint64_t) tmp;
    sched_setaffinity(0, sizeof(cpu_set_t), &current);
    return 0;
#endif
    return 1;
}

int access_arm_write(PciDeviceIndex dev, const int cpu_id, uint32_t reg, uint64_t data)
{
#if defined(__ARM_ARCH_7A__)
    uint32_t tmp = (uint32_t) data;
    uint32_t copro_op = 0;
    uint32_t copro_reg = 0;
    switch(reg)
    {
        case A15_PMC0:
        case A15_PMC1:
        case A15_PMC2:
        case A15_PMC3:
        case A15_PMC4:
        case A15_PMC5:
            if ((reg >= 0x000) && (reg <= 0x014))
            {
                __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"((reg & 0xFF)/4) );
            }
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 2" : : "r" (tmp));
            break;
        case A15_PERFEVTSEL0:
        case A15_PERFEVTSEL1:
        case A15_PERFEVTSEL2:
        case A15_PERFEVTSEL3:
        case A15_PERFEVTSEL4:
        case A15_PERFEVTSEL5:
            if ((reg >= 0x400) && (reg <= 0x414))
            {
                __asm__ volatile("mcr p15, 0, %0, c9, c12, 5" : : "r"((reg & 0xFF)/4) );
            }
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 1" : : "r" (tmp));
            break;
        case A15_CYCLES:
            __asm__ volatile("mcr p15, 0, %0, c9, c13, 0" : : "r" (tmp));
            break;
        case A15_COUNT_ENABLE:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 1" : : "r" (tmp));
            break;
        case A15_COUNT_CLEAR:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 2" : : "r" (tmp));
            break;
        case A15_OVERFLOW_FLAGS:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 3" : : "r" (tmp));
            break;
        case A15_OVERFLOW_STATUS:
            __asm__ volatile("mcr p15, 0, %0, c9, c14, 3" : "=r" (tmp));
            break;
        case A15_PERF_CONTROL_CTRL:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 0" : : "r" (tmp));
            break;
        case A15_EVENTS0:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 6" : : "r" (tmp));
            break;
        case A15_EVENTS1:
            __asm__ volatile("mcr p15, 0, %0, c9, c12, 7" : : "r" (tmp));
            break;
        default:
            return -EINVAL;
            break;
    }
#endif
    return 0;
}

int access_arm_check(PciDeviceIndex dev, int cpu_id)
{
    return 1;
}
