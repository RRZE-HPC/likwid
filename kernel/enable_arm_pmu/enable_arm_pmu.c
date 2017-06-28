#include <linux/module.h>   /* Needed by all modules */
#include <linux/kernel.h>   /* Needed for KERN_INFO */
#include <linux/init.h>     /* Needed for the macros */
#define DRIVER_AUTHOR "Johannes Hofmann <johannes.hofmann@fau.de>"
#define DRIVER_DESC   "Driver to activate ARM PMUs for user-space"
#define DRVR_NAME "enable_arm_pmu"
/* ARMv7 defines */
#define PERF_DEF_OPTS             (1U<<0)
#define PERF_OPT_RESET_CYCLES         ((1U<<2))
#define PERF_OPT_RESET_COUNTER         ((1U<<1))
#define PERF_OPT_DIV64             (1U<<3)


/* ARMv8 defines */

// Shift and mask to determine number of available counters
#define ARMV8_PMCR_N_SHIFT 11
#define ARMV8_PMCR_N_MASK 0x1F

#define ARMV8_PMCR_MASK         0x3f
#define ARMV8_PMCR_E            (1U << 0) /*  Enable all counters */
#define ARMV8_PMCR_P            (1U << 1) /*  Reset all counters */
#define ARMV8_PMCR_C            (1U << 2) /*  Cycle counter reset */
#define ARMV8_PMCR_D            (1U << 3) /*  CCNT counts every 64th cpu cycle */
#define ARMV8_PMCR_X            (1U << 4) /*  Export to ETM */
#define ARMV8_PMCR_DP           (1U << 5) /*  Disable CCNT if non-invasive debug*/

#define ARMV8_PMUSERENR_EN_EL0  (1 << 0) /*  EL0 access enable */
#define ARMV8_PMUSERENR_CR      (1 << 2) /*  Cycle counter read enable */
#define ARMV8_PMUSERENR_ER      (1 << 3) /*  Event counter read enable */

#define ARMV8_PMCNTENSET_EL0_ENABLE (1<<31) /*  Enable Perf count reg */

#define PMCR_RL0_RESET  0x41013000

#if !defined(__arm__) && !defined(__aarch64__)
#error Module can only be compiled on ARM machines.
#endif



static u32 nrCounters = 0;
static u32 cntMask = 0x0;



static void get_num_counters(void)
{
    int i = 0;
    u32 tmp = 0;
#if defined(__ARM_ARCH_7A__)
    asm volatile("MRC p15, 0, %0, c9, c12, 0" :: "r"(tmp));
    nrCounters = tmp;
#endif
#if defined(__ARM_ARCH_8A) && defined(__aarch64__)
    asm volatile ("mrs %0, PMCR_EL0" : "=&r" (tmp) ::);
    nrCounters = (tmp >> ARMV8_PMCR_N_SHIFT) & ARMV8_PMCR_N_MASK;
#endif
    // Mask for all counters
    for (i = 0; i < nrCounters; i++)
        cntMask |= (1<<i);
    // add cycle counter bit
    cntMask |= (1<<31);

}

static inline u32 armv8pmu_pmcr_read(void)
{
    u64 val=0;
    asm volatile("mrs %0, pmcr_el0" : "=r" (val));
    return (u32)val;
}
static inline void armv8pmu_pmcr_write(u32 val)
{
        val &= ARMV8_PMCR_MASK;
            isb();
                asm volatile("msr pmcr_el0, %0" : : "r" ((u64)val));
}

static void enable_cpu_counters(void* data)
{
    u32 mask = 0x0;
    printk(KERN_INFO "[" DRVR_NAME "] enabling user-mode PMU access on CPU #%d for %d counters\n", smp_processor_id(), nrCounters);
#if defined(__ARM_ARCH_8A) && defined(__aarch64__)
    mask |= ARMV8_PMUSERENR_EN_EL0|ARMV8_PMUSERENR_ER|ARMV8_PMUSERENR_CR;

    asm volatile("msr pmuserenr_el0, %0" : : "r"((u64)mask));
    armv8pmu_pmcr_write(ARMV8_PMCR_P | ARMV8_PMCR_C);

    asm volatile("msr pmintenset_el1, %0" : : "r" ((u64)~cntMask));
    asm volatile("msr pmcntenset_el0, %0" : : "r" (ARMV8_PMCNTENSET_EL0_ENABLE));
    armv8pmu_pmcr_write(armv8pmu_pmcr_read() | ARMV8_PMCR_E);
#endif
#ifdef __ARM_ARCH_7A
    /* Enable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(1));
    /* Program PMU and enable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(PERF_DEF_OPTS | PERF_OPT_DIV64));
    asm volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(cntMask));
#endif

}

static void disable_cpu_counters(void* data)
{
    printk(KERN_INFO "[" DRVR_NAME "] disabling user-mode PMU access on CPU #%d\n", smp_processor_id());
#if defined(__ARM_ARCH_8A) && defined(__aarch64__)
    asm volatile("msr pmcntenset_el0, %0" : : "r"(~cntMask));
    armv8pmu_pmcr_write(armv8pmu_pmcr_read() |~ARMV8_PMCR_E);
    asm volatile("msr pmuserenr_el0, %0" : : "r"((u64)0));
#endif
#ifdef __ARM_ARCH_7A__
    /* Program PMU and disable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(0));
    asm volatile("mcr p15, 0, %0, c9, c12, 2" :: "r"(~cntMask));
    /* Disable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(0));
#endif

}

static int __init init_arm_pmu(void)
{
    get_num_counters();
    printk(KERN_INFO "[" DRVR_NAME "] Loading module for %d counters\n", nrCounters);
    on_each_cpu(enable_cpu_counters, NULL, 1);
    return 0;
}

static void __exit cleanup_arm_pmu(void)
{
    printk(KERN_INFO "[" DRVR_NAME "] Unloading module for %d counters\n", nrCounters);
    on_each_cpu(disable_cpu_counters, NULL, 1);
}

module_init(init_arm_pmu);
module_exit(cleanup_arm_pmu);

MODULE_LICENSE("GPL");

MODULE_AUTHOR(DRIVER_AUTHOR);
MODULE_DESCRIPTION(DRIVER_DESC);
