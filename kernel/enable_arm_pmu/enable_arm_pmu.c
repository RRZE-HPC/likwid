/*
 *  Read PMC in kernel mode.
 */
#include <linux/module.h>   /* Needed by all modules */
#include <linux/kernel.h>   /* Needed for KERN_INFO */
#include <linux/smp.h>

#define PERF_DEF_OPTS             (1U<<0)
#define PERF_OPT_RESET_CYCLES         ((1U<<2))
#define PERF_OPT_RESET_COUNTER         ((1U<<1))
#define PERF_OPT_DIV64             (1U<<3)


#define PMCR_RL0_RESET  0x41013000

#define MODULE_PARAM(type, name, value, desc) \
    type name = value; \
    module_param(name, type, 0664); \
    MODULE_PARM_DESC(name, desc)

MODULE_PARAM(int, debug, 0, "Debug output");

#define DRVR_NAME "enable_arm_pmu"


static u32 nrCounters = 0x0;

static void
enable_cpu_counters(void *data)
{
    int i;
    u32 set_mask = 0x0;
//    asm volatile("MRC p15,0,%0,c9,c12,0" :: "r"(nrCounters));
#ifdef __ARM_ARCH_8A
#ifdef __aarch64__
    asm volatile("MRS %0, PMCR_EL0" : "=&r" (nrCounters) :: );
    nrCounters = (nrCounters>>11) & (0x1F);
#endif
#endif
    printk(KERN_INFO "[" DRVR_NAME "] enabling user-mode PMU access on CPU #%d for %d counters\n",
            smp_processor_id(), nrCounters);

    for (i = 0; i < nrCounters; i++)
        set_mask |= (1<<i);
    set_mask |= (1<<31);

#ifdef __ARM_ARCH_8A
#ifdef __aarch64__
    /* Enable user-mode access to counters. */
    asm volatile("MSR PMUSERENR_EL0, %0" :: "r"(0xF));
    /* Program PMU and enable all counters */
    asm volatile("MSR PMCR_EL0, %0" :: "r"(PMCR_RL0_RESET | PERF_DEF_OPTS | PERF_OPT_RESET_COUNTER));
    asm volatile("MSR PMCNTENSET_EL0, %0" :: "r"(set_mask));
#else
    /* Enable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(1));
    /* Program PMU and enable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(PMCR_RL0_RESET | PERF_DEF_OPTS | PERF_OPT_RESET_COUNTER));
    asm volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(set_mask));
#endif
#endif

#ifdef __ARM_ARCH_7A
    /* Enable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(1));
    /* Program PMU and enable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(PERF_DEF_OPTS | PERF_OPT_DIV64));
    asm volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(set_mask));
#endif
}

static void
disable_cpu_counters(void *data)
{
    int i;
    u32 clear_mask = 0x0;
    printk(KERN_INFO "[" DRVR_NAME "] disabling user-mode PMU access on CPU #%d\n",
            smp_processor_id());

    for (i = 0; i < nrCounters; i++)
        clear_mask |= (1<<i);
    clear_mask |= (1<<31);

#ifdef __ARM_ARCH_8A
#ifdef __aarch64__
    asm volatile("MSR PMCR_EL0, %0" :: "r"(0));
    asm volatile("MSR PMCNTENCLR_EL0, %0" :: "r"(clear_mask));
//    asm volatile("mcr p15, 0, %0, c9, c12, 2" :: "r"(clear_mask));
    asm volatile("MSR PMUSERENR_EL0, %0" :: "r"(0));
#else
    /* Program PMU and disable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(0));
    asm volatile("mcr p15, 0, %0, c9, c12, 2" :: "r"(clear_mask));
    /* Disable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(0));
#endif
#endif

#ifdef __ARM_ARCH_7A__
    /* Program PMU and disable all counters */
    asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(0));
    asm volatile("mcr p15, 0, %0, c9, c12, 2" :: "r"(clear_mask));
    /* Disable user-mode access to counters. */
    asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(0));
#endif
}

int start_module(void)
{
    on_each_cpu(enable_cpu_counters, NULL, 1);
    printk(KERN_INFO "[" DRVR_NAME "] initialized\n");
    return 0;
}
void stop_module(void)
{
    on_each_cpu(disable_cpu_counters, NULL, 1);
    printk(KERN_INFO "[" DRVR_NAME "] unloaded\n");
}

module_init(start_module);
module_exit(stop_module)

MODULE_AUTHOR("Johannes Hofmann <johannes.hofmann@fau.de>");
MODULE_DESCRIPTION("Enable PMU from userspace");
MODULE_LICENSE("GPL");
