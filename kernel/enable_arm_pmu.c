/*
 *  Read PMC in kernel mode.
 */
#include <linux/module.h>   /* Needed by all modules */
#include <linux/kernel.h>   /* Needed for KERN_INFO */
#include <linux/smp.h>

#define PERF_DEF_OPTS 			(1)
#define PERF_OPT_RESET_CYCLES 		(2 | 4)
#define PERF_OPT_DIV64 			(8)

#define MODULE_PARAM(type, name, value, desc) \
    type name = value; \
    module_param(name, type, 0664); \
    MODULE_PARM_DESC(name, desc)

MODULE_PARAM(int, debug, 0, "Debug output");

#define DRVR_NAME "enable_arm_pmu"

#if !defined(__arm__)
#error Module can only be compiled on ARM platforms.
#endif

static void
enable_cpu_counters(void *data)
{
	printk(KERN_INFO "[" DRVR_NAME "] enabling user-mode PMU access on CPU #%d\n",
			smp_processor_id());

	/* Enable user-mode access to counters. */
	asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(1));
	/* Program PMU and enable all counters */
	asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(PERF_DEF_OPTS | PERF_OPT_DIV64));
	asm volatile("mcr p15, 0, %0, c9, c12, 1" :: "r"(0x8000000f));
}

static void
disable_cpu_counters(void *data)
{
	printk(KERN_INFO "[" DRVR_NAME "] disabling user-mode PMU access on CPU #%d\n",
			smp_processor_id());

	/* Program PMU and disable all counters */
	asm volatile("mcr p15, 0, %0, c9, c12, 0" :: "r"(0));
	asm volatile("mcr p15, 0, %0, c9, c12, 2" :: "r"(0x8000000f));
	/* Disable user-mode access to counters. */
	asm volatile("mcr p15, 0, %0, c9, c14, 0" :: "r"(0));
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
