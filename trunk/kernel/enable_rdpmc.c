/*  
 *  Read PMC in kernel mode.
 */
#include <linux/module.h>   /* Needed by all modules */
#include <linux/kernel.h>   /* Needed for KERN_INFO */

#define MODULE_PARAM(type, name, value, desc)			\
	type name = value;					\
	module_param(name, type, 0664);				\
	MODULE_PARM_DESC(name, desc)

MODULE_PARAM(int, debug, 0, "Debug output");


static void printc4(void) {
    uint64_t output;
    // Read back CR4 to check the bit.
    __asm__("\t mov %%cr4,%0" : "=r"(output));
    printk(KERN_INFO "%llu", output);
}

static void setc4b8(void * info) {
    // Set CR4, Bit 8 (9th bit from the right)  to enable
	__asm__("push   %rax\n\t"
            "mov    %cr4,%rax;\n\t"
            "or     $(1 << 8),%rax;\n\t"
            "mov    %rax,%cr4;\n\t"
            "wbinvd\n\t"
            "pop    %rax"
    );

    if (debug) {
        printk(KERN_INFO "Processor %d, RDPMC_ENABLE_BIT=%d\n", smp_processor_id(), printc4());
    }
}

static void clearc4b8(void * info) {
    printc4();
	__asm__("push   %rax\n\t"
        	"push   %rbx\n\t"
            "mov    %cr4,%rax;\n\t"
            "mov  $(1 << 8), %rbx\n\t"
            "not  %rbx\n\t"
            "and   %rbx, %rax;\n\t"
            "mov    %rax,%cr4;\n\t"
            "wbinvd\n\t"
            "pop    %rbx\n\t"
            "pop    %rax\n\t"
    );
    
    if (debug) {
        printk(KERN_INFO "Processor %d, RDPMC_ENABLE_BIT=%d\n", smp_processor_id(), printc4());
    }
}



int start_module(void)
{
    on_each_cpu(setc4b8, NULL, 0);
    return 0;
}
void stop_module(void)
{
    on_each_cpu(clearc4b8, NULL, 0);
}

module_init(start_module);
module_exit(stop_module)

MODULE_AUTHOR("Thomas Roehl <Thomas.Roehl@fau.de>");
MODULE_DESCRIPTION("Enable RDPMC for userspace");
MODULE_LICENSE("GPL");
