#ifndef LIKWID_CPUID_H
#define LIKWID_CPUID_H

/* This was taken from the linux kernel
 * Kernel version 3.19
 * File: arch/x86/boot/cpuflags.c
*/


#if defined(__i386__) && defined(__PIC__)
# define EBX_REG "=r"
#else
# define EBX_REG "=b"
#endif

#define CPUID                              \
    __asm__ volatile(".ifnc %%ebx,%3 ; movl  %%ebx,%3 ; .endif  \n\t" \
                     "cpuid                                     \n\t" \
                     ".ifnc %%ebx,%3 ; xchgl %%ebx,%3 ; .endif  \n\t" \
                     : "=a" (eax), "=c" (ecx), "=d" (edx), EBX_REG (ebx) \
                     : "a" (eax) \
                     )


#endif
