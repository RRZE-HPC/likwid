#ifndef SETFREQ_DAEMON
#define SETFREQ_DAEMON

#if defined(__i386__) && defined(__PIC__)
# define EBX_REG "=r"
#else
# define EBX_REG "=b"
#endif

#ifndef __clang__
#define CPUID(eax,ebx,ecx,edx)                            \
    __asm__ volatile(".ifnc %%ebx,%3 ; movl  %%ebx,%3 ; .endif  \n\t" \
                     "cpuid                                     \n\t" \
                     ".ifnc %%ebx,%3 ; xchgl %%ebx,%3 ; .endif  \n\t" \
                     : "=a" (eax), "=c" (ecx), "=d" (edx), EBX_REG (ebx) \
                     : "a" (eax), "c" (ecx) \
                     )
#else
#define CPUID(eax,ebx,ecx,edx)         \
    __asm__ volatile("cpuid" : "=a" (eax), "=c" (ecx), "=d" (edx), EBX_REG (ebx) : "a" (eax), "c" (ecx) );
#endif

extern int do_pstate (int argn, char** argv);
extern int do_cpufreq (int argn, char** argv);

#endif
