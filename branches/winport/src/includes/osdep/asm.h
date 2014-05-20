#ifndef CPUID_ASM_H
#define CPUID_ASM_H

#include <stdint.h>

#include <types.h>


#ifdef WIN32

#define CPUID          \
	__asm {            \
		push eax       \
		push ecx       \
		mov eax, eax   \
		mov ecx, ecx   \
		cpuid          \
		mov eax, eax   \
		mov ebx, ebx   \
		mov ecx, ecx   \
		mov edx, edx   \
		pop ecx        \
		pop eax }

#define ASM_BSR(fieldWidth, number)  \
	__asm {                          \
		mov eax, number              \
		bsr ecx, eax                 \
		mov fieldwidth, ecx          \
	}

/* TODO Test on Windows */
#define RDTSC(cpu_c)                 \
    __asm  {                         \
        rdtsc                        \
        movl eax, cpu_c.int32.lo     \
        movl edx, cpu_c.int32.hi }


#else /* WIN32 */

/* this was taken from the linux kernel */
#define CPUID                         \
    __asm__ volatile ("cpuid"         \
        : "=a" (eax),                 \
          "=b" (ebx),                 \
          "=c" (ecx),                 \
          "=d" (edx)                  \
        : "0" (eax), "2" (ecx))

#define ASM_BSR(fieldWidth, number)          \
    __asm__ volatile ("bsr %%eax, %%ecx\n\t" \
            : "=c" (fieldWidth)              \
            : "a"(number))

#define RDTSC(cpu_c)                 \
    __asm__ volatile( "rdtsc\n\t"    \
            "movl %%eax, %0\n\t"     \
            "movl %%edx, %1\n\t"     \
            : "=r" ((cpu_c).int32.lo), "=r" ((cpu_c).int32.hi) \
: : "%eax", "%edx")


#endif /* WIN32 */

#endif /* CPUID_ASM_H */
