/*
 *	The PCI Library -- Types and Format Strings
 *
 *	Copyright (c) 1997--2008 Martin Mares <mj@ucw.cz>
 *
 *	Can be freely distributed and used under the terms of the GNU GPL.
 */

#include <sys/types.h>

#ifndef PCI_HAVE_Uxx_TYPES

#ifdef PCI_OS_WINDOWS
#include <windef.h>
typedef BYTE u8;
typedef WORD u16;
typedef DWORD u32;
#elif defined(PCI_HAVE_STDINT_H) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L)
#include <stdint.h>
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
#else
typedef u_int8_t u8;
typedef u_int16_t u16;
typedef u_int32_t u32;
#endif

#ifdef PCI_HAVE_64BIT_ADDRESS
#include <limits.h>
#if ULONG_MAX > 0xffffffff
typedef unsigned long u64;
#define PCI_U64_FMT "l"
#else
typedef unsigned long long u64;
#define PCI_U64_FMT "ll"
#endif
#endif

#endif	/* PCI_HAVE_Uxx_TYPES */

#ifdef PCI_HAVE_64BIT_ADDRESS
typedef u64 pciaddr_t;
#define PCIADDR_T_FMT "%08" PCI_U64_FMT "x"
#define PCIADDR_PORT_FMT "%04" PCI_U64_FMT "x"
#else
typedef u32 pciaddr_t;
#define PCIADDR_T_FMT "%08x"
#define PCIADDR_PORT_FMT "%04x"
#endif

#ifdef PCI_ARCH_SPARC64
/* On sparc64 Linux the kernel reports remapped port addresses and IRQ numbers */
#undef PCIADDR_PORT_FMT
#define PCIADDR_PORT_FMT PCIADDR_T_FMT
#define PCIIRQ_FMT "%08x"
#else
#define PCIIRQ_FMT "%d"
#endif

#if defined(__GNUC__) && __GNUC__ > 2
#define PCI_PRINTF(x,y) __attribute__((format(printf, x, y)))
#else
#define PCI_PRINTF(x,y)
#endif
