#include <stdlib.h>
#include <stdio.h>

#include <likwid.h>

int likwid_getMajorVersion(void)
{
    return (int) VERSION ;
}

int likwid_getMinorVersion(void)
{
    return (int) RELEASE ;
}

int likwid_getBugfixVersion(void)
{
    return (int) MINORVERSION ;
}

int likwid_getNvidiaSupport(void)
{
#ifdef LIKWID_WITH_NVMON
    return 1;
#else
    return 0;
#endif
}

int likwid_getRocmSupport(void)
{
#ifdef LIKWID_WITH_ROCMON
    return 1;
#else
    return 0;
#endif
}

int likwid_getMaxSupportedThreads(void)
{
    return (int) MAX_NUM_THREADS;
}

int likwid_getMaxSupportedSockets(void)
{
    return (int) MAX_NUM_NODES;
}
