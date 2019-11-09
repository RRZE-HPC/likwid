#ifndef _REALTIME_H_
#define _REALTIME_H_

#if defined _OPENMP
#include <omp.h>
#elif defined MPI_INT
#include <mpi.h>
#elif defined WIN32|WIN64
#include <Windows.h>
#include <stdio.h>
#include <stdlib.h>
#define Li2Double(x) \
    ((double)((x).HighPart) * 4.294967296E9 + (double)((x).LowPart))
#else
#include <sys/time.h>
#include <time.h>
#endif

double GetRealTime (void)
{
/* is there a common MPI define if MPI is active? MPI does not work */
/* crude fix: look for MPI_INT */
#if defined _OPENMP
    return omp_get_wtime();

#elif defined MPI_INT
    return MPI_Wtime();

#elif defined WIN32
    LARGE_INTEGER time, freq;

    if (QueryPerformanceCounter(&time) == 0)
    {
        DWORD err = GetLastError();
        LPVOID buf;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
                      NULL,
                      err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf,
                      0, NULL);
        printf("QueryPerformanceCounter() failed with error %d: %s \n",err, buf);
        exit(1);
    }

    if (QueryPerformanceFrequency(&freq) == 0)
    {
        DWORD err = GetLastError();
        LPVOID buf;
        FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM,
                      NULL,
                      err,
                      MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                      (LPTSTR) &buf,
                      0, NULL);
        printf("QueryPerformanceFrequency() failed with error %d: %s \n",err, buf);
        exit(1);
    }
    
    return Li2Double(time) / Li2Double(freq);

#elif defined SUNOS
    hrtime_t t;
    t = gethrtime();
    return (((double)t)*1.0E-9);

#else
    struct timeval tv;
    gettimeofday(&tv, (struct timezone*)0);
    return ((double)tv.tv_sec + (double)tv.tv_usec / 1000000.0 );

#endif
}
#endif
