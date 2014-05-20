#ifndef OSDEP_CPUID_H
#define OSDEP_CPUID_H

#include <stdint.h>

#ifdef WIN32

// code taken from http://www.dalun.com/blogs/05.01.2007.htm

#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <tchar.h>

typedef BOOL (WINAPI *LPFN_GLPI)(
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION,
    PDWORD);

static uint32_t numberOfProcessors()
{
    BOOL done;
    BOOL rc;
    DWORD returnLength;
    DWORD procCoreCount;
    DWORD byteOffset;
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer, ptr;
    LPFN_GLPI Glpi;

    Glpi = (LPFN_GLPI) GetProcAddress(
                            GetModuleHandle(TEXT("kernel32")),
                            "GetLogicalProcessorInformation");
    if (NULL == Glpi)
    {
        _tprintf(
            TEXT("GetLogicalProcessorInformation is not supported.\n"));
        exit(1);
    }

    done = FALSE;
    buffer = NULL;
    returnLength = 0;

    while (!done)
    {
        rc = Glpi(buffer, &returnLength);

        if (FALSE == rc)
        {
            if (GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            {
                if (buffer)
                    free(buffer);

                buffer=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(
                        returnLength);

                if (NULL == buffer)
                {
                    _tprintf(TEXT("Allocation failure\n"));
                    return (2);
                }
            }
            else
            {
                _tprintf(TEXT("Error %d\n"), GetLastError());
                return (3);
            }
        }
        else done = TRUE;
    }

    procCoreCount = 0;
    byteOffset = 0;

    ptr=buffer;
    while (byteOffset < returnLength)
    {
        switch (ptr->Relationship)
        {
            case RelationProcessorCore:
                procCoreCount++;
                break;

            default:
                break;
        }
        byteOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        ptr++;
    }

    free (buffer);

	return procCoreCount;
}

#else /* WIN32 */

#include <stdio.h>
#include <stdlib.h>
#include <error.h>

uint32_t numberOfProcessors() 
{
    int ret;
    FILE* pipe;

    uint32_t numProcs = -1;

    /* First determine the number of processors accessible */
    pipe = popen("cat /proc/cpuinfo | grep ^processor | wc -l", "r");
    ret = fscanf(pipe, "%u\n", &numProcs);


    if ((ret == EOF) && ferror(pipe))
    {
        ERROR;
    }

    ret = pclose(pipe);

    if (ret < 0)
    {
        ERROR;
    }

    return numProcs;
}

#endif /* WIN32 */

#endif /* OSDEP_CPUID_H */
