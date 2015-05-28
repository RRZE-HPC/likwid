/*
 * =======================================================================================
 *
 *      Filename:  pthread-overload.c
 *
 *      Description:  Overloaded library for pthread_create call. 
 *                    Implements pinning of threads together with likwid-pin.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2013 Jan Treibig 
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <dlfcn.h>
#include <sched.h>
#include <bits/pthreadtypes.h>
#include <sys/types.h>
#include <errno.h>
#include <dirent.h>
#include <unistd.h>
#include <string.h>

#ifdef COLOR
#include <textcolor.h>
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define LLU_CAST  (unsigned long long)

extern int pthread_setaffinity_np(pthread_t thread, size_t cpusetsize, const cpu_set_t *cpuset);

static char * sosearchpaths[] = {
#ifdef LIBPTHREAD
    TOSTRING(LIBPTHREAD),
#endif
    "/lib64/tls/libpthread.so.0",/* sles9 x86_64 */
    "libpthread.so.0",           /* Ubuntu */
    NULL
};

int
pthread_create(pthread_t* thread,
        const pthread_attr_t* attr,
        void* (*start_routine)(void *),
        void * arg)
{
    void *handle;
    char *error;
    int (*rptc) (pthread_t *, const pthread_attr_t *, void* (*start_routine)(void *), void *);
    int ret;
    static int reallpthrindex = 0;
    static int npinned = 0;
    static int ncalled = 0;
    static int overflow = 0;
    static int silent = 0;
    static int pin_ids[MAX_NUM_THREADS];
    static uint64_t skipMask = 0;


    /* On first entry: Get Evironment Variable and initialize pin_ids */
    if (ncalled == 0)
    {
        char *str;
        char *token, *saveptr;
        char *delimiter = ",";
        int i = 0;
        int ncpus = 0;

        str = getenv("LIKWID_SKIP");
        if (str != NULL)
        {
            skipMask = strtoul(str, &str, 16);
        }
        else
        {
            printf("[pthread wrapper] ERROR: Environment Variabel LIKWID_SKIP not set!\n");
        }

        if ( skipMask == 0 )
        {
            dlerror();    /* Clear any existing error */
            dlsym(RTLD_DEFAULT,"__kmpc_begin");

            if (( dlerror()) == NULL)  {
                skipMask = 0x1;
            }
        }

        if (getenv("LIKWID_SILENT") != NULL)
        {
            silent = 1;
        }
#ifdef COLOR
        else
        {
            color_on(BRIGHT, COLOR);
        }
#endif

        if (!silent)
        {
            printf("[pthread wrapper] ");
        }

        str = getenv("LIKWID_PIN");
        if (str != NULL)
        {
            token = str;
            while (token)
            {
                token = strtok_r(str,delimiter,&saveptr);
                str = NULL;
                if (token)
                {
                    ncpus++;
                    pin_ids[i++] = strtoul(token, &token, 10);
                }
            }
            ncpus--; /* last ID is the first (the process was pinned to) */
        }
        else
        {
            printf("[pthread wrapper] ERROR: Environment Variabel LIKWID_PIN not set!\n");
        }

        if (!silent)
        {
            printf("[pthread wrapper] PIN_MASK: ");

            for (int i=0;i<ncpus;i++)
            {
                printf("%d->%d  ",i,pin_ids[i]);
            }
            printf("\n");
            printf("[pthread wrapper] SKIP MASK: 0x%llX\n",LLU_CAST skipMask);
        }

        overflow = ncpus;
    }
    else
    {
#ifdef COLOR
        if (!silent)
        {
            color_on(BRIGHT, COLOR);
        }
#endif
    }

    /* Handle dll related stuff */
    do
    {
        handle = dlopen(sosearchpaths[reallpthrindex], RTLD_LAZY);
        if (handle)
        {
            break;
        }
        if (sosearchpaths[reallpthrindex] != NULL) 
        {
            reallpthrindex++;
        }
    }

    while (sosearchpaths[reallpthrindex] != NULL);

    if (!handle)
    {
        printf("%s\n", dlerror());
        return -1;
    }

    dlerror();    /* Clear any existing error */
    rptc = dlsym(handle, "pthread_create");

    if ((error = dlerror()) != NULL)
    {
        printf("%s\n", error);
        return -2;
    }

    ret = (*rptc)(thread, attr, start_routine, arg);

    /* After thread creation pin the thread */
    if (ret == 0)
    {
        cpu_set_t cpuset;

        if ((ncalled<64) && (skipMask&(1ULL<<(ncalled))))
        {
            if (!silent)
            {
                printf("\tthreadid %lu -> SKIP \n", *thread);
            }
        }
        else
        {
            CPU_ZERO(&cpuset);
            CPU_SET(pin_ids[npinned], &cpuset);
            pthread_setaffinity_np(*thread, sizeof(cpu_set_t), &cpuset);

            if (npinned == overflow)
            {
                if (!silent)
                {
                    printf("Roundrobin placement triggered\n");
                    printf("\tthreadid %lu -> core %d - OK", *thread, pin_ids[npinned]);
                }
                npinned = 0;
            }
            else
            {
                if (!silent)
                {
                    printf("\tthreadid %lu -> core %d - OK", *thread, pin_ids[npinned]);
                }
                npinned++;
            }

            if (!silent)
            {
#ifdef COLOR
                color_reset();
#endif
                printf("\n");
            }
        }
    }

    fflush(stdout);
    ncalled++;
    dlclose(handle);

    return ret;
}

