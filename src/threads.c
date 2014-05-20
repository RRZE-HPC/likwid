/*
 * =======================================================================================
 *
 *      Filename:  threads.c
 *
 *      Description:  High level interface to pthreads
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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

/* #####   HEADER FILE INCLUDES   ######################################### */

#include <stdlib.h>
#include <stdio.h>

#include <error.h>
#include <types.h>
#include <threads.h>


/* #####   EXPORTED VARIABLES   ########################################### */

pthread_barrier_t threads_barrier;
ThreadData* threads_data;
ThreadGroup* threads_groups;

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static pthread_t* threads = NULL;
static pthread_attr_t attr;
static int numThreads = 0;


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
threads_init(int numberOfThreads)
{
    int i;
    numThreads = numberOfThreads;

    threads = (pthread_t*) malloc(numThreads * sizeof(pthread_t));
    threads_data = (ThreadData*) malloc(numThreads * sizeof(ThreadData));

    for(i = 0; i < numThreads; i++)
    {
        threads_data[i].numberOfThreads = numThreads;
        threads_data[i].globalNumberOfThreads = numThreads;
        threads_data[i].globalThreadId = i;
        threads_data[i].threadId = i;
    }

    pthread_barrier_init(&threads_barrier, NULL, numThreads);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
}


void 
threads_create(void *(*startRoutine)(void*))
{
    int i;

    for(i = 0; i < numThreads; i++)
    {
        pthread_create(&threads[i],
                &attr,
                startRoutine,
                (void*) &threads_data[i]); 
    }
}

void 
threads_createGroups(int numberOfGroups)
{
    int i;
    int j;
    int numThreadsPerGroup;
    int globalId = 0;

    if (numThreads % numberOfGroups)
    {
        ERROR_PRINT(Not enough threads %d to create %d groups,numThreads,numberOfGroups);
    }
    else 
    {
        numThreadsPerGroup = numThreads / numberOfGroups;
    }

    threads_groups = (ThreadGroup*) malloc(numberOfGroups *
            sizeof(ThreadGroup));

    for (i = 0; i < numberOfGroups; i++)
    {
        threads_groups[i].numberOfThreads = numThreadsPerGroup;
        threads_groups[i].threadIds = (int*) malloc(numThreadsPerGroup *
                sizeof(int));

        for (j = 0; j < numThreadsPerGroup; j++)
        {
            threads_data[globalId].threadId = j;
            threads_data[globalId].groupId = i;
            threads_data[globalId].numberOfGroups = numberOfGroups;
            threads_data[globalId].numberOfThreads = numThreadsPerGroup;
            threads_groups[i].threadIds[j] = globalId++;
        }
    }
}


void 
threads_registerDataAll(ThreadUserData* data, threads_copyDataFunc func)
{
    int i;

    if (func == NULL)
    {
        for(i = 0; i < numThreads; i++)
        {
            threads_data[i].data = (*data);
        }
    }
    else
    {
        for(i = 0; i < numThreads; i++)
        {
            func( data, &threads_data[i].data);
        }
    }
}

void
threads_registerDataThread(int threadId,
        ThreadUserData* data,
        threads_copyDataFunc func)
{
    if (func == NULL)
    {
        threads_data[threadId].data = (*data);
    }
    else
    {
        func( data, &threads_data[threadId].data);
    }
}

void
threads_registerDataGroup(int groupId,
        ThreadUserData* data,
        threads_copyDataFunc func)
{
    int i;

    if (func == NULL)
    {
        for (i = 0; i < threads_groups[groupId].numberOfThreads; i++)
        {
            threads_data[threads_groups[groupId].threadIds[i]].data = (*data);
        }
    }
    else
    {
        for (i = 0; i < threads_groups[groupId].numberOfThreads; i++)
        {
            func( data,
                    &threads_data[threads_groups[groupId].threadIds[i]].data);
        }
    }
}

void
threads_join(void)
{
    int i;

    for(i=0; i < numThreads; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_attr_destroy(&attr);
    pthread_barrier_destroy(&threads_barrier);
}

void
threads_destroy(void)
{
    free(threads_data);
    free(threads_groups);
    free(threads);
}
