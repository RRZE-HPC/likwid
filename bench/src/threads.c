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
 *      Copyright (C) 2016 RRZE, University Erlangen-Nuremberg
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <errno.h>
#include <strUtil.h>
#include <threads.h>

/* #####   EXPORTED VARIABLES   ########################################### */

pthread_barrier_t threads_barrier;
ThreadData *threads_data;
ThreadGroup *threads_groups;

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static pthread_t *threads = NULL;
static pthread_attr_t attr;
static int numThreads = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE  ################## */

static int count_characters(const char *str, char character)
{
    if (str == 0)
        return 0;
    const char *p = str;
    int count     = 0;

    do {
        if (*p == character)
            count++;
    } while (*(p++));

    return count;
}

void *dummy_function(void *arg)
{
    (void)arg;
    return NULL;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

// TODO threads_test isn't used anymore. Can we just remove it?
int threads_test()
{
    int cnt = 0;
    pthread_t pid;
    int likwid_pin = count_characters(getenv("LIKWID_PIN"), ',');
    int max_cpus   = sysconf(_SC_NPROCESSORS_CONF);
    int max        = likwid_pin;
    if (likwid_pin == 0) {
        max = max_cpus;
    }
    while (cnt < max) {
        pthread_create(&pid, NULL, dummy_function, NULL);
        cnt++;
    }
    return cnt;
}

void threads_init(int numberOfThreads)
{
    int i;
    numThreads   = numberOfThreads;

    threads      = (pthread_t *)malloc(numThreads * sizeof(pthread_t));
    threads_data = (ThreadData *)malloc(numThreads * sizeof(ThreadData));

    for (i = 0; i < numThreads; i++) {
        threads_data[i].numberOfThreads       = numThreads;
        threads_data[i].globalNumberOfThreads = numThreads;
        threads_data[i].globalThreadId        = i;
        threads_data[i].threadId              = i;
    }

    pthread_barrier_init(&threads_barrier, NULL, numThreads);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
}

void threads_create(void *(*startRoutine)(void *))
{
    int i;

    for (i = 0; i < numThreads; i++) {
        pthread_create(&threads[i], &attr, startRoutine, (void *)&threads_data[i]);
    }
}

void threads_createGroups(int numberOfGroups, Workgroup *groups)
{
    int i;
    int j;
    int globalId   = 0;

    threads_groups = (ThreadGroup *)malloc(numberOfGroups * sizeof(ThreadGroup));
    if (!threads_groups) {
        fprintf(stderr, "ERROR: Cannot allocate thread groups - %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    for (i = 0; i < numberOfGroups; i++) {
        threads_groups[i].numberOfThreads = groups[i].numberOfThreads;
        threads_groups[i].threadIds =
            (int *)malloc(threads_groups[i].numberOfThreads * sizeof(int));
        if (!threads_groups[i].threadIds) {
            fprintf(stderr,
                "ERROR: Cannot allocate threadID list for thread groups - %s\n",
                strerror(errno));
            exit(EXIT_FAILURE);
        }

        for (j = 0; j < threads_groups[i].numberOfThreads; j++) {
            threads_data[globalId].threadId        = j;
            threads_data[globalId].groupId         = i;
            threads_data[globalId].numberOfGroups  = numberOfGroups;
            threads_data[globalId].numberOfThreads = threads_groups[i].numberOfThreads;
            threads_groups[i].threadIds[j]         = globalId++;
        }
    }
}

void threads_registerDataAll(ThreadUserData *data, threads_copyDataFunc func)
{
    int i;

    if (func == NULL) {
        for (i = 0; i < numThreads; i++) {
            threads_data[i].data = (*data);
        }
    } else {
        for (i = 0; i < numThreads; i++) {
            func(data, &threads_data[i].data);
        }
    }
}

void threads_registerDataThread(int threadId, ThreadUserData *data, threads_copyDataFunc func)
{
    if (func == NULL) {
        threads_data[threadId].data = (*data);
    } else {
        func(data, &threads_data[threadId].data);
    }
}

void threads_registerDataGroup(int groupId, ThreadUserData *data, threads_copyDataFunc func)
{
    int i;

    if (func == NULL) {
        for (i = 0; i < threads_groups[groupId].numberOfThreads; i++) {
            threads_data[threads_groups[groupId].threadIds[i]].data = (*data);
        }
    } else {
        for (i = 0; i < threads_groups[groupId].numberOfThreads; i++) {
            func(data, &threads_data[threads_groups[groupId].threadIds[i]].data);
        }
    }
}

size_t threads_updateIterations(int groupId, size_t demandIter)
{
    int i             = 0;
    size_t iterations = threads_data[0].data.iter;
    if (demandIter > 0) {
        iterations = demandIter;
    }
    iterations = (iterations < MIN_ITERATIONS ? MIN_ITERATIONS : iterations);

    for (i = 0; i < threads_groups[groupId].numberOfThreads; i++) {
        threads_data[threads_groups[groupId].threadIds[i]].data.iter   = iterations;
        threads_data[threads_groups[groupId].threadIds[i]].data.cycles = 0;
        threads_data[threads_groups[groupId].threadIds[i]].cycles      = 0;
        threads_data[threads_groups[groupId].threadIds[i]].time        = 0;
    }
    return iterations;
}

void threads_join(void)
{
    int i = 0;

    for (i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }
}

void threads_destroy(int numberOfGroups, int numberOfStreams)
{
    (void)numberOfStreams;

    int i = 0, j = 0;
    pthread_attr_destroy(&attr);
    pthread_barrier_destroy(&threads_barrier);

    for (i = 0; i < numberOfGroups; i++) {
        for (j = 0; j < threads_groups[i].numberOfThreads; j++) {
            free(threads_data[threads_groups[i].threadIds[j]].data.processors);
            free(threads_data[threads_groups[i].threadIds[j]].data.streams);
        }
        free(threads_groups[i].threadIds);
    }
    free(threads_groups);
    free(threads);
}
