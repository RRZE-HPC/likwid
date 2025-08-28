/*
 * =======================================================================================
 *
 *      Filename:  barrier.c
 *
 *      Description:  Implementation of threaded spin loop barrier
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

#include <barrier.h>
#include <errno.h>

/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define CACHELINE_SIZE 64

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static BarrierGroup *groups;
static int currentGroupId = 0;
static int maxGroupId     = 0;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int barrier_registerGroup(int numThreads)
{
    int ret;

    if (currentGroupId > maxGroupId) {
        fprintf(
            stderr, "ERROR: Group ID %d larger than maxGroupID %d\n", currentGroupId, maxGroupId);
    }

    groups[currentGroupId].numberOfThreads = numThreads;
    ret                                    = posix_memalign(
        (void **)&groups[currentGroupId].groupBval, CACHELINE_SIZE, numThreads * 32 * sizeof(int));

    if (ret < 0) {
        fprintf(stderr, "ERROR: Cannot register thread group - %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    return currentGroupId++;
}

void barrier_registerThread(BarrierData *barr, int groupId, int threadId)
{
    int ret;
    int i;
    int j = 1;
    if (groupId > currentGroupId) {
        fprintf(stderr, "ERROR: Group not yet registered");
    }
    if (threadId > groups[groupId].numberOfThreads) {
        fprintf(stderr, "ERROR: Thread ID %d too large\n", threadId);
    }

    barr->numberOfThreads = groups[groupId].numberOfThreads;
    barr->offset          = 0;
    barr->val             = 1;
    barr->bval            = groups[groupId].groupBval;
    ret                   = posix_memalign(
        (void **)&(barr->index), CACHELINE_SIZE, barr->numberOfThreads * sizeof(int));

    if (ret < 0) {
        fprintf(stderr, "ERROR: Cannot register thread - %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }

    barr->index[0] = threadId;

    for (i = 0; i < barr->numberOfThreads; i++) {
        if (!(i == threadId)) {
            barr->index[j++] = i;
        }
    }
}

void barrier_init(int numberOfGroups)
{
    maxGroupId = numberOfGroups - 1;
    groups     = (BarrierGroup *)malloc(numberOfGroups * sizeof(BarrierGroup));
    if (!groups) {
        fprintf(stderr, "ERROR: Cannot allocate barrier - %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
}

void barrier_synchronize(BarrierData *barr)
{
    int i;

    barr->bval[barr->index[0] * 32 + barr->offset * 16] = barr->val;

    for (i = 1; i < barr->numberOfThreads; i++) {
        while (barr->bval[barr->index[i] * 32 + barr->offset * 16] != barr->val) {
#if defined(__arm__) || defined(__ARM_ARCH_8A)
            __asm__("nop");
#endif
#if defined(__i386__) || defined(__i486__) || defined(__i586__) || defined(__i686__) ||            \
    defined(__x86_64)
            __asm__("pause");
#endif
#ifdef _ARCH_PCC
            __asm__("noop");
#endif
        }
    }

    if (barr->offset) {
        barr->val = !barr->val;
    }
    barr->offset = !barr->offset;
}

void barrier_destroy(BarrierData *barr)
{
    if (currentGroupId > maxGroupId) {
        fprintf(
            stderr, "ERROR: Group ID %d larger than maxGroupID %d\n", currentGroupId, maxGroupId);
    }
    free(barr->index);
    free(groups[currentGroupId].groupBval);
}
