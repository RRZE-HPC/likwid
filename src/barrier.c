/*
 * ===========================================================================
 *
 *      Filename:  barrier.c
 *
 *      Description:  Implementation of threaded spin loop barrier
 *
 *      Version:  <VERSION>
 *      Created:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Company:  RRZE Erlangen
 *      Project:  likwid
 *      Copyright:  Copyright (c) 2010, Jan Treibig
 *
 *      This program is free software; you can redistribute it and/or modify
 *      it under the terms of the GNU General Public License, v2, as
 *      published by the Free Software Foundation
 *     
 *      This program is distributed in the hope that it will be useful,
 *      but WITHOUT ANY WARRANTY; without even the implied warranty of
 *      MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *      GNU General Public License for more details.
 *     
 *      You should have received a copy of the GNU General Public License
 *      along with this program; if not, write to the Free Software
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 *
 * ===========================================================================
 */

/* #####   HEADER FILE INCLUDES   ######################################### */
#include <stdlib.h>
#include <stdio.h>

#include <error.h>
#include <types.h>
#include <barrier.h>

/* #####   EXPORTED VARIABLES   ########################################### */


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */

#define CACHELINE_SIZE 64

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static BarrierGroup* groups;
static int currentGroupId = 0;
static int maxGroupId = 0;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

int
barrier_registerGroup(int numThreads)
{
    int ret;

    if (currentGroupId > maxGroupId)
    {
        ERROR_MSG(Group ID too large);
    }

    groups[currentGroupId].numberOfThreads = numThreads;
    ret = posix_memalign(
            (void**) &groups[currentGroupId].groupBval,
            CACHELINE_SIZE, 
            numThreads * 32 * sizeof(int));

    if (ret < 0)
    {
        ERROR;
    }


    return currentGroupId++;
}

void
barrier_registerThread(BarrierData* barr, int groupId, int threadId)
{
    int ret;
    int i;
    int j = 1;
    if (groupId > currentGroupId)
    {
        ERROR_MSG(Group not yet registered);
    }
    if (threadId > groups[groupId].numberOfThreads)
    {
        ERROR_PMSG(Thread ID %d too large,threadId);
    }

    barr->numberOfThreads = groups[groupId].numberOfThreads;
    barr->offset = 0;
    barr->val = 1;
    barr->bval =  groups[groupId].groupBval;
    ret = posix_memalign(
            (void**) &(barr->index),
            CACHELINE_SIZE, 
            barr->numberOfThreads * sizeof(int));

    if (ret < 0)
    {
        ERROR;
    }


    barr->index[0] = threadId;

    for (i = 0; i < barr->numberOfThreads; i++)
    {
        if (!(i == threadId))
        {
            barr->index[j++] = i;
        }
    }
}


void
barrier_init(int numberOfGroups) 
{ 
    maxGroupId = numberOfGroups-1;
    groups = (BarrierGroup*) malloc(numberOfGroups * sizeof(BarrierGroup));
}

void
barrier_synchronize(BarrierData* barr)
{
    int i;

    barr->bval[barr->index[0] * 32 +  barr->offset * 16] = barr->val;

    for (i = 1; i < barr->numberOfThreads; i++)
    {
        while (barr->bval[barr->index[i] * 32 + barr->offset * 16] != barr->val)
        {
            __asm__ ("pause");
        }
    } 
    
    if (barr->offset)
    {
        barr->val = !barr->val;
    }
    barr->offset = !barr->offset;
}


