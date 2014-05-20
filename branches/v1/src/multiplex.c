/*
 * ===========================================================================
 *
 *      Filename:  multiplex.c
 *
 *      Description:  Utility routines for strings. Depends on bstring lib.
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/time.h>

#include <timer.h>
#include <perfmon.h>
#include <multiplex.h>

static int currentCollection = -1;
static MultiplexCollections* multiplex_set = NULL;
static CyclesData timeData;
static int  multiplex_useMarker = 0;

void
multiplex_swapEventSet ()
{
    int threadId;
    PerfmonEventSet* collection;

    /* collection from last run */
    collection = multiplex_set->collections + currentCollection;

    for (threadId = 0; threadId < perfmon_numThreads; threadId++)
    {
        /* Stop counters */
        if (!multiplex_useMarker) perfmon_stopCountersThread(threadId);
        /* Accumulate counters */
        for (int i=0; i<collection->numberOfEvents; i++)
        {
//            collection->events[i].result[threadId] += 
 //               (double) perfmon_threadData[threadId].counters[collection->events[i].index].counterData;
        }
    }

    /* switch to next collection */
    if( currentCollection == multiplex_set->numberOfCollections-1)
    {
        currentCollection = 0;
    }
    else
    {
        currentCollection++;
    }
    collection = multiplex_set->collections + currentCollection;

    for (threadId = 0; threadId < perfmon_numThreads; threadId++)
    {
        /* Reconfigure counters */
        for (int i=0; i<collection->numberOfEvents; i++)
        {
            perfmon_setupCounterThread(threadId,
                    collection->events[i].event.eventId,
                    collection->events[i].event.umask,
                    collection->events[i].index);
        }

        /* Start counters */
       if (!multiplex_useMarker)  perfmon_startCountersThread(threadId);
    }
}

void
multiplex_init(MultiplexCollections* set)
{
    int i;

    multiplex_set = set;

    for (i=0;i<multiplex_set->numberOfCollections; i++)
    {
//        perfmon_initEventset(multiplex_set->collections+i);
    }
}

void
multiplex_start(int useMarker)
{
    struct itimerval val;
    struct sigaction sa;

    multiplex_useMarker = useMarker;

    val.it_interval.tv_sec = 0;
    val.it_interval.tv_usec = 500;
    val.it_value.tv_sec = 0; 
    val.it_value.tv_usec = 100;

    sa.sa_handler = multiplex_swapEventSet;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    if (sigaction(SIGALRM, &sa, NULL) == -1)
    {
        /* Handle error */;
        perror("sigaction");
    }

    setitimer(ITIMER_REAL, &val,0);
    timer_startCycles(&timeData);
}

void
multiplex_stop()
{
    struct itimerval val;

    val.it_interval.tv_sec = 0;
    val.it_interval.tv_usec = 0;
    val.it_value.tv_sec = 0; 
    val.it_value.tv_usec = 0;

    timer_stopCycles(&timeData);
    setitimer(ITIMER_REAL, &val,0);

    multiplex_set->time = timer_printCyclesTime(&timeData);
}



