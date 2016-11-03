/*
 * =======================================================================================
 *
 *      Filename:  hashTable.c
 *
 *      Description: Hashtable implementation based on SGLIB.
 *                   Used for Marker API result handling.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com
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

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <ghash.h>
#include <bstrlib.h>
#include <types.h>
#include <hashTable.h>
#include <error.h>
#include <likwid.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

typedef struct {
    pthread_t tid;
    uint32_t coreId;
    GHashTable* hashTable;
} ThreadList;

static ThreadList* threadList[MAX_NUM_THREADS];

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
hashTable_init()
{
    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        threadList[i] = NULL;
    }
}

void
hashTable_initThread(int coreID)
{
    ThreadList* resPtr = threadList[coreID];
    /* check if thread was already initialized */
    if (resPtr == NULL)
    {
        resPtr = (ThreadList*) malloc(sizeof(ThreadList));
        /* initialize structure */
        resPtr->tid =  pthread_self();
        resPtr->coreId  = coreID;
        resPtr->hashTable = g_hash_table_new(g_str_hash, g_str_equal);
        threadList[coreID] = resPtr;
    }
}

int
hashTable_get(bstring label, LikwidThreadResults** resEntry)
{
    int coreID = likwid_getProcessorId();
    ThreadList* resPtr = threadList[coreID];

    /* check if thread was already initialized */
    if (resPtr == NULL)
    {
        resPtr = (ThreadList*) malloc(sizeof(ThreadList));
        /* initialize structure */
        resPtr->tid =  pthread_self();
        resPtr->coreId  = coreID;
        resPtr->hashTable = g_hash_table_new(g_str_hash, g_str_equal);
        threadList[coreID] = resPtr;
    }

    (*resEntry) = g_hash_table_lookup(resPtr->hashTable, (gpointer) bdata(label));

    /* if region is not known create new region and add to hashtable */
    if ( (*resEntry) == NULL )
    {
        (*resEntry) = (LikwidThreadResults*) malloc(sizeof(LikwidThreadResults));
        (*resEntry)->label = bstrcpy (label);
        (*resEntry)->time = 0.0;
        (*resEntry)->count = 0;
        for (int i=0; i< NUM_PMC; i++)
        {
            (*resEntry)->PMcounters[i] = 0.0;
            (*resEntry)->StartPMcounters[i] = 0.0;
        }

        g_hash_table_insert(
                resPtr->hashTable,
                (gpointer) g_strdup(bdata(label)),
                (gpointer) (*resEntry));
    }

    return coreID;
}

void
hashTable_finalize(int* numThreads, int* numRegions, LikwidResults** results)
{
    int threadId = 0;
    uint32_t numberOfThreads = 0;
    uint32_t numberOfRegions = 0;
    GHashTable* regionLookup;

    regionLookup = g_hash_table_new(g_str_hash, g_str_equal);
    /* determine number of active threads */
    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        if (threadList[i] != NULL)
        {
            numberOfThreads++;
            uint32_t threadNumberOfRegions = g_hash_table_size(threadList[i]->hashTable);

            /*  Determine maximum number of regions */
            if (numberOfRegions < threadNumberOfRegions)
            {
                numberOfRegions = threadNumberOfRegions;
            }
        }
    }

    /* allocate data structures */
    (*results) = (LikwidResults*) malloc(numberOfRegions * sizeof(LikwidResults));
    if (!(*results))
    {
        fprintf(stderr, "Failed to allocate %lu bytes for the results\n",
                numberOfRegions * sizeof(LikwidResults));
    }
    else
    {
        for ( uint32_t i=0; i < numberOfRegions; i++ )
        {
            (*results)[i].time = (double*) malloc(numberOfThreads * sizeof(double));
            if (!(*results)[i].time)
            {
                fprintf(stderr, "Failed to allocate %lu bytes for the time storage\n",
                        numberOfThreads * sizeof(double));
                break;
            }
            (*results)[i].count = (uint32_t*) malloc(numberOfThreads * sizeof(uint32_t));
            if (!(*results)[i].count)
            {
                fprintf(stderr, "Failed to allocate %lu bytes for the count storage\n",
                        numberOfThreads * sizeof(uint32_t));
                break;
            }
            (*results)[i].cpulist = (int*) malloc(numberOfThreads * sizeof(int));
            if (!(*results)[i].count)
            {
                fprintf(stderr, "Failed to allocate %lu bytes for the cpulist storage\n",
                        numberOfThreads * sizeof(int));
                break;
            }
            (*results)[i].counters = (double**) malloc(numberOfThreads * sizeof(double*));
            if (!(*results)[i].counters)
            {
                fprintf(stderr, "Failed to allocate %lu bytes for the counter result storage\n",
                        numberOfThreads * sizeof(double*));
                break;
            }

            for ( uint32_t j=0; j < numberOfThreads; j++ )
            {
                (*results)[i].time[j] = 0.0;
                (*results)[i].count[j] = 0;
                (*results)[i].cpulist[j] = -1;
                (*results)[i].counters[j] = (double*) malloc(NUM_PMC * sizeof(double));
                if (!(*results)[i].counters)
                {
                    fprintf(stderr, "Failed to allocate %lu bytes for the counter result storage for thread %d\n",
                            NUM_PMC * sizeof(double), j);
                    break;
                }
                else
                {
                    for ( uint32_t k=0; k < NUM_PMC; k++ )
                    {
                        (*results)[i].counters[j][k] = 0.0;
                    }
                }
            }
        }
    }

    uint32_t regionIds[numberOfRegions];
    uint32_t currentRegion = 0;

    for (int core=0; core<MAX_NUM_THREADS; core++)
    {
        ThreadList* resPtr = threadList[core];

        if (resPtr != NULL)
        {
            LikwidThreadResults* threadResult  = NULL;
            GHashTableIter iter;
            gpointer key, value;
            g_hash_table_iter_init (&iter, resPtr->hashTable);

            /* iterate over all regions in thread */
            while (g_hash_table_iter_next (&iter, &key, &value))
            {
                threadResult = (LikwidThreadResults*) value;
                uint32_t* regionId = (uint32_t*) g_hash_table_lookup(regionLookup, key);

                /* is region not yet registered */
                if ( regionId == NULL )
                {
                    (*results)[currentRegion].tag = bstrcpy (threadResult->label);
                    (*results)[currentRegion].groupID = threadResult->groupID;
                    regionIds[currentRegion] = currentRegion;
                    regionId = regionIds + currentRegion;
                    g_hash_table_insert(regionLookup, g_strdup(key), (regionIds+currentRegion));
                    currentRegion++;
                }

                (*results)[*regionId].count[threadId] = threadResult->count;
                (*results)[*regionId].time[threadId] = threadResult->time;
                (*results)[*regionId].cpulist[threadId] = threadResult->cpuID;

                for ( int j=0; j < NUM_PMC; j++ )
                {
                    (*results)[*regionId].counters[threadId][j] = threadResult->PMcounters[j];
                }
                //bdestroy(threadResult->label);
                //free(threadResult);
            }

            threadId++;
            /*g_hash_table_destroy(resPtr->hashTable);
            free(resPtr);
            threadList[core] = NULL;*/
        }
    }
    g_hash_table_destroy(regionLookup);
    regionLookup = NULL;
    (*numThreads) = numberOfThreads;
    (*numRegions) = numberOfRegions;
}

void hashTable_updateOverflows(int coreID, int ctr, int overflows)
{
    GHashTableIter i;
    gpointer k, v;
    g_hash_table_iter_init(&i, threadList[coreID]->hashTable);
    while(g_hash_table_iter_next(&i, &k, &v) == TRUE)
    {
        LikwidThreadResults *res = (LikwidThreadResults *)v;
        if (res->state == REGION_RUNNING)
        {
            DEBUG_PRINT(DEBUGLEV_DETAIL, Adding %d overflows to region %s, overflows, (char*)k);
            res->StartOverflows[ctr] -= overflows;
        }
    }
}


void __attribute__((destructor (102))) hashTable_finalizeDestruct(void)
{
    for (int core=0; core<MAX_NUM_THREADS; core++)
    {
        ThreadList* resPtr = threadList[core];
        if (resPtr != NULL)
        {
            g_hash_table_destroy(resPtr->hashTable);
            free(resPtr);
            threadList[core] = NULL;
        }
    }
}

