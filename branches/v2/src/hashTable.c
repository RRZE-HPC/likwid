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
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2012 Jan Treibig 
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

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>

#include <sglib.h>
#include <bstrlib.h>
#include <types.h>
#include <hashTable.h>
#include <likwid.h>

typedef struct {
    pthread_t tid;
    uint32_t coreId;
    LikwidThreadResults* hashTable[HASH_TABLE_SIZE];
    uint32_t currentMaxSize;
    uint32_t numberOfRegions;
} ThreadList;


static ThreadList* threadList[MAX_NUM_THREADS];

static unsigned int hashFunction(LikwidThreadResults* item)
{
    const char* str =  bdata(item->label);
    unsigned int len = blength(item->label);
    unsigned int b    = 378551;
    unsigned int a    = 63689;
    unsigned int hash = 0;
    unsigned int i    = 0;

    for(i = 0; i < len; str++, i++)
    {
        hash = hash * a + (*str);
        a    = a * b;
    }

    return hash;
}

/* ======================================================================== */
#define SLIST_COMPARATOR(e1, e2)    bstrncmp((e1)->label,(e2)->label,100)

SGLIB_DEFINE_LIST_PROTOTYPES(LikwidThreadResults,SLIST_COMPARATOR , next)
SGLIB_DEFINE_LIST_FUNCTIONS(LikwidThreadResults,SLIST_COMPARATOR , next)
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(LikwidThreadResults, HASH_TABLE_SIZE, hashFunction)
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(LikwidThreadResults, HASH_TABLE_SIZE, hashFunction)

void
hashTable_init()
{
    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        threadList[i] = NULL;
    }
}


int
hashTable_get(bstring label, LikwidThreadResults** resEntry)
{
    LikwidThreadResults li;
    int coreID = likwid_getProcessorId();

    ThreadList* resPtr = threadList[coreID];
    li.label = label;

    /* check if thread was already initialized */
    if (resPtr == NULL)
    {
        resPtr = (ThreadList*) malloc(sizeof(ThreadList));
        /* initialize structure */
        resPtr->tid =  pthread_self();
        resPtr->coreId  = coreID;
        resPtr->numberOfRegions = 0;
        sglib_hashed_LikwidThreadResults_init(resPtr->hashTable);
        threadList[coreID] = resPtr;
    }

    /* if region is not known create new region and add to hashtable */
    if (((*resEntry) = sglib_hashed_LikwidThreadResults_find_member(resPtr->hashTable, &li)) == NULL) 
    {
        (*resEntry) = (LikwidThreadResults*) malloc(sizeof(LikwidThreadResults));
        (*resEntry)->label = bstrcpy (label);
        (*resEntry)->time = 0.0;
        (*resEntry)->count = 0;
        resPtr->numberOfRegions++; 
        for (int i=0; i< NUM_PMC; i++) (*resEntry)->PMcounters[i] = 0.0;
        sglib_hashed_LikwidThreadResults_add(resPtr->hashTable, (*resEntry));
    }

    return coreID;
}

void
hashTable_finalize(int* numThreads, int* numRegions, LikwidResults** results)
{
    int init = 0;
    int threadId = 0;
    int regionId = 0;
    uint32_t numberOfThreads = 0;
    uint32_t numberOfRegions = 0;
    struct sglib_hashed_LikwidThreadResults_iterator hash_it;

    /* determine number of threads */
    for (int i=0; i<MAX_NUM_THREADS; i++)
    {
        if (threadList[i] != NULL)
        {
            numberOfThreads++;
            if (!init)
            {
                /* determine number of regions */
                numberOfRegions = threadList[i]->numberOfRegions;
                init = 1;
            } 
            else
            {
                if (numberOfRegions != threadList[i]->numberOfRegions)
                {
                    printf("Different number of regions!! %d\n",threadList[i]->numberOfRegions);
                }
            }
        }
    }

    init = 0;

    for (int core=0; core<MAX_NUM_THREADS; core++)
    {
        ThreadList* resPtr = threadList[core];
        if (resPtr != NULL)
        {

            resPtr->numberOfRegions=0;
            LikwidThreadResults* hash  = NULL;

            if (!init)
            {
                init =1;
                for(hash=sglib_hashed_LikwidThreadResults_it_init(&hash_it,resPtr->hashTable);
                        hash!=NULL; hash=sglib_hashed_LikwidThreadResults_it_next(&hash_it))
                {
                    resPtr->numberOfRegions++;
                }

                if( resPtr->numberOfRegions != numberOfRegions)
                {
                    printf("Different number of regions!!\n");
                }

                /* allocate data structure */
                (*results) = (LikwidResults*) malloc(numberOfRegions * sizeof(LikwidResults));

                for ( uint32_t i=0; i < numberOfRegions; i++ )
                {
                    (*results)[i].time = (double*) malloc(numberOfThreads * sizeof(double));
                    (*results)[i].count = (uint32_t*) malloc(numberOfThreads * sizeof(uint32_t));
                    (*results)[i].counters = (double**) malloc(numberOfThreads * sizeof(double*));

                    for ( uint32_t j=0; j < numberOfThreads; j++ )
                    {
                        (*results)[i].counters[j] = (double*) malloc(NUM_PMC * sizeof(double));
                    }
                }
            }

            regionId = 0;
            /* iterate over all regions in thread */
            for ( hash=sglib_hashed_LikwidThreadResults_it_init(&hash_it,resPtr->hashTable);
                    hash!=NULL; hash=sglib_hashed_LikwidThreadResults_it_next(&hash_it) )
            {
                (*results)[regionId].tag = bstrcpy (hash->label);
                (*results)[regionId].count[threadId] = hash->count;
                (*results)[regionId].time[threadId] = hash->time;

                for ( int j=0; j < NUM_PMC; j++ )
                {
                    (*results)[regionId].counters[threadId][j] = hash->PMcounters[j];
                }

                regionId++;
            }

            threadId++;
        }
    }

    (*numThreads) = numberOfThreads;
    (*numRegions) = numberOfRegions;
}


