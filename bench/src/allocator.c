/*
 * =======================================================================================
 *
 *      Filename:  allocator.c
 *
 *      Description:  Implementation of allocator module.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2015 RRZE, University Erlangen-Nuremberg
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
#include <string.h>

#include <allocator_types.h>
#include <allocator.h>
#include <likwid.h>

/* #####   EXPORTED VARIABLES   ########################################### */


/* #####   MACROS  -  LOCAL TO THIS SOURCE FILE   ######################### */


/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int numberOfAllocatedVectors = 0;
static allocation* allocList;
static AffinityDomains_t domains = NULL;

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */


/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void
allocator_init(int numVectors)
{
    allocList = (allocation*) malloc(numVectors * sizeof(allocation));
    domains = get_affinityDomains();
}


void
allocator_finalize()
{
    int i;

    for (i=0; i<numberOfAllocatedVectors; i++)
    {
        free(allocList[i].ptr);
        allocList[i].ptr = NULL;
        allocList[i].size = 0;
        allocList[i].offset = 0;
    }
    numberOfAllocatedVectors = 0;
}

void
allocator_allocateVector(
        void** ptr,
        int alignment,
        uint64_t size,
        int offset,
        DataType type,
        bstring domainString)
{
    int i;
    size_t bytesize = 0;
    const AffinityDomain* domain = NULL;
    int errorCode;
    int elements = 0;

    switch ( type )
    {
        case SINGLE:
            bytesize = (size+offset) * sizeof(float);
            elements = alignment / sizeof(float);
            break;

        case DOUBLE:
            bytesize = (size+offset) * sizeof(double);
            elements = alignment / sizeof(double);
            break;
    }

    for (i=0;i<domains->numberOfAffinityDomains;i++)
    {
        if (biseq(domainString, domains->domains[i].tag))
        {
            domain = domains->domains + i;
        }
    }
    if (!domain)
    {
        fprintf(stderr, "Error: Cannot use desired domain %s for vector placement, Domain %s does not exist.\n",
                        bdata(domainString), bdata(domainString));
        exit(EXIT_FAILURE);
    }

    errorCode =  posix_memalign(ptr, alignment, bytesize);

    if (errorCode)
    {
        if (errorCode == EINVAL)
        {
            fprintf(stderr,
                    "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM)
        {
            fprintf(stderr,
                    "Error: Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if ((*ptr) == NULL)
    {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    allocList[numberOfAllocatedVectors].ptr = *ptr;
    allocList[numberOfAllocatedVectors].size = bytesize;
    allocList[numberOfAllocatedVectors].offset = offset;
    allocList[numberOfAllocatedVectors].type = type;
    numberOfAllocatedVectors++;

    affinity_pinProcess(domain->processorList[0]);
    printf("Allocate: Process running on core %d (Domain %s) - Vector length %llu Offset %d Alignment %llu\n",
            affinity_processGetProcessorId(),
            bdata(domain->tag),
            LLU_CAST size,
            offset,
            LLU_CAST elements);

    switch ( type )
    {
        case SINGLE:
            {
                float* sptr = (float*) (*ptr);
                sptr += offset;

                for ( uint64_t i=0; i < size; i++ )
                {
                    sptr[i] = 1.0;
                }
                *ptr = (void*) sptr;

            }
            break;

        case DOUBLE:
            {
                double* dptr = (double*) (*ptr);
                dptr += offset;

                for ( uint64_t i=0; i < size; i++ )
                {
                    dptr[i] = 1.0;
                }
                *ptr = (void*) dptr;
            }
            break;
    }
}

