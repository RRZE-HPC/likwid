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
#include <stdio.h>
#include <string.h>

#include <allocator_types.h>
#include <allocator.h>
#include <likwid.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int numberOfAllocatedVectors = 0;
static allocation* allocList;
static AffinityDomains_t domains = NULL;

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

size_t
allocator_dataTypeLength(DataType type)
{
    switch (type)
    {
        case INT:
            return sizeof(int);
            break;
        case SINGLE:
            return sizeof(float);
            break;
        case DOUBLE:
            return sizeof(double);
            break;
        default:
            return 0;
    }
    return 0;
}

void
allocator_allocateVector(
        void** ptr,
        int alignment,
        uint64_t size,
        off_t offset,
        DataType type,
        int stride,
        bstring domainString,
        InitMethod init_method,
        uint64_t init_method_arg,
        int init_per_thread)
{
    int i;
    size_t bytesize = 0;
    const AffinityDomain* domain = NULL;
    int errorCode;
    int elements = 0;
    affinity_init();

    size_t typesize = allocator_dataTypeLength(type);
    bytesize = (size+offset) * typesize;
    elements = alignment / typesize;

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
    printf("Allocate: Process running on hwthread %d (Domain %s) - Vector length %llu/%llu Offset %llu Alignment %llu\n",
            affinity_processGetProcessorId(),
            bdata(domain->tag),
            LLU_CAST size,
            LLU_CAST bytesize,
            offset,
            LLU_CAST elements);

    if (!init_per_thread)
    {
        allocator_initVector(ptr, size, offset, type, stride, init_method, init_method_arg, true);
    }
}

void allocator_initVector(void** ptr,
        uint64_t size,
        off_t offset,
        DataType type,
        int stride,
        InitMethod init_method,
        uint64_t init_method_arg,
        bool fill)
{
    switch ( type )
    {
        case INT:
            {
                int* iptr = (int*) (*ptr);
                iptr += offset;

                switch ( init_method ) {
                    case CONSTANT_ONE:
                        for ( uint64_t i=0; fill && i < size; i++ )
                        {
                            iptr[i] = 1;
                        }
                        break;
                    case INDEX_STRIDE:
                        for ( int64_t i=0; fill && i < size; i++ )
                        {
                            iptr[i] = (int) ((i + stride) % size);
                        }
                        break;
                    case LINKED_LIST:
			;
                        /* init_method_arg is guaranteed to be a non-zero multiple of sizeof(int) or linked lists items */
                        const int64_t ll_int_item_size = init_method_arg / sizeof(int);
                        const int64_t ll_items = size / init_method_arg;

                        for ( int64_t i=0; fill && i < ll_items; i++ )
                        {
                            iptr[i * ll_int_item_size] = i * init_method_arg;
                        }

                        /* Use Sattolo's algorithm to create single-cycle permutation */
                        struct drand48_data rng_state;
                        srand48_r(0, &rng_state);

                        int64_t i = ll_items;
                        while ( i > 1 && fill )
                        {
                            i--;

                            long j;
                            mrand48_r(&rng_state, &j);
                            j = abs(j) % i;

                            /* swap */
                            const int tmp = iptr[i * ll_int_item_size];
                            iptr[i * ll_int_item_size] = iptr[j * ll_int_item_size];
                            iptr[j * ll_int_item_size] = tmp;
                        }
                        break;
                }

                *ptr = (void*) iptr;
            }
            break;

        case SINGLE:
            {
                float* sptr = (float*) (*ptr);
                sptr += offset;

                for ( uint64_t i=0; fill && i < size; i++ )
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

                for ( uint64_t i=0; fill && i < size; i++ )
                {
                    dptr[i] = 1.0;
                }
                *ptr = (void*) dptr;
            }
            break;
    }
}
