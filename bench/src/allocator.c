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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <allocator.h>
#include <allocator_types.h>
#include <likwid.h>

/* #####   VARIABLES  -  LOCAL TO THIS SOURCE FILE   ###################### */

static int numberOfAllocatedVectors = 0;
static allocation *allocList;
static AffinityDomains_t domains = NULL;

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */

void allocator_init(int numVectors)
{
    allocList = (allocation *)malloc(numVectors * sizeof(allocation));
    domains   = get_affinityDomains();
}

void allocator_finalize()
{
    int i;

    for (i = 0; i < numberOfAllocatedVectors; i++) {
        free(allocList[i].ptr);
        allocList[i].ptr    = NULL;
        allocList[i].size   = 0;
        allocList[i].offset = 0;
    }
    numberOfAllocatedVectors = 0;
}

size_t allocator_dataTypeLength(DataType type)
{
    switch (type) {
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

void allocator_allocateVector(void **ptr, int alignment, uint64_t size, int offset, DataType type,
    int stride, bstring domainString, int init_per_thread)
{
    (void)stride;

    size_t bytesize              = 0;
    const AffinityDomain *domain = NULL;
    int errorCode;
    int elements = 0;
    affinity_init();

    size_t typesize = allocator_dataTypeLength(type);
    bytesize        = (size + offset) * typesize;
    elements        = alignment / typesize;

#pragma GCC diagnostic ignored "-Wnonnull"
    for (size_t i = 0; i < domains->numberOfAffinityDomains; i++) {
        if (strcmp(domains->domains[i].tag, bdata(domainString)) == 0) {
            domain = domains->domains + i;
        }
    }
    if (!domain) {
        fprintf(stderr,
            "Error: Cannot use desired domain %s for vector placement, Domain %s does not exist.\n",
            bdata(domainString),
            bdata(domainString));
        exit(EXIT_FAILURE);
    }

    errorCode = posix_memalign(ptr, alignment, bytesize);

    if (errorCode) {
        if (errorCode == EINVAL) {
            fprintf(stderr, "Error: Alignment parameter is not a power of two\n");
            exit(EXIT_FAILURE);
        }
        if (errorCode == ENOMEM) {
            fprintf(stderr, "Error: Insufficient memory to fulfill the request\n");
            exit(EXIT_FAILURE);
        }
    }

    if ((*ptr) == NULL) {
        fprintf(stderr, "Error: posix_memalign failed!\n");
        exit(EXIT_FAILURE);
    }

    allocList[numberOfAllocatedVectors].ptr    = *ptr;
    allocList[numberOfAllocatedVectors].size   = bytesize;
    allocList[numberOfAllocatedVectors].offset = offset;
    allocList[numberOfAllocatedVectors].type   = type;
    numberOfAllocatedVectors++;

    affinity_pinProcess(domain->processorList[0]);
    printf("Allocate: Process running on hwthread %d (Domain %s) - Vector length %llu/%llu Offset "
           "%d Alignment %llu\n",
        affinity_processGetProcessorId(),
        domain->tag,
        LLU_CAST size,
        LLU_CAST bytesize,
        offset,
        LLU_CAST elements);

    if (!init_per_thread) {
        switch (type) {
        case INT: {
            int *sptr = (int *)(*ptr);
            sptr += offset;

            for (uint64_t i = 0; i < size; i++) {
                sptr[i] = 1;
            }
            *ptr = (void *)sptr;

        } break;

        case SINGLE: {
            float *sptr = (float *)(*ptr);
            sptr += offset;

            for (uint64_t i = 0; i < size; i++) {
                sptr[i] = 1.0;
            }
            *ptr = (void *)sptr;

        } break;

        case DOUBLE: {
            double *dptr = (double *)(*ptr);
            dptr += offset;

            for (uint64_t i = 0; i < size; i++) {
                dptr[i] = 1.0;
            }
            *ptr = (void *)dptr;
        } break;
        }
    }
}
