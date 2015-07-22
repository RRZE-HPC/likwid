/*
 * =======================================================================================
 *
 *      Filename:  strUtil.c
 *
 *      Description:  Utility string routines building upon bstrlib
 *
 *      Version:   4.0
 *      Released:  22.7.2015
 *
 *      Author:   Jan Treibig (jt), jan.treibig@gmail.com.
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
#include <strUtil.h>
#include <math.h>

static int str2int(const char* str)
{
    char* endptr;
    errno = 0;
    unsigned long val;
    val = strtoul(str, &endptr, 10);

    if ((errno == ERANGE && val == LONG_MAX)
        || (errno != 0 && val == 0))
    {
        fprintf(stderr, "Value in string out of range\n");
        return -EINVAL;
    }

    if (endptr == str)
    {
        fprintf(stderr, "No digits were found\n");
        return -EINVAL;
    }

    return (int) val;
}

uint64_t bstr_to_doubleSize(const_bstring str, DataType type)
{
    bstring unit = bmidstr(str, blength(str)-2, 2);
    bstring sizeStr = bmidstr(str, 0, blength(str)-2);
    uint64_t sizeU = str2int(bdata(sizeStr));
    uint64_t junk = 0;
    uint64_t bytesize = 0;

    switch (type)
    {
        case SINGLE:
            bytesize = 4;
            break;

        case DOUBLE:
            bytesize = 8;
            break;
    }

    if ((biseqcstr(unit, "kB"))||(biseqcstr(unit, "KB")))
    {
        junk = (sizeU *1000)/bytesize;
    }
    else if (biseqcstr(unit, "MB"))
    {
        junk = (sizeU *1000000)/bytesize;
    }
    else if (biseqcstr(unit, "GB"))
    {
        junk = (sizeU *1000000000)/bytesize;
    }
    else if (biseqcstr(unit, "B"))
    {
        junk = (sizeU)/bytesize;
    }

    return junk;
}

void bstr_to_workgroup(Workgroup* group, const_bstring str, DataType type, int numberOfStreams)
{
    uint32_t i;
    int parseStreams = 0;
    bstring threadInfo;
    bstring streams= bformat("0");
    struct bstrList* tokens;
    struct bstrList* subtokens;
    AffinityDomains_t domains;
    AffinityDomain* domain = NULL;

    /* split the workgroup into the thread and the streams part */
    tokens = bsplit(str,'-');

    if (tokens->qty == 2)
    {
        threadInfo = bstrcpy(tokens->entry[0]);
        streams = bstrcpy(tokens->entry[1]);
        parseStreams = 1;
    }
    else if (tokens->qty == 1)
    {
        threadInfo = bstrcpy(tokens->entry[0]);
    }
    else
    {
        fprintf(stderr, "Error in parsing workgroup string\n");
    }

    bstrListDestroy (tokens);
    tokens = bsplit(threadInfo,':');

    if (tokens->qty == 5)
    {
        uint32_t maxNumThreads;
        int chunksize;
        int stride;
        int counter;
        int currentId = 0;
        int startId = 0;

        domains = get_affinityDomains();
        for (i = 0; i < domains->numberOfAffinityDomains; i++)
        {
            if (bstrcmp(domains->domains[i].tag, tokens->entry[0]) == BSTR_OK)
            {
                domain = &(domains->domains[i]);
                break;
            }
        }

        if (domain == NULL)
        {
            fprintf(stderr, "Error: Domain %s not available on current machine.\nTry likwid-bench -p for supported domains.",
                bdata(tokens->entry[0]));
            exit(EXIT_FAILURE);
        }

        group->size = bstr_to_doubleSize(tokens->entry[1], type);
        group->numberOfThreads = str2int(bdata(tokens->entry[2]));
        chunksize = str2int(bdata(tokens->entry[3]));
        stride = str2int(bdata(tokens->entry[4]));
        maxNumThreads = ceil((double)domain->numberOfProcessors / stride) * chunksize;

        if (group->numberOfThreads > maxNumThreads)
        {
            fprintf(stderr, "Warning: More threads (%d) requested than CPUs in domain %s fulfilling expression (%d).\n",
                    group->numberOfThreads, bdata(tokens->entry[0]), maxNumThreads);
        }

        group->processorIds = (int*) malloc(group->numberOfThreads * sizeof(int));

        counter = chunksize;

        counter = 0;
        for (int j=0; j<group->numberOfThreads; j+=chunksize)
        {
            for(i=0;i<chunksize && j+i<group->numberOfThreads ;i++)
            {
                group->processorIds[startId++] = domain->processorList[counter+i];
            }
            counter += stride;
            if (counter >= domain->numberOfProcessors)
            {
                counter = counter-domain->numberOfProcessors;
            }
        }
    }
    else if (tokens->qty == 3)
    {
        domains = get_affinityDomains();
        for (i = 0; i < domains->numberOfAffinityDomains; i++)
        {
            if (bstrcmp(domains->domains[i].tag, tokens->entry[0]) == BSTR_OK)
            {
                domain = &(domains->domains[i]);
                break;
            }
        }

        if (domain == NULL)
        {
            fprintf(stderr, "Error: Domain %s not available on current machine.\nTry likwid-bench -p for supported domains.",
                    bdata(tokens->entry[0]));
            exit(EXIT_FAILURE);
        }

        group->size = bstr_to_doubleSize(tokens->entry[1], type);
        group->numberOfThreads = str2int(bdata(tokens->entry[2]));

        if (group->numberOfThreads > domain->numberOfProcessors)
        {
            fprintf(stderr, "Warning: More threads (%d) requested than CPUs in domain %s (%d).\n",
                    group->numberOfThreads, bdata(tokens->entry[0]), domain->numberOfProcessors);
        }

        group->processorIds = (int*) malloc(group->numberOfThreads * sizeof(int));

        for (i=0; i<group->numberOfThreads; i++)
        {
            group->processorIds[i] = domain->processorList[i % domain->numberOfProcessors];
        }
    }
    else if (tokens->qty == 2)
    {
        domains = get_affinityDomains();
        for (i = 0; i < domains->numberOfAffinityDomains; i++)
        {
            if (bstrcmp(domains->domains[i].tag, tokens->entry[0]) == BSTR_OK)
            {
                domain = &(domains->domains[i]);
                break;
            }
        }

        if (domain == NULL)
        {
            fprintf(stderr, "Error: Domain %s not available on current machine.\nTry likwid-bench -p for supported domains.",
                            bdata(tokens->entry[0]));
            exit(EXIT_FAILURE);
        }

        group->size = bstr_to_doubleSize(tokens->entry[1], type);
        group->numberOfThreads = domain->numberOfProcessors;
        group->processorIds = (int*) malloc(group->numberOfThreads * sizeof(int));

        for (i=0; i<group->numberOfThreads; i++)
        {
            group->processorIds[i] = domain->processorList[i];
        }
    }
    else
    {
        fprintf(stderr, "Error in parsing workgroup string\n");
    }

    bstrListDestroy(tokens);

    /* parse stream list */
    if (parseStreams)
    {
        tokens = bsplit(streams,',');

        if (tokens->qty < numberOfStreams)
        {
            fprintf(stderr, "Error: Testcase requires at least %d streams\n", numberOfStreams);
        }

        group->streams = (Stream*) malloc(numberOfStreams * sizeof(Stream));

        for (i=0;i<(uint32_t) tokens->qty;i++)
        {
            subtokens = bsplit(tokens->entry[i],':');

            if ( subtokens->qty == 3 )
            {
                int index = str2int(bdata(subtokens->entry[0]));
                if (index >= numberOfStreams)
                {
                    fprintf(stderr, "Error: Stream index %d out of range\n",index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = str2int(bdata(subtokens->entry[2]));
            }
            else if ( subtokens->qty == 2 )
            {
                int index = str2int(bdata(subtokens->entry[0]));
                if (index >= numberOfStreams)
                {
                    fprintf(stderr, "Error: Stream index %d out of range\n",index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = 0;
            }
            else
            {
                fprintf(stderr, "Error: Cannot parse stream placement defintition in %s\n", bdata(str));
            }

            bstrListDestroy(subtokens);
        }

        bstrListDestroy(tokens);
    }
    else
    {
        group->streams = (Stream*) malloc(numberOfStreams * sizeof(Stream));

        for (i=0; i< (uint32_t)numberOfStreams; i++)
        {
            group->streams[i].domain = domain->tag;
            group->streams[i].offset = 0;
        }
    }

    group->size /= numberOfStreams;
    return;
}
