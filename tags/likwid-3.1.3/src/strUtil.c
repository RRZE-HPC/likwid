/*
 * =======================================================================================
 *
 *      Filename:  strUtil.c
 *
 *      Description:  Utility routines for strings. Depends on bstring lib.
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:  Jan Treibig (jt), jan.treibig@gmail.com
 *      Project:  likwid
 *
 *      Copyright (C) 2014 Jan Treibig
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
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

#include <error.h>
#include <types.h>
#include <bstrlib.h>
#include <strUtil.h>
#include <affinity.h>
#include <cpuid.h>
#include <pci.h>

/* #####   FUNCTION DEFINITIONS  -  LOCAL TO THIS SOURCE FILE   ########### */
static int
cpu_count(cpu_set_t* set)
{
    uint32_t i;
    int s = 0;
    const __cpu_mask *p = set->__bits;
    const __cpu_mask *end = &set->__bits[sizeof(cpu_set_t) / sizeof (__cpu_mask)];

    while (p < end)
    {
        __cpu_mask l = *p++;

        if (l == 0)
        {
            continue;
        }

        for (i=0; i< (sizeof(__cpu_mask)*8); i++)
        {
            if (l&(1UL<<i))
            {
            s++;
            }
        }
    }

    return s;
}

/* #####   FUNCTION DEFINITIONS  -  EXPORTED FUNCTIONS   ################## */
int str2int(const char* str)
{
    char* endptr;
    errno = 0;
    unsigned long val;
    val = strtoul(str, &endptr, 10);
    if ((errno == ERANGE && val == LONG_MAX )
        || (errno != 0 && val == 0))
    {
        ERROR;
    }

    if (endptr == str)
    {
        ERROR_PRINT(Cannot parse string %s to digits, str);
    }

    return (int) val;
}

uint32_t
bstr_to_cpuset_physical(uint32_t* threads,  const_bstring q)
{
    int i;
    unsigned int rangeBegin;
    unsigned int rangeEnd;
    uint32_t numThreads=0;
    struct bstrList* tokens;
    struct bstrList* subtokens;

    tokens = bsplit(q,',');

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],'-');

        if( subtokens->qty == 1 )
        {
            threads[numThreads] = str2int((char *) bdata(subtokens->entry[0]));
            numThreads++;
        }
        else if ( subtokens->qty == 2 )
        {
            rangeBegin = str2int((char*) bdata(subtokens->entry[0]));
            rangeEnd = str2int((char*) bdata(subtokens->entry[1]));

            if (!(rangeBegin <= rangeEnd))
            {
                ERROR_PRINT(Range End %d bigger than begin %d, rangeEnd, rangeBegin);
            }

            while (rangeBegin <= rangeEnd) {
                threads[numThreads] = rangeBegin;
                numThreads++;
                rangeBegin++;
            }
        }
        else
        {
            ERROR_PLAIN_PRINT(Parse Error);
        }
        bstrListDestroy(subtokens);
    }
    if (numThreads > MAX_NUM_THREADS)
    {
        ERROR_PRINT(Number Of threads %d too large, numThreads);
    }

    bstrListDestroy(tokens);

    return numThreads;
}

uint32_t
bstr_to_cpuset_logical(uint32_t* threads,  const_bstring q)
{
    int i;
    uint32_t j;
    int id;
    uint32_t tmpThreads[MAX_NUM_THREADS];
    int globalNumThreads=0;
    uint32_t numThreads=0;
    struct bstrList* tokens;
    struct bstrList* subtokens;
    const AffinityDomain* domain;

    tokens = bsplit(q,'@');

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],':');

        if ( subtokens->qty == 2 )
        {
            domain =  affinity_getDomain(subtokens->entry[0]);

            if (!domain)
            {
                ERROR_PRINT(Unknown domain ##%s##,bdata(subtokens->entry[0]));
            }

            numThreads = bstr_to_cpuset_physical(tmpThreads, subtokens->entry[1]);

            for (j=0; j<numThreads; j++)
                {
                if (! (tmpThreads[j] >= domain->numberOfProcessors))
                {
                    id = (tmpThreads[j]/domain->numberOfCores) +
                        (tmpThreads[j]%domain->numberOfCores) * cpuid_topology.numThreadsPerCore;
                    threads[globalNumThreads++] = domain->processorList[id];
                }
                else
                {
                    ERROR_PRINT(Too many threads requested. Avaialable 0-%d,domain->numberOfProcessors-1);
                }
            }
        }
        else
        {
            ERROR_PLAIN_PRINT(Parse Error);
        }
        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);

    return globalNumThreads;
}

#define PRINT_EXPR_ERR printf("SYNTAX ERROR: Expression must have the format E:<thread domain>:<num threads>[:chunk size>:<stride>]\n")

uint32_t
bstr_to_cpuset_expression(uint32_t* threads,  const_bstring qi)
{
    int i;
    uint32_t j;
    bstring q = (bstring) qi;
    int globalNumThreads=0;
    uint32_t numThreads=0;
    struct bstrList* tokens;
    struct bstrList* subtokens;
    const AffinityDomain* domain;

    bdelete (q, 0, 2);
    tokens = bsplit(q,'@');

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],':');

        if ( subtokens->qty == 2 )
        {
            domain =  affinity_getDomain(subtokens->entry[0]);

            if (!domain)
            {
                ERROR_PRINT(Unknown domain ##%s##,bdata(subtokens->entry[0]));
            }

            numThreads = str2int(bdata(subtokens->entry[1]));

            if (numThreads > domain->numberOfProcessors)
            {
                ERROR_PRINT(Invalid processor id requested. Avaialable 0-%d,
                            domain->numberOfProcessors-1);
            }

            for (j=0; j<numThreads; j++)
            {
                threads[globalNumThreads++] = domain->processorList[j];
            }
        }
        else if ( subtokens->qty == 4 )
        {
            int counter;
            int currentId = 0;
            int startId = 0;
            int chunksize =  str2int(bdata(subtokens->entry[2]));
            int stride =  str2int(bdata(subtokens->entry[3]));
            domain = affinity_getDomain(subtokens->entry[0]);

            if (!domain)
            {
                ERROR_PRINT(Unknown domain ##%s##,bdata(subtokens->entry[0]));
            }

            numThreads = str2int(bdata(subtokens->entry[1]));

            if (numThreads > domain->numberOfProcessors)
            {
                ERROR_PRINT(Invalid number of processors requested. Available 0-%d,
                            domain->numberOfProcessors-1);
            }


            counter = 0;
            for (j=0; j<numThreads; j+=chunksize)
            {
                for(i=0;i<chunksize && j+i<numThreads ;i++)
                {
                    threads[globalNumThreads++] = domain->processorList[counter+i];
                }
                counter += stride;
                if (counter >= domain->numberOfProcessors)
                {
                    counter = 0;
                }
            }
        }
        else
        {
            PRINT_EXPR_ERR;
            ERROR_PLAIN_PRINT(Parse Error);
        }
        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);

    return globalNumThreads;
}

uint32_t
bstr_to_cpuset_scatter(uint32_t* threads,  const_bstring qi)
{
    int domainId = 0;
    int id = 0;
    int threadId = 0;
    bstring q = (bstring) qi;
    bstring domaintag;
    int globalNumThreads=0;
    struct bstrList* subtokens;
    int numberOfDomains = 0;
    AffinityDomain* domain;
    AffinityDomain* tmpDomainPtr;

    domain = (AffinityDomain*) malloc(cpuid_topology.numHWThreads * sizeof(AffinityDomain));

    subtokens = bsplit(q,':');

    if ( subtokens->qty == 2 )
    {
        for(int i =0;;i++)
        {
            domaintag = bformat("%s%d",bdata(subtokens->entry[0]),i);
            tmpDomainPtr = (AffinityDomain*) affinity_getDomain(domaintag);

            if (tmpDomainPtr == NULL)
            {
                break;
            }
            else
            {
                memcpy(domain+i,tmpDomainPtr,sizeof(AffinityDomain));
                numberOfDomains++;
            }
        }

        threads[globalNumThreads++] = domain[domainId].processorList[0];

        for (uint32_t i=1; i<cpuid_topology.numHWThreads; i++)
        {
            domainId = i%numberOfDomains;

            if (domainId == 0)
            {
                threadId++;
            }

            id = (threadId/domain->numberOfCores) +
                (threadId%domain->numberOfCores) * cpuid_topology.numThreadsPerCore;

            threads[globalNumThreads++] = domain[domainId].processorList[id];
        }
    }
    else
    {
        PRINT_EXPR_ERR;
        ERROR_PLAIN_PRINT(Parse Error);
    }
    bstrListDestroy(subtokens);
    free(domain);

    return globalNumThreads;
}



#define CPUSET_ERROR  \
    if (cpuid_isInCpuset()) {  \
        ERROR_PLAIN_PRINT(You are running inside a cpuset. In cpusets only logical pinning inside set is allowed!);  \
    }



int
bstr_to_cpuset(int* threadsIN,  const_bstring q)
{
    uint32_t i;
    int num=0;
    int cpuMapping[cpuid_topology.numHWThreads];
    cpu_set_t cpu_set;
    uint32_t numThreads;
    bstring domainStr = bformat("NSCM");
    const_bstring  scatter = bformat("scatter");
    struct bstrList* tokens;
    CPU_ZERO(&cpu_set);
    sched_getaffinity(0,sizeof(cpu_set_t), &cpu_set);
    uint32_t* threads = (uint32_t*) threadsIN;

    if (binchr (q, 0, domainStr) !=  BSTR_ERR)
    {
        CPUSET_ERROR;

        if (binstr (q, 0 , scatter ) !=  BSTR_ERR)
        {
          numThreads =  bstr_to_cpuset_scatter(threads,q);
        }
        else if (bstrchr (q, 'E') !=  BSTR_ERR)
        {
          numThreads =  bstr_to_cpuset_expression(threads,q);
        }
        else
        {
          numThreads =  bstr_to_cpuset_logical(threads,q);
        }
    }
    else if (bstrchr (q, 'L') !=  BSTR_ERR)
    {
        uint32_t count = cpu_count(&cpu_set);

        tokens = bsplit(q,':');
        numThreads = bstr_to_cpuset_physical(threads,tokens->entry[1]);

        for (i=0; i <  cpuid_topology.numHWThreads; i++)
        {
            if (CPU_ISSET(i,&cpu_set))
            {
                cpuMapping[num++]=i;
            }
        }

        for (i=0; i < numThreads; i++)
        {
            if (!(threads[i] >= count))
            {
                threads[i] = cpuMapping[threads[i]];
            }
            else
            {
                fprintf(stderr, "Available CPUs: ");
                for (int j=0; j< num-1;j++)
                {
                    fprintf(stderr, "%d,", cpuMapping[j]);
                }
                fprintf(stderr, "%d\n", cpuMapping[num-1]);
                ERROR_PRINT(Index %d out of range.,threads[i]);
            }
        }
        bstrListDestroy(tokens);
    }
    else
    {
        CPUSET_ERROR;
        numThreads = bstr_to_cpuset_physical(threads,q);
    }

    bdestroy(domainStr);
    return (int) numThreads;
}


void
bstr_to_eventset(StrUtilEventSet* set, const_bstring q)
{
    int i;
    struct bstrList* tokens;
    struct bstrList* subtokens;

    tokens = bsplit(q,',');
    set->numberOfEvents = tokens->qty;
    set->events = (StrUtilEvent*)
    malloc(set->numberOfEvents * sizeof(StrUtilEvent));

    for (i=0;i<tokens->qty;i++)
    {
        subtokens = bsplit(tokens->entry[i],':');

        if ( subtokens->qty != 2 )
        {
          
            fprintf(stderr, "Cannot parse event string %s, probably missing counter name\n"
                          ,bdata(tokens->entry[i]));
            fprintf(stderr, "Format: <eventName>:<counter>,...\n");
            msr_finalize();
            pci_finalize();
            exit(EXIT_FAILURE);

        }
        else
        {
            set->events[i].eventName = bstrcpy(subtokens->entry[0]);
            set->events[i].counterName = bstrcpy(subtokens->entry[1]);
        }

        bstrListDestroy(subtokens);
    }

    bstrListDestroy(tokens);
}

FILE*
bstr_to_outstream(const_bstring argString, bstring filter)
{
    int i;
    char* cstr;
    FILE* STREAM;
    struct bstrList* tokens;
    bstring base;
    bstring suffix = bfromcstr(".");
    bstring filename;

    /* configure filter */
    tokens = bsplit(argString,'.');

    if (tokens->qty < 2)
    {
        fprintf(stderr, "Outputfile has no filetype suffix!\n");
        fprintf(stderr, "Add suffix .txt for raw output or any supported filter suffix.\n");
        exit(EXIT_FAILURE);
    }

    base = bstrcpy(tokens->entry[0]);

    if (biseqcstr(tokens->entry[1],"txt"))
    {
        bassigncstr(filter, "NO");
    }
    else
    {
        bassigncstr(filter, TOSTRING(LIKWIDFILTERPATH));
        bconchar(filter,'/');
        bconcat(filter,tokens->entry[1]);
    }

    bconcat(suffix,tokens->entry[1]);
    bstrListDestroy(tokens);

    tokens = bsplit(base,'_');

    if (tokens->qty < 1)
    {
        ERROR_PLAIN_PRINT(Error in parsing file string);
    }

    filename = bstrcpy(tokens->entry[0]);

    for (i=1; i<tokens->qty; i++)
    {
        if (biseqcstr(tokens->entry[i],"%j"))
        {
            cstr = getenv("PBS_JOBID");
            if (cstr != NULL) 
            {
                bcatcstr(filename, "_");
                bcatcstr(filename, cstr);
            }
        }
        else if (biseqcstr(tokens->entry[i],"%r"))
        {
            cstr = getenv("PMI_RANK");
            if (cstr == NULL) 
            {
                cstr = getenv("OMPI_COMM_WORLD_RANK");
            }
            if (cstr != NULL) 
            {
                bcatcstr(filename, "_");
                bcatcstr(filename, cstr);
            }
        }
        else if (biseqcstr(tokens->entry[i],"%h"))
        {
            cstr = (char*) malloc(HOST_NAME_MAX * sizeof(char));
            gethostname(cstr,HOST_NAME_MAX);
            bcatcstr(filename, "_");
            bcatcstr(filename, cstr);
            free(cstr);
        }
        else if (biseqcstr(tokens->entry[i],"%p"))
        {
            bstring pid = bformat("_%d",getpid());
            bconcat(filename, pid);
            bdestroy(pid);
        }
        else 
        {
            ERROR_PLAIN_PRINT(Unsupported placeholder in filename!);
        }
    }

    if (biseqcstr(filter,"NO"))
    {
        bconcat(filename, suffix);
    }
    else
    {
        bcatcstr(filter, " ");
        bcatcstr(filename, ".tmp");
        bconcat(filter, filename);
    }

    bstrListDestroy(tokens);
    STREAM = fopen(bdata(filename),"w");
    bdestroy(filename);
    bdestroy(suffix);
    bdestroy(base);

    return STREAM;
}


uint64_t
bstr_to_doubleSize(const_bstring str, DataType type)
{
    bstring unit = bmidstr(str, blength(str)-2, 2);
    bstring sizeStr = bmidstr(str, 0, blength(str)-2);
    uint64_t sizeU = str2int(bdata(sizeStr));
    uint64_t junk = 0;
    uint64_t bytesize = 0;

    switch (type)
    {
        case SINGLE:
        case SINGLE_RAND:
            bytesize = sizeof(float);
            break;

        case DOUBLE:
        case DOUBLE_RAND:
            bytesize = sizeof(double);
            break;
    }

    if (biseqcstr(unit, "kB")) {
        junk = (sizeU *1024)/bytesize;
    } else if (biseqcstr(unit, "MB")) {
        junk = (sizeU *1024*1024)/bytesize;
    } else if (biseqcstr(unit, "GB")) {
        junk = (sizeU *1024*1024*1024)/bytesize;
    }

    return junk;
}

void
bstr_to_interval(const_bstring str, struct timespec* interval)
{
    int size;
    int pos;
    bstring ms = bformat("ms");

    if ((pos = bstrrchr (str, 's')) != BSTR_ERR)
    {
        if (pos != (blength(str)-1))
        {
            fprintf(stderr, "You need to specify a time unit s or ms like 200ms\n");
            msr_finalize();
            exit(EXIT_FAILURE);
        }

        /* unit is ms */
        if (binstrr (str, blength(str), ms) != BSTR_ERR)
        {
            bstring sizeStr = bmidstr(str, 0, blength(str)-2);
            size = str2int(bdata(sizeStr));
            if (size >= 1000)
            {
                interval->tv_sec = size/1000;
                interval->tv_nsec = (size%1000) * 1.E06;
            }
            else
            {
                interval->tv_sec = 0L;
                interval->tv_nsec = size * 1.E06;
            }
        }
        /* unit is s */
        else 
        {
            bstring sizeStr = bmidstr(str, 0, blength(str)-1);
            size = str2int(bdata(sizeStr));
            interval->tv_sec = size;
            interval->tv_nsec = 0L;
        }
    }
    else
    {
        fprintf(stderr, "You need to specify a time unit s or ms like 200ms\n");
        msr_finalize();
        exit(EXIT_FAILURE);
    }
}


void
bstr_to_workgroup(Workgroup* group,
    const_bstring str,
    DataType type,
    int numberOfStreams)
{
    uint32_t i;
    int parseStreams = 0;
    bstring threadInfo;
    bstring streams= bformat("0");
    struct bstrList* tokens;
    struct bstrList* subtokens;
    const AffinityDomain* domain;

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
        ERROR_PLAIN_PRINT(Error in parsing workgroup string);
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

        domain = affinity_getDomain(tokens->entry[0]);

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
        maxNumThreads = (domain->numberOfProcessors / stride) * chunksize;

        if (group->numberOfThreads > maxNumThreads)
        {
          fprintf(stderr, "Error: Domain %s supports only up to %d threads with used expression.\n",
                        bdata(tokens->entry[0]), maxNumThreads);
          exit(EXIT_FAILURE);
        }

        group->processorIds = (int*) malloc(group->numberOfThreads * sizeof(int));

        counter = chunksize;

        for (i=0; i<group->numberOfThreads; i++)
        {
            if (counter)
            {
                group->processorIds[i] = domain->processorList[currentId++];
            }
            else
            {
                startId += stride;
                currentId = startId;
                group->processorIds[i] = domain->processorList[currentId++];
                counter = chunksize;
            }
            counter--;
        }
    }
    else if (tokens->qty == 3)
    {
        domain = affinity_getDomain(tokens->entry[0]);

        if (domain == NULL)
        {
            fprintf(stderr, "Error: Domain %s not available on current machine.\n", bdata(tokens->entry[0]));
            fprintf(stderr, "Try likwid-bench -p for supported domains.\n");
            exit(EXIT_FAILURE);
        }

        group->size = bstr_to_doubleSize(tokens->entry[1], type);
        group->numberOfThreads = str2int(bdata(tokens->entry[2]));

        if (group->numberOfThreads > domain->numberOfProcessors)
        {
            fprintf(stderr, "Error: Domain %s supports only up to %d threads.\n",
                            bdata(tokens->entry[0]),domain->numberOfProcessors);
            exit(EXIT_FAILURE);
        }

        group->processorIds = (int*) malloc(group->numberOfThreads * sizeof(int));

        for (i=0; i<group->numberOfThreads; i++)
        {
            group->processorIds[i] = domain->processorList[i];
        }
    }
    else if (tokens->qty == 2)
    {
        domain = affinity_getDomain(tokens->entry[0]);

        if (domain == NULL)
        {
            fprintf(stderr, "Error: Domain %s not available on current machine.\n",
                            bdata(tokens->entry[0]));
            fprintf(stderr, "Try likwid-bench -p for supported domains.\n");
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
    ERROR_PLAIN_PRINT(Error in parsing workgroup string);
    }

    bstrListDestroy(tokens);

    /* parse stream list */
    if (parseStreams)
    {
        tokens = bsplit(streams,',');

        if (tokens->qty < numberOfStreams)
        {
            ERROR_PRINT(Testcase requires at least %d streams, numberOfStreams);
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
                    ERROR_PRINT(Stream Index %d out of range,index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = str2int(bdata(subtokens->entry[2]));
            }
            else if ( subtokens->qty == 2 )
            {
                int index = str2int(bdata(subtokens->entry[0]));
                if (index >= numberOfStreams)
                {
                    ERROR_PRINT(Stream Index %d out of range,index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = 0;
            }
            else
            {
                ERROR_PLAIN_PRINT(Error in parsing event string);
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
}


#define INIT_SECURE_INPUT_LENGTH 256

bstring
bSecureInput (int maxlen, char* vgcCtx) {
    int i, m, c = 1;
    bstring b, t;
    int termchar = 0;

    if (!vgcCtx) return NULL;

    b = bfromcstralloc (INIT_SECURE_INPUT_LENGTH, "");

    for (i=0; ; i++)
    {
        if (termchar == c)
        {
            break;
        }
        else if ((maxlen > 0) && (i >= maxlen))
        {
            b = NULL;
            return b;
        }
        else
        {
            c = *(vgcCtx++);
        }

        if (EOF == c)
        {
            break;
        }

        if (i+1 >= b->mlen) {

            /* Double size, but deal with unusual case of numeric
             overflows */

            if ((m = b->mlen << 1)   <= b->mlen &&
                (m = b->mlen + 1024) <= b->mlen &&
                (m = b->mlen + 16)   <= b->mlen &&
                (m = b->mlen + 1)    <= b->mlen)
            {
                t = NULL;
            }
            else
            {
                t = bfromcstralloc (m, "");
            }

            if (t)
            {
                memcpy (t->data, b->data, i);
            }

            bdestroy (b); /* Clean previous buffer */
            b = t;
            if (!b)
            {
                return b;
            }
        }

        b->data[i] = (unsigned char) c;
    }

    i--;
    b->slen = i;
    b->data[i] = (unsigned char) '\0';
    return b;
}


int
bJustifyCenter (bstring b, int width) 
{
    unsigned char space  = ' ';
    int alignSpace = (width - b->slen) / 2;
    int restSpace = (width - b->slen) % 2;
    if (width <= 0) return -__LINE__;

    if (b->slen <= width)
    {
        binsertch (b, 0, alignSpace, space);
    }

    binsertch (b, b->slen , alignSpace+restSpace, space);

    return BSTR_OK;
}


