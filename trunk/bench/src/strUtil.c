
#include <strUtil.h>

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

    if (biseqcstr(unit, "kB"))
    {
        junk = (sizeU *1024)/bytesize;
    }
    else if (biseqcstr(unit, "MB"))
    {
        junk = (sizeU *1024*1024)/bytesize;
    }
    else if (biseqcstr(unit, "GB"))
    {
        junk = (sizeU *1024*1024*1024)/bytesize;
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
    AffinityDomain* domain;

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
            fprintf(stderr, "Error: Domain %s supports only up to %d threads.\n",
                    bdata(tokens->entry[0]), domain->numberOfProcessors);
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
            fprintf(stderr, "Testcase requires at least %d streams\n", numberOfStreams);
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
                    fprintf(stderr, "Stream Index %d out of range\n",index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = str2int(bdata(subtokens->entry[2]));
            }
            else if ( subtokens->qty == 2 )
            {
                int index = str2int(bdata(subtokens->entry[0]));
                if (index >= numberOfStreams)
                {
                    fprintf(stderr, "Stream Index %d out of range\n",index);
                }
                group->streams[index].domain = bstrcpy(subtokens->entry[1]);
                group->streams[index].offset = 0;
            }
            else
            {
                fprintf(stderr, "Error in parsing event string\n");
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
